"""Fetch, preprocess, and load Wakeman–Henson ``ds000117`` MEG data.

The module turns raw BIDS recordings into whitened ``obs × channel × time``
containers for the trajectory tutorials. Expensive Maxwell/ICA preprocessing is
persisted as epochs plus one empty-room covariance per participant.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.io.dataset import BIDSDataset
from coco_pipe.io.structures import DataContainer

DATASET_ID = "ds000117"
DATASET_SNAPSHOT = "1.0.6"
TASK = "facerecognition"
SESSION = "meg"
DATATYPE = "meg"
RUNS: tuple[str, ...] = ("01", "02", "03", "04", "05", "06")

# Elekta fine-calibration and crosstalk correction, shipped inside the dataset's
# derivatives tree rather than at the root. Maxwell filtering needs both, so the
# fetch selectors name them explicitly even though everything else under
# ``derivatives/`` is excluded.
CALIBRATION_FILE = "derivatives/meg_derivatives/sss_cal.dat"
CROSSTALK_FILE = "derivatives/meg_derivatives/ct_sparse.fif"

# ds000117 records EOG and ECG on three of its EEG channels but types them as
# EEG, so picking "eog"/"ecg" silently drops them and ICA has nothing to
# correlate artifacts against. They have to be retyped before anything else.
EOG_ECG_CHANNELS: dict[str, str] = {"EEG061": "eog", "EEG062": "eog", "EEG063": "ecg"}

LABEL_NAMES: dict[int, str] = {1: "Famous", 2: "Unfamiliar", 3: "Scrambled"}
BIDS_NAMES: dict[int, str] = {label: name.lower() for label, name in LABEL_NAMES.items()}
DEFAULT_CONDITIONS: tuple[int, ...] = tuple(LABEL_NAMES)

# Built-in Neuromag VectorView sensor-position selections provided by MNE.
# These describe helmet regions, not cortical source-space ROIs.
MEG_SENSOR_SETS: dict[str, tuple[str, ...] | None] = {
    "all_sensors": None,
    "sensors_occipital": ("Left-occipital", "Right-occipital"),
    "sensors_temporal": ("Left-temporal", "Right-temporal"),
    "sensors_occipito_temporal": (
        "Left-occipital",
        "Right-occipital",
        "Left-temporal",
        "Right-temporal",
    ),
    # Right-lateralised variants. The face response is right-dominant, so these
    # ask whether the effects survive in the hemisphere that carries them
    # rather than being diluted by the contralateral sensors.
    "sensors_right_temporal": ("Right-temporal",),
    "sensors_right_occipital": ("Right-occipital",),
    "sensors_right_occipito_temporal": ("Right-occipital", "Right-temporal"),
}

# ``ds000117`` events.tsv carries a ``stim_type`` string *and* a ``trigger``
# code; only the trigger distinguishes the three presentations of each face.
# Verified exhaustive over all 96 MEG runs of snapshot 1.0.6 (14,140 trials):
# no trigger outside this table occurs, and ``stim_type`` never disagrees.
TRIGGER_INFO: dict[int, dict[str, Any]] = {
    5: {"condition_id": 1, "repetition": "first"},
    6: {"condition_id": 1, "repetition": "immediate"},
    7: {"condition_id": 1, "repetition": "delayed"},
    13: {"condition_id": 2, "repetition": "first"},
    14: {"condition_id": 2, "repetition": "immediate"},
    15: {"condition_id": 2, "repetition": "delayed"},
    17: {"condition_id": 3, "repetition": "first"},
    18: {"condition_id": 3, "repetition": "immediate"},
    19: {"condition_id": 3, "repetition": "delayed"},
}


def _fetch_wakeman_henson(
    target_dir: str | Path,
    *,
    subjects: str | Sequence[str] | None = None,
    empty_room: bool = True,
    **download_kwargs: Any,
) -> Path:
    """Fetch the MEG subset of ``ds000117`` from OpenNeuro, snapshot-pinned.

    ``subjects=None`` pulls every participant — 85 GB. The selectors follow
    ``openneuro-py``'s gitignore-style matching, where a wildcard-free path also
    matches everything beneath it, so ``"sub-01/ses-meg"`` takes the whole MEG
    session and nothing else. The calibration and crosstalk files have to be
    named explicitly because they sit under ``derivatives/``, which is otherwise
    entirely excluded.
    """
    import openneuro

    if subjects is None:
        selectors = [f"sub-*/ses-{SESSION}"]
    else:
        if isinstance(subjects, str):
            subjects = [subjects]
        selectors = [f"sub-{str(s).removeprefix('sub-')}/ses-{SESSION}" for s in subjects]
    include = [*selectors, CALIBRATION_FILE, CROSSTALK_FILE]
    if empty_room:
        include.append("sub-emptyroom")

    target = Path(target_dir).expanduser()
    openneuro.download(
        dataset=DATASET_ID,
        tag=DATASET_SNAPSHOT,
        target_dir=target,
        include=include,
        **download_kwargs,
    )
    return target


def _discover_meg_subjects(
    bids_root: str | Path,
    *,
    pattern: str = "*_meg.fif",
) -> tuple[str, ...]:
    """Return subjects that contain a Wakeman–Henson MEG session.

    ``pattern`` selects which file makes a subject count: the default finds raw
    recordings in a fetched ``ds000117`` root, and ``"*_epo.fif"`` finds
    preprocessed subjects in a derivatives root.
    """
    root = Path(bids_root)
    if not root.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {root}")
    subjects = []
    for subject_dir in sorted(root.glob("sub-*")):
        if subject_dir.name == "sub-emptyroom":
            continue
        meg_dir = subject_dir / f"ses-{SESSION}" / DATATYPE
        if meg_dir.is_dir() and any(meg_dir.glob(pattern)):
            subjects.append(subject_dir.name.removeprefix("sub-"))
    return tuple(subjects)


def map_events(events: pd.DataFrame) -> pd.DataFrame:
    """Map raw ``ds000117`` event rows onto analysis conditions and repetitions.

    Adds ``condition`` (``Famous``/``Unfamiliar``/``Scrambled``), ``condition_id``
    (the integer code the epochs are written with) and ``repetition``
    (``first``/``immediate``/``delayed``). Repetition is kept for balancing and
    for the repetition-suppression diagnostics; it is deliberately *not* folded
    into the condition labels, so the three conditions stay comparable in trial
    count.

    Raises on any trigger outside :data:`TRIGGER_INFO`, and on any row where the
    dataset's own ``stim_type`` string disagrees with the trigger — an unmapped
    or mislabelled event must fail rather than quietly drop out of the analysis.
    """
    required = {"onset_sample", "stim_type", "trigger"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"ds000117 events are missing columns: {missing}")

    triggers = events["trigger"].astype(int)
    unknown = sorted(set(triggers) - set(TRIGGER_INFO))
    if unknown:
        raise ValueError(f"Unknown ds000117 triggers: {unknown}")

    mapped = events.copy()
    mapped["condition_id"] = [TRIGGER_INFO[t]["condition_id"] for t in triggers]
    mapped["condition"] = [LABEL_NAMES[c] for c in mapped["condition_id"]]
    mapped["repetition"] = [TRIGGER_INFO[t]["repetition"] for t in triggers]

    disagree = mapped["condition"] != mapped["stim_type"].astype(str)
    if disagree.any():
        rows = mapped.loc[disagree, ["trigger", "stim_type", "condition"]].head()
        raise ValueError(f"trigger and stim_type disagree:\n{rows}")
    return mapped


def preprocessing_config(**overrides: Any) -> dict[str, Any]:
    """Return options for the fixed Maxwell → ICA → epoch pipeline."""
    config = {
        "origin": (0.0, 0.0, 0.04),
        "destination": (0.0, 0.0, 0.04),
        "st_duration": None,
        "movement_compensation": True,
        "chpi_t_step_min": 0.1,
        "noise_cov_method": "shrunk",
        "ica_l_freq": 1.0,
        "ica_h_freq": 40.0,
        "n_ica_components": 40,
        "l_freq": 0.1,
        "h_freq": 40.0,
        "sfreq": 250.0,
        "tmin": -0.2,
        "tmax": 0.8,
        "baseline": (-0.2, 0.0),
        "reject": None,
        "reject_decim": 4,
        "random_state": 42,
    }
    unknown = sorted(set(overrides) - set(config))
    if unknown:
        raise TypeError(f"Unknown preprocessing options: {unknown}")
    config.update(overrides)
    return config


def epochs_path(derivatives_root: str | Path, subject: str) -> Path:
    """Location of a subject's preprocessed epochs inside a derivatives root."""
    return _derivative_path(derivatives_root, subject, "epo.fif")


def cov_path(derivatives_root: str | Path, subject: str) -> Path:
    """Location of a subject's empty-room noise covariance."""
    return _derivative_path(derivatives_root, subject, "cov.fif")


def _derivative_path(derivatives_root: str | Path, subject: str, suffix: str) -> Path:
    sub = str(subject).removeprefix("sub-")
    return (
        Path(derivatives_root)
        / f"sub-{sub}"
        / f"ses-{SESSION}"
        / DATATYPE
        / f"sub-{sub}_ses-{SESSION}_task-{TASK}_{suffix}"
    )


def _preprocess_subject(
    bids_root: str | Path,
    subject: str,
    *,
    derivatives_root: str | Path,
    runs: Sequence[str] = RUNS,
    config: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Run Maxwell, ICA, rejection, and empty-room covariance for one participant."""
    import mne
    from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos
    from mne.preprocessing import ICA, find_bad_channels_maxwell, maxwell_filter
    from mne_bids import BIDSPath, read_raw_bids

    config = dict(config) if config is not None else preprocessing_config()
    root = Path(bids_root)
    sub = str(subject).removeprefix("sub-")
    raw_reference = None
    destination = epochs_path(derivatives_root, sub)
    if destination.exists() and not overwrite:
        print(f"sub-{sub}: epochs already present, skipping → {destination}")
        return destination

    calibration = root / CALIBRATION_FILE
    cross_talk = root / CROSSTALK_FILE
    for path in (calibration, cross_talk):
        if not path.exists():
            raise FileNotFoundError(
                f"Maxwell filtering needs {path}. Re-run `load_wakeman_henson(..., "
                "prepare=True)` to complete the download."
            )

    n_events_total = 0
    epochs_per_run = []

    for run in runs:
        run = str(run)
        bids_path = BIDSPath(
            subject=sub,
            session=SESSION,
            task=TASK,
            run=run,
            datatype=DATATYPE,
            root=root,
            suffix=DATATYPE,
            extension=".fif",
        )
        with warnings.catch_warnings():
            # ds000117 predates the `trial_type` rename; mne-bids reads the
            # legacy `stim_type` column but says so once per run.
            warnings.simplefilter("ignore")
            raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data(verbose=False)
        raw.set_channel_types(EOG_ECG_CHANNELS, verbose=False)
        raw.pick(["meg", "eog", "ecg"], verbose=False)

        events_tsv = bids_path.copy().update(suffix="events", extension=".tsv").fpath
        table = map_events(pd.read_csv(events_tsv, sep="\t"))
        n_events_total += len(table)
        # events.tsv onsets are relative to the start of the recording; MNE
        # event samples are absolute, hence the `first_samp` offset.
        events = np.column_stack(
            [
                raw.first_samp + table["onset_sample"].to_numpy(int),
                np.zeros(len(table), int),
                table["condition_id"].to_numpy(int),
            ]
        )
        metadata = table[["condition", "condition_id", "repetition", "trigger", "stim_file"]].copy()
        metadata["run"] = run
        metadata = metadata.reset_index(drop=True)

        # 1 — bad and flat channels
        raw.info["bads"] = []
        noisy, flat = find_bad_channels_maxwell(
            raw,
            calibration=str(calibration),
            cross_talk=str(cross_talk),
            origin=config["origin"],
            coord_frame="head",
            verbose=False,
        )
        raw.info["bads"] = sorted(set(noisy) | set(flat))

        # The empty-room covariance has to inherit this run's sensor selection,
        # so keep the pre-Maxwell reference before the filter rewrites the info.
        if raw_reference is None:
            raw_reference = raw.copy().crop(tmax=min(30.0, raw.times[-1]))

        # 2 — head position, then Maxwell filter onto a common head position.
        # `destination` is what makes subjects comparable at all; `head_pos`
        # additionally removes within-run drift.
        head_pos = None
        if config["movement_compensation"]:
            amplitudes = compute_chpi_amplitudes(
                raw, t_step_min=config["chpi_t_step_min"], verbose=False
            )
            locs = compute_chpi_locs(raw.info, amplitudes, verbose=False)
            head_pos = compute_head_pos(raw.info, locs, verbose=False)

        raw = maxwell_filter(
            raw,
            calibration=str(calibration),
            cross_talk=str(cross_talk),
            st_duration=config["st_duration"],
            origin=config["origin"],
            destination=config["destination"],
            head_pos=head_pos,
            coord_frame="head",
            verbose=False,
        )
        raw.info["bads"] = []

        # 3 — ICA on a band-limited copy, applied to the analysis band below
        raw_for_ica = raw.copy().filter(
            l_freq=config["ica_l_freq"],
            h_freq=config["ica_h_freq"],
            fir_design="firwin",
            verbose=False,
        )
        ica = ICA(
            n_components=config["n_ica_components"],
            method="infomax",
            fit_params=dict(extended=True),
            random_state=config["random_state"],
            max_iter="auto",
            verbose=False,
        )
        ica.fit(raw_for_ica, picks="meg", verbose=False)
        eog_components, _ = ica.find_bads_eog(raw_for_ica, verbose=False)
        ecg_components, _ = ica.find_bads_ecg(raw_for_ica, method="ctps", verbose=False)
        ica.exclude = sorted(set(eog_components) | set(ecg_components))

        # 4 — analysis band, artifact subtraction, resample
        raw.filter(
            l_freq=config["l_freq"],
            h_freq=config["h_freq"],
            fir_design="firwin",
            verbose=False,
        )
        if ica.exclude:
            ica.apply(raw, verbose=False)
        raw, events = raw.resample(config["sfreq"], events=events, verbose=False)

        # 5 — epoch
        epochs_per_run.append(
            mne.Epochs(
                raw,
                events=events,
                event_id={BIDS_NAMES[c]: c for c in DEFAULT_CONDITIONS},
                tmin=config["tmin"],
                tmax=config["tmax"],
                baseline=config["baseline"],
                metadata=metadata,
                picks="meg",
                preload=True,
                verbose=False,
            )
        )

    epochs = mne.concatenate_epochs(epochs_per_run, verbose=False)

    # A fixed peak-to-peak threshold rejects a subject for being loud rather
    # than a trial for being anomalous: sub-06's *median* epoch exceeds the
    # conventional magnetometer limit, which would discard 85% of good data.
    # autoreject derives the threshold from each subject's own distribution.
    reject = config["reject"]
    if reject is None:
        from autoreject import get_rejection_threshold

        reject = get_rejection_threshold(
            epochs,
            decim=config["reject_decim"],
            random_state=config["random_state"],
            verbose=False,
        )
    epochs.drop_bad(reject=reject, verbose=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(destination, overwrite=True, verbose=False)

    _write_empty_room_covariance(
        root,
        sub,
        raw_reference=raw_reference,
        config=config,
        calibration=calibration,
        cross_talk=cross_talk,
        destination=cov_path(derivatives_root, sub),
    )

    print(
        f"sub-{sub}: {len(epochs)}/{n_events_total} epochs kept "
        f"→ {destination}"
    )
    return destination


def _write_empty_room_covariance(
    bids_root: Path,
    subject: str,
    *,
    raw_reference: Any,
    config: Mapping[str, Any],
    calibration: Path,
    cross_talk: Path,
    destination: Path,
) -> None:
    """Apply matching preprocessing and save the empty-room covariance."""
    import mne
    from mne.preprocessing import maxwell_filter, maxwell_filter_prepare_emptyroom
    from mne_bids import BIDSPath, read_raw_bids

    bids_path = BIDSPath(
        subject=subject,
        session=SESSION,
        task=TASK,
        run=RUNS[0],
        datatype=DATATYPE,
        root=bids_root,
        suffix=DATATYPE,
        extension=".fif",
    )
    empty_room_path = bids_path.find_empty_room(verbose=False)
    if empty_room_path is None:
        raise FileNotFoundError(
            f"No empty-room recording associated with sub-{subject}. Re-fetch with "
            "`empty_room=True`; the tutorial pipeline requires whitening."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_er = read_raw_bids(empty_room_path, verbose=False)
    raw_er.load_data(verbose=False)
    raw_er = maxwell_filter_prepare_emptyroom(
        raw_er, raw=raw_reference, bads="from_raw", annotations="keep", verbose=False
    )
    raw_er = maxwell_filter(
        raw_er,
        calibration=str(calibration),
        cross_talk=str(cross_talk),
        st_duration=config["st_duration"],
        coord_frame="meg",
        verbose=False,
    )
    raw_er.filter(
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
        fir_design="firwin",
        verbose=False,
    )
    raw_er.resample(config["sfreq"], verbose=False)

    cov = mne.compute_raw_covariance(
        raw_er, method=config["noise_cov_method"], rank="info", verbose=False
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    mne.write_cov(destination, cov, overwrite=True, verbose=False)


def _load_wakeman_henson_container(
    derivatives_root: str | Path,
    *,
    subjects: str | Sequence[str] | None = None,
    conditions: Sequence[int] = DEFAULT_CONDITIONS,
    notch_freq: float | None = None,
    sensor_set: str = "all_sensors",
) -> DataContainer:
    """Load preprocessed Wakeman–Henson epochs through ``coco-pipe``.

    Returns the same ``(obs, channel, time)`` container as
    :func:`~pca_neural_trajectories.eegbci.load_eegbci_container`, so the
    trajectory code downstream is unchanged. Epochs are read back as written by
    by preprocessing — the window, baseline and sampling rate are
    fixed at preprocessing time, not here.

    Per-trial ``condition``, ``repetition``, ``trigger``, ``stim_file`` and
    ``run`` are attached as observation-aligned coordinates, since
    ``BIDSDataset`` carries the epoch labels but not their metadata.

    ``sensor_set`` can retain all channels or an MNE VectorView helmet selection.
    The retained channels are whitened by each subject's restricted empty-room
    covariance, putting magnetometers and gradiometers on one footing before PCA.
    """
    try:
        vectorview_selections = MEG_SENSOR_SETS[sensor_set]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sensor_set {sensor_set!r}; choose {tuple(MEG_SENSOR_SETS)}."
        ) from exc

    unknown = sorted({int(c) for c in conditions} - set(BIDS_NAMES))
    if unknown:
        raise ValueError(f"Unknown Wakeman–Henson conditions: {unknown}")
    selected = [int(c) for c in conditions]

    if subjects is None:
        resolved = list(_discover_meg_subjects(derivatives_root, pattern="*_epo.fif"))
    elif isinstance(subjects, str):
        resolved = [subjects.removeprefix("sub-")]
    else:
        resolved = [str(s).removeprefix("sub-") for s in subjects]
    resolved = sorted(resolved)

    dataset = BIDSDataset(
        root=Path(derivatives_root),
        task=TASK,
        session=SESSION,
        datatype=DATATYPE,
        suffix="epo",
        mode="load_existing",
        event_id={BIDS_NAMES[c]: c for c in selected},
        subjects=resolved,
    )
    container = dataset.load()

    if vectorview_selections is not None:
        import mne

        info = mne.read_epochs(
            epochs_path(derivatives_root, resolved[0]),
            preload=False,
            verbose=False,
        ).info
        selected_names = set(
            mne.read_vectorview_selection(vectorview_selections, info=info)
        )
        available_names = np.asarray(container.coords["channel"]).astype(str)
        keep = np.asarray([name in selected_names for name in available_names])
        if not keep.any():
            raise RuntimeError(
                f"Sensor set {sensor_set!r} has no channels in the loaded data."
            )
        container = container.isel(channel=keep)

    # Restrict the signals before computing the subject-wise whitener. This keeps
    # excluded helmet locations from contributing to a regional analysis.
    _whiten_container(container, derivatives_root, resolved)
    container.meta["whitened"] = True
    container.meta["whitening"] = "subject-wise empty-room covariance"

    if notch_freq is not None:
        import mne

        times = np.asarray(container.coords["time"], dtype=float)
        sfreq = 1.0 / float(np.median(np.diff(times)))
        if not 0 < notch_freq < sfreq / 2:
            raise ValueError(
                f"notch_freq must lie between 0 and Nyquist ({sfreq / 2:g} Hz)."
            )
        container.X = mne.filter.notch_filter(
            np.asarray(container.X),
            Fs=sfreq,
            freqs=notch_freq,
            method="iir",
            verbose=False,
        )
        container.meta["notch_freq_hz"] = float(notch_freq)

    metadata = _trial_metadata(derivatives_root, resolved, conditions=selected)
    if len(metadata) != container.X.shape[0]:
        raise RuntimeError(
            f"Trial metadata ({len(metadata)} rows) does not match the loaded epochs "
            f"({container.X.shape[0]}). The derivatives root and the epochs files disagree."
        )
    for column in metadata.columns:
        container.coords[column] = metadata[column].to_numpy()

    container.meta["sensor_set"] = sensor_set
    container.meta["n_sensors"] = int(container.X.shape[1])
    return container


def _whiten_container(
    container: DataContainer,
    derivatives_root: str | Path,
    subjects: Sequence[str],
) -> None:
    """Whiten each subject's channels in place using its empty-room covariance.

    Per subject rather than pooled: SSS leaves each recording with its own rank,
    and the covariance is only valid for the sensor set it was estimated on.
    """
    import mne

    channels = np.asarray(container.coords["channel"]).astype(str)
    subject_ids = np.asarray(container.coords["subject"]).astype(str)
    for subject in subjects:
        path = cov_path(derivatives_root, subject)
        if not path.exists():
            raise FileNotFoundError(
                f"No noise covariance at {path}. Re-run `load_wakeman_henson(..., "
                "prepare=True)` with the same root."
            )
        cov = mne.read_cov(path, verbose=False)
        info = mne.read_epochs(epochs_path(derivatives_root, subject), preload=False).info
        n_available_meg = len(
            mne.pick_types(info, meg=True, eeg=False, ref_meg=False, exclude=[])
        )
        channel_indices = [info.ch_names.index(name) for name in channels]
        info = mne.pick_info(info, channel_indices, copy=True, verbose=False)
        cov = mne.pick_channels_cov(
            cov,
            include=channels.tolist(),
            exclude=[],
            ordered=True,
            copy=True,
            verbose=False,
        )
        rank = None
        if len(channels) < n_available_meg:
            # A small SSS sensor subset can be numerically rank deficient even
            # when it has fewer channels than the full-system SSS rank. Keep the
            # more conservative of the covariance and acquisition-info ranks.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Something went wrong in the data-driven estimation",
                )
                data_rank = mne.compute_rank(
                    cov,
                    info=info,
                    tol=1e-8,
                    tol_kind="relative",
                    on_rank_mismatch="ignore",
                    verbose=False,
                )["meg"]
            info_rank = mne.compute_rank(
                cov,
                rank="info",
                info=info,
                on_rank_mismatch="ignore",
                verbose=False,
            )["meg"]
            rank = {"meg": min(data_rank, info_rank)}

        whitener, ch_names = mne.cov.compute_whitener(
            cov, info, picks="meg", rank=rank, return_rank=False, verbose=False
        )
        # `compute_whitener` orders its output by `ch_names`; the container may
        # not, so reindex rather than assume.
        order = [ch_names.index(name) for name in channels]
        whitener = whitener[np.ix_(order, order)]

        mask = subject_ids == str(subject).removeprefix("sub-")
        if not mask.any():
            raise RuntimeError(f"sub-{subject} has a covariance but no trials in the container.")
        container.X[mask] = np.einsum("ij,njt->nit", whitener, container.X[mask])


def _trial_metadata(
    derivatives_root: str | Path,
    subjects: Sequence[str],
    *,
    conditions: Sequence[int] = DEFAULT_CONDITIONS,
) -> pd.DataFrame:
    """Read per-trial epoch metadata without loading the epoch data itself.

    Subjects are read in the given order and the condition filter mirrors the
    one ``BIDSDataset`` applies, so the rows line up with the loaded container.
    """
    import mne

    selected = {int(c) for c in conditions}
    frames = []
    for subject in subjects:
        path = epochs_path(derivatives_root, subject)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessed epochs not found: {path}")
        table = mne.read_epochs(path, preload=False, verbose=False).metadata
        frames.append(table[table["condition_id"].isin(selected)])
    return pd.concat(frames, ignore_index=True)


def load_wakeman_henson(
    root: str | Path = "data/wakeman_henson",
    *,
    subjects: str | Sequence[str] | None = None,
    prepare: bool = False,
    spectral: bool = False,
    conditions: Sequence[int] = DEFAULT_CONDITIONS,
    notch_freq: float | None = None,
    sensor_set: str = "all_sensors",
) -> DataContainer:
    """Return ready-to-analyse Wakeman–Henson MEG epochs.

    The first call can explicitly download and preprocess selected participants
    with ``prepare=True``. Later calls load the saved epochs and their empty-room
    covariance directly. ``spectral=True`` uses a separate padded 0.5--90 Hz
    derivative set for filtering and Hilbert envelopes. ``notch_freq`` applies
    an optional IIR notch after participant-specific whitening. ``sensor_set``
    defaults to all sensors and can select one of the named VectorView helmet
    regions in :data:`MEG_SENSOR_SETS`.
    """
    project_root = Path(root)
    raw_root = project_root / "raw"
    derivatives_root = project_root / (
        "derivatives_spectral" if spectral else "derivatives"
    )

    if prepare:
        if subjects is None:
            raise ValueError("Pass explicit subjects when prepare=True, for example ('01',).")
        selected = [subjects] if isinstance(subjects, str) else list(subjects)
        _fetch_wakeman_henson(raw_root, subjects=selected)
        config = preprocessing_config()
        if spectral:
            config.update(
                h_freq=90.0,
                tmin=-1.2,
                tmax=1.8,
                baseline=None,
            )
        for subject in selected:
            _preprocess_subject(
                raw_root,
                subject,
                derivatives_root=derivatives_root,
                config=config,
            )
    elif not derivatives_root.exists():
        raise FileNotFoundError(
            f"Prepared MEG data not found at {derivatives_root}. Run "
            "load_wakeman_henson(root, subjects=('01',), prepare=True"
            f", spectral={spectral}) once."
        )

    return _load_wakeman_henson_container(
        derivatives_root,
        subjects=subjects,
        conditions=conditions,
        notch_freq=notch_freq,
        sensor_set=sensor_set,
    )


__all__ = [
    "BIDS_NAMES",
    "DEFAULT_CONDITIONS",
    "LABEL_NAMES",
    "MEG_SENSOR_SETS",
    "RUNS",
    "load_wakeman_henson",
    "preprocessing_config",
]
