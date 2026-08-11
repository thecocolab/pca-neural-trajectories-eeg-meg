"""PhysioNet EEGBCI metadata, preprocessing, and loading helpers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from coco_pipe.io.dataset import BIDSDataset
from coco_pipe.io.structures import DataContainer

LABEL_INFO: dict[int, dict[str, str]] = {
    3: {
        "name": "Left Hand (Exec)",
        "mode": "exec",
        "effector": "left_hand",
        "laterality": "unilateral",
    },
    4: {
        "name": "Right Hand (Exec)",
        "mode": "exec",
        "effector": "right_hand",
        "laterality": "unilateral",
    },
    5: {
        "name": "Left Hand (Imag)",
        "mode": "imag",
        "effector": "left_hand",
        "laterality": "unilateral",
    },
    6: {
        "name": "Right Hand (Imag)",
        "mode": "imag",
        "effector": "right_hand",
        "laterality": "unilateral",
    },
    7: {
        "name": "Hands (Exec)",
        "mode": "exec",
        "effector": "hands",
        "laterality": "bilateral",
    },
    8: {
        "name": "Feet (Exec)",
        "mode": "exec",
        "effector": "feet",
        "laterality": "bilateral",
    },
    9: {
        "name": "Hands (Imag)",
        "mode": "imag",
        "effector": "hands",
        "laterality": "bilateral",
    },
    10: {
        "name": "Feet (Imag)",
        "mode": "imag",
        "effector": "feet",
        "laterality": "bilateral",
    },
}

LABEL_NAMES: dict[int, str] = {label: metadata["name"] for label, metadata in LABEL_INFO.items()}

BIDS_NAMES: dict[int, str] = {
    label: f"{metadata['effector']}_{metadata['mode']}" for label, metadata in LABEL_INFO.items()
}

ALL_CONDITIONS: tuple[int, ...] = tuple(LABEL_INFO)
DEFAULT_CONDITIONS: tuple[int, ...] = ALL_CONDITIONS

# ICLabel's own class names for the components we subtract. These strings must
# match ``mne_icalabel.config.ICALABEL_METHODS_NUMERICAL_TO_STRING['iclabel']``.
ARTIFACT_IC_LABELS: tuple[str, ...] = ("eye blink", "heart beat")


def setup_data_bids(
    subjects: int | Sequence[int] = 1,
    runs: Sequence[int] = tuple(range(1, 15)),
    root: str | Path = "./PhysioNet_EEGBCI/BIDS",
    force_reconvert: bool = False,
    clean_artifacts: bool = True,
    n_ica_components: int = 20,
    ic_label_threshold: float = 0.7,
    data_path: str | Path | None = None,
) -> Path:
    """Download PhysioNet EEGBCI data, clean it, and convert it to BIDS.

    Idempotent: re-running on an already-converted BIDS root only re-processes
    runs whose BrainVision file is missing (unless ``force_reconvert=True``).

    Cleaning pipeline (paper §6 / §7.2 ─ PCA amplifies whatever variance is
    in the data, so artifacts must be controlled *before* the trajectory
    analysis stage):

    1. Detect and interpolate bad channels (LOF) before referencing so a bad
       sensor cannot leak into the common-average reference or the ICA.
    2. Common-average reference.
    3. Analysis band-pass filter 0.5-40 Hz (FIR); the low-pass also removes
       residual muscle and line noise, so no separate notch is needed.
    4. Optional ICA artifact removal via ``mne-icalabel``: fits an extended
       infomax ``ICA`` (``n_ica_components`` components) on a 1 Hz high-passed
       copy -- the algorithm and band ICLabel was trained on -- and subtracts
       the components labelled ``eye blink`` or ``heart beat`` at a confidence
       of at least ``ic_label_threshold``. The unmixing matrix is a spatial
       operator, so fitting on the 1 Hz copy and applying to the band-passed
       recording is valid as long as both share channels and reference.

    Parameters
    ----------
    subjects : int or sequence of int
    runs : sequence of int
    root : str or Path
    force_reconvert : bool
        Reprocess runs whose BIDS output already exists on disk.
    clean_artifacts : bool, default True
        Whether to run ICA-based cleaning. Disable for a "raw" baseline
        comparison.
    n_ica_components : int, default 20
        Number of ICA components to fit. Lower for speed, higher for finer
        artifact isolation.
    ic_label_threshold : float, default 0.7
        Minimum ICLabel confidence required to subtract a component.
    data_path : str or Path, optional
        Where the raw PhysioNet download lives. Defaults to ``~/mne_data``; point
        it at external storage when the internal disk is tight.

    Returns
    -------
    Path
        Resolved BIDS root directory.
    """
    import mne
    from mne.preprocessing import ICA
    from mne_bids import BIDSPath, write_raw_bids
    from mne_icalabel import label_components

    if isinstance(subjects, int):
        subjects = [subjects]
    mne_data_path = Path(data_path).expanduser() if data_path else Path.home() / "mne_data"
    bids_root = Path(root)
    bids_root.mkdir(parents=True, exist_ok=True)
    print(f"EEGBCI: subjects={list(subjects)}, runs={list(runs)}, root={bids_root}")

    # Mapping from run number → MNE annotation → coco-pipe event id.
    # Run 1 is eyes-open rest; run 2 is eyes-closed rest; runs 3–14 alternate
    # 4 motor-imagery paradigms.
    annotation_maps = {
        1: ({"T0": "rest_eyes_open"}, {"rest_eyes_open": 0}),
        2: ({"T0": "rest_eyes_closed"}, {"rest_eyes_closed": 1}),
    }
    for r in (3, 7, 11):
        annotation_maps[r] = (
            {"T0": "rest", "T1": "left_hand_exec", "T2": "right_hand_exec"},
            {"rest": 2, "left_hand_exec": 3, "right_hand_exec": 4},
        )
    for r in (4, 8, 12):
        annotation_maps[r] = (
            {"T0": "rest", "T1": "left_hand_imag", "T2": "right_hand_imag"},
            {"rest": 2, "left_hand_imag": 5, "right_hand_imag": 6},
        )
    for r in (5, 9, 13):
        annotation_maps[r] = (
            {"T0": "rest", "T1": "hands_exec", "T2": "feet_exec"},
            {"rest": 2, "hands_exec": 7, "feet_exec": 8},
        )
    for r in (6, 10, 14):
        annotation_maps[r] = (
            {"T0": "rest", "T1": "hands_imag", "T2": "feet_imag"},
            {"rest": 2, "hands_imag": 9, "feet_imag": 10},
        )

    for sub_id in subjects:
        raw_fnames = mne.datasets.eegbci.load_data(sub_id, runs, path=mne_data_path)
        for fname in raw_fnames:
            stem = Path(fname).stem
            run_num = int(stem.split("R")[-1])
            bids_path = BIDSPath(
                subject=f"{sub_id:03d}",
                session="01",
                task="motorimagery",
                run=f"{run_num:02d}",
                datatype="eeg",
                root=bids_root,
                suffix="eeg",
            )
            if not force_reconvert and bids_path.copy().update(extension=".vhdr").fpath.exists():
                continue

            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage(mne.channels.make_standard_montage("standard_1005"), verbose=False)
            raw.pick_types(eeg=True, verbose=False)

            # Interpolate bad channels before referencing/ICA so a bad sensor
            # cannot contaminate the common-average reference or the ICA.
            bads = mne.preprocessing.find_bad_channels_lof(raw, verbose=False)
            if bads:
                raw.info["bads"] = bads
                raw.interpolate_bads(reset_bads=True, verbose=False)

            raw.set_eeg_reference("average", projection=False, verbose=False)

            if clean_artifacts:
                # ICLabel was trained on extended-infomax ICA over data
                # high-passed at 1 Hz, so fit on a dedicated 1 Hz copy. The
                # upper edge is bounded by the ~80 Hz Nyquist of 160 Hz EEGBCI
                # data (below ICLabel's ideal 100 Hz), but the high-pass and
                # algorithm are what matter for reliable labels.
                raw_for_ica = raw.copy().filter(
                    l_freq=1.0, h_freq=None, fir_design="firwin", verbose=False
                )
                ica = ICA(
                    n_components=min(n_ica_components, len(raw.ch_names) - 1),
                    method="infomax",
                    fit_params=dict(extended=True),
                    random_state=42,
                    max_iter="auto",
                    verbose=False,
                )
                ica.fit(raw_for_ica, verbose=False)
                ic_labels = label_components(raw_for_ica, ica, method="iclabel")
                # Exclude confidently-labelled ocular and cardiac ICs; muscle is
                # handled by the 40 Hz analysis low-pass applied below. The
                # label strings are ICLabel's own vocabulary — matching anything
                # else silently excludes nothing at all.
                ica.exclude = [
                    i
                    for i, (label, prob) in enumerate(
                        zip(ic_labels["labels"], ic_labels["y_pred_proba"], strict=True)
                    )
                    if label in ARTIFACT_IC_LABELS and prob >= ic_label_threshold
                ]

            # Analysis band-pass, then subtract the labelled artifact ICs.
            raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin", verbose=False)
            if clean_artifacts and ica.exclude:
                raw = ica.apply(raw, verbose=False)

            mapping, event_id = annotation_maps.get(run_num, ({}, {}))
            existing = set(raw.annotations.description)
            mapping = {k: v for k, v in mapping.items() if k in existing}
            if mapping:
                raw.annotations.rename(mapping)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_raw_bids(
                    raw,
                    bids_path,
                    event_id=event_id,
                    overwrite=True,
                    allow_preload=True,
                    format="BrainVision",
                    verbose=False,
                )

    print(f"EEGBCI BIDS root: {bids_root}")
    return bids_root


def load_eegbci_container(
    bids_root: str | Path,
    *,
    subjects: str | Sequence[str] | None = None,
    runs: str | Sequence[str | int] | None = None,
    conditions: Sequence[int] = DEFAULT_CONDITIONS,
    tmin: float = -1.0,
    tmax: float = 2.5,
    baseline: tuple[float | None, float | None] | None = (-1.0, 0.0),
) -> DataContainer:
    """Load selected EEGBCI motor-task epochs through ``coco-pipe``.

    Subject identifiers are BIDS identifiers without the ``sub-`` prefix.
    Integer run identifiers are normalized to two-digit BIDS run strings.
    """
    unknown = sorted(set(int(c) for c in conditions) - set(BIDS_NAMES))
    if unknown:
        raise ValueError(f"Unknown EEGBCI conditions: {unknown}")

    normalized_runs: str | list[str] | None
    if runs is None or isinstance(runs, str):
        normalized_runs = runs
    else:
        normalized_runs = [f"{int(run):02d}" for run in runs]

    normalized_subjects: str | list[str] | None
    if subjects is None or isinstance(subjects, str):
        normalized_subjects = subjects
    else:
        normalized_subjects = [str(subject).removeprefix("sub-") for subject in subjects]

    event_id = {BIDS_NAMES[int(condition)]: int(condition) for condition in conditions}
    dataset = BIDSDataset(
        root=Path(bids_root),
        task="motorimagery",
        datatype="eeg",
        mode="epochs",
        event_id=event_id,
        subjects=normalized_subjects,
        runs=normalized_runs,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
    )
    return dataset.load()


def format_pair_keys(
    data_dict: dict[tuple[int, int], np.ndarray],
    label_names: dict[int, str],
    *,
    exclude_pairs: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """Format integer condition-pair keys and apply explicit exclusions.

    Turns the ``(3, 4)`` keys ``trajectory_separation`` returns into readable
    ``"Left Hand (Exec) vs Right Hand (Exec)"`` labels for plotting.
    """
    excluded = set(exclude_pairs)
    formatted: dict[str, np.ndarray] = {}
    for pair, values in data_dict.items():
        if len(pair) != 2:
            raise ValueError(f"Condition-pair key must contain two labels; got {pair!r}.")
        missing = [label for label in pair if label not in label_names]
        if missing:
            raise ValueError(f"Pair labels missing from `label_names`: {missing}")
        name = f"{label_names[pair[0]]} vs {label_names[pair[1]]}"
        if name not in excluded:
            formatted[name] = np.asarray(values)
    return formatted


__all__ = [
    "ALL_CONDITIONS",
    "BIDS_NAMES",
    "DEFAULT_CONDITIONS",
    "LABEL_INFO",
    "LABEL_NAMES",
    "format_pair_keys",
    "load_eegbci_container",
    "setup_data_bids",
]
