"""Shared helpers for the PCA neural-trajectory tutorial suite.

This module is the single source of truth for:

- EEGBCI condition metadata (``LABEL_INFO``, ``LABEL_NAMES``, ``CONDITION_COLORS``).
- BIDS conversion of the PhysioNet EEGBCI data via MNE.
- Loading the converted BIDS dataset into a coco-pipe ``DataContainer``.
- Building the long-format per-trial / per-condition / per-pair metric tables
  (paper §3.2 Table 2).
- Saving and loading per-subject reducers + trajectory container so the
  advanced and decoding tutorials can pick up from the main tutorial's output.

These are *helpers*, not coco-pipe public API — anything that should live in
coco-pipe upstream is marked with ``TODO(coco-pipe upstream):``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import mne
import numpy as np
import pandas as pd

from coco_pipe.dim_reduction import (
    DimReduction,
    trajectory_acceleration,
    trajectory_auc_speed,
    trajectory_cohesion,
    trajectory_curvature,
    trajectory_displacement,
    trajectory_distance_from_center,
    trajectory_intra_spread,
    trajectory_path_length,
    trajectory_separation,
    trajectory_speed,
    trajectory_tortuosity,
    trajectory_turning_angle,
)
from coco_pipe.io.dataset import BIDSDataset
from coco_pipe.io.structures import DataContainer


# ---------------------------------------------------------------------------
# Trial metadata expansion (single source of truth for downstream plots)
# ---------------------------------------------------------------------------

def make_trial_meta(
    labels: np.ndarray,
    label_info: dict[int, dict[str, str]],
    label_names: dict[int, str],
) -> dict[str, np.ndarray]:
    """Return a dict of per-trial metadata columns derived from the provided mapping.

    Use the returned dict directly as ``meta=`` for ``coco_pipe.viz.interactive.plot_embedding``.

    Parameters
    ----------
    labels : array of int
        Integer condition labels, one per trial.
    label_info : dict
        Mapping ``int_label -> {name, mode, effector, laterality}``.
    label_names : dict
        Mapping ``int_label -> readable_name``.

    Returns
    -------
    dict
        Keys: ``Label ID``, ``Condition``, ``Exec vs Imag``, ``Right vs Left``,
        ``Unilateral vs Bilateral``, ``Effector``.
    """
    labels = np.asarray(labels).astype(int)

    def _col(field: str, fallback: str = "unknown") -> np.ndarray:
        return np.array([label_info.get(int(x), {}).get(field, fallback) for x in labels])

    side = np.array(
        [
            "left" if int(x) in (3, 5) else "right" if int(x) in (4, 6) else "bilateral"
            for x in labels
        ]
    )
    return {
        "Label ID": labels,
        "Condition": np.array([label_names.get(int(x), str(int(x))) for x in labels]),
        "Exec vs Imag": _col("mode"),
        "Right vs Left": side,
        "Unilateral vs Bilateral": _col("laterality"),
        "Effector": _col("effector"),
    }


def format_pair_keys(
    data_dict: dict[tuple[int, int], np.ndarray],
    label_names: dict[int, str],
) -> dict[str, np.ndarray]:
    """Format pair tuple keys into readable strings.

    Parameters
    ----------
    data_dict : dict
        Mapping ``(label_a, label_b) -> array``
    label_names : dict
        Mapping ``int_label -> readable_name``

    Returns
    -------
    dict
        Mapping ``"Label A vs Label B" -> array``
    """
    return {
        f"{label_names.get(k[0], str(k[0]))} vs {label_names.get(k[1], str(k[1]))}": v
        for k, v in data_dict.items()
    }


# ---------------------------------------------------------------------------
# PhysioNet EEGBCI download + BIDS conversion
# ---------------------------------------------------------------------------


def setup_data_bids(
    subjects: int | Sequence[int] = 1,
    runs: Sequence[int] = tuple(range(1, 15)),
    root: str | Path = "./PhysioNet_EEGBCI/BIDS",
    force_reconvert: bool = False,
    clean_artifacts: bool = True,
    line_freq: tuple[float, ...] | None = (50.0, 60.0),
    n_ica_components: int = 20,
) -> Path:
    """Download PhysioNet EEGBCI data, clean it, and convert it to BIDS.

    Idempotent: re-running on an already-converted BIDS root only re-processes
    runs whose BrainVision file is missing (unless ``force_reconvert=True``).

    Cleaning pipeline (paper §6 / §7.2 ─ PCA amplifies whatever variance is
    in the data, so artifacts must be controlled *before* the trajectory
    analysis stage):

    1. Bandpass filter 0.5–40 Hz (FIR).
    2. Notch filter at ``line_freq`` for residual line noise (50/60 Hz).
    3. Optional ICA-based blink removal: fits an ``ICA`` with
       ``n_ica_components`` components, locates blink ICs by correlating
       with Fp1/Fp2 (frontal channels; EEGBCI has no dedicated EOG), and
       subtracts them.

    Parameters
    ----------
    subjects : int or sequence of int
    runs : sequence of int
    root : str or Path
    force_reconvert : bool
        Reprocess runs whose BIDS output already exists on disk.
    clean_artifacts : bool, default True
        Whether to run notch + ICA cleaning. Disable for a "raw" baseline
        comparison.
    line_freq : tuple of float or None, default (50.0, 60.0)
        Frequencies to notch out. Set to ``None`` to skip notch filtering.
    n_ica_components : int, default 20
        Number of ICA components to fit. Lower for speed, higher for finer
        artifact isolation.

    Returns
    -------
    Path
        Resolved BIDS root directory.
    """
    from mne.preprocessing import ICA
    from mne_bids import BIDSPath, write_raw_bids

    if isinstance(subjects, int):
        subjects = [subjects]
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
        raw_fnames = mne.datasets.eegbci.load_data(
            sub_id, runs, path=Path.home() / "mne_data"
        )
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
            if (
                not force_reconvert
                and bids_path.copy().update(extension=".vhdr").fpath.exists()
            ):
                continue

            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage(
                mne.channels.make_standard_montage("standard_1005"), verbose=False
            )
            raw.pick_types(eeg=True, verbose=False)
            raw.set_eeg_reference("average", projection=False, verbose=False)
            raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin", verbose=False)

            if clean_artifacts:
                # Notch filter for residual line noise. The 40 Hz lowpass
                # already rejects 50/60 Hz, but we run the notch explicitly
                # for the cases (e.g. when the bandpass cutoff is raised)
                # where it would otherwise leak in.
                if line_freq:
                    notch_freqs = [
                        f for f in line_freq if f < raw.info["sfreq"] / 2
                    ]
                    if notch_freqs:
                        raw.notch_filter(
                            freqs=notch_freqs, picks="eeg", verbose=False
                        )

                # ICA blink, muscle, and heart removal via mne-icalabel.
                n_comp = min(n_ica_components, len(raw.ch_names) - 1)
                ica = ICA(
                    n_components=n_comp,
                    random_state=42,
                    max_iter="auto",
                    verbose=False,
                )
                ica.fit(raw, verbose=False)
                
                from mne_icalabel import label_components
                ic_labels = label_components(raw, ica, method="iclabel")
                    
                # Automatically exclude 'eye' and 'heart' components.
                # Note: We rely on the 40Hz lowpass filter to handle 'muscle' noise.
                exclude_idx = [
                    i for i, label in enumerate(ic_labels["labels"]) 
                    if label in ["eye", "heart"]
                    ]
                        
                if exclude_idx:
                    ica.exclude = exclude_idx
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




# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


# Per-trial scalar metrics: (column name, callable taking (traj, time) → scalar).
# All 13 paper Table 2 metrics live somewhere in this module; the *narrative*
# subset taught in the main tutorial is just a few of these, but the report
# carries them all.
_PER_TRIAL_REDUCERS = [
    ("trajectory_length", lambda traj, t: float(np.nanmean(trajectory_path_length(traj)))),
    ("straight_distance", lambda traj, t: float(np.nanmean(trajectory_displacement(traj, final=True)))),
    ("tortuosity", lambda traj, t: float(np.nanmean(trajectory_tortuosity(traj)))),
    ("mean_speed", lambda traj, t: float(np.nanmean(trajectory_speed(traj, time=t)))),
    ("max_speed", lambda traj, t: float(np.nanmax(trajectory_speed(traj, time=t)))),
    ("auc_speed", lambda traj, t: float(np.nanmean(trajectory_auc_speed(traj, time=t)))),
    ("mean_curvature", lambda traj, t: float(np.nanmean(trajectory_curvature(traj)))),
    ("max_curvature", lambda traj, t: float(np.nanmax(trajectory_curvature(traj)))),
    ("mean_acceleration", lambda traj, t: float(np.nanmean(trajectory_acceleration(traj)))),
    ("mean_turning_angle", lambda traj, t: float(np.nanmean(trajectory_turning_angle(traj)))),
    ("mean_distance_from_center", lambda traj, t: float(np.nanmean(trajectory_distance_from_center(traj)))),
]


def compute_per_trial_scalars(
    trajectories: np.ndarray,
    times: np.ndarray,
    subjects: np.ndarray,
    conditions: np.ndarray,
) -> pd.DataFrame:
    """Compute per-trial scalar metrics in long format.

    Parameters
    ----------
    trajectories : np.ndarray, shape (n_trials, n_times, n_dims)
    times : np.ndarray, shape (n_times,)
    subjects : np.ndarray, shape (n_trials,)
        Per-trial subject identifier (any hashable).
    conditions : np.ndarray, shape (n_trials,)
        Per-trial condition label (int).

    Returns
    -------
    pd.DataFrame
        Columns: ``subject``, ``condition``, ``trial``, ``metric``, ``value``.
    """
    if trajectories.ndim != 3:
        raise ValueError(
            f"`trajectories` must be (n_trials, n_times, n_dims); got {trajectories.shape}."
        )

    rows: list[dict[str, Any]] = []
    for trial_idx in range(trajectories.shape[0]):
        traj = trajectories[trial_idx][None]  # add singleton trial axis for the metric API
        sub = subjects[trial_idx]
        cond = int(conditions[trial_idx])
        for name, fn in _PER_TRIAL_REDUCERS:
            try:
                val = fn(traj, times)
            except Exception:
                val = float("nan")
            rows.append(
                {
                    "subject": sub,
                    "condition": cond,
                    "trial": trial_idx,
                    "metric": name,
                    "value": val,
                }
            )
    return pd.DataFrame(rows)


def compute_per_condition_scalars(
    trajectories: np.ndarray,
    subjects: np.ndarray,
    conditions: np.ndarray,
) -> pd.DataFrame:
    """Compute within-condition spread metrics (cohesion, intra_spread).

    Returns one row per (subject, condition, metric).
    """
    rows: list[dict[str, Any]] = []
    subj_arr = np.asarray(subjects)
    cond_arr = np.asarray(conditions).astype(int)
    for sub in np.unique(subj_arr):
        for cond in np.unique(cond_arr):
            mask = (subj_arr == sub) & (cond_arr == cond)
            if not mask.any():
                continue
            stack = trajectories[mask]
            try:
                cohesion = float(np.nanmean(trajectory_cohesion(stack)))
                intra = float(np.nanmean(trajectory_intra_spread(stack)))
            except Exception:
                cohesion, intra = float("nan"), float("nan")
            rows.extend(
                [
                    {"subject": sub, "condition": cond, "metric": "mean_cohesion", "value": cohesion},
                    {"subject": sub, "condition": cond, "metric": "mean_intra_spread", "value": intra},
                ]
            )
    return pd.DataFrame(rows)


def compute_separation_pair_scalars(
    trajectories: np.ndarray,
    times: np.ndarray,
    subjects: np.ndarray,
    conditions: np.ndarray,
    methods: Sequence[str] = ("centroid", "mahalanobis"),
) -> pd.DataFrame:
    """Per-subject, per-condition-pair separation peak / peak-time / AUC.

    For each method in ``methods``, calls ``trajectory_separation`` on the
    trials for one subject and extracts scalar summaries per condition pair.

    Returns
    -------
    pd.DataFrame
        Columns: ``subject``, ``method``, ``pair`` (string ``"A_vs_B"``),
        ``label_a``, ``label_b``, ``metric``, ``value``.
        Metric values: ``peak_separation``, ``peak_separation_time``,
        ``auc_separation``.
    """
    subj_arr = np.asarray(subjects)
    cond_arr = np.asarray(conditions).astype(int)
    rows: list[dict[str, Any]] = []
    for sub in np.unique(subj_arr):
        mask = subj_arr == sub
        traj_sub = trajectories[mask]
        cond_sub = cond_arr[mask]
        if np.unique(cond_sub).size < 2:
            continue
        for method in methods:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sep = trajectory_separation(traj_sub, cond_sub, method=method)
            for (a, b), curve in sep.items():
                curve = np.asarray(curve, dtype=float)
                if not np.any(np.isfinite(curve)):
                    continue
                peak_idx = int(np.nanargmax(curve))
                peak = float(curve[peak_idx])
                peak_time = float(times[peak_idx])
                auc = float(np.trapezoid(curve, times))
                pair = f"{int(a)}_vs_{int(b)}"
                for metric_name, val in [
                    ("peak_separation", peak),
                    ("peak_separation_time", peak_time),
                    ("auc_separation", auc),
                ]:
                    rows.append(
                        {
                            "subject": sub,
                            "method": method,
                            "pair": pair,
                            "label_a": int(a),
                            "label_b": int(b),
                            "metric": metric_name,
                            "value": val,
                        }
                    )
    return pd.DataFrame(rows)


def compute_separation_timecourses(
    trajectories: np.ndarray,
    conditions: np.ndarray,
    methods: Sequence[str] = ("centroid", "mahalanobis"),
) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    """Pooled-across-subjects separation timecourses per condition pair.

    Returns
    -------
    dict
        ``{method: {(a, b): timecourse_array}}``. Use the result directly with
        ``coco_pipe.viz.interactive.plot_trajectory_separation``.
    """
    cond_arr = np.asarray(conditions).astype(int)
    out: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for method in methods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sep = trajectory_separation(trajectories, cond_arr, method=method)
        out[method] = {(int(a), int(b)): np.asarray(curve) for (a, b), curve in sep.items()}
    return out


def permutation_null_separation_auc(
    trajectories: np.ndarray,
    times: np.ndarray,
    conditions: np.ndarray,
    group_a: Sequence[int],
    group_b: Sequence[int],
    n_perm: int = 200,
    rng: Optional[np.random.Generator] = None,
    method: str = "centroid",
    window: Optional[tuple[float, float]] = None,
) -> tuple[float, np.ndarray]:
    """Label-shuffle null on between-group centroid separation AUC.

    Returns
    -------
    observed_auc : float
    null_aucs : np.ndarray
        AUC values under ``n_perm`` random label shuffles.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    cond = np.asarray(conditions).astype(int)
    group_a_arr = np.asarray(group_a).astype(int)
    group_b_arr = np.asarray(group_b).astype(int)

    if window is None:
        mask = np.ones_like(times, dtype=bool)
    else:
        mask = (times >= window[0]) & (times <= window[1])

    def _auc(labels_array: np.ndarray) -> float:
        ma = np.isin(labels_array, group_a_arr)
        mb = np.isin(labels_array, group_b_arr)
        if not ma.any() or not mb.any():
            return float("nan")
        ca = trajectories[ma].mean(axis=0)
        cb = trajectories[mb].mean(axis=0)
        d = np.linalg.norm(ca - cb, axis=-1)
        return float(np.trapezoid(d[mask], times[mask]))

    observed = _auc(cond)
    null = np.full(n_perm, np.nan, dtype=float)
    for i in range(n_perm):
        shuffled = rng.permutation(cond)
        null[i] = _auc(shuffled)
    null = null[np.isfinite(null)]
    return observed, null


# ---------------------------------------------------------------------------
# Artifact save / load (bridge between tutorials)
# ---------------------------------------------------------------------------


@dataclass
class TutorialArtifacts:
    """Bundle of objects produced by the main tutorial and consumed by
    the advanced and decoding tutorials."""

    shared_reducer: DimReduction
    trajectory_container: DataContainer
    per_subject_reducers: dict[Any, DimReduction]
    scalar_metrics: pd.DataFrame
    continuous_metrics: pd.DataFrame
    separation_pair_scalars: pd.DataFrame


def save_artifacts(
    artifacts_dir: Path | str,
    *,
    shared_reducer: DimReduction,
    trajectory_container: DataContainer,
    per_subject_reducers: dict[Any, DimReduction],
    scalar_metrics: pd.DataFrame,
    continuous_metrics: pd.DataFrame,
    separation_pair_scalars: pd.DataFrame,
    extra_csvs: Optional[dict[str, pd.DataFrame]] = None,
) -> Path:
    """Persist all main-tutorial outputs to ``artifacts_dir``.

    Layout::

        artifacts_dir/
            shared_reducer.pkl                  (DimReduction.save)
            trajectory_pc_scores.pkl            (DataContainer.save)
            reducer_sub-<id>.pkl                (per-subject reducers)
            scalar_metrics.csv
            continuous_metrics.csv
            separation_pair_scalars.csv
            <extra>.csv ...
    """
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)

    shared_reducer.save(out / "shared_reducer.pkl")
    trajectory_container.save(out / "trajectory_pc_scores.pkl")
    for sub_id, reducer in per_subject_reducers.items():
        reducer.save(out / f"reducer_sub-{sub_id}.pkl")

    scalar_metrics.to_csv(out / "scalar_metrics.csv", index=False)
    continuous_metrics.to_csv(out / "continuous_metrics.csv", index=False)
    separation_pair_scalars.to_csv(out / "separation_pair_scalars.csv", index=False)

    if extra_csvs:
        for name, df in extra_csvs.items():
            df.to_csv(out / f"{name}.csv", index=False)

    print(f"Saved artifacts → {out}")
    return out


def load_artifacts(
    artifacts_dir: Path | str,
    method: str = "PCA",
) -> TutorialArtifacts:
    """Inverse of :func:`save_artifacts`.

    Raises a clear ``RuntimeError`` if the artifacts directory is missing —
    advanced / decoding tutorials display this message verbatim to tell the
    reader which notebook to run first.
    """
    src = Path(artifacts_dir)
    if not src.exists():
        raise RuntimeError(
            f"Artifacts directory {src} not found.\n"
            f"Run the main tutorial `tutorials/tutorial_pca_trajectories_eegbci.ipynb` "
            "first to generate it. This tutorial assumes per-subject reducers and "
            "the PC-score container are already on disk."
        )

    shared_reducer = DimReduction.load(src / "shared_reducer.pkl", method=method)
    trajectory_container = DataContainer.load(src / "trajectory_pc_scores.pkl")

    per_subject_reducers: dict[Any, DimReduction] = {}
    for p in sorted(src.glob("reducer_sub-*.pkl")):
        sub_token = p.stem.split("-", 1)[-1]
        try:
            sub_id: Any = int(sub_token)
        except ValueError:
            sub_id = sub_token
        per_subject_reducers[sub_id] = DimReduction.load(p, method=method)

    return TutorialArtifacts(
        shared_reducer=shared_reducer,
        trajectory_container=trajectory_container,
        per_subject_reducers=per_subject_reducers,
        scalar_metrics=pd.read_csv(src / "scalar_metrics.csv"),
        continuous_metrics=pd.read_csv(src / "continuous_metrics.csv"),
        separation_pair_scalars=pd.read_csv(src / "separation_pair_scalars.csv"),
    )
