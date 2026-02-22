"""
Tutorial: End-to-End EEG Trajectory Analysis (Linear Walkthrough)

This script intentionally uses a linear stage-by-stage flow:
Stage A -> Stage I

Only primitive reusable operations are implemented as helpers.
"""

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mne_bids import BIDSPath, write_raw_bids
from plotly.subplots import make_subplots

from coco_pipe.dim_reduction import (
    DimReduction,
    trajectory_curvature,
    trajectory_separation,
    trajectory_speed,
)
from coco_pipe.io.dataset import BIDSDataset
from coco_pipe.io.structures import DataContainer
from coco_pipe.report import PlotlyElement, Report, Section, TableElement
from coco_pipe.viz.plotly_utils import (
    plot_channel_traces_interactive,
    plot_embedding_interactive,
    plot_trajectory_interactive,
)

LABEL_INFO = {
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

LABEL_NAMES = {k: v["name"] for k, v in LABEL_INFO.items()}

CONDITION_COLORS = {
    3: "#1f77b4",
    4: "#d62728",
    5: "#17becf",
    6: "#ff7f0e",
    7: "#2ca02c",
    8: "#9467bd",
    9: "#8dd3c7",
    10: "#fb8072",
}

def setup_data_bids(
    subjects: Union[int, List[int]] = 1,
    runs: Sequence[int] = tuple(range(1, 15)),
    root: str = "./PhysioNet_EEGBCI/BIDS",
    force_reconvert: bool = False,
) -> None:
    """Download PhysioNet EEGBCI data and convert to BIDS."""
    if isinstance(subjects, int):
        subjects = [subjects]

    bids_root = Path(root)
    print(f"Downloading/converting EEGBCI for subjects={subjects}, runs={list(runs)}")

    for sub_id in subjects:
        raw_fnames = mne.datasets.eegbci.load_data(
            sub_id,
            runs,
            path=Path.home() / "mne_data",
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
                print(f"Skipping sub-{sub_id:03d} run-{run_num:02d}; BIDS already exists.")
                continue

            raw = mne.io.read_raw_edf(fname, preload=True)
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
            raw.pick_types(eeg=True, verbose=False)
            raw.set_eeg_reference("average", projection=False, verbose=False)
            raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin", verbose=False)

            if run_num == 1:
                mapping = {"T0": "rest_eyes_open"}
                event_id = {"rest_eyes_open": 0}
            elif run_num == 2:
                mapping = {"T0": "rest_eyes_closed"}
                event_id = {"rest_eyes_closed": 1}
            elif run_num in [3, 7, 11]:
                mapping = {"T0": "rest", "T1": "left_hand_exec", "T2": "right_hand_exec"}
                event_id = {"rest": 2, "left_hand_exec": 3, "right_hand_exec": 4}
            elif run_num in [4, 8, 12]:
                mapping = {"T0": "rest", "T1": "left_hand_imag", "T2": "right_hand_imag"}
                event_id = {"rest": 2, "left_hand_imag": 5, "right_hand_imag": 6}
            elif run_num in [5, 9, 13]:
                mapping = {"T0": "rest", "T1": "hands_exec", "T2": "feet_exec"}
                event_id = {"rest": 2, "hands_exec": 7, "feet_exec": 8}
            elif run_num in [6, 10, 14]:
                mapping = {"T0": "rest", "T1": "hands_imag", "T2": "feet_imag"}
                event_id = {"rest": 2, "hands_imag": 9, "feet_imag": 10}
            else:
                mapping = {}
                event_id = {}

            existing_annotations = set(raw.annotations.description)
            mapping = {k: v for k, v in mapping.items() if k in existing_annotations}
            if mapping:
                raw.annotations.rename(mapping)

            write_raw_bids(
                raw,
                bids_path,
                event_id=event_id,
                overwrite=True,
                allow_preload=True,
                format="BrainVision",
            )

    print(f"BIDS conversion complete: {bids_root}")


def load_dataset(bids_root: Path, runs: Optional[List[str]] = None) -> DataContainer:
    """Load epoched motor imagery data through BIDSDataset."""
    event_id = {
        "left_hand_exec": 3,
        "right_hand_exec": 4,
        "left_hand_imag": 5,
        "right_hand_imag": 6,
        "hands_exec": 7,
        "feet_exec": 8,
        "hands_imag": 9,
        "feet_imag": 10,
    }
    ds = BIDSDataset(
        root=bids_root,
        task="motorimagery",
        datatype="eeg",
        mode="epochs",
        event_id=event_id,
        runs=runs,
        tmin=-1.0,
        tmax=3.0,
        baseline=(-1.0, 0.0),
    )
    return ds.load()


def assert_and_describe_shapes(container: DataContainer) -> None:
    """Primitive helper: validate and print expected tensor shapes."""
    if container.X.ndim != 3:
        raise ValueError(
            f"Expected epoched data with 3 dims (obs, channel, time), got {container.X.shape}"
        )
    if container.y is None:
        raise ValueError("Container has no labels (y).")
    if "time" not in container.coords:
        raise ValueError("Container is missing time coordinate.")
    print(f"Epoched tensor shape: {container.X.shape}  (obs, channel, time)")


def compute_auc(values: np.ndarray, times: np.ndarray) -> float:
    """Primitive helper: trapezoidal AUC."""
    if len(values) < 2 or len(times) < 2:
        return np.nan
    return float(np.trapz(values, times))


def shuffle_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Primitive helper: shuffled label vector."""
    return rng.permutation(labels)


def contrast_distance(
    trajectories: np.ndarray,
    labels: np.ndarray,
    group_a: Sequence[int],
    group_b: Sequence[int],
) -> Optional[np.ndarray]:
    """Primitive helper: centroid distance over time for two label groups."""
    mask_a = np.isin(labels, np.array(group_a))
    mask_b = np.isin(labels, np.array(group_b))
    if not mask_a.any() or not mask_b.any():
        return None
    centroid_a = trajectories[mask_a].mean(axis=0)
    centroid_b = trajectories[mask_b].mean(axis=0)
    return np.linalg.norm(centroid_a - centroid_b, axis=-1)


def save_plotly(fig: go.Figure, path: Path) -> None:
    """Primitive helper: save Plotly figure as HTML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------------
    # Stage A: Setup + data access
    # -------------------------------------------------------------------------
    bids_dir = Path("./PhysioNet_EEGBCI/BIDS")
    runs = list(range(3, 15))
    subjects = list(range(1, 11))

    output_root = Path("./outputs/tutorial_trajectories")
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    metrics_dir = output_root / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    setup_data_bids(subjects=subjects, runs=runs, root=str(bids_dir), force_reconvert=False)

    # -------------------------------------------------------------------------
    # Stage B: Load epoched container
    # -------------------------------------------------------------------------
    runs_to_load = [f"{r:02d}" for r in runs]
    print(f"Loading runs={runs_to_load}")
    container = load_dataset(bids_dir, runs=runs_to_load)
    assert_and_describe_shapes(container)

    labels = np.asarray(container.y).astype(int)
    times = np.asarray(container.coords["time"])
    channels = np.asarray(container.coords["channel"])
    n_trials = container.X.shape[0]

    # -------------------------------------------------------------------------
    # Stage C: Build condition metadata table
    # -------------------------------------------------------------------------
    trial_meta = pd.DataFrame({"label": labels})
    trial_meta["condition"] = trial_meta["label"].map(
        lambda x: LABEL_NAMES.get(int(x), str(int(x)))
    )
    trial_meta["mode"] = trial_meta["label"].map(
        lambda x: LABEL_INFO.get(int(x), {}).get("mode", "unknown")
    )
    trial_meta["effector"] = trial_meta["label"].map(
        lambda x: LABEL_INFO.get(int(x), {}).get("effector", "unknown")
    )
    trial_meta["laterality"] = trial_meta["label"].map(
        lambda x: LABEL_INFO.get(int(x), {}).get("laterality", "unknown")
    )

    condition_summary = (
        trial_meta.groupby(["label", "condition", "mode", "effector", "laterality"])
        .size()
        .reset_index(name="n_trials")
        .sort_values("label")
    )
    condition_summary.to_csv(tables_dir / "condition_summary.csv", index=False)
    trial_meta.to_csv(tables_dir / "trial_metadata.csv", index=False)

    # -------------------------------------------------------------------------
    # Stage D: Sensor-space inspection first (before PCA)
    # -------------------------------------------------------------------------
    analysis_tmin = 0.0
    analysis_tmax = 1.5
    report = Report(title="Neural Trajectory Tutorial: End-to-End Analysis")
    report.add_container(container, show_coords=False, show_dist=False)

    # Figure 1: Epoched QC timecourses
    preferred_channels = ["C3", "Cz", "C4"]
    selected_channels = [ch for ch in preferred_channels if ch in channels]
    if not selected_channels:
        selected_channels = list(channels[: min(3, len(channels))])

    c_raw_centroids = container.aggregate(by="y", method="mean")
    fig01 = plot_channel_traces_interactive(
        data=c_raw_centroids.X,
        times=np.asarray(c_raw_centroids.coords["time"]),
        group_labels=np.asarray(c_raw_centroids.coords["obs"]).astype(int),
        channel_names=np.asarray(c_raw_centroids.coords["channel"]).astype(str),
        selected_channels=selected_channels,
        group_name_map=LABEL_NAMES,
        color_map=CONDITION_COLORS,
        title="Figure 1: Epoched Data Sanity Plot",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        base_height=300,
        row_height=220,
    )
    fig01_height_px = int(fig01.layout.height) if fig01.layout.height is not None else 800
    fig01_path = figures_dir / "fig01_epoched_qc.html"
    save_plotly(fig01, fig01_path)

    sec01 = Section("Figure 1: Epoched Data Sanity", icon="1")
    sec01.add_markdown(
        "Condition-averaged epoched traces over representative channels to verify event alignment. "
        "Visual inspection shows most variance is concentrated in the first ~1.5s."
    )
    sec01.add_element(PlotlyElement(fig01, height=f"{fig01_height_px}px"))
    sec01.add_element(TableElement(condition_summary, title="Condition Summary"))
    report.add_section(sec01)
    del c_raw_centroids

    # -------------------------------------------------------------------------
    # Stage E: Restrict analysis window to 0.0-1.5s before PCA
    # -------------------------------------------------------------------------
    time_mask = (times >= analysis_tmin - 1e-8) & (times <= analysis_tmax + 1e-8)
    container = container.isel(time=time_mask)
    times = np.asarray(container.coords["time"])
    labels = np.asarray(container.y).astype(int)
    channels = np.asarray(container.coords["channel"])
    n_trials = container.X.shape[0]
    n_times = container.X.shape[2]

    print(
        f"Cropped container to analysis window [{analysis_tmin:.1f}, {analysis_tmax:.1f}]s: "
        f"shape={container.X.shape}"
    )

    sec_window = Section("Analysis Window Selection", icon="W")
    sec_window.add_markdown(
        f"All downstream PCA and trajectory analyses are restricted to "
        f"**{analysis_tmin:.1f}-{analysis_tmax:.1f}s** based on Figure 1. "
        f"The in-memory container is cropped to this window before dimensionality reduction."
    )
    sec_window.add_element(
        TableElement(
            {
                "analysis_tmin_s": analysis_tmin,
                "analysis_tmax_s": analysis_tmax,
                "n_timepoints_window": n_times,
            },
            title="Window Parameters",
        )
    )
    report.add_section(sec_window)

    # -------------------------------------------------------------------------
    # Stage F: Fit PCA on cropped window only
    # -------------------------------------------------------------------------
    c_stacked = container.stack(dims=("obs", "time"), new_dim="obs")
    c_stacked_z = c_stacked.zscore(dim="obs")
    state_matrix_z = c_stacked_z.X

    print(
        f"State matrix shape (cropped window): {state_matrix_z.shape}  "
        f"(n_trials*n_times, n_channels) = ({n_trials}*{n_times}, {container.X.shape[1]})"
    )

    reducer_3d = DimReduction(method="PCA", n_components=3, random_state=42)
    embedding_3d = reducer_3d.fit_transform(state_matrix_z)

    n_scree_components = min(20, state_matrix_z.shape[1])
    reducer_scree = DimReduction(
        method="PCA",
        n_components=n_scree_components,
        random_state=42,
    )
    _ = reducer_scree.fit_transform(state_matrix_z)

    # -------------------------------------------------------------------------
    # Stage G: Reconstruct trajectories from cropped-window PCA scores
    # -------------------------------------------------------------------------
    reduced_coords = c_stacked_z.coords.copy()
    reduced_coords["component"] = np.array(["PC1", "PC2", "PC3"])
    c_reduced = replace(
        c_stacked_z,
        X=embedding_3d,
        dims=("obs", "component"),
        coords=reduced_coords,
    )

    c_traj_reduced = c_reduced.unstack("obs")
    c_traj_reduced = replace(
        c_traj_reduced,
        y=labels,
        coords={**c_traj_reduced.coords, "time": times},
    )
    trajectories = c_traj_reduced.X
    sample_labels = np.repeat(c_traj_reduced.y, n_times)

    c_centroids = c_traj_reduced.aggregate(by="y", method="mean")
    unique_labels = np.asarray(c_centroids.coords["obs"]).astype(int)

    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
    # Figure 2a: one point per trial (time-mean of PC coordinates)
    trial_emb = trajectories.mean(axis=1)
    trial_labels = labels.astype(int)
    trial_label_names = np.array([LABEL_NAMES.get(int(x), str(int(x))) for x in trial_labels])
    trial_mode = np.array([LABEL_INFO[int(x)]["mode"] for x in trial_labels])
    trial_laterality = np.array([LABEL_INFO[int(x)]["laterality"] for x in trial_labels])
    trial_effector = np.array([LABEL_INFO[int(x)]["effector"] for x in trial_labels])
    trial_side = np.array(
        [
            "left" if int(x) in (3, 5) else "right" if int(x) in (4, 6) else "bilateral"
            for x in trial_labels
        ]
    )
    trial_mode_side = np.array(
        [f"{m}_{s}" for m, s in zip(trial_mode, trial_side, strict=False)]
    )
    trial_mode_effector = np.array(
        [f"{m}_{e}" for m, e in zip(trial_mode, trial_effector, strict=False)]
    )
    trial_meta = {
        "Label ID": trial_labels,
        "Exec vs Imag": trial_mode,
        "Right vs Left": trial_side,
        "Unilateral vs Bilateral": trial_laterality,
        "Effector": trial_effector,
        "Exec/Imag x Side": trial_mode_side,
        "Exec/Imag x Effector": trial_mode_effector,
    }
    fig02a = plot_embedding_interactive(
        embedding=trial_emb[:, :2],
        labels=trial_label_names,
        meta=trial_meta,
        title="Figure 2a: PC1-PC2 with switchable coloring (one point per trial, time-mean)",
        dimensions=2,
    )
    fig02a_path = figures_dir / "fig02a_scatter_trial_mean_pc12.html"
    save_plotly(fig02a, fig02a_path)

    # Figure 2b: all trial x time states
    max_scatter_points = min(25000, embedding_3d.shape[0])
    scatter_idx = rng.choice(embedding_3d.shape[0], size=max_scatter_points, replace=False)
    emb_scatter = embedding_3d[scatter_idx]
    lbl_scatter = sample_labels[scatter_idx].astype(int)
    lbl_scatter_names = np.array([LABEL_NAMES.get(int(x), str(int(x))) for x in lbl_scatter])

    mode_scatter = np.array([LABEL_INFO[int(x)]["mode"] for x in lbl_scatter])
    laterality_scatter = np.array([LABEL_INFO[int(x)]["laterality"] for x in lbl_scatter])
    effector_scatter = np.array([LABEL_INFO[int(x)]["effector"] for x in lbl_scatter])
    side_scatter = np.array(
        [
            "left" if int(x) in (3, 5) else "right" if int(x) in (4, 6) else "bilateral"
            for x in lbl_scatter
        ]
    )
    mode_side_scatter = np.array(
        [f"{m}_{s}" for m, s in zip(mode_scatter, side_scatter, strict=False)]
    )
    mode_effector_scatter = np.array(
        [f"{m}_{e}" for m, e in zip(mode_scatter, effector_scatter, strict=False)]
    )
    scatter_meta = {
        "Label ID": lbl_scatter,
        "Exec vs Imag": mode_scatter,
        "Right vs Left": side_scatter,
        "Unilateral vs Bilateral": laterality_scatter,
        "Effector": effector_scatter,
        "Exec/Imag x Side": mode_side_scatter,
        "Exec/Imag x Effector": mode_effector_scatter,
    }
    fig02b = plot_embedding_interactive(
        embedding=emb_scatter[:, :2],
        labels=lbl_scatter_names,
        meta=scatter_meta,
        title="Figure 2b: PC1-PC2 with switchable coloring (all trial x time states)",
        dimensions=2,
    )
    fig02b_path = figures_dir / "fig02b_scatter_all_states_pc12.html"
    save_plotly(fig02b, fig02b_path)

    sec02 = Section("Figures 2a-2b: Baseline Scatter", icon="2")
    sec02.add_markdown(
        "2a gives one point per trial (time-mean). 2b shows dense trial x time states."
    )
    sec02.add_element(PlotlyElement(fig02a, height="460px"))
    sec02.add_element(PlotlyElement(fig02b, height="460px"))
    report.add_section(sec02)

    # Figure 3: start with a single 3D contrast
    exec_imag_groups = np.where(np.isin(labels, [3, 4, 7, 8]), "Exec", "Imag")
    c_exec_imag = c_traj_reduced.aggregate(by=exec_imag_groups, method="mean")
    obs_exec_imag = np.asarray(c_exec_imag.coords["obs"]).astype(str)
    exec_curve = c_exec_imag.X[np.where(obs_exec_imag == "Exec")[0][0]]
    imag_curve = c_exec_imag.X[np.where(obs_exec_imag == "Imag")[0][0]]

    label_centroids = {}
    centroid_labels = np.asarray(c_centroids.coords["obs"]).astype(int)
    for i_lbl, lbl in enumerate(centroid_labels):
        label_centroids[int(lbl)] = c_centroids.X[i_lbl]
    required_for_requested_contrasts = [3, 4, 5]
    missing_required = [
        lbl for lbl in required_for_requested_contrasts if lbl not in label_centroids
    ]
    if missing_required:
        raise ValueError(
            "Missing required labels for requested contrasts: "
            f"{missing_required}. Available labels: {sorted(label_centroids.keys())}"
        )

    first_curve_names = ["Exec", "Imag"]
    first_curve_stack = np.vstack([exec_curve, imag_curve])
    first_groups = np.repeat(first_curve_names, len(times))
    first_times = np.tile(times, len(first_curve_names))

    fig03 = plot_trajectory_interactive(
        X=first_curve_stack,
        times=first_times,
        groups=first_groups,
        title="Figure 3: Exec vs Imag combining Left/Right and Hands/Feet",
        dimensions=3,
        smooth_window=8,
        mode="lines",
        line_width=8,
        group_cmaps={"Exec": "Blues", "Imag": "Oranges"},
        show_group_colorbars=True,
    )
    fig03.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    fig03_path = figures_dir / "fig03_exec_vs_imag_combined_effectors.html"
    save_plotly(fig03, fig03_path)

    sec03 = Section("Figure 3: Exec vs Imag combining Left/Right and Hands/Feet", icon="3")
    sec03.add_element(PlotlyElement(fig03, height="650px"))
    report.add_section(sec03)

    # Figure 4: Hands vs Feet (pooled exec+imag)
    hf_idx = np.where(np.isin(labels, [7, 8, 9, 10]))[0]
    if hf_idx.size == 0:
        raise ValueError("Missing labels for Hands vs Feet contrast (expected 7/8/9/10).")
    c_hf = c_traj_reduced.isel(obs=hf_idx)
    hf_groups = np.where(np.isin(c_hf.y, [7, 9]), "Hands", "Feet")
    c_hf_centroids = c_hf.aggregate(by=hf_groups, method="mean")
    obs_hf = np.asarray(c_hf_centroids.coords["obs"]).astype(str)
    if "Hands" not in obs_hf or "Feet" not in obs_hf:
        raise ValueError(
            "Could not build Hands vs Feet pooled centroids from labels 7/8/9/10."
        )
    hands_curve = c_hf_centroids.X[np.where(obs_hf == "Hands")[0][0]]
    feet_curve = c_hf_centroids.X[np.where(obs_hf == "Feet")[0][0]]
    hf_curve_names = ["Hands", "Feet"]
    hf_curve_stack = np.vstack([hands_curve, feet_curve])
    hf_groups_plot = np.repeat(hf_curve_names, len(times))
    hf_times_plot = np.tile(times, len(hf_curve_names))

    fig04 = plot_trajectory_interactive(
        X=hf_curve_stack,
        times=hf_times_plot,
        groups=hf_groups_plot,
        title="Figure 4: Hands vs Feet combining Imag+Exec",
        dimensions=3,
        smooth_window=8,
        mode="lines",
        line_width=8,
        group_cmaps={"Hands": "Purples", "Feet": "Oranges"},
        show_group_colorbars=True,
    )
    fig04.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    fig04_path = figures_dir / "fig04_hands_vs_feet_combining_imag_exec.html"
    save_plotly(fig04, fig04_path)

    sec04 = Section("Figure 4: Hands vs Feet combining Imag+Exec", icon="4")
    sec04.add_element(PlotlyElement(fig04, height="650px"))
    report.add_section(sec04)

    # Figure 5: Left vs Right (pooled exec+imag)
    lr_idx = np.where(np.isin(labels, [3, 4, 5, 6]))[0]
    if lr_idx.size == 0:
        raise ValueError("Missing labels for Left vs Right contrast (expected 3/4/5/6).")
    c_lr = c_traj_reduced.isel(obs=lr_idx)
    lr_groups = np.where(np.isin(c_lr.y, [3, 5]), "Left", "Right")
    c_lr_centroids = c_lr.aggregate(by=lr_groups, method="mean")
    obs_lr = np.asarray(c_lr_centroids.coords["obs"]).astype(str)
    if "Left" not in obs_lr or "Right" not in obs_lr:
        raise ValueError("Could not build Left vs Right pooled centroids from labels 3/4/5/6.")
    left_curve = c_lr_centroids.X[np.where(obs_lr == "Left")[0][0]]
    right_curve = c_lr_centroids.X[np.where(obs_lr == "Right")[0][0]]
    lr_curve_names = ["Left", "Right"]
    lr_curve_stack = np.vstack([left_curve, right_curve])
    lr_groups_plot = np.repeat(lr_curve_names, len(times))
    lr_times_plot = np.tile(times, len(lr_curve_names))

    fig05 = plot_trajectory_interactive(
        X=lr_curve_stack,
        times=lr_times_plot,
        groups=lr_groups_plot,
        title="Figure 5: Left vs Right combining Imag+Exec",
        dimensions=3,
        smooth_window=8,
        mode="lines",
        line_width=8,
        group_cmaps={"Left": "Greens", "Right": "Reds"},
        show_group_colorbars=True,
    )
    fig05.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    fig05_path = figures_dir / "fig05_left_vs_right_combining_imag_exec.html"
    save_plotly(fig05, fig05_path)

    sec05 = Section("Figure 5: Left vs Right combining Imag+Exec", icon="5")
    sec05.add_element(PlotlyElement(fig05, height="650px"))
    report.add_section(sec05)

    # Figure 6: Label 3 vs Label 5
    curve_3 = label_centroids[3]
    curve_5 = label_centroids[5]
    curves_35_names = [LABEL_NAMES[3], LABEL_NAMES[5]]
    curves_35_stack = np.vstack([curve_3, curve_5])
    groups_35 = np.repeat(curves_35_names, len(times))
    times_35 = np.tile(times, len(curves_35_names))

    fig06 = plot_trajectory_interactive(
        X=curves_35_stack,
        times=times_35,
        groups=groups_35,
        title="Figure 6: Left Hand Exec vs Left Hand Imag (3 vs 5)",
        dimensions=3,
        smooth_window=8,
        mode="lines",
        line_width=8,
        group_cmaps={LABEL_NAMES[3]: "Blues", LABEL_NAMES[5]: "Oranges"},
        show_group_colorbars=True,
    )
    fig06.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    fig06_path = figures_dir / "fig06_left_exec_vs_left_imag_3_vs_5.html"
    save_plotly(fig06, fig06_path)

    sec06 = Section("Figure 6: Left Hand Exec vs Left Hand Imag (3 vs 5)", icon="6")
    sec06.add_element(PlotlyElement(fig06, height="650px"))
    report.add_section(sec06)

    # Figure 7: Label 3 vs Label 4
    curve_4 = label_centroids[4]
    curves_34_names = [LABEL_NAMES[3], LABEL_NAMES[4]]
    curves_34_stack = np.vstack([curve_3, curve_4])
    groups_34 = np.repeat(curves_34_names, len(times))
    times_34 = np.tile(times, len(curves_34_names))

    fig07 = plot_trajectory_interactive(
        X=curves_34_stack,
        times=times_34,
        groups=groups_34,
        title="Figure 7: Left Hand Exec vs Right Hand Exec (3 vs 4)",
        dimensions=3,
        smooth_window=8,
        mode="lines",
        line_width=8,
        group_cmaps={LABEL_NAMES[3]: "Blues", LABEL_NAMES[4]: "Reds"},
        show_group_colorbars=True,
    )
    fig07.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    fig07_path = figures_dir / "fig07_left_exec_vs_right_exec_3_vs_4.html"
    save_plotly(fig07, fig07_path)

    sec07 = Section("Figure 7: Left Hand Exec vs Right Hand Exec (3 vs 4)", icon="7")
    sec07.add_element(PlotlyElement(fig07, height="650px"))
    report.add_section(sec07)

    # Figure 8: labels 3, 4, 5, 6 together
    labels_3456 = [lbl for lbl in [3, 4, 5, 6] if lbl in label_centroids]
    names_3456 = [LABEL_NAMES[lbl] for lbl in labels_3456]
    curves_3456 = np.vstack([label_centroids[lbl] for lbl in labels_3456])
    groups_3456 = np.repeat(names_3456, len(times))
    times_3456 = np.tile(times, len(names_3456))

    fig08_traj = plot_trajectory_interactive(
        X=curves_3456,
        times=times_3456,
        groups=groups_3456,
        title="Figure 8: Left/Right with Exec+Imag shown together (3, 4, 5, 6)",
        dimensions=3,
        smooth_window=8,
        mode="lines",
        line_width=7,
        group_cmaps={
            LABEL_NAMES[3]: "Blues",
            LABEL_NAMES[4]: "Reds",
            LABEL_NAMES[5]: "Oranges",
            LABEL_NAMES[6]: "Greens",
        },
        show_group_colorbars=False,
    )
    fig08_traj.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    fig08_traj_path = figures_dir / "fig08_left_right_exec_imag_together_3_4_5_6.html"
    save_plotly(fig08_traj, fig08_traj_path)

    sec08_traj = Section("Figure 8: Left/Right with Exec+Imag shown together (3, 4, 5, 6)", icon="8")
    sec08_traj.add_element(PlotlyElement(fig08_traj, height="650px"))
    report.add_section(sec08_traj)

    # -------------------------------------------------------------------------
    # Stage G (continued): Metric computation and condition contrasts
    # -------------------------------------------------------------------------
    focus_labels = [lbl for lbl in [3, 4, 5, 6] if lbl in unique_labels]
    # Curvature can spike when instantaneous speed is very small; use a robust cap for plotting.
    if focus_labels:
        focus_mask = np.isin(labels, np.array(focus_labels))
        curvature_all = trajectory_curvature(trajectories[focus_mask])
        curvature_plot_cap = float(np.nanpercentile(curvature_all, 99.0))
        if not np.isfinite(curvature_plot_cap) or curvature_plot_cap <= 0:
            curvature_plot_cap = 1.0
    else:
        curvature_plot_cap = 1.0

    fig05 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Speed by condition", "Curvature by condition"),
    )
    for lbl in focus_labels:
        cond_trials = trajectories[labels == lbl]
        if cond_trials.shape[0] == 0:
            continue
        speed_trials = trajectory_speed(cond_trials, dt=dt)
        curvature_trials = trajectory_curvature(cond_trials)
        curvature_trials_plot = np.clip(curvature_trials, a_min=0.0, a_max=curvature_plot_cap)

        mean_speed = speed_trials.mean(axis=0)
        sem_speed = speed_trials.std(axis=0) / np.sqrt(speed_trials.shape[0])
        mean_curv = curvature_trials_plot.mean(axis=0)
        sem_curv = curvature_trials_plot.std(axis=0) / np.sqrt(curvature_trials_plot.shape[0])

        color = CONDITION_COLORS.get(int(lbl), "#444444")
        name = LABEL_NAMES.get(int(lbl), str(int(lbl)))

        fig05.add_trace(
            go.Scatter(
                x=times,
                y=mean_speed,
                mode="lines",
                name=f"{name} speed",
                line=dict(color=color, width=2),
                error_y=dict(type="data", array=sem_speed, visible=True),
                legendgroup=str(lbl),
            ),
            row=1,
            col=1,
        )
        fig05.add_trace(
            go.Scatter(
                x=times,
                y=mean_curv,
                mode="lines",
                name=f"{name} curvature",
                line=dict(color=color, width=2, dash="dot"),
                error_y=dict(type="data", array=sem_curv, visible=True),
                legendgroup=str(lbl),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    window_edges = np.linspace(analysis_tmin, analysis_tmax, 4)
    for w_start, w_end in zip(window_edges[:-1], window_edges[1:], strict=False):
        fig05.add_vrect(
            x0=float(w_start),
            x1=float(w_end),
            fillcolor="gray",
            opacity=0.06,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )
        fig05.add_vrect(
            x0=float(w_start),
            x1=float(w_end),
            fillcolor="gray",
            opacity=0.06,
            layer="below",
            line_width=0,
            row=2,
            col=1,
        )

    fig05.update_xaxes(range=[analysis_tmin, analysis_tmax], row=1, col=1)
    fig05.update_xaxes(title_text="Time (s)", range=[analysis_tmin, analysis_tmax], row=2, col=1)
    fig05.update_yaxes(title_text="Speed (a.u./s)", row=1, col=1)
    fig05.update_yaxes(
        title_text=f"Curvature (a.u., clipped @ p99={curvature_plot_cap:.2f})",
        range=[0.0, curvature_plot_cap * 1.05],
        row=2,
        col=1,
    )
    fig05.update_layout(
        title="Figure 9: Kinematic Metrics Over Time",
        template="plotly_white",
        height=700,
    )
    fig05_path = figures_dir / "fig09_speed_curvature_timecourses.html"
    save_plotly(fig05, fig05_path)

    sec05 = Section("Figure 9: Speed and Curvature", icon="9")
    sec05.add_markdown(
        "Mean +/- SEM speed and curvature across key unilateral conditions."
    )
    sec05.add_element(PlotlyElement(fig05, height="700px"))
    report.add_section(sec05)

    # Figure 10: Separation-over-time with AUC table
    contrast_definitions = {
        "Left Exec vs Right Exec": ([3], [4]),
        "Left Imag vs Right Imag": ([5], [6]),
        "Hands Exec vs Feet Exec": ([7], [8]),
        "Hands Imag vs Feet Imag": ([9], [10]),
        "Exec vs Imag (pooled)": ([3, 4, 7, 8], [5, 6, 9, 10]),
    }
    pairwise_separations = trajectory_separation(c_traj_reduced.X, c_traj_reduced.y)
    fig06 = go.Figure()
    sep_rows = []
    for contrast_name, (group_a, group_b) in contrast_definitions.items():
        if len(group_a) == 1 and len(group_b) == 1:
            pair_key = tuple(sorted((group_a[0], group_b[0])))
            dist_curve = pairwise_separations.get(pair_key)
        elif contrast_name == "Exec vs Imag (pooled)":
            dist_curve = np.linalg.norm(exec_curve - imag_curve, axis=-1)
        else:
            dist_curve = contrast_distance(trajectories, labels, group_a, group_b)
        if dist_curve is None:
            continue
        fig06.add_trace(
            go.Scatter(
                x=times,
                y=dist_curve,
                mode="lines",
                name=contrast_name,
                line=dict(width=3),
            )
        )

        peak_idx = int(np.nanargmax(dist_curve))
        row = {
            "contrast": contrast_name,
            "peak_time_s": float(times[peak_idx]),
            "peak_distance": float(dist_curve[peak_idx]),
            "mean_distance": float(np.nanmean(dist_curve)),
        }
        for w_start, w_end in [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]:
            mask = (times >= w_start) & (times <= w_end)
            row[f"auc_{w_start:.0f}_{w_end:.0f}s"] = compute_auc(dist_curve[mask], times[mask])
        sep_rows.append(row)

    fig06.update_layout(
        title="Figure 10: Separation Over Time Across Conditions",
        xaxis_title="Time (s)",
        yaxis_title="Centroid distance (a.u.)",
        template="plotly_white",
        height=500,
    )
    fig06_path = figures_dir / "fig10_separation_curves.html"
    save_plotly(fig06, fig06_path)

    sep_df = pd.DataFrame(sep_rows).sort_values("contrast")
    sep_df.to_csv(tables_dir / "separation_auc_summary.csv", index=False)

    sec06 = Section("Figure 10: Separation Curves", icon="10")
    sec06.add_markdown(
        "Time-resolved centroid separation with AUC summaries across predefined windows."
    )
    sec06.add_element(PlotlyElement(fig06, height="500px"))
    sec06.add_element(TableElement(sep_df, title="Separation AUC Summary"))
    report.add_section(sec06)

    # -------------------------------------------------------------------------
    # Stage H: Quality diagnostics + robustness checks
    # -------------------------------------------------------------------------
    # Quality metrics are expensive; evaluate on a subset.
    score_n = min(1200, state_matrix_z.shape[0])
    score_idx = rng.choice(state_matrix_z.shape[0], size=score_n, replace=False)
    score_X = state_matrix_z[score_idx]
    k_neighbors = max(5, min(20, score_n - 2))

    # Use DimReduction plotting APIs directly on a dedicated subset fit.
    reducer_diag = DimReduction(method="PCA", n_components=3, random_state=42)
    _ = reducer_diag.fit_transform(score_X)

    score_payload = reducer_diag.score(score_X, n_neighbors=k_neighbors)
    quality_metrics = score_payload.get("metrics", {})

    metrics_df = pd.DataFrame(
        [
            {"metric": k, "value": float(v)}
            for k, v in quality_metrics.items()
            if isinstance(v, (float, int, np.floating, np.integer))
        ]
    )
    metrics_df.to_csv(metrics_dir / "quality_metrics.csv", index=False)

    fig07 = reducer_diag.plot(
        mode="metrics",
        X=score_X,
        interactive=True,
        title="Figure 11: Embedding Quality Metrics",
    )
    fig07.update_layout(height=420)

    fig07_scree = reducer_scree.plot(
        mode="diagnostics",
        interactive=True,
    )
    fig07_scree.update_layout(title="Figure 11b: PCA Scree / Cumulative Variance")
    fig07_scree.update_layout(height=350)
    fig07_scree_path = figures_dir / "fig11b_scree.html"
    save_plotly(fig07, figures_dir / "fig11_quality_metrics.html")
    save_plotly(fig07_scree, fig07_scree_path)

    fig08 = reducer_diag.plot(
        mode="shepard",
        X=score_X,
        interactive=True,
        title="Figure 12: Shepard Diagram",
    )
    fig08_path = figures_dir / "fig12_shepard.html"
    save_plotly(fig08, fig08_path)

    sec07 = Section("Figures 11-12: Diagnostics", icon="11")
    sec07.add_markdown(
        "Includes scree, preservation metrics (trustworthiness/continuity/LCMC), and Shepard distance diagnostics."
    )
    sec07.add_element(PlotlyElement(fig07, height="420px"))
    sec07.add_element(PlotlyElement(fig07_scree, height="350px"))
    sec07.add_element(PlotlyElement(fig08, height="450px"))
    sec07.add_element(TableElement(metrics_df, title="Quality Metrics"))
    report.add_section(sec07)

    # Robustness: permutation test on Exec vs Imag pooled separation AUC (0-2s)
    observed_curve = contrast_distance(
        trajectories,
        labels,
        [3, 4, 7, 8],
        [5, 6, 9, 10],
    )
    robustness_rows = []
    if observed_curve is not None:
        eval_mask = (times >= 0.0) & (times <= 2.0)
        observed_auc = compute_auc(observed_curve[eval_mask], times[eval_mask])

        n_perm = 200
        null_aucs = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            shuffled = shuffle_labels(labels, rng)
            shuffled_curve = contrast_distance(
                trajectories,
                shuffled,
                [3, 4, 7, 8],
                [5, 6, 9, 10],
            )
            if shuffled_curve is None:
                null_aucs[i] = np.nan
            else:
                null_aucs[i] = compute_auc(shuffled_curve[eval_mask], times[eval_mask])
        null_aucs = null_aucs[np.isfinite(null_aucs)]
        p_value = float((np.sum(null_aucs >= observed_auc) + 1) / (len(null_aucs) + 1))

        robustness_rows.append(
            {
                "contrast": "Exec vs Imag (pooled)",
                "window": "0-2s",
                "observed_auc": float(observed_auc),
                "n_permutations": int(len(null_aucs)),
                "p_value": p_value,
            }
        )

        fig_perm = go.Figure()
        fig_perm.add_trace(
            go.Histogram(
                x=null_aucs,
                nbinsx=30,
                name="Null AUC",
                marker_color="#9ecae1",
                opacity=0.8,
            )
        )
        fig_perm.add_vline(
            x=float(observed_auc),
            line_width=3,
            line_dash="dash",
            line_color="#d62728",
        )
        fig_perm.update_layout(
            title="Figure 13: Permutation Robustness (Exec vs Imag AUC, 0-2s)",
            xaxis_title="AUC under label shuffle",
            yaxis_title="Count",
            template="plotly_white",
            height=420,
        )
        fig_perm_path = figures_dir / "fig13_permutation_robustness.html"
        save_plotly(fig_perm, fig_perm_path)

        robustness_df = pd.DataFrame(robustness_rows)
        robustness_df.to_csv(metrics_dir / "robustness_permutation.csv", index=False)

        sec08 = Section("Figure 13: Robustness Check", icon="13")
        sec08.add_markdown(
            "Label-shuffle test on separation AUC. A small p-value indicates non-random condition separation."
        )
        sec08.add_element(PlotlyElement(fig_perm, height="420px"))
        sec08.add_element(TableElement(robustness_df, title="Permutation Summary"))
        report.add_section(sec08)

    # Optional method comparison: Figure-2 style scatter + Left/Right (Imag+Exec) per method
    compare_rows = []
    compare_scatter_trial_figs: Dict[str, go.Figure] = {}
    compare_scatter_state_figs: Dict[str, go.Figure] = {}
    compare_lr_figs: Dict[str, go.Figure] = {}

    compare_trial_n = min(60, n_trials)
    compare_trial_idx = np.sort(rng.choice(n_trials, size=compare_trial_n, replace=False))
    c_compare = container.isel(obs=compare_trial_idx)
    compare_trial_labels = np.asarray(c_compare.y).astype(int)
    compare_times = np.asarray(c_compare.coords["time"])
    compare_n_times = c_compare.X.shape[2]
    compare_trial_label_names = np.array(
        [LABEL_NAMES.get(int(x), str(int(x))) for x in compare_trial_labels]
    )
    compare_trial_mode = np.array([LABEL_INFO[int(x)]["mode"] for x in compare_trial_labels])
    compare_trial_laterality = np.array(
        [LABEL_INFO[int(x)]["laterality"] for x in compare_trial_labels]
    )
    compare_trial_effector = np.array(
        [LABEL_INFO[int(x)]["effector"] for x in compare_trial_labels]
    )
    compare_trial_side = np.array(
        [
            "left"
            if int(x) in (3, 5)
            else "right"
            if int(x) in (4, 6)
            else "bilateral"
            for x in compare_trial_labels
        ]
    )
    compare_trial_mode_side = np.array(
        [f"{m}_{s}" for m, s in zip(compare_trial_mode, compare_trial_side, strict=False)]
    )
    compare_trial_mode_effector = np.array(
        [
            f"{m}_{e}"
            for m, e in zip(compare_trial_mode, compare_trial_effector, strict=False)
        ]
    )
    compare_trial_meta = {
        "Label ID": compare_trial_labels,
        "Exec vs Imag": compare_trial_mode,
        "Right vs Left": compare_trial_side,
        "Unilateral vs Bilateral": compare_trial_laterality,
        "Effector": compare_trial_effector,
        "Exec/Imag x Side": compare_trial_mode_side,
        "Exec/Imag x Effector": compare_trial_mode_effector,
    }

    c_compare_stacked = c_compare.stack(dims=("obs", "time"), new_dim="obs")
    c_compare_stacked_z = c_compare_stacked.zscore(dim="obs")
    compare_X = c_compare_stacked_z.X
    compare_sample_labels = np.repeat(compare_trial_labels, compare_n_times).astype(int)
    compare_label_names = np.array(
        [LABEL_NAMES.get(int(x), str(int(x))) for x in compare_sample_labels]
    )
    compare_mode = np.array([LABEL_INFO[int(x)]["mode"] for x in compare_sample_labels])
    compare_laterality = np.array(
        [LABEL_INFO[int(x)]["laterality"] for x in compare_sample_labels]
    )
    compare_effector = np.array([LABEL_INFO[int(x)]["effector"] for x in compare_sample_labels])
    compare_side = np.array(
        [
            "left"
            if int(x) in (3, 5)
            else "right"
            if int(x) in (4, 6)
            else "bilateral"
            for x in compare_sample_labels
        ]
    )
    compare_mode_side = np.array(
        [f"{m}_{s}" for m, s in zip(compare_mode, compare_side, strict=False)]
    )
    compare_mode_effector = np.array(
        [f"{m}_{e}" for m, e in zip(compare_mode, compare_effector, strict=False)]
    )
    compare_meta = {
        "Label ID": compare_sample_labels,
        "Exec vs Imag": compare_mode,
        "Right vs Left": compare_side,
        "Unilateral vs Bilateral": compare_laterality,
        "Effector": compare_effector,
        "Exec/Imag x Side": compare_mode_side,
        "Exec/Imag x Effector": compare_mode_effector,
    }

    scatter_trial_titles = {
        "UMAP": "Figure 14a: UMAP 2a (one point per trial, time-mean)",
        "PHATE": "Figure 14b: PHATE 2a (one point per trial, time-mean)",
        "Isomap": "Figure 14c: Isomap 2a (one point per trial, time-mean)",
    }
    scatter_state_titles = {
        "UMAP": "Figure 14d: UMAP 2b (all trial x time states)",
        "PHATE": "Figure 14e: PHATE 2b (all trial x time states)",
        "Isomap": "Figure 14f: Isomap 2b (all trial x time states)",
    }
    lr_titles = {
        "UMAP": "Figure 14g: UMAP Left vs Right combining Imag+Exec",
        "PHATE": "Figure 14h: PHATE Left vs Right combining Imag+Exec",
        "Isomap": "Figure 14i: Isomap Left vs Right combining Imag+Exec",
    }

    for method_name in ["UMAP", "PHATE", "Isomap"]:
        try:
            reducer_cmp = DimReduction(method=method_name, n_components=3, random_state=42)
            emb_cmp_3d = reducer_cmp.fit_transform(compare_X)
            method_key = method_name.lower()

            cmp_coords = c_compare_stacked_z.coords.copy()
            cmp_coords["component"] = np.array(["Dim1", "Dim2", "Dim3"])
            c_cmp_reduced = replace(
                c_compare_stacked_z,
                X=emb_cmp_3d,
                dims=("obs", "component"),
                coords=cmp_coords,
            )
            c_cmp_traj = c_cmp_reduced.unstack("obs")
            c_cmp_traj = replace(
                c_cmp_traj,
                y=compare_trial_labels,
                coords={**c_cmp_traj.coords, "time": compare_times},
            )

            # 2a-style scatter (one point per trial = time-mean)
            emb_cmp_trial = c_cmp_traj.X.mean(axis=1)
            fig_scatter_trial = plot_embedding_interactive(
                embedding=emb_cmp_trial[:, :2],
                labels=compare_trial_label_names,
                meta=compare_trial_meta,
                title=scatter_trial_titles[method_name],
                dimensions=2,
            )
            save_plotly(
                fig_scatter_trial,
                figures_dir / f"fig14_{method_key}_scatter_trial_mean_pc12.html",
            )
            compare_scatter_trial_figs[method_name] = fig_scatter_trial

            # 2b-style scatter (all trial x time states)
            fig_scatter_state = plot_embedding_interactive(
                embedding=emb_cmp_3d[:, :2],
                labels=compare_label_names,
                meta=compare_meta,
                title=scatter_state_titles[method_name],
                dimensions=2,
            )
            save_plotly(
                fig_scatter_state,
                figures_dir / f"fig14_{method_key}_scatter_all_states_pc12.html",
            )
            compare_scatter_state_figs[method_name] = fig_scatter_state

            lr_idx_cmp = np.where(np.isin(c_cmp_traj.y, [3, 4, 5, 6]))[0]
            if lr_idx_cmp.size == 0:
                raise ValueError("missing labels 3/4/5/6 for Left/Right pooled trajectory")
            c_lr_cmp = c_cmp_traj.isel(obs=lr_idx_cmp)
            lr_groups_cmp = np.where(np.isin(c_lr_cmp.y, [3, 5]), "Left", "Right")
            c_lr_centroids_cmp = c_lr_cmp.aggregate(by=lr_groups_cmp, method="mean")
            obs_lr_cmp = np.asarray(c_lr_centroids_cmp.coords["obs"]).astype(str)
            if "Left" not in obs_lr_cmp or "Right" not in obs_lr_cmp:
                raise ValueError("could not compute both pooled Left and Right centroids")

            left_curve_cmp = c_lr_centroids_cmp.X[np.where(obs_lr_cmp == "Left")[0][0]]
            right_curve_cmp = c_lr_centroids_cmp.X[np.where(obs_lr_cmp == "Right")[0][0]]
            lr_curve_stack_cmp = np.vstack([left_curve_cmp, right_curve_cmp])
            lr_groups_plot_cmp = np.repeat(["Left", "Right"], len(compare_times))
            lr_times_plot_cmp = np.tile(compare_times, 2)

            fig_lr = plot_trajectory_interactive(
                X=lr_curve_stack_cmp,
                times=lr_times_plot_cmp,
                groups=lr_groups_plot_cmp,
                title=lr_titles[method_name],
                dimensions=3,
                smooth_window=8,
                mode="lines",
                line_width=8,
                group_cmaps={"Left": "Greens", "Right": "Reds"},
                show_group_colorbars=True,
            )
            fig_lr.update_layout(
                scene=dict(xaxis_title="Dim1", yaxis_title="Dim2", zaxis_title="Dim3")
            )
            save_plotly(
                fig_lr,
                figures_dir / f"fig14_{method_key}_left_vs_right_combining_imag_exec.html",
            )
            compare_lr_figs[method_name] = fig_lr

            cmp_score_n = min(800, compare_X.shape[0])
            cmp_idx = rng.choice(compare_X.shape[0], size=cmp_score_n, replace=False)
            cmp_payload = reducer_cmp.score(
                compare_X[cmp_idx],
                X_emb=emb_cmp_3d[cmp_idx],
                n_neighbors=max(5, min(15, cmp_score_n - 2)),
            )
            cmp_metrics = cmp_payload.get("metrics", {})
            row = {"method": method_name, "status": "ok"}
            for key in ["trustworthiness", "continuity", "lcmc", "shepard_correlation"]:
                row[key] = cmp_metrics.get(key, np.nan)
            compare_rows.append(row)
        except Exception as exc:
            compare_rows.append({"method": method_name, "status": f"unavailable: {exc}"})

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(metrics_dir / "optional_method_comparison.csv", index=False)

    available_scatter_trial = [
        m for m in ["UMAP", "PHATE", "Isomap"] if m in compare_scatter_trial_figs
    ]
    available_scatter_state = [
        m for m in ["UMAP", "PHATE", "Isomap"] if m in compare_scatter_state_figs
    ]
    available_lr = [m for m in ["UMAP", "PHATE", "Isomap"] if m in compare_lr_figs]
    if available_scatter_trial or available_scatter_state or available_lr:
        sec09 = Section("Optional Method Comparison", icon="O")
        sec09.add_markdown(
            "For each method: 2a-style trial-mean scatter, 2b-style all-state scatter, and Left/Right pooled trajectory."
        )
        for method_name in ["UMAP", "PHATE", "Isomap"]:
            if method_name in compare_scatter_trial_figs:
                sec09.add_element(
                    PlotlyElement(compare_scatter_trial_figs[method_name], height="460px")
                )
            if method_name in compare_scatter_state_figs:
                sec09.add_element(
                    PlotlyElement(compare_scatter_state_figs[method_name], height="460px")
                )
            if method_name in compare_lr_figs:
                sec09.add_element(PlotlyElement(compare_lr_figs[method_name], height="650px"))
        sec09.add_element(TableElement(compare_df, title="Method Comparison Summary"))
        report.add_section(sec09)

    # -------------------------------------------------------------------------
    # Stage I: Export report and artifact registry
    # -------------------------------------------------------------------------
    report_path = output_root / "neural_trajectory_report.html"
    report.save(str(report_path))

    artifact_index = pd.DataFrame(
        [
            {"type": "figure", "path": str(p)}
            for p in sorted(figures_dir.glob("*.html"))
        ]
        + [{"type": "table", "path": str(p)} for p in sorted(tables_dir.glob("*.csv"))]
        + [{"type": "metric", "path": str(p)} for p in sorted(metrics_dir.glob("*.csv"))]
    )
    artifact_index.to_csv(output_root / "artifact_index.csv", index=False)

    print(f"Report saved: {report_path}")
    print(f"Figures directory: {figures_dir}")
    print(f"Tables directory: {tables_dir}")
    print(f"Metrics directory: {metrics_dir}")
