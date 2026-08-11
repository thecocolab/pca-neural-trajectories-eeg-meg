"""Main PCA neural-trajectory analysis on PhysioNet EEGBCI (109 subjects).

Headless full-cohort counterpart to the main tutorial notebook
(`tutorials/tutorial_eegbci_main.ipynb`). Runs the full 10-step
PCA-trajectory workflow and produces a standalone HTML report
that mirrors the same steps as the tutorial notebook.

Outputs (under ``--output``)::

    main_figure.svg / .html        — 5-panel analysis composite
    report.html                    — 10-step interactive companion report
    artifacts/                     — trial/ERP + per-subject reducers, container, CSVs

Usage
-----
::

    python scripts/analysis_eegbci_main.py \\
        --subjects 1 2 3 ... 109 \\
        --output outputs/eegbci_main \\
        --n-perm 1000

By default runs on all 109 subjects with ``n_perm=1000`` for the
permutation null. Pass ``--subjects 1 2 3`` to dry-run on a small
slice (matches the tutorial mode).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from coco_pipe.dim_reduction import (
    DimReduction,
    apply_pca_score_baseline,
    flip_pc_scores_for_consistency,
    paired_condition_stats,
    trajectory_dispersion,
    trajectory_speed,
)
from coco_pipe.dim_reduction.evaluation import (
    TrajectoryResult,
    permutation_null_separation_auc,
)
from coco_pipe.report import PlotlyElement, Report, Section
from coco_pipe.report.elements import (
    AccordionElement,
    InteractiveTableElement,
    MarkdownElement,
)
from coco_pipe.viz.interactive import (
    plot_bar,
    plot_distribution_groups,
    plot_heatmap,
    plot_null_interval_summary,
    plot_scree,
    plot_shepard_diagram,
    plot_trajectory,
    plot_trajectory_metric_series,
    plot_trajectory_separation,
)
from plotly.subplots import make_subplots

from pca_neural_trajectories import (
    LABEL_NAMES,
    format_pair_keys,
    load_eegbci_container,
    save_artifacts,
    setup_data_bids,
    write_manifest,
)

# ---------------------------------------------------------------------------
# EEGBCI condition metadata (mirrors tutorials/tutorial_eegbci_main.ipynb)
# ---------------------------------------------------------------------------
CONDITION_COLORS: dict[int, str] = {
    3: "#0072B2",
    5: "#0072B2",
    7: "#0072B2",
    9: "#0072B2",
    4: "#D55E00",
    6: "#D55E00",
    8: "#D55E00",
    10: "#D55E00",
}

CONDITION_LINESTYLES: dict[int, str] = {
    3: "solid",
    4: "solid",
    7: "solid",
    8: "solid",
    5: "dash",
    6: "dash",
    9: "dash",
    10: "dash",
}

# Keep the full-cohort analysis identical to the introductory notebook. The
# subject count changes, but the selected conditions and every analysis step do
# not.
ANALYSIS_CONDITIONS: tuple[int, ...] = (3, 4, 5, 6)

EXCLUDED_PAIRS: tuple[str, ...] = (
    "Left Hand (Exec) vs Right Hand (Imag)",
    "Right Hand (Exec) vs Left Hand (Imag)",
)

PAIR_COLORS: dict[str, str] = {
    "Left Hand (Imag) vs Right Hand (Imag)": "#d62728",
    "Left Hand (Exec) vs Right Hand (Exec)": "gray",
    "Right Hand (Exec) vs Right Hand (Imag)": "#009E73",
    "Left Hand (Exec) vs Left Hand (Imag)": "#CC79A7",
}


def _baseline_scores(
    X: np.ndarray,
    times: np.ndarray,
    pc_names: list[str],
) -> np.ndarray:
    """Baseline-correct per-trial PC scores exactly as in notebook Step 6.

    ``X`` is (trial, time, component); the coco-pipe helpers take
    (component, time) frames, hence the transposes.
    """
    out = np.empty_like(X)
    for trial_index in range(X.shape[0]):
        scores = apply_pca_score_baseline(
            times,
            pd.DataFrame(X[trial_index].T, index=pc_names),
            baseline_min_ms=-0.2,
            baseline_max_ms=0.0,
        )
        out[trial_index] = scores.to_numpy().T
    return out


def _resolve_subjects(spec: list[int] | None) -> list[int]:
    if spec is None or not spec:
        return list(range(1, 110))
    return [int(s) for s in spec]


def run_main_analysis(
    subjects: list[int] | None,
    output: Path | str,
    bids_root: Path | str = "PhysioNet_EEGBCI/BIDS",
    n_perm: int = 1000,
    n_components: int = 10,
    analysis_window: tuple[float, float] = (-0.2, 1.0),
    random_state: int = 42,
    force_reconvert: bool = False,
    clean_artifacts: bool = True,
    n_ica_components: int = 20,
) -> dict:
    """Run the main 10-step PCA-trajectory workflow across the full cohort.

    See module docstring for output layout.
    """
    rng = np.random.default_rng(random_state)
    subject_list = _resolve_subjects(subjects)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out / "artifacts"

    print("=== EEGBCI main analysis ===")
    print(f"subjects: {subject_list[0]}..{subject_list[-1]} ({len(subject_list)} total)")
    print(f"output:   {out}")

    # --------------------------------------------------------------- Steps 1 and 2
    # Step 1: represent each state in the 64-channel EEG sensor space.
    # Step 2: run the same cleaning, loading, crop, and normalization as the
    # notebook. Only the number of requested subjects differs.
    setup_data_bids(
        subjects=subject_list,
        runs=list(range(3, 15)),
        root=bids_root,
        force_reconvert=force_reconvert,
        clean_artifacts=clean_artifacts,
        n_ica_components=n_ica_components,
    )

    container = load_eegbci_container(
        bids_root,
        subjects=[f"{subject:03d}" for subject in subject_list],
        conditions=ANALYSIS_CONDITIONS,
        runs=[f"{r:02d}" for r in range(3, 15)],
        tmin=analysis_window[0] - 1.0,
        tmax=analysis_window[1] + 1.5,
        baseline=(-1.0, 0.0),
    )

    times_full = np.asarray(container.coords["time"])
    time_mask = (times_full >= analysis_window[0] - 1e-8) & (
        times_full <= analysis_window[1] + 1e-8
    )
    container = container.isel(time=time_mask).zscore(
        dim=("obs", "time"), eps=1e-5
    )
    times = np.asarray(container.coords["time"])
    subject_ids = np.asarray(container.coords["subject"])
    trial_labels = np.asarray(container.y).astype(int)
    analyzed_subjects = np.unique(subject_ids)
    n_analyzed = len(analyzed_subjects)
    if n_analyzed == 0:
        raise RuntimeError("No analyzable EEGBCI subjects were loaded.")
    print(f"analyzable subjects: {n_analyzed}/{len(subject_list)}")
    print(f"analysis tensor shape: {container.X.shape}")

    # --------------------------------------------------------------------- Step 3
    # Make one subject-by-condition ERP before stacking, alongside the pooled
    # single-trial stack used by the second shared PCA.
    plot_groups = np.asarray(
        [
            f"{subject}|{condition}"
            for subject, condition in zip(subject_ids, trial_labels, strict=True)
        ]
    )
    container_plot_erp = container.aggregate(by=plot_groups, stats="mean")
    aggregate_keys = np.asarray(container_plot_erp.coords["obs"]).astype(str)
    container_plot_erp.y = np.asarray(
        [int(key.split("|")[1]) for key in aggregate_keys]
    )
    container_plot_erp.coords["subject"] = np.asarray(
        [key.split("|")[0] for key in aggregate_keys]
    )

    container_plot_pooled_trials = container.stack(
        dims=("obs", "time"), new_dim="obs"
    )
    container_plot_pooled_erp = container_plot_erp.stack(
        dims=("obs", "time"), new_dim="obs"
    )
    print(f"Single-trial stacked shape: {container_plot_pooled_trials.X.shape}")
    print(f"ERP stacked shape:          {container_plot_pooled_erp.X.shape}")

    per_subject_fit_stacks = {
        sub_id: container.isel(obs=subject_ids == sub_id).stack(
            dims=("obs", "time"), new_dim="obs"
        )
        for sub_id in analyzed_subjects
    }

    # --------------------------------------------------------------------- Step 4
    # Match the notebook's two shared models: single-trial PCA for separation
    # and kinematics, ERP PCA for group figures and embedding validation.
    print("Fitting shared single-trial and ERP PCAs …")
    shared_reducer_trials = DimReduction(
        method="PCA", n_components=n_components, random_state=random_state
    )
    shared_reducer_erp = DimReduction(
        method="PCA", n_components=n_components, random_state=random_state
    )
    shared_scores_trials = shared_reducer_trials.fit_transform(
        container_plot_pooled_trials.X
    )
    shared_scores_erp = shared_reducer_erp.fit_transform(container_plot_pooled_erp.X)
    evr_trials = np.asarray(
        shared_reducer_trials.get_diagnostics().get("explained_variance_ratio_")
    )
    evr_erp = np.asarray(
        shared_reducer_erp.get_diagnostics().get("explained_variance_ratio_")
    )

    pc_names = [f"PC{i + 1}" for i in range(n_components)]
    print("Fitting per-subject PCA …")
    per_subject_reducers: dict = {}
    per_subject_scores: dict = {}
    per_subject_scores_flipped: dict = {}
    for sub_id, stack in per_subject_fit_stacks.items():
        reducer = DimReduction(
            method="PCA", n_components=n_components, random_state=random_state
        )
        scores = reducer.fit_transform(stack.X)
        per_subject_reducers[sub_id] = reducer
        per_subject_scores[sub_id] = scores

        times_tiled = np.tile(times, scores.shape[0] // len(times))
        score_frame = pd.DataFrame(scores.T, index=pc_names)
        per_subject_scores_flipped[sub_id] = flip_pc_scores_for_consistency(
            scores=score_frame,
            time=times_tiled,
            flip_window_ms=(0.0, 1.0),
        ).values.T

    # --------------------------------------------------------------------- Step 5
    pr_trials = shared_reducer_trials.get_diagnostics().get("participation_ratio_")
    pr_erp = shared_reducer_erp.get_diagnostics().get("participation_ratio_")
    print(
        "Single-trial cumulative variance @ 3 PCs: "
        f"{float(np.sum(evr_trials[:3])):.2%}; PR={float(pr_trials):.2f}"
    )
    print(
        "ERP cumulative variance @ 3 PCs: "
        f"{float(np.sum(evr_erp[:3])):.2%}; PR={float(pr_erp):.2f}"
    )
    fig_scree_trials = plot_scree(evr_trials)
    fig_scree_trials.update_layout(title="Step 5a — Scree Plot (Single-Trial PCA)")
    fig_scree_erp = plot_scree(evr_erp)
    fig_scree_erp.update_layout(title="Step 5b — Scree Plot (ERP PCA)")

    # --------------------------------------------------------------------- Step 6
    container_pc_trials = container_plot_pooled_trials.with_features(
        shared_scores_trials,
        names=pc_names,
        new_dim_name="component",
    )
    container_traj_trials = container_pc_trials.unstack("obs")
    container_pc_erp = container_plot_pooled_erp.with_features(
        shared_scores_erp,
        names=pc_names,
        new_dim_name="component",
    )
    container_traj_erp = container_pc_erp.unstack("obs")
    container_traj_trials.X = _baseline_scores(
        container_traj_trials.X, times, pc_names
    )
    container_traj_erp.X = _baseline_scores(container_traj_erp.X, times, pc_names)

    # --------------------------------------------------------------------- Step 7
    container_traj_trials_agg = container_traj_trials.aggregate(
        by="y", stats=["mean", "sem"]
    )
    container_traj_erp_agg = container_traj_erp.aggregate(
        by="y", stats=["mean", "sem"]
    )
    traj_mean_trials = container_traj_trials_agg.X[:, 0, ...]
    traj_sem_trials = container_traj_trials_agg.X[:, 1, ...]
    traj_mean_erp = container_traj_erp_agg.X[:, 0, ...]
    traj_sem_erp = container_traj_erp_agg.X[:, 1, ...]
    print(f"Single-trial mean / SEM shapes: {traj_mean_trials.shape} / {traj_sem_trials.shape}")
    print(f"ERP mean / SEM shapes:          {traj_mean_erp.shape} / {traj_sem_erp.shape}")
    unique_conds = np.asarray(container_traj_trials_agg.coords["obs"]).astype(int)
    cond_names = np.asarray([LABEL_NAMES[condition] for condition in unique_conds])
    color_map_str = {
        LABEL_NAMES[key]: value for key, value in CONDITION_COLORS.items()
    }
    dash_map_str = {
        LABEL_NAMES[key]: value for key, value in CONDITION_LINESTYLES.items()
    }

    fig_traj_2d = plot_trajectory(
        X=traj_mean_erp[..., :2],
        times=times,
        labels=cond_names,
        sem=traj_sem_erp[..., :2],
        color_map=color_map_str,
        linestyle_map=dash_map_str,
        title="Step 7 — Subject-mean ERP trajectories (PC1–PC2)",
        dimensions=2,
        smooth_window=30,
    )
    fig_traj_3d = plot_trajectory(
        X=traj_mean_erp[..., :3],
        times=times,
        labels=cond_names,
        color_map=color_map_str,
        linestyle_map=dash_map_str,
        title="Step 7 — Subject-mean ERP trajectories (PC1–PC2–PC3)",
        dimensions=3,
        smooth_window=12,
        show_markers=False,
        add_start_end_markers=True,
    )

    # --------------------------------------------------------------------- Step 8
    traj_res_trials = TrajectoryResult(
        trajectories=container_traj_trials.X,
        times=times,
        subjects=np.asarray(container_traj_trials.coords["subject"]),
        conditions=np.asarray(container_traj_trials.y).astype(int),
    )
    traj_res_erp = TrajectoryResult(
        trajectories=container_traj_erp.X,
        times=times,
        subjects=np.asarray(
            container_traj_erp.coords.get(
                "subject", np.ones(container_traj_erp.X.shape[0])
            )
        ),
        conditions=np.asarray(container_traj_erp.y).astype(int),
    )
    sep_timecourses_trials = traj_res_trials.get_separation_timecourses(
        methods=["centroid", "mahalanobis"]
    )
    sep_timecourses_erp = traj_res_erp.get_separation_timecourses(methods=["centroid"])

    fig_sep_c = plot_trajectory_separation(
        format_pair_keys(
            sep_timecourses_trials["centroid"],
            LABEL_NAMES,
            exclude_pairs=EXCLUDED_PAIRS,
        ),
        times=times,
        title="Step 8a — Euclidean Separation Timecourse (Single-Trial PCA)",
        color_map=PAIR_COLORS,
        smooth_window=10,
    )
    fig_sep_m = plot_trajectory_separation(
        format_pair_keys(
            sep_timecourses_trials["mahalanobis"],
            LABEL_NAMES,
            exclude_pairs=EXCLUDED_PAIRS,
        ),
        times=times,
        title="Step 8b — Mahalanobis Separation Timecourse (Single-Trial PCA)",
        color_map=PAIR_COLORS,
        smooth_window=10,
    )
    fig_sep_erp = plot_trajectory_separation(
        format_pair_keys(
            sep_timecourses_erp["centroid"],
            LABEL_NAMES,
            exclude_pairs=EXCLUDED_PAIRS,
        ),
        times=times,
        title="Step 8c — Euclidean Separation Timecourse (ERP PCA)",
        color_map=PAIR_COLORS,
        smooth_window=10,
    )

    pair_scalars = traj_res_trials.get_separation_pair_scalars(
        methods=["centroid", "mahalanobis"]
    )
    pair_scalars["pair_name"] = pair_scalars["pair"].apply(
        lambda pair: (
            f"{LABEL_NAMES[int(pair.split('_vs_')[0])]} vs "
            f"{LABEL_NAMES[int(pair.split('_vs_')[1])]}"
        )
    )
    pair_scalars = pair_scalars[
        ~pair_scalars["pair_name"].isin(EXCLUDED_PAIRS)
    ].copy()

    peak_figures = {}
    for method, title, ylabel in (
        ("centroid", "Centroid", "Peak Distance (a.u.)"),
        ("mahalanobis", "Mahalanobis", "Mahalanobis Distance (D)"),
    ):
        peak_mask = (pair_scalars["method"] == method) & (
            pair_scalars["metric"] == "peak_separation"
        )
        peak_summary = pair_scalars.loc[peak_mask].groupby("pair_name")[
            "value"
        ].agg(["mean", "sem"])
        peak_figures[method] = plot_bar(
            scores=peak_summary["mean"],
            errors=peak_summary["sem"],
            title=f"Step 8d — Peak Separation per Pair ({title})",
            yaxis_title=ylabel,
        )

    _auc_mask = pair_scalars["metric"] == "auc_separation"
    auc_pivot = pair_scalars.loc[_auc_mask].pivot_table(
        index="pair_name", columns="method", values="value", aggfunc="mean"
    )
    fig_auc_heat = plot_heatmap(
        auc_pivot,
        annotate=True,
        annotation_format=".2f",
        title="Step 8d — Mean AUC per pair × method",
        xaxis_title="method",
        yaxis_title="condition pair",
    )

    auc_centroid_mask = (pair_scalars["method"] == "centroid") & (
        pair_scalars["metric"] == "auc_separation"
    )
    auc_long = pair_scalars.loc[auc_centroid_mask].copy()
    paired_stats = paired_condition_stats(
        scalar_df=auc_long,
        conditions=list(auc_long["pair_name"].unique()),
        condition_col="pair_name",
        subject_col="subject",
        metric_col="metric",
        value_col="value",
    )

    # --------------------------------------------------------------------- Step 9
    # The notebook computes every metric from the baselined shared single-trial
    # PCA trajectories. Keep that exact basis here as well.
    trial_labels = np.asarray(container_traj_trials.y).astype(int)
    all_speeds = trajectory_speed(container_traj_trials.X, time=times)
    condition_labels = np.asarray([LABEL_NAMES[label] for label in trial_labels])
    fig_speed = plot_trajectory_metric_series(
        series=all_speeds,
        times=times[1:],
        labels=condition_labels,
        title="Step 9a — Trajectory Speed per Condition",
        ylabel="Speed (a.u./s)",
        color_map=color_map_str,
        linestyle_map=dash_map_str,
        smooth_window=10,
    )

    spread_dict = {
        LABEL_NAMES[condition]: trajectory_dispersion(
            container_traj_trials.X[trial_labels == condition]
        )
        for condition in unique_conds
    }
    fig_spread = plot_trajectory_metric_series(
        spread_dict,
        times=times,
        title="Step 9b — Within-Condition Spread (Dispersion)",
        ylabel="Within-Group Spread (a.u.)",
        color_map=color_map_str,
        linestyle_map=dash_map_str,
        smooth_window=2,
    )

    all_distances = np.linalg.norm(traj_res_trials.trajectories, axis=-1)
    fig_distance = plot_trajectory_metric_series(
        series=all_distances,
        times=times,
        labels=condition_labels,
        title="Step 9c — Distance from Baseline Origin (Magnitude)",
        ylabel="Distance from Origin (a.u.)",
        color_map=color_map_str,
        linestyle_map=dash_map_str,
        smooth_window=5,
    )

    scalar_metrics = traj_res_trials.get_per_trial_scalars()
    condition_scalars = traj_res_trials.get_per_condition_scalars()
    scalar_metrics_all = pd.concat(
        [scalar_metrics.assign(level="trial"), condition_scalars.assign(level="condition")],
        ignore_index=True,
    )
    scalar_metrics_all["condition_name"] = scalar_metrics_all["condition"].map(LABEL_NAMES)

    path_mask = scalar_metrics_all["metric"] == "trajectory_length"
    path_summary = scalar_metrics_all.loc[path_mask].groupby("condition_name")[
        "value"
    ].agg(["mean", "sem"])
    fig_path = plot_bar(
        scores=path_summary["mean"],
        errors=path_summary["sem"],
        title="Step 9 — Mean path length per condition",
        yaxis_title="Path Length (a.u.)",
    )

    tortuosity = scalar_metrics_all[
        scalar_metrics_all["metric"] == "tortuosity"
    ]
    tortuosity_labels = [LABEL_NAMES[condition] for condition in unique_conds]
    fig_tort = plot_distribution_groups(
        groups=[
            tortuosity.loc[
                tortuosity["condition_name"] == label, "value"
            ].values
            for label in tortuosity_labels
        ],
        labels=tortuosity_labels,
        kind="violin",
        color=[color_map_str[label] for label in tortuosity_labels],
        title='Step 9 — Trajectory Tortuosity (Neural "Wandering")',
        yaxis_title="Ratio (Path Length / Displacement)",
    )

    # Keep the artifact table long-form while deriving it from the same shared
    # single-trial trajectories used by the notebook's speed figure.
    continuous_rows = []
    for subject in analyzed_subjects:
        for condition in unique_conds:
            rows = (subject_ids == subject) & (trial_labels == condition)
            if not rows.any():
                continue
            subject_speed = np.nanmean(
                trajectory_speed(container_traj_trials.X[rows], time=times), axis=0
            )
            continuous_rows.extend(
                {
                    "subject": subject,
                    "condition": condition,
                    "condition_name": LABEL_NAMES[condition],
                    "metric": "speed",
                    "time": float(time),
                    "value": float(value),
                }
                for time, value in zip(times[1:], subject_speed, strict=True)
            )
    continuous_metrics = pd.DataFrame.from_records(continuous_rows)

    # -------------------------------------------------------------------- Step 10
    print(f"Permutation null on separation AUC (n_perm={n_perm}) …")
    score_n = min(1500, shared_scores_erp.shape[0])
    score_idx = rng.choice(shared_scores_erp.shape[0], size=score_n, replace=False)
    score_payload = shared_reducer_erp.score(
        X_emb=shared_scores_erp[score_idx, :3],
        X=container_plot_pooled_erp.X[score_idx],
        n_neighbors=min(20, score_n - 2),
        metrics=["trustworthiness", "continuity"],
    )
    quality = score_payload.get("metrics", {})
    obs_auc, null_auc = permutation_null_separation_auc(
        result=traj_res_trials,
        group_a=[3, 4],
        group_b=[5, 6],
        n_perm=n_perm,
        rng=rng,
        window=(0.0, 1.0),
    )
    p_value = float((np.sum(null_auc >= obs_auc) + 1) / (len(null_auc) + 1))
    print(
        f"trustworthiness={quality.get('trustworthiness', float('nan')):.3f}, "
        f"continuity={quality.get('continuity', float('nan')):.3f}, "
        f"p(AUC)={p_value:.4f}"
    )

    fig_shepard = plot_shepard_diagram(
        X_orig=container_plot_pooled_erp.X[score_idx],
        X_emb=shared_scores_erp[score_idx, :3],
        sample_size=300,
        title="Step 10 — Shepard diagram (3D vs 64D)",
    )

    # Same null-summary frame the tutorial builds for this plot.
    _null_frame = pd.DataFrame(
        {
            "Model": ["PCA Trajectory"],
            "Metric": ["AUC"],
            "Observed": [obs_auc],
            "NullLower": [float(np.percentile(null_auc, 2.5))],
            "NullUpper": [float(np.percentile(null_auc, 97.5))],
        }
    )
    fig_null = plot_null_interval_summary(
        _null_frame,
        title=(
            f"Step 10 — Permutation Null (Exec vs Imag)<br>"
            f"Empirical p-value: {p_value:.3f}"
        ),
    )

    # ------------------------------------------------------------------ Composite
    print("Assembling 5-panel main composite …")
    fig_main = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "scene"}],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy"}],
        ],
        subplot_titles=(
            "A — Scree",
            "B — Trial-mean PC1–PC2",
            "C — 3D trajectories",
            "D — Speed timecourse",
            "E — Separation (centroid + Mahalanobis)",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Panel A: ERP scree, matching the notebook's group-figure basis.
    fig_main.add_trace(
        go.Bar(
            x=[f"PC{i + 1}" for i in range(len(evr_erp))],
            y=evr_erp,
            name="Var.",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Panel B: trial-mean PC1–PC2 scatter
    trial_means = container_traj_trials.X[..., :3].mean(axis=1)
    for cond in unique_conds:
        mask = trial_labels == cond
        fig_main.add_trace(
            go.Scatter(
                x=trial_means[mask, 0],
                y=trial_means[mask, 1],
                mode="markers",
                marker=dict(color=CONDITION_COLORS.get(cond, "#444"), size=4, opacity=0.5),
                name=LABEL_NAMES[cond],
                legendgroup=LABEL_NAMES[cond],
            ),
            row=1,
            col=2,
        )

    # Panel C: 3D group-mean trajectories
    for cond, traj in zip(unique_conds, traj_mean_erp, strict=True):
        fig_main.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines",
                line=dict(color=CONDITION_COLORS.get(cond, "#444"), width=5),
                name=LABEL_NAMES[cond],
                legendgroup=LABEL_NAMES[cond],
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    # Panel D: the notebook's shared single-trial PCA speed with trial-level SEM.
    speed_mean = [
        np.nanmean(all_speeds[trial_labels == condition], axis=0)
        for condition in unique_conds
    ]
    speed_sem = [
        np.nanstd(all_speeds[trial_labels == condition], axis=0)
        / np.sqrt(np.sum(trial_labels == condition))
        for condition in unique_conds
    ]
    for cond, mean_curve, sem_curve in zip(
        unique_conds, speed_mean, speed_sem, strict=True
    ):
        fig_main.add_trace(
            go.Scatter(
                x=times[1:],
                y=mean_curve,
                mode="lines",
                line=dict(color=CONDITION_COLORS.get(cond, "#444"), width=2),
                error_y=dict(
                    type="data", array=sem_curve, visible=True, thickness=0.5
                ),
                name=LABEL_NAMES[cond],
                legendgroup=LABEL_NAMES[cond],
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Panel E: separation timecourses (centroid + Mahalanobis overlaid)
    for method, dash in [("centroid", "solid"), ("mahalanobis", "dash")]:
        formatted = format_pair_keys(
            sep_timecourses_trials[method],
            LABEL_NAMES,
            exclude_pairs=EXCLUDED_PAIRS,
        )
        for pair_name, curve in formatted.items():
            fig_main.add_trace(
                go.Scatter(
                    x=times,
                    y=curve,
                    mode="lines",
                    line=dict(width=2, dash=dash),
                    name=f"{pair_name} ({method})",
                ),
                row=2,
                col=3,
            )

    fig_main.update_layout(
        title=f"PCA Neural Trajectory Analysis — EEGBCI ({n_analyzed} subjects)",
        height=820,
        width=1400,
        template="plotly_white",
    )

    fig_main.write_image(str(out / "main_figure.png"), scale=2)
    fig_main.write_image(str(out / "main_figure.svg"))
    fig_main.write_html(str(out / "main_figure.html"), include_plotlyjs="inline")

    # ------------------------------------------------------------------ Report (10-step)
    report = Report(
        title=f"EEGBCI — PCA Neural Trajectories ({n_analyzed} subjects)",
        asset_urls="inline",
    )

    # ── Overview ────────────────────────────────────────────────────────────
    sec_overview = Section("Overview", icon="O")
    sec_overview.add_element(
        MarkdownElement(
            f"This companion report runs the full **10-step PCA neural-trajectory workflow** "
            f"from *'A Primer on Low-Dimensional Neural Dynamics'* across the full cohort "
            f"({n_analyzed} analyzable subjects from {len(subject_list)} requested).\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Subjects | {n_analyzed} analyzable / {len(subject_list)} requested |\n"
            f"| Conditions | {', '.join(LABEL_NAMES[c] for c in unique_conds)} |\n"
            f"| Components fit | {n_components} (3 used for visualization) |\n"
            f"| Single-trial variance @ 3 PCs | {float(np.sum(evr_trials[:3])):.2%} |\n"
            f"| ERP variance @ 3 PCs | {float(np.sum(evr_erp[:3])):.2%} |\n"
            f"| Single-trial / ERP PR | {float(pr_trials):.2f} / {float(pr_erp):.2f} |\n"
            f"| Trustworthiness | {quality.get('trustworthiness', float('nan')):.3f} |\n"
            f"| Continuity | {quality.get('continuity', float('nan')):.3f} |\n"
            f"| Permutation p (AUC, n_perm={n_perm}) | {p_value:.4f} |\n\n"
            f"> Each section below corresponds to one numbered step in the tutorial notebook "
            f"`tutorials/tutorial_eegbci_main.ipynb`."
        )
    )
    report.add_section(sec_overview)

    # ── Step 1 — Choose Your Representation ─────────────────────────────────
    sec1 = Section("Step 1 — Choose Your Representation", icon="1")
    sec1.add_element(
        MarkdownElement(
            "The first question in any neural trajectory analysis is: *what is the "
            '"state" of the brain at time t?*\n\n'
            "We choose the **sensor-space voltage pattern** — a vector of 64 simultaneous "
            "EEG channel amplitudes:\n\n"
            "**x**(t) = [v₁(t), v₂(t), …, v₆₄(t)]ᵀ ∈ ℝ⁶⁴\n\n"
            "Each time point is therefore a single point in a 64-dimensional state space, "
            "and each trial traces out a **trajectory** through that space. Dimensionality "
            "reduction (PCA) will reveal a low-dimensional manifold — typically 2–4 "
            "dimensions — that captures the dominant variance across trials and conditions.\n\n"
            "**Dataset:** PhysioNet EEGBCI — "
            f"{n_analyzed} analyzable subjects and the same four conditions selected in "
            "the notebook: left/right hand execution and left/right hand imagination."
        )
    )
    report.add_section(sec1)

    # ── Step 2 — Preprocess ─────────────────────────────────────────────────
    sec2 = Section("Step 2 — Preprocess & Normalize", icon="2")
    sec2.add_element(
        MarkdownElement(
            "> **Why preprocessing is critical for PCA:** PCA does not know what is "
            '"neural signal" and what is "noise" — it simply finds the directions of '
            "maximum variance. If you don't remove eye blinks and heartbeats, those huge "
            "artifacts will completely dominate Principal Component 1!\n\n"
            "Before extracting neural trajectories, raw EEG data must be cleaned and "
            "epoched. Three sequential phases are applied:\n\n"
            "**Phase 1 — Download & Preprocess (MNE-Python)**\n\n"
            "The `setup_data_bids` helper executes a state-of-the-art MNE preprocessing "
            "pipeline and writes the epoched data to a standardized BIDS directory:\n\n"
            "1. **Bad-channel interpolation** — outlier channels (LOF) are "
            "interpolated before referencing so a bad sensor cannot contaminate "
            "the common-average reference or the ICA.\n"
            "2. **Common-average reference.**\n"
            "3. **ICA artifact removal** — a 20-component extended-infomax ICA is "
            "fit on a 1 Hz high-passed copy (the algorithm and band `mne-icalabel` "
            "expects) to identify and subtract **blinks** and **heartbeats**.\n"
            "4. **Bandpass filter** 0.5–40 Hz (FIR) — attenuates slow drifts; the "
            "40 Hz low-pass also suppresses broadband muscle and residual line "
            "noise, so no separate notch is needed.\n\n"
            "**Phase 2 — Load into coco-pipe**\n\n"
            "The preprocessed BIDS directory is parsed by `BIDSDataset`, which produces "
            "a `DataContainer` with shape `[Trials × Channels × Time]`.\n\n"
            "**Phase 3 — Crop, Baseline, and Z-score**\n\n"
            "- **Baseline** sensors in the loader using the pre-stimulus interval.\n"
            "- **Crop** to the common `-0.2s–1.0s` analysis window.\n"
            "- **Z-score** each channel across observations to prevent high-amplitude "
            "sensors from dominating PCA variance.\n"
            "- **Baseline again in PC space** over `-0.2s–0.0s` after projection."
        )
    )
    report.add_section(sec2)

    # ── Step 3 — Reshape ────────────────────────────────────────────────────
    sec3 = Section("Step 3 — Reshape into (Samples × Features)", icon="3")
    sec3.add_element(
        MarkdownElement(
            "To apply PCA, our 3-dimensional neural data must be reshaped into a flat "
            "2-dimensional matrix. Two crucial methodological choices govern this step:\n\n"
            "- **Single-Trial Preservation:** We do *not* average trials before PCA. "
            "Every single timepoint of every trial is treated as a separate independent "
            "observation, capturing true biological variance across individual repetitions.\n"
            "- **Subject Normalization:** We explore two model types — a *Shared PCA* "
            "(all subjects pooled) and *Per-Subject PCAs* — to disentangle between-subject "
            "differences from within-condition neural dynamics.\n\n"
            "> **The Tensor Flattening Math:** When loaded, data is a 3D tensor "
            "`[Total Trials (T), Channels (C), Time (N)]`, where T = N_trials × N_subjects. "
            "`DataContainer.stack()` flattens this to `[(T × N) Observations, C Features]`. "
            "Because `coco-pipe` tracks the metadata expansion, we can cleanly `unstack()` "
            "PCA scores back into 3D trajectories later — losslessly.\n\n"
            "The data produces **three kinds of stacks**: pooled single trials, pooled "
            "subject-by-condition ERPs, and one single-trial stack per subject. This is "
            "the exact Step 3 layout used in the notebook."
        )
    )
    report.add_section(sec3)

    # ── Step 4 — Fit PCA ─────────────────────────────────────────────────────
    sec4 = Section("Step 4 — Fit PCA Models", icon="4")
    sec4.add_element(
        MarkdownElement(
            "Step 4 fits the same three model sets as the notebook:\n\n"
            "#### 4a. Two Shared PCAs\n\n"
            "The **single-trial shared PCA** drives separation and kinematic metrics. The "
            "**subject-level ERP shared PCA** drives group trajectory figures and embedding "
            "quality. Both bases are common across subjects.\n\n"
            "#### 4b. Per-Subject PCAs\n\n"
            "A separate single-trial PCA is also fit and sign-aligned for each subject, "
            "matching the notebook and preserving the reducer artifacts for companion work. "
            "The introductory notebook's Step 9 metrics themselves remain in the shared "
            "single-trial basis.\n\n"
            '> **The Group-Level Alignment Caveat:** Subject 1\'s "PC1" is not necessarily '
            'the same spatial direction as Subject 2\'s "PC1" — components can swap order '
            "or rotate based on individual head geometries. `flip_pc_scores_for_consistency` "
            "fixes *sign* flips, but cannot fix component swaps. This is why we maintain "
            "the two shared representations and the per-subject reducer artifacts."
        )
    )
    report.add_section(sec4)

    # ── Step 5 — Select components ──────────────────────────────────────────
    sec5 = Section("Step 5 — Select Components", icon="5")
    sec5.add_element(
        MarkdownElement(
            "How many Principal Components do we actually need? For trajectory "
            "visualization, **2–3 PCs typically suffice** for EEG/MEG. We evaluate "
            "this using two complementary methods:\n\n"
            "1. **Scree Plot** — visual plot of the Explained Variance Ratio (EVR) per "
            'component. Look for the "elbow" where additional PCs yield diminishing '
            "returns.\n"
            "2. **Participation Ratio (PR)** — a robust mathematical metric giving a "
            "single number representing the *effective dimensionality* of the dataset. "
            "Unlike an arbitrary variance cutoff, PR mathematically quantifies how many "
            "dimensions the neural data is *actually* using.\n\n"
            f"| Representation | Variance @ 3 PCs | Variance @ 5 PCs | PR |\n"
            f"|---|---:|---:|---:|\n"
            f"| Single trials | {float(np.sum(evr_trials[:3])):.2%} | "
            f"{float(np.sum(evr_trials[:5])):.2%} | {float(pr_trials):.2f} |\n"
            f"| Subject ERPs | {float(np.sum(evr_erp[:3])):.2%} | "
            f"{float(np.sum(evr_erp[:5])):.2%} | {float(pr_erp):.2f} |\n\n"
            "We retain 3 PCs for visualization and all fitted components for metrics, "
            "exactly as in the notebook."
        )
    )
    sec5.add_element(PlotlyElement(fig_scree_trials, height="420px"))
    sec5.add_element(PlotlyElement(fig_scree_erp, height="420px"))
    report.add_section(sec5)

    # ── Step 6 — Project, unstack, baseline ─────────────────────────────────
    sec6 = Section("Step 6 — Project, Unstack & Baseline", icon="6")
    sec6.add_element(
        MarkdownElement(
            "> **Why do we need post-processing in PC-space?** PCA is a purely mathematical "
            "rotation. It does not guarantee that trajectories start at the origin (0, 0, 0), "
            "nor does it guarantee consistent eigenvector sign across subjects. We enforce "
            "these constraints manually.\n\n"
            "Both shared PCA score matrices are projected back into biological structure "
            "via `with_features` + `unstack()`: one trial trajectory container and one "
            "subject-by-condition ERP trajectory container. "
            "Two post-processing corrections follow:\n\n"
            "**1. PC-Space Baselining** (`apply_pca_score_baseline`)\n\n"
            "Subtracts each trial's mean over the true pre-stimulus window, anchoring the "
            "start of each trajectory at **(0, 0, 0)** exactly at stimulus onset. "
            "*(Note: Step 2 centered raw channels to correct sensor drift; this secondary "
            "pass anchors the components themselves after the PCA rotation.)*\n\n"
            "**2. Sign-Alignment — Per-Subject Only** (`flip_pc_scores_for_consistency`)\n\n"
            'PCA eigenvectors are arbitrarily signed (a PC pointing "up" is '
            'mathematically identical to one pointing "down"). We flip the signs of the '
            "**per-subject** scores for consistent reducer artifacts. The shared PCA scores "
            "used by the notebook's metrics are not sign-flipped "
            "because a single, globally shared basis has no cross-subject sign inconsistencies."
        )
    )
    report.add_section(sec6)

    # ── Step 7 — Group-mean trajectories ────────────────────────────────────
    sec7 = Section("Step 7 — Group-Mean Trajectories", icon="7")
    sec7.add_element(
        MarkdownElement(
            "This is the headline state-space visualization of the pipeline. We compute "
            "two quantities for each condition:\n\n"
            "- **Centroid Trajectory:** Mean of the subject-level ERP trajectories for a "
            "condition in the shared ERP PCA basis.\n"
            "- **Uncertainty Envelope:** Standard Error across subject-level ERPs.\n\n"
            "> **How to interpret these interactive plots:**\n"
            "> - **2D (PC1 vs PC2):** The translucent shading represents the SEM envelope — "
            "how stable the neural representation is at each moment.\n"
            "> - **3D (PC1 vs PC2 vs PC3):** Click and drag to rotate. Examine from "
            "different angles to spot when and where condition representations diverge.\n"
            "> - **The Baseline Anchor:** All trajectories originate tightly near (0, 0, 0) "
            "before branching out in response to the task — a direct consequence of the "
            "pre-stimulus PC-space baseline from Step 6."
        )
    )
    sec7.add_element(PlotlyElement(fig_traj_2d, height="520px"))
    sec7.add_element(PlotlyElement(fig_traj_3d, height="620px"))
    report.add_section(sec7)

    # ── Step 8 — Compare conditions ─────────────────────────────────────────
    sec8 = Section("Step 8 — Compare Conditions", icon="8")
    sec8.add_element(
        MarkdownElement(
            "To quantify how distinct different cognitive tasks are, we calculate the "
            "geometric **trajectory separation** over time under two distance definitions:\n\n"
            "- **Euclidean (Centroid):** Raw geometric distance between condition means "
            "in the PCA space.\n"
            "- **Mahalanobis:** Distance scaled by the trial-to-trial covariance, "
            "penalizing axes with high within-condition noise.\n\n"
            "> **How to interpret separation metrics:**\n"
            "> - **Timecourse:** Look for the precise moment the curve lifts from zero — "
            "this is the latency at which the brain successfully discriminates between tasks.\n"
            "> - **AUC (Area Under Curve):** Captures *sustained* discriminability over "
            "the entire task window.\n"
            "> - **Peak Separation:** Captures the maximum instantaneous neural divergence.\n\n"
            "We extract Peak and AUC per subject and run **paired t-tests with FDR "
            "correction** (`paired_condition_stats`) to rigorously test whether some task "
            "pairs are statistically more separable than others. Single-trial PCA supplies "
            "centroid and Mahalanobis separation; ERP PCA supplies the additional centroid "
            "timecourse, matching notebook panels 8a–8d."
        )
    )
    sec8.add_element(PlotlyElement(fig_sep_c, height="420px"))
    sec8.add_element(PlotlyElement(fig_sep_m, height="420px"))
    sec8.add_element(PlotlyElement(fig_sep_erp, height="420px"))
    sec8.add_element(PlotlyElement(peak_figures["centroid"], height="380px"))
    sec8.add_element(PlotlyElement(peak_figures["mahalanobis"], height="380px"))
    sec8.add_element(PlotlyElement(fig_auc_heat, height="420px"))
    sec8.add_element(
        InteractiveTableElement(paired_stats, title="Paired t-tests on AUC (centroid, FDR)")
    )
    report.add_section(sec8)

    # ── Step 9 — Trajectory metrics ─────────────────────────────────────────
    sec9 = Section("Step 9 — Trajectory Metrics & Dynamical Systems", icon="9")
    sec9.add_element(
        MarkdownElement(
            "While 3D visualizations are excellent for intuition, statistical rigor requires "
            "quantifying the geometric properties of these trajectories. We map the geometry "
            "of the neural state space to concrete cognitive interpretations using **Narrative "
            "Metrics**. Every quantity below is computed from the baselined shared "
            "single-trial PCA trajectories, as in the notebook.\n\n"
            "> **The Cognitive Interpretation Cheat Sheet:**\n"
            "> - **Speed:** Moment-to-moment rate of change. Peak speed reflects rapid "
            "transitions between population states; often correlates with reaction time "
            "and decision commitment.\n"
            "> - **Distance from Origin:** Magnitude of the current state relative to the "
            "PC-space baseline.\n"
            "> - **Path Length:** Total distance traveled in state space — a proxy for "
            "the amount of representational change a condition demands (cognitive effort).\n"
            "> - **Tortuosity:** Ratio of total path length to straight-line displacement. "
            'High tortuosity indicates a "wandering" neural representation, e.g., due '
            "to task ambiguity.\n"
            "> - **Dispersion:** Across-trial variability. Low dispersion = highly "
            "consistent neural processing.\n\n"
            "*All per-trial and per-condition metrics exposed by `TrajectoryResult` are "
            "exported to the appendix table below.*"
        )
    )
    sec9.add_element(PlotlyElement(fig_speed, height="400px"))
    sec9.add_element(PlotlyElement(fig_spread, height="400px"))
    sec9.add_element(PlotlyElement(fig_distance, height="400px"))
    sec9.add_element(PlotlyElement(fig_path, height="380px"))
    sec9.add_element(PlotlyElement(fig_tort, height="450px"))
    appendix = AccordionElement("Appendix: full 13-metric grid (per-subject scalars)", open=False)
    appendix.add_element(
        InteractiveTableElement(
            scalar_metrics_all,
            title="All scalar metrics (level / metric / subject / condition / value)",
            selector_columns=["level", "metric", "condition_name"],
        )
    )
    sec9.add_element(appendix)
    report.add_section(sec9)

    # ── Step 10 — Validate ──────────────────────────────────────────────────
    sec10 = Section("Step 10 — Validate", icon="10")
    sec10.add_element(
        MarkdownElement(
            "Before drawing cognitive conclusions from lower-dimensional neural trajectories, "
            "we must rigorously validate that the dimensionality reduction did not destroy the "
            "true geometry of the neural state space, and that condition separations are not "
            "merely statistical noise.\n\n"
            "> **Validation Metrics:**\n"
            "> - **Trustworthiness:** Measures if PCA artificially crushed distant "
            "high-dimensional states close together in 3D (false neighbors). "
            "Scores > 0.85 are considered excellent.\n"
            "> - **Continuity:** Measures if PCA artificially tore close high-dimensional "
            "neighbors far apart in 3D.\n"
            "> - **Shepard Diagram:** Scatter of original 64D distances vs embedded 3D "
            "distances. A tight diagonal indicates perfect distance preservation.\n"
            "> - **Permutation Null (Separation AUC):** Condition labels are shuffled "
            "thousands of times to build a null distribution of trajectory separation. "
            "If the observed AUC exceeds the 95th percentile of the null, the neural "
            "separation is statistically significant (p < 0.05).\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Trustworthiness | **{quality.get('trustworthiness', float('nan')):.3f}** |\n"
            f"| Continuity | **{quality.get('continuity', float('nan')):.3f}** |\n"
            f"| Observed AUC (Execution vs Imagination) | **{float(obs_auc):.3f}** |\n"
            f"| Empirical p-value (n_perm={n_perm}) | **{p_value:.4f}** |\n\n"
            f"The permutation null tests the notebook's Execution vs Imagination contrast "
            f"(groups 3+4 vs 5+6) — condition labels are shuffled, separation AUC "
            f"recomputed, and the observed statistic is compared to the resulting null "
            f"distribution."
        )
    )
    sec10.add_element(PlotlyElement(fig_shepard, height="420px"))
    sec10.add_element(PlotlyElement(fig_null, height="380px"))
    report.add_section(sec10)

    # ── Main composite ──────────────────────────────────────────────────────
    sec_fig = Section("Main Composite — 5-Panel Figure", icon="F")
    sec_fig.add_element(
        MarkdownElement(
            "The 5-panel manuscript composite uses the same representation split as the "
            "notebook: ERP scree and group trajectories, plus shared single-trial PCA "
            "scatter, speed, and condition separation."
        )
    )
    sec_fig.add_element(PlotlyElement(fig_main, height="820px"))
    report.add_section(sec_fig)

    report.save(str(out / "report.html"))

    # ------------------------------------------------------------------ Save artifacts
    save_artifacts(
        artifacts_dir,
        shared_reducer=shared_reducer_trials,
        trajectory_container=container_traj_trials,
        per_subject_reducers=per_subject_reducers,
        scalar_metrics=scalar_metrics_all,
        continuous_metrics=continuous_metrics,
        separation_pair_scalars=pair_scalars,
        extra_csvs={"paired_stats": paired_stats},
    )
    shared_reducer_erp.save(artifacts_dir / "shared_reducer_erp.pkl")

    return {
        "p_value": p_value,
        "observed_auc": float(obs_auc),
        "quality": quality,
        "n_subjects": n_analyzed,
        "subjects": analyzed_subjects.tolist(),
        "output": out,
    }


def _completed(manifest_path: Path) -> bool:
    """True when a previous run already finished at this output root."""
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        default=None,
        help="Subject IDs (default: 1..109).",
    )
    parser.add_argument(
        "--output",
        default="outputs/eegbci_main",
        help="Output root (default: outputs/eegbci_main).",
    )
    parser.add_argument(
        "--bids-root",
        default="PhysioNet_EEGBCI/BIDS",
        help="BIDS root for the EEGBCI conversion.",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--force-reconvert",
        action="store_true",
        help="Re-clean and re-convert runs whose BIDS output already exists. "
        "Required after any change to the cleaning pipeline, which is otherwise "
        "skipped for already-converted runs.",
    )
    parser.add_argument(
        "--no-clean-artifacts",
        action="store_false",
        dest="clean_artifacts",
        help="Skip ICA artifact removal, for a raw-baseline comparison.",
    )
    parser.add_argument("--n-ica-components", type=int, default=20)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast check: one subject and at most 10 permutations.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Re-run even if this output root already holds a completed run.",
    )
    args = parser.parse_args(argv)

    subjects = args.subjects
    n_perm = args.n_perm
    if args.smoke:
        subjects = subjects or [1]
        n_perm = min(n_perm, 10)

    manifest_path = Path(args.output) / "run_manifest.json"
    if args.resume and _completed(manifest_path):
        print(f"Completed run found at {manifest_path}; nothing to do.")
        return

    settings = {**vars(args), "subjects": subjects, "n_perm": n_perm}
    write_manifest(manifest_path, settings, status="running")

    try:
        result = _run(args, subjects, n_perm)
    except Exception as error:
        write_manifest(
            manifest_path,
            settings,
            status="failed",
            extra={"error": f"{type(error).__name__}: {error}"},
        )
        raise
    write_manifest(manifest_path, settings, status="complete", extra=result)
    print("=== done ===")
    print(result)


def _run(args: argparse.Namespace, subjects: list[int] | None, n_perm: int) -> dict:
    return run_main_analysis(
        subjects=subjects,
        output=args.output,
        bids_root=args.bids_root,
        n_perm=n_perm,
        n_components=args.n_components,
        random_state=args.random_state,
        force_reconvert=args.force_reconvert,
        clean_artifacts=args.clean_artifacts,
        n_ica_components=args.n_ica_components,
    )


if __name__ == "__main__":
    main()
