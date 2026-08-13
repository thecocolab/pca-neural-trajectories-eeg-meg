"""Main PCA neural-trajectory analysis on PhysioNet EEGBCI (109 subjects).

Headless full-cohort counterpart to the main tutorial notebook
(`tutorials/tutorial_eegbci_main.ipynb`). Runs the full 10-step
PCA-trajectory workflow and produces a standalone HTML report
that mirrors the same steps as the tutorial notebook.

Outputs (under ``--output``)::

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
    CalloutElement,
    ContainerElement,
    Element,
    InteractiveTableElement,
    MarkdownElement,
    StatCardElement,
    TabsElement,
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


def _stack(*elements: Element) -> ContainerElement:
    """Bundle several elements into one container (used as a tab panel)."""
    box = ContainerElement()
    for element in elements:
        box.add_element(element)
    return box


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

    # ------------------------------------------------------------------ Report (10-step)
    report = Report(
        title=f"EEGBCI — PCA Neural Trajectories ({n_analyzed} subjects)",
        asset_urls="inline",
    )
    report.add_summary_card(
        {
            "Subjects": n_analyzed,
            "Variance @ 3 PCs": f"{float(np.sum(evr_trials[:3])):.1%}",
            "Participation ratio": f"{float(pr_trials):.2f}",
            "Trustworthiness": f"{quality.get('trustworthiness', float('nan')):.3f}",
            "Permutation p": f"{p_value:.4f}",
        }
    )

    # ── Overview ────────────────────────────────────────────────────────────
    sec_overview = Section(
        "Overview",
        icon="O",
        description="Full 10-step PCA neural-trajectory workflow across the cohort",
        metadata={
            "Subjects": f"{n_analyzed} analyzable / {len(subject_list)} requested",
            "Conditions": ", ".join(LABEL_NAMES[c] for c in unique_conds),
            "Components": f"{n_components} fit, 3 visualized",
            "Window": f"{analysis_window[0]:g}–{analysis_window[1]:g} s",
            "Permutations": str(n_perm),
        },
    )
    sec_overview.add_element(
        MarkdownElement(
            "Companion report to *'A Primer on Low-Dimensional Neural Dynamics'*. "
            "Each section below is one numbered step of "
            "`tutorials/tutorial_eegbci_main.ipynb`, run headless on the full cohort."
        )
    )
    sec_overview.add_columns(
        [
            StatCardElement(
                "Variance @ 3 PCs (trials)",
                f"{float(np.sum(evr_trials[:3])):.1%}",
                color="blue",
            ),
            StatCardElement(
                "Variance @ 3 PCs (ERP)",
                f"{float(np.sum(evr_erp[:3])):.1%}",
                color="green",
            ),
            StatCardElement("PR trials / ERP", f"{pr_trials:.2f} / {pr_erp:.2f}", color="purple"),
            StatCardElement(
                "Trust. / Cont.",
                f"{quality.get('trustworthiness', float('nan')):.3f} / "
                f"{quality.get('continuity', float('nan')):.3f}",
                color="yellow",
            ),
        ]
    )
    report.add_section(sec_overview)

    # ── Step 1 — Choose Your Representation ─────────────────────────────────
    sec1 = Section(
        "Step 1 — Choose Your Representation",
        icon="1",
        description="Sensor-space voltage patterns as points in a 64-D state space",
    )
    sec1.add_element(
        MarkdownElement(
            'The first question in any trajectory analysis is: *what is the brain\'s "state" '
            "at time t?* We take it to be the instantaneous sensor-space voltage pattern.\n\n"
            "**x**(t) = [v₁(t), v₂(t), …, v₆₄(t)]ᵀ ∈ ℝ⁶⁴"
        )
    )
    sec1.add_element(
        CalloutElement(
            "Every time point is one point in a 64-dimensional space, and every trial traces "
            "a **trajectory** through it. PCA then reveals the low-dimensional manifold — "
            "typically 2–4 dimensions — carrying the dominant variance.",
            kind="info",
            title="Why this representation",
        )
    )
    sec1.add_element(
        MarkdownElement(
            f"**Data** — PhysioNet EEGBCI, {n_analyzed} analyzable subjects, four conditions: "
            "left/right hand execution and left/right hand imagination."
        )
    )
    report.add_section(sec1)

    # ── Step 2 — Preprocess ─────────────────────────────────────────────────
    sec2 = Section(
        "Step 2 — Preprocess & Normalize",
        icon="2",
        description="MNE cleaning → BIDS → crop, baseline, z-score",
    )
    sec2.add_element(
        CalloutElement(
            "PCA does not know signal from noise — it only finds directions of maximum "
            "variance. Leave blinks and heartbeats in, and they *will* own PC1.",
            kind="warning",
            title="Why preprocessing is critical for PCA",
        )
    )
    sec2.add_element(
        TabsElement(
            {
                "1 · Clean (MNE)": MarkdownElement(
                    "`setup_data_bids` runs the MNE preprocessing pipeline and writes epochs "
                    "to a standardized BIDS directory:\n\n"
                    "1. **Bad-channel interpolation** — outlier channels (LOF) are "
                    "interpolated before referencing, so a bad sensor cannot contaminate "
                    "the common-average reference or the ICA.\n"
                    "2. **Common-average reference.**\n"
                    "3. **ICA artifact removal** — a 20-component extended-infomax ICA fit on "
                    "a 1 Hz high-passed copy (the algorithm and band `mne-icalabel` expects) "
                    "identifies and subtracts **blinks** and **heartbeats**.\n"
                    "4. **Bandpass 0.5–40 Hz (FIR)** — attenuates slow drifts; the 40 Hz "
                    "low-pass also suppresses broadband muscle and residual line noise, so "
                    "no separate notch is needed."
                ),
                "2 · Load": MarkdownElement(
                    "`BIDSDataset` parses the preprocessed directory into a `DataContainer` "
                    "of shape `[Trials × Channels × Time]`."
                ),
                "3 · Normalize": MarkdownElement(
                    "- **Baseline** sensors in the loader using the pre-stimulus interval.\n"
                    f"- **Crop** to the common `{analysis_window[0]:g}s–{analysis_window[1]:g}s` "
                    "analysis window.\n"
                    "- **Z-score** each channel across observations, so high-amplitude sensors "
                    "cannot dominate PCA variance.\n"
                    "- **Baseline again in PC space** over `-0.2s–0.0s` after projection."
                ),
            }
        )
    )
    report.add_section(sec2)

    # ── Step 3 — Reshape ────────────────────────────────────────────────────
    sec3 = Section(
        "Step 3 — Reshape into (Samples × Features)",
        icon="3",
        description="Flatten the 3-D tensor without averaging trials away",
    )
    sec3.add_element(
        MarkdownElement(
            "PCA needs a flat 2-D matrix. Two methodological choices govern the reshape:\n\n"
            "- **Single-trial preservation** — trials are *not* averaged before PCA. Every "
            "timepoint of every trial is an independent observation, keeping the true "
            "biological variance across repetitions.\n"
            "- **Subject normalization** — we fit both a *shared PCA* (all subjects pooled) "
            "and *per-subject PCAs*, disentangling between-subject differences from "
            "within-condition dynamics.\n\n"
            "This yields **three stacks**: pooled single trials, pooled subject-by-condition "
            "ERPs, and one single-trial stack per subject."
        )
    )
    sec3.add_element(
        AccordionElement("The tensor-flattening math", open=False).add_markdown(
            "Loaded data is a 3-D tensor `[Total Trials (T), Channels (C), Time (N)]`, where "
            "T = N_trials × N_subjects. `DataContainer.stack()` flattens it to "
            "`[(T × N) Observations, C Features]`. Because `coco-pipe` tracks the metadata "
            "expansion, PCA scores can be `unstack()`-ed back into 3-D trajectories "
            "losslessly."
        )
    )
    report.add_section(sec3)

    # ── Step 4 — Fit PCA ─────────────────────────────────────────────────────
    sec4 = Section(
        "Step 4 — Fit PCA Models",
        icon="4",
        description="Two shared bases plus one sign-aligned PCA per subject",
    )
    sec4.add_element(
        TabsElement(
            {
                "4a · Shared PCAs": MarkdownElement(
                    "The **single-trial shared PCA** drives separation and kinematic metrics; "
                    "the **subject-level ERP shared PCA** drives the group trajectory figures "
                    "and embedding quality. Both bases are common across subjects."
                ),
                "4b · Per-subject PCAs": MarkdownElement(
                    "A separate single-trial PCA is fit and sign-aligned per subject, matching "
                    "the notebook and preserving reducer artifacts for companion work. Step 9 "
                    "metrics stay in the shared single-trial basis."
                ),
            }
        )
    )
    sec4.add_element(
        CalloutElement(
            "Subject 1's \"PC1\" need not be the same spatial direction as subject 2's — "
            "components can swap order or rotate with head geometry. "
            "`flip_pc_scores_for_consistency` fixes *sign* flips but cannot fix component "
            "swaps, which is why we keep the two shared representations alongside the "
            "per-subject reducers.",
            kind="warning",
            title="The group-level alignment caveat",
        )
    )
    report.add_section(sec4)

    # ── Step 5 — Select components ──────────────────────────────────────────
    sec5 = Section(
        "Step 5 — Select Components",
        icon="5",
        description="Scree elbow and participation ratio agree on ~3 PCs",
    )
    sec5.add_element(
        MarkdownElement(
            "How many PCs do we actually need? For EEG/MEG trajectory visualization, "
            "**2–3 typically suffice**. We check with two complementary methods: the "
            "**scree plot** (explained variance per component — look for the elbow) and the "
            "**participation ratio** (a single number for the *effective* dimensionality, "
            "rather than an arbitrary variance cutoff)."
        )
    )
    sec5.add_columns(
        [
            StatCardElement(
                "Trials · var @ 3 / 5 PCs",
                f"{float(np.sum(evr_trials[:3])):.1%} / {float(np.sum(evr_trials[:5])):.1%}",
                color="blue",
            ),
            StatCardElement("Trials · PR", f"{float(pr_trials):.2f}", color="blue"),
            StatCardElement(
                "ERP · var @ 3 / 5 PCs",
                f"{float(np.sum(evr_erp[:3])):.1%} / {float(np.sum(evr_erp[:5])):.1%}",
                color="green",
            ),
            StatCardElement("ERP · PR", f"{float(pr_erp):.2f}", color="green"),
        ]
    )
    sec5.add_element(
        TabsElement(
            {
                "Single trials": PlotlyElement(fig_scree_trials, height="420px"),
                "Subject ERPs": PlotlyElement(fig_scree_erp, height="420px"),
            }
        )
    )
    sec5.add_element(
        MarkdownElement(
            "We retain 3 PCs for visualization and all fitted components for metrics, "
            "exactly as in the notebook."
        )
    )
    report.add_section(sec5)

    # ── Step 6 — Project, unstack, baseline ─────────────────────────────────
    sec6 = Section(
        "Step 6 — Project, Unstack & Baseline",
        icon="6",
        description="Anchor trajectories at the origin and align signs",
    )
    sec6.add_element(
        MarkdownElement(
            "Both shared score matrices go back into biological structure via `with_features` "
            "+ `unstack()` — one trial trajectory container, one subject-by-condition ERP "
            "container — followed by two corrections in PC space."
        )
    )
    sec6.add_element(
        CalloutElement(
            "PCA is a purely mathematical rotation. It guarantees neither that trajectories "
            "start at the origin nor that eigenvector signs are consistent across subjects. "
            "Both constraints are enforced by hand.",
            kind="info",
            title="Why post-process in PC space",
        )
    )
    sec6.add_element(
        TabsElement(
            {
                "1 · PC-space baselining": MarkdownElement(
                    "`apply_pca_score_baseline` subtracts each trial's mean over the true "
                    "pre-stimulus window, anchoring every trajectory at **(0, 0, 0)** at "
                    "stimulus onset. Step 2 centered *raw channels* to correct sensor drift; "
                    "this second pass anchors the *components* after the PCA rotation."
                ),
                "2 · Sign alignment (per-subject)": MarkdownElement(
                    'PCA eigenvectors are arbitrarily signed — a PC pointing "up" is '
                    'mathematically identical to one pointing "down". '
                    "`flip_pc_scores_for_consistency` flips the **per-subject** scores for "
                    "consistent reducer artifacts. Shared PCA scores are left untouched: a "
                    "single global basis has no cross-subject sign inconsistency."
                ),
            }
        )
    )
    report.add_section(sec6)

    # ── Step 7 — Group-mean trajectories ────────────────────────────────────
    sec7 = Section(
        "Step 7 — Group-Mean Trajectories",
        icon="7",
        description="Condition centroids with SEM envelopes in the shared ERP basis",
    )
    sec7.add_element(
        MarkdownElement(
            "The headline state-space visualization. Per condition we compute the **centroid "
            "trajectory** (mean of subject-level ERP trajectories in the shared ERP basis) and "
            "an **uncertainty envelope** (SEM across subjects)."
        )
    )
    sec7.add_element(
        TabsElement(
            {
                "2D — PC1 vs PC2": PlotlyElement(fig_traj_2d, height="520px"),
                "3D — PC1/PC2/PC3": PlotlyElement(fig_traj_3d, height="620px"),
            }
        )
    )
    sec7.add_element(
        CalloutElement(
            "**2D** — translucent shading is the SEM envelope: how stable the representation "
            "is at each moment. **3D** — drag to rotate; different angles reveal when and "
            "where conditions diverge. **Baseline anchor** — all trajectories start tightly "
            "near (0, 0, 0) before branching, a direct consequence of the Step 6 baseline.",
            kind="tip",
            title="How to read these plots",
        )
    )
    report.add_section(sec7)

    # ── Step 8 — Compare conditions ─────────────────────────────────────────
    sec8 = Section(
        "Step 8 — Compare Conditions",
        icon="8",
        description="Separation timecourses, peak/AUC per subject, paired tests",
    )
    sec8.add_element(
        MarkdownElement(
            "How distinct are the tasks? We measure geometric **trajectory separation** over "
            "time under two distance definitions: **Euclidean (centroid)**, the raw distance "
            "between condition means, and **Mahalanobis**, scaled by trial-to-trial covariance "
            "so noisy axes are penalized."
        )
    )
    sec8.add_element(
        TabsElement(
            {
                "Timecourses": _stack(
                    PlotlyElement(fig_sep_c, height="420px"),
                    PlotlyElement(fig_sep_m, height="420px"),
                    PlotlyElement(fig_sep_erp, height="420px"),
                ),
                "Per-subject peaks": _stack(
                    PlotlyElement(peak_figures["centroid"], height="380px"),
                    PlotlyElement(peak_figures["mahalanobis"], height="380px"),
                ),
                "AUC matrix": PlotlyElement(fig_auc_heat, height="420px"),
                "Paired stats": InteractiveTableElement(
                    paired_stats, title="Paired t-tests on AUC (centroid, FDR)"
                ),
            }
        )
    )
    sec8.add_element(
        CalloutElement(
            "**Timecourse** — the moment the curve lifts off zero is the latency at which the "
            "brain discriminates the tasks. **AUC** — sustained discriminability over the task "
            "window. **Peak** — maximum instantaneous divergence.",
            kind="tip",
            title="How to read separation metrics",
        )
    )
    sec8.add_element(
        MarkdownElement(
            "Peak and AUC are extracted per subject and tested with **paired t-tests + FDR "
            "correction** (`paired_condition_stats`). Single-trial PCA supplies centroid and "
            "Mahalanobis separation; ERP PCA supplies the additional centroid timecourse, "
            "matching notebook panels 8a–8d."
        )
    )
    report.add_section(sec8)

    # ── Step 9 — Trajectory metrics ─────────────────────────────────────────
    sec9 = Section(
        "Step 9 — Trajectory Metrics & Dynamical Systems",
        icon="9",
        description="Geometry of the state space, translated into cognitive terms",
    )
    sec9.add_element(
        MarkdownElement(
            "3-D views are good for intuition; rigor needs numbers. Every quantity below is "
            "computed from the baselined shared single-trial PCA trajectories, as in the "
            "notebook."
        )
    )
    sec9.add_element(
        CalloutElement(
            "**Speed** — rate of state change; peaks reflect rapid population transitions and "
            "often track reaction time. **Distance from origin** — state magnitude relative to "
            "the PC-space baseline. **Path length** — total distance travelled, a proxy for "
            "representational change (cognitive effort). **Tortuosity** — path length over "
            'straight-line displacement; high values mean a "wandering" representation. '
            "**Dispersion** — across-trial variability; low means consistent processing.",
            kind="tip",
            title="Cognitive interpretation cheat sheet",
        )
    )
    sec9.add_element(
        TabsElement(
            {
                "Speed": PlotlyElement(fig_speed, height="400px"),
                "Dispersion": PlotlyElement(fig_spread, height="400px"),
                "Distance": PlotlyElement(fig_distance, height="400px"),
                "Path length": PlotlyElement(fig_path, height="380px"),
                "Tortuosity": PlotlyElement(fig_tort, height="450px"),
            }
        )
    )
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
    sec10 = Section(
        "Step 10 — Validate",
        icon="10",
        description="Embedding faithfulness and a permutation null on separation",
    )
    sec10.add_element(
        MarkdownElement(
            "Before drawing cognitive conclusions we check that the reduction preserved the "
            "geometry of the state space, and that condition separation is not noise."
        )
    )
    sec10.add_columns(
        [
            StatCardElement(
                "Trustworthiness",
                f"{quality.get('trustworthiness', float('nan')):.3f}",
                color="green" if quality.get("trustworthiness", 0.0) > 0.85 else "yellow",
                delta="> 0.85 is excellent",
            ),
            StatCardElement(
                "Continuity",
                f"{quality.get('continuity', float('nan')):.3f}",
                color="green" if quality.get("continuity", 0.0) > 0.85 else "yellow",
            ),
            StatCardElement(
                "Observed AUC",
                f"{float(obs_auc):.3f}",
                color="purple",
                delta="Execution vs Imagination",
            ),
            StatCardElement(
                "Empirical p",
                f"{p_value:.4f}",
                color="green" if p_value < 0.05 else "red",
                delta=f"n_perm = {n_perm}",
            ),
        ]
    )
    sec10.add_element(
        CalloutElement(
            "**Trustworthiness** — did PCA crush distant 64-D states together in 3-D (false "
            "neighbors)? **Continuity** — did it tear close neighbors apart? **Shepard "
            "diagram** — 64-D vs 3-D distances; a tight diagonal means faithful preservation. "
            "**Permutation null** — condition labels are shuffled to build a null separation "
            "AUC; an observed value beyond the 95th percentile is significant.",
            kind="info",
            title="What these validations test",
        )
    )
    sec10.add_element(
        TabsElement(
            {
                "Shepard diagram": PlotlyElement(fig_shepard, height="420px"),
                "Permutation null": PlotlyElement(fig_null, height="380px"),
            }
        )
    )
    sec10.add_element(
        MarkdownElement(
            "The null tests the notebook's Execution vs Imagination contrast (groups 3+4 vs "
            "5+6): labels are shuffled, separation AUC recomputed, and the observed statistic "
            "compared to the resulting distribution."
        )
    )
    report.add_section(sec10)

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
