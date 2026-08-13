"""Run and save the complete MEG Faces main trajectory analysis.

This is the headless counterpart to ``tutorials/tutorial_megfaces_main.ipynb``.
It follows the same numbered workflow, analysis choices, figures, participant-
level summaries, within-participant permutation null, component sensitivity,
and focused two-condition PCA spaces. It also saves every table, fitted reducer,
analysis array, provenance manifest, and a fully self-contained HTML report.

The script loads already-prepared, subject-wise noise-whitened Wakeman--Henson
MEG derivatives. It never downloads or preprocesses data implicitly.

Examples
--------
python scripts/analysis_megfaces_main.py
python scripts/analysis_megfaces_main.py --subjects 01 02 03 --n-perm 1000
python scripts/analysis_megfaces_main.py --smoke
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from coco_pipe.dim_reduction import DimReduction
from coco_pipe.report import PlotlyElement, Report, Section
from coco_pipe.report.elements import (
    AccordionElement,
    CalloutElement,
    InteractiveTableElement,
    MarkdownElement,
    StatCardElement,
    TabsElement,
)
from coco_pipe.viz.interactive import (
    plot_scree,
    plot_timecourses,
    plot_trajectory,
    plot_trajectory_metric_series,
)
from plotly.subplots import make_subplots

from pca_neural_trajectories import facet_figures, overlay_figures, write_manifest
from pca_neural_trajectories.wakeman_henson import (
    LABEL_NAMES,
    MEG_SENSOR_SETS,
    _load_wakeman_henson_container,
)

CONDITIONS = (1, 2, 3)
N_DISPLAY_COMPONENTS = 3
ACTIVE_WINDOW = (0.0, 0.6)
FOCUSED_PAIRS = {
    "Famous vs Unfamiliar": (1, 2),
    "Famous vs Scrambled": (1, 3),
}

CONDITION_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
CONDITION_FILLS = {
    1: "rgba(0, 114, 178, 0.15)",
    2: "rgba(213, 94, 0, 0.15)",
    3: "rgba(0, 158, 115, 0.15)",
}
CONTRAST_COLORS = {
    "Faces vs Scrambled": "#7B2CBF",
    "Famous vs Unfamiliar": "#D55E00",
    "Famous vs Scrambled": "#7B2CBF",
}
CONTRAST_FILLS = {
    "Faces vs Scrambled": "rgba(123, 44, 191, 0.15)",
    "Famous vs Unfamiliar": "rgba(213, 94, 0, 0.15)",
    "Famous vs Scrambled": "rgba(123, 44, 191, 0.15)",
}


def _contrast_curves(
    trajectories: np.ndarray,
    trial_labels: np.ndarray,
    trial_subjects: np.ndarray,
    unique_subjects: np.ndarray,
    baseline_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the notebook's two baseline-relative participant contrasts."""
    curves: dict[str, list[np.ndarray]] = {
        "Faces vs Scrambled": [],
        "Famous vs Unfamiliar": [],
    }
    for subject in unique_subjects:
        means = {
            condition: trajectories[
                (trial_subjects == subject) & (trial_labels == condition)
            ].mean(axis=0)
            for condition in CONDITIONS
        }
        faces = 0.5 * (means[1] + means[2])
        distances = {
            "Faces vs Scrambled": np.linalg.norm(faces - means[3], axis=1),
            "Famous vs Unfamiliar": np.linalg.norm(means[1] - means[2], axis=1),
        }
        for name, distance in distances.items():
            curves[name].append(distance - distance[baseline_mask].mean())
    return {name: np.asarray(values) for name, values in curves.items()}


def _completed(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete"


def run_megfaces_main_analysis(
    *,
    subjects: list[str],
    derivatives_root: Path,
    output: Path,
    metric_pca_mode: str = "both",
    n_components: int = 10,
    n_perm: int = 200,
    seed: int = 42,
    sensor_set: str = "all_sensors",
) -> dict[str, object]:
    """Run the notebook-equivalent MEG Faces analysis and save its full bundle."""
    if metric_pca_mode not in {"shared", "subject", "both"}:
        raise ValueError("metric_pca_mode must be 'shared', 'subject', or 'both'.")
    if n_components < 5:
        raise ValueError("n_components must be at least 5 for the 2/3/5-PC sensitivity.")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")

    subjects = [str(subject).removeprefix("sub-").zfill(2) for subject in subjects]
    output = Path(output) / sensor_set
    figures_dir = output / "figures"
    reducers_dir = output / "reducers"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reducers_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("=== MEG Faces main analysis ===")
    print(f"subjects:         {', '.join(subjects)}")
    print(f"derivatives root: {derivatives_root}")
    print(f"output:           {output}")

    # ------------------------------------------------------------------ Steps 1--2
    # The neural state is the retained MEG sensor pattern. The loader applies
    # each participant's empty-room whitener, so a second channel z-score would
    # erase the intended noise scaling and is deliberately not performed.
    container = _load_wakeman_henson_container(
        derivatives_root,
        subjects=subjects,
        conditions=CONDITIONS,
        sensor_set=sensor_set,
    )
    X = np.asarray(container.X, dtype=np.float32)
    times = np.asarray(container.coords["time"], dtype=float)
    channels = np.asarray(container.coords["channel"]).astype(str)
    trial_subjects = np.asarray(container.coords["subject"]).astype(str)
    labels = np.asarray(container.y, dtype=int)
    repetitions = np.asarray(container.coords["repetition"]).astype(str)
    unique_subjects = np.unique(trial_subjects)

    missing = {
        subject: [
            LABEL_NAMES[condition]
            for condition in CONDITIONS
            if not np.any(
                (trial_subjects == subject) & (labels == condition)
            )
        ]
        for subject in unique_subjects
    }
    missing = {subject: names for subject, names in missing.items() if names}
    if missing:
        raise RuntimeError(f"Participants missing required conditions: {missing}")
    if len(unique_subjects) < 2:
        raise RuntimeError(
            "At least two analyzable participants are required for group SEMs."
        )

    # Resolve the required inline assets before the expensive PCA fits. If the
    # coco-pipe asset cache has not been populated and no network is available,
    # fail here instead of completing the analysis and producing a non-portable
    # CDN-backed report.
    report = Report(
        title=(
            f"MEG Faces Main Analysis - {sensor_set} "
            f"({len(unique_subjects)} participants)"
        ),
        asset_urls="inline",
    )

    n_trials, n_sensors, n_times = X.shape
    baseline_mask = times < 0
    active_mask = (times >= ACTIVE_WINDOW[0]) & (times <= ACTIVE_WINDOW[1])
    if not baseline_mask.any() or active_mask.sum() < 2:
        raise RuntimeError(
            "The loaded epoch must include pre-stimulus samples and 0--0.6 s."
        )
    active_times = times[active_mask]
    print(f"data shape: {X.shape} (trial, sensor, time)")
    print(f"subjects analyzed: {', '.join(unique_subjects)}")
    print(f"whitening: {container.meta.get('whitening', 'not recorded')}")

    condition_names = pd.Series(
        [LABEL_NAMES[condition] for condition in labels], name="condition"
    )
    trial_counts = (
        pd.crosstab(pd.Series(trial_subjects, name="subject"), condition_names)
        .reindex(columns=[LABEL_NAMES[condition] for condition in CONDITIONS])
        .reset_index()
    )
    repetition_counts = pd.crosstab(condition_names, repetitions).reset_index()

    sensor_magnitude = {condition: [] for condition in CONDITIONS}
    for subject in unique_subjects:
        for condition in CONDITIONS:
            rows = (trial_subjects == subject) & (labels == condition)
            evoked = X[rows].mean(axis=0)
            sensor_magnitude[condition].append(
                np.sqrt(np.mean(evoked**2, axis=0))
            )

    # One curve per participant, grouped by condition: plot_timecourses draws
    # the group mean with an SEM band.
    sensor_curves = np.concatenate(
        [np.asarray(sensor_magnitude[condition]) for condition in CONDITIONS]
    )[:, np.newaxis, :]
    sensor_groups = [
        LABEL_NAMES[condition]
        for condition in CONDITIONS
        for _ in sensor_magnitude[condition]
    ]
    fig_sensor = plot_timecourses(
        sensor_curves,
        times=times,
        channel_names=["RMS"],
        group_labels=sensor_groups,
        palette={LABEL_NAMES[c]: CONDITION_COLORS[c] for c in CONDITIONS},
        error_style="band",
        xlabel="Time (s)",
        ylabel="RMS response (noise-normalised units)",
        title="Whitened sensor-space response magnitude",
    )
    fig_sensor.add_vline(x=0, line_dash="dash", line_color="black")
    fig_sensor.update_layout(
        title="Whitened sensor-space response magnitude",
        xaxis_title="Time (s)",
        yaxis_title="RMS response (noise-normalised units)",
        legend_title_text="Condition",
        template="plotly_white",
    )

    # ------------------------------------------------------------------ Steps 3--5
    pooled = X.transpose(0, 2, 1).reshape(n_trials * n_times, n_sensors)
    shared_pca = DimReduction(
        method="PCA", n_components=n_components, random_state=seed
    )
    scores_flat = shared_pca.fit_transform(pooled)
    diagnostics = shared_pca.get_diagnostics()
    explained_variance = np.asarray(diagnostics["explained_variance_ratio_"])
    shared_variance = pd.DataFrame(
        {
            "component": np.arange(1, len(explained_variance) + 1),
            "explained_variance_ratio": explained_variance,
            "cumulative_variance": np.cumsum(explained_variance),
        }
    )
    fig_scree = plot_scree(explained_variance)
    fig_scree.update_layout(
        title="Shared PCA: explained and cumulative variance",
        template="plotly_white",
    )

    scores = scores_flat.reshape(n_trials, n_times, n_components)
    scores_baselined = scores - scores[:, baseline_mask].mean(axis=1, keepdims=True)
    baseline_offset = float(
        np.abs(scores_baselined[:, baseline_mask].mean(axis=1)).mean()
    )

    subject_scores_baselined: np.ndarray | None = None
    subject_pcas: dict[str, DimReduction] = {}
    subject_variance_rows: list[dict[str, object]] = []
    if metric_pca_mode in {"subject", "both"}:
        subject_scores_baselined = np.empty_like(scores_baselined)
        for subject in unique_subjects:
            rows = trial_subjects == subject
            X_subject = X[rows]
            n_subject_trials = X_subject.shape[0]
            subject_matrix = X_subject.transpose(0, 2, 1).reshape(
                n_subject_trials * n_times, n_sensors
            )
            subject_pca = DimReduction(
                method="PCA", n_components=n_components, random_state=seed
            )
            subject_flat = subject_pca.fit_transform(subject_matrix)
            subject_traj = subject_flat.reshape(
                n_subject_trials, n_times, n_components
            )
            subject_traj -= subject_traj[:, baseline_mask].mean(
                axis=1, keepdims=True
            )
            subject_pcas[subject] = subject_pca
            subject_scores_baselined[rows] = subject_traj
            evr = np.asarray(
                subject_pca.get_diagnostics()["explained_variance_ratio_"]
            )
            subject_variance_rows.append(
                {
                    "subject": subject,
                    "PC1": evr[0],
                    "PC2": evr[1],
                    "PC3": evr[2],
                    "cumulative_3pc": evr[:3].sum(),
                }
            )
    subject_variance = pd.DataFrame(subject_variance_rows)

    subject_condition = {
        (subject, condition): scores_baselined[
            (trial_subjects == subject) & (labels == condition)
        ].mean(axis=0)
        for subject in unique_subjects
        for condition in CONDITIONS
    }

    # Channels are principal components, groups are conditions: one stacked
    # panel per PC with the across-participant mean and SEM band.
    pc_curves = np.stack(
        [
            subject_condition[subject, condition][:, :N_DISPLAY_COMPONENTS].T
            for condition in CONDITIONS
            for subject in unique_subjects
        ]
    )
    pc_groups = [
        LABEL_NAMES[condition] for condition in CONDITIONS for _ in unique_subjects
    ]
    fig_pc = plot_timecourses(
        pc_curves,
        times=times,
        channel_names=[f"PC{index + 1}" for index in range(N_DISPLAY_COMPONENTS)],
        group_labels=pc_groups,
        palette={LABEL_NAMES[c]: CONDITION_COLORS[c] for c in CONDITIONS},
        n_cols=1,
        error_style="band",
        xlabel="Time (s)",
        ylabel="Score (a.u.)",
        title="Principal-component time courses",
    )
    fig_pc.add_vline(x=0, line_dash="dash", line_color="black")
    fig_pc.update_xaxes(title_text="Time (s)", row=N_DISPLAY_COMPONENTS, col=1)
    fig_pc.update_yaxes(title_text="Score (a.u.)")
    fig_pc.update_layout(
        height=720, title="Principal-component time courses", template="plotly_white"
    )

    loadings = np.asarray(shared_pca.get_components())[:N_DISPLAY_COMPONENTS]
    loading_rows: list[dict[str, object]] = []
    for pc, weights in enumerate(loadings, start=1):
        strongest = np.argsort(np.abs(weights))[-8:][::-1]
        loading_rows.extend(
            {
                "component": f"PC{pc}",
                "sensor": channels[index],
                "loading": weights[index],
            }
            for index in strongest
        )
    loading_table = pd.DataFrame(loading_rows)

    # ------------------------------------------------------------------ Step 6
    group_trajectories = np.stack(
        [
            np.stack(
                [subject_condition[subject, condition] for subject in unique_subjects]
            ).mean(axis=0)
            for condition in CONDITIONS
        ]
    )
    group_sem = np.stack(
        [
            np.stack(
                [subject_condition[subject, condition] for subject in unique_subjects]
            ).std(axis=0, ddof=1)
            / np.sqrt(len(unique_subjects))
            for condition in CONDITIONS
        ]
    )
    plot_labels = np.array([LABEL_NAMES[condition] for condition in CONDITIONS])
    color_map = {
        LABEL_NAMES[condition]: CONDITION_COLORS[condition]
        for condition in CONDITIONS
    }
    fig_2d = plot_trajectory(
        X=group_trajectories[..., :2],
        times=times,
        labels=plot_labels,
        color_map=color_map,
        title="Equal-participant mean trajectories: PC1-PC2",
        dimensions=2,
        smooth_window=12,
        show_markers=False,
        add_start_end_markers=True,
    )
    fig_3d = plot_trajectory(
        X=group_trajectories[..., :3],
        times=times,
        labels=plot_labels,
        color_map=color_map,
        title="Equal-participant mean trajectories: PC1-PC2-PC3",
        dimensions=3,
        smooth_window=12,
        show_markers=False,
        linewidth=10,
        add_start_end_markers=True,
        height=700,
    )

    # ------------------------------------------------------------------ Steps 7--8
    metric_trajectory_spaces: dict[str, np.ndarray] = {}
    if metric_pca_mode in {"shared", "both"}:
        metric_trajectory_spaces["Shared PCA"] = scores_baselined[
            ..., :N_DISPLAY_COMPONENTS
        ]
    if metric_pca_mode in {"subject", "both"}:
        assert subject_scores_baselined is not None
        metric_trajectory_spaces["Subject PCA"] = subject_scores_baselined[
            ..., :N_DISPLAY_COMPONENTS
        ]
    contrast_curves_by_space = {
        space: _contrast_curves(
            trajectories,
            labels,
            trial_subjects,
            unique_subjects,
            baseline_mask,
        )
        for space, trajectories in metric_trajectory_spaces.items()
    }
    contrast_names = ["Faces vs Scrambled", "Famous vs Unfamiliar"]
    # Each panel layers two coco-pipe figures: the per-participant curves
    # underneath, the group mean with its SEM band on top.
    contrast_panels = {}
    for space, space_curves in contrast_curves_by_space.items():
        for name in contrast_names:
            curves = np.asarray(space_curves[name])
            participants = plot_trajectory_metric_series(
                curves,
                times=times,
                labels=np.array([f"sub-{subject}" for subject in unique_subjects]),
                color_map={
                    f"sub-{subject}": CONTRAST_COLORS[name]
                    for subject in unique_subjects
                },
                title=f"{space}: {name}",
                ylabel="Baseline-relative distance",
            )
            participants.update_traces(line={"width": 1}, showlegend=False)
            group = plot_timecourses(
                curves[:, np.newaxis, :],
                times=times,
                channel_names=[name],
                group_labels=[name] * len(curves),
                palette={name: CONTRAST_COLORS[name]},
                error_style="band",
                xlabel="Time (s)",
                ylabel="Baseline-relative distance",
                title=f"{space}: {name}",
            )
            contrast_panels[f"{space}: {name}"] = overlay_figures(
                [participants, group], opacities=[0.28, None]
            )
    fig_contrasts = facet_figures(
        contrast_panels,
        n_cols=2,
        title="Planned contrasts: participants and group mean",
        row_height=380,
    )
    fig_contrasts.add_vline(x=0, line_dash="dash", line_color="black")
    fig_contrasts.add_hline(y=0, line_dash="dot", line_color="grey")
    fig_contrasts.update_xaxes(title_text="Time (s)")
    fig_contrasts.update_yaxes(
        title_text="Distance relative to baseline (a.u.)", col=1
    )
    fig_contrasts.update_layout(
        height=390 * len(contrast_curves_by_space),
        title="Planned contrasts across PCA metric spaces",
        template="plotly_white",
    )

    contrast_summary_rows: list[dict[str, object]] = []
    contrast_timeseries_rows: list[dict[str, object]] = []
    for space, space_curves in contrast_curves_by_space.items():
        for name, curves in space_curves.items():
            for subject, curve in zip(unique_subjects, curves, strict=True):
                contrast_timeseries_rows.extend(
                    {
                        "pca_space": space,
                        "subject": subject,
                        "contrast": name,
                        "time_s": time,
                        "baseline_relative_separation": value,
                    }
                    for time, value in zip(times, curve, strict=True)
                )
                active_curve = curve[active_mask]
                peak_index = int(np.argmax(active_curve))
                contrast_summary_rows.append(
                    {
                        "pca_space": space,
                        "subject": subject,
                        "contrast": name,
                        "auc_0_600ms": np.trapezoid(active_curve, active_times),
                        "peak_separation": active_curve[peak_index],
                        "peak_time_s": active_times[peak_index],
                    }
                )
    contrast_summary = pd.DataFrame(contrast_summary_rows)
    contrast_timeseries = pd.DataFrame(contrast_timeseries_rows)
    contrast_group_summary = (
        contrast_summary.groupby(["pca_space", "contrast"], as_index=False)
        .agg(
            mean_auc=("auc_0_600ms", "mean"),
            sem_auc=("auc_0_600ms", "sem"),
            mean_peak_time_s=("peak_time_s", "mean"),
        )
    )

    observed_auc: dict[str, dict[str, float]] = {}
    null_auc: dict[str, dict[str, np.ndarray | list[float]]] = {}
    for space in metric_trajectory_spaces:
        observed_auc[space] = {
            name: float(
                np.trapezoid(
                    curves.mean(axis=0)[active_mask], active_times
                )
            )
            for name, curves in contrast_curves_by_space[space].items()
        }
        null_auc[space] = {name: [] for name in contrast_names}

    for _ in range(n_perm):
        shuffled = labels.copy()
        for subject in unique_subjects:
            rows = np.flatnonzero(trial_subjects == subject)
            shuffled[rows] = rng.permutation(shuffled[rows])
        for space, trajectories in metric_trajectory_spaces.items():
            permuted = _contrast_curves(
                trajectories,
                shuffled,
                trial_subjects,
                unique_subjects,
                baseline_mask,
            )
            for name, curves in permuted.items():
                null_auc[space][name].append(
                    float(
                        np.trapezoid(
                            curves.mean(axis=0)[active_mask], active_times
                        )
                    )
                )

    inference_rows: list[dict[str, object]] = []
    null_export: dict[str, np.ndarray] = {}
    for space, space_nulls in null_auc.items():
        for name, values in space_nulls.items():
            array = np.asarray(values, dtype=float)
            null_auc[space][name] = array
            null_export[f"{space}__{name}"] = array
            inference_rows.append(
                {
                    "pca_space": space,
                    "contrast": name,
                    "observed_auc": observed_auc[space][name],
                    "null_mean": array.mean(),
                    "empirical_p": (
                        1 + np.sum(array >= observed_auc[space][name])
                    )
                    / (n_perm + 1),
                    "n_permutations": n_perm,
                }
            )
    inference = pd.DataFrame(inference_rows)

    fig_null = make_subplots(
        rows=len(null_auc),
        cols=2,
        subplot_titles=[
            f"{space}: {contrast}"
            for space in null_auc
            for contrast in contrast_names
        ],
    )
    for row, (space, space_nulls) in enumerate(null_auc.items(), start=1):
        for column, name in enumerate(contrast_names, start=1):
            values = np.asarray(space_nulls[name])
            fig_null.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=25,
                    marker_color=CONTRAST_COLORS[name],
                    opacity=0.72,
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
            fig_null.add_vline(
                x=observed_auc[space][name],
                line_color="black",
                line_width=3,
                annotation_text="observed",
                row=row,
                col=column,
            )
    fig_null.update_xaxes(title_text="Baseline-relative separation AUC")
    fig_null.update_yaxes(title_text="Permutations", col=1)
    fig_null.update_layout(
        height=360 * len(null_auc),
        title="Within-participant permutation nulls",
        template="plotly_white",
    )

    # ------------------------------------------------------------------ Steps 9--10
    speed_summary_rows: list[dict[str, object]] = []
    speed_timeseries_rows: list[dict[str, object]] = []
    speed_curves_by_space: dict[str, dict[str, np.ndarray]] = {}
    for space, trajectories in metric_trajectory_spaces.items():
        speed_curves_by_space[space] = {}
        for condition in CONDITIONS:
            speed_curves = []
            for subject in unique_subjects:
                subject_mean = trajectories[
                    (trial_subjects == subject) & (labels == condition)
                ].mean(axis=0)
                velocity = np.gradient(subject_mean, times, axis=0)
                speed = np.linalg.norm(velocity, axis=1)
                speed_curves.append(speed)
                speed_timeseries_rows.extend(
                    {
                        "pca_space": space,
                        "subject": subject,
                        "condition": LABEL_NAMES[condition],
                        "time_s": time,
                        "speed": value,
                    }
                    for time, value in zip(times, speed, strict=True)
                )
                active_speed = speed[active_mask]
                peak_index = int(np.argmax(active_speed))
                speed_summary_rows.append(
                    {
                        "pca_space": space,
                        "subject": subject,
                        "condition": LABEL_NAMES[condition],
                        "mean_speed_0_600ms": active_speed.mean(),
                        "peak_speed": active_speed[peak_index],
                        "peak_time_s": active_times[peak_index],
                    }
                )
            speed_curves_by_space[space][LABEL_NAMES[condition]] = np.asarray(
                speed_curves
            )

    # One panel per PCA space: participant speed curves under the condition
    # means and their SEM bands.
    speed_panels = {}
    for space, condition_curves in speed_curves_by_space.items():
        stacked = np.concatenate(list(condition_curves.values()))
        groups = [
            name for name, curves in condition_curves.items() for _ in range(len(curves))
        ]
        participants = plot_trajectory_metric_series(
            stacked,
            times=times,
            labels=np.array(groups),
            color_map={LABEL_NAMES[c]: CONDITION_COLORS[c] for c in CONDITIONS},
            title=space,
            ylabel="Speed (a.u./s)",
        )
        participants.update_traces(line={"width": 1}, showlegend=False)
        group = plot_timecourses(
            stacked[:, np.newaxis, :],
            times=times,
            channel_names=["speed"],
            group_labels=groups,
            palette={LABEL_NAMES[c]: CONDITION_COLORS[c] for c in CONDITIONS},
            error_style="band",
            xlabel="Time (s)",
            ylabel="Speed (a.u./s)",
            title=space,
        )
        speed_panels[space] = overlay_figures(
            [participants, group], opacities=[0.25, None]
        )
    fig_speed = facet_figures(
        speed_panels,
        n_cols=1,
        title="Trajectory speed: participants and condition means",
        row_height=380,
        shared_yaxes=False,
    )
    fig_speed.add_vline(x=0, line_dash="dash", line_color="black")
    fig_speed.update_xaxes(title_text="Time (s)")
    fig_speed.update_yaxes(title_text="Speed (a.u./s)")
    fig_speed.update_layout(
        height=360 * len(metric_trajectory_spaces),
        title="Trajectory speed across PCA metric spaces",
        template="plotly_white",
    )
    speed_summary = pd.DataFrame(speed_summary_rows)
    speed_timeseries = pd.DataFrame(speed_timeseries_rows)

    reference_curves = {
        name: curves.mean(axis=0)
        for name, curves in _contrast_curves(
            scores_baselined[..., :3],
            labels,
            trial_subjects,
            unique_subjects,
            baseline_mask,
        ).items()
    }
    sensitivity_rows: list[dict[str, object]] = []
    for component_count in (2, 3, 5):
        candidates = _contrast_curves(
            scores_baselined[..., :component_count],
            labels,
            trial_subjects,
            unique_subjects,
            baseline_mask,
        )
        for name, curves in candidates.items():
            active_curve = curves.mean(axis=0)[active_mask]
            sensitivity_rows.append(
                {
                    "contrast": name,
                    "components": component_count,
                    "r_with_3pc_timecourse": np.corrcoef(
                        active_curve, reference_curves[name][active_mask]
                    )[0, 1],
                    "peak_time_s": active_times[int(np.argmax(active_curve))],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    # ------------------------------------------------------------------ Step 11
    focused_results: dict[str, dict[str, object]] = {}
    focused_pcas: dict[str, DimReduction] = {}
    focused_trajectory_figures: dict[str, go.Figure] = {}
    for pair_name, pair in FOCUSED_PAIRS.items():
        pair_mask = np.isin(labels, pair)
        X_pair = X[pair_mask]
        labels_pair = labels[pair_mask]
        subjects_pair = trial_subjects[pair_mask]
        n_pair_trials = X_pair.shape[0]
        pair_matrix = X_pair.transpose(0, 2, 1).reshape(
            n_pair_trials * n_times, n_sensors
        )
        pair_pca = DimReduction(
            method="PCA", n_components=n_components, random_state=seed
        )
        pair_flat = pair_pca.fit_transform(pair_matrix)
        pair_scores = pair_flat.reshape(n_pair_trials, n_times, n_components)
        pair_scores -= pair_scores[:, baseline_mask].mean(axis=1, keepdims=True)
        subject_pair_means = {
            (subject, condition): pair_scores[
                (subjects_pair == subject) & (labels_pair == condition)
            ].mean(axis=0)
            for subject in unique_subjects
            for condition in pair
        }
        pair_group = np.stack(
            [
                np.stack(
                    [
                        subject_pair_means[subject, condition]
                        for subject in unique_subjects
                    ]
                ).mean(axis=0)
                for condition in pair
            ]
        )
        pair_sem = np.stack(
            [
                np.stack(
                    [
                        subject_pair_means[subject, condition]
                        for subject in unique_subjects
                    ]
                ).std(axis=0, ddof=1)
                / np.sqrt(len(unique_subjects))
                for condition in pair
            ]
        )
        separation = np.stack(
            [
                np.linalg.norm(
                    subject_pair_means[subject, pair[0]][:, :3]
                    - subject_pair_means[subject, pair[1]][:, :3],
                    axis=1,
                )
                for subject in unique_subjects
            ]
        )
        separation -= separation[:, baseline_mask].mean(axis=1, keepdims=True)
        focused_pcas[pair_name] = pair_pca
        focused_results[pair_name] = {
            "pair": pair,
            "scores": pair_scores,
            "group": pair_group,
            "sem": pair_sem,
            "separation": separation,
            "variance_3pc": np.asarray(
                pair_pca.get_diagnostics()["explained_variance_ratio_"]
            )[:3].sum(),
        }
        pair_labels = np.array([LABEL_NAMES[condition] for condition in pair])
        focused_trajectory_figures[pair_name] = plot_trajectory(
            X=pair_group[..., :3],
            times=times,
            labels=pair_labels,
            color_map={
                LABEL_NAMES[condition]: CONDITION_COLORS[condition]
                for condition in pair
            },
            title=f"Focused shared PCA: {pair_name}",
            dimensions=3,
            show_markers=False,
            smooth_window=12,
            add_start_end_markers=True,
        )

    focused_variance = pd.DataFrame(
        [
            {
                "focused_space": pair_name,
                "variance_explained_3pc": result["variance_3pc"],
            }
            for pair_name, result in focused_results.items()
        ]
    )
    focused_summary_rows: list[dict[str, object]] = []
    focused_timeseries_rows: list[dict[str, object]] = []
    for pair_name, result in focused_results.items():
        for subject, curve in zip(
            unique_subjects, result["separation"], strict=True
        ):
            focused_timeseries_rows.extend(
                {
                    "focused_space": pair_name,
                    "subject": subject,
                    "time_s": time,
                    "baseline_relative_separation": value,
                }
                for time, value in zip(times, curve, strict=True)
            )
            active_curve = curve[active_mask]
            peak_index = int(np.argmax(active_curve))
            focused_summary_rows.append(
                {
                    "focused_space": pair_name,
                    "subject": subject,
                    "auc_0_600ms": np.trapezoid(active_curve, active_times),
                    "peak_separation": active_curve[peak_index],
                    "peak_time_s": active_times[peak_index],
                }
            )
    focused_summary = pd.DataFrame(focused_summary_rows)
    focused_timeseries = pd.DataFrame(focused_timeseries_rows)
    focused_group_summary = (
        focused_summary.groupby("focused_space", as_index=False)
        .agg(
            mean_auc=("auc_0_600ms", "mean"),
            sem_auc=("auc_0_600ms", "sem"),
            mean_peak_time_s=("peak_time_s", "mean"),
        )
    )

    # Same two-layer construction as the planned contrasts, per focused pair.
    focused_panels = {}
    for pair_name, result in focused_results.items():
        curves = np.asarray(result["separation"])
        participants = plot_trajectory_metric_series(
            curves,
            times=times,
            labels=np.array([f"sub-{subject}" for subject in unique_subjects]),
            color_map={
                f"sub-{subject}": CONTRAST_COLORS[pair_name]
                for subject in unique_subjects
            },
            title=pair_name,
            ylabel="Baseline-relative distance",
        )
        participants.update_traces(line={"width": 1}, showlegend=False)
        group = plot_timecourses(
            curves[:, np.newaxis, :],
            times=times,
            channel_names=[pair_name],
            group_labels=[pair_name] * len(curves),
            palette={pair_name: CONTRAST_COLORS[pair_name]},
            error_style="band",
            xlabel="Time (s)",
            ylabel="Baseline-relative distance",
            title=pair_name,
        )
        focused_panels[pair_name] = overlay_figures(
            [participants, group], opacities=[0.28, None]
        )
    fig_focused_sep = facet_figures(
        focused_panels,
        n_cols=2,
        title="Focused-PCA separation: participants and group mean",
        row_height=380,
    )
    fig_focused_sep.add_vline(x=0, line_dash="dash", line_color="black")
    fig_focused_sep.add_hline(y=0, line_dash="dot", line_color="grey")
    fig_focused_sep.update_xaxes(title_text="Time (s)")
    fig_focused_sep.update_yaxes(
        title_text="Distance relative to baseline (a.u.)", row=1, col=1
    )
    fig_focused_sep.update_layout(
        height=470,
        title="Focused shared-PCA separation",
        template="plotly_white",
    )

    # ------------------------------------------------------------------ Step 12: save everything
    tables = {
        "trial_counts": trial_counts,
        "repetition_counts": repetition_counts,
        "shared_variance": shared_variance,
        "subject_variance": subject_variance,
        "strongest_sensor_loadings": loading_table,
        "planned_contrast_timeseries": contrast_timeseries,
        "planned_contrasts": contrast_summary,
        "planned_contrasts_group": contrast_group_summary,
        "within_subject_null": inference,
        "speed_timeseries": speed_timeseries,
        "speed_summary": speed_summary,
        "component_sensitivity": sensitivity,
        "focused_variance": focused_variance,
        "focused_contrast_timeseries": focused_timeseries,
        "focused_contrasts": focused_summary,
        "focused_contrasts_group": focused_group_summary,
    }
    figures = {
        "sensor_space_qc": fig_sensor,
        "shared_pca_scree": fig_scree,
        "pc_timecourses": fig_pc,
        "shared_trajectories_2d": fig_2d,
        "shared_trajectories_3d": fig_3d,
        "planned_contrasts": fig_contrasts,
        "within_subject_null": fig_null,
        "trajectory_speed": fig_speed,
        "focused_separation": fig_focused_sep,
        "focused_famous_unfamiliar": focused_trajectory_figures[
            "Famous vs Unfamiliar"
        ],
        "focused_famous_scrambled": focused_trajectory_figures[
            "Famous vs Scrambled"
        ],
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    static_export_error: str | None = None
    for name, figure in figures.items():
        figure.write_html(figures_dir / f"{name}.html", include_plotlyjs="cdn")
        if static_export_error is None:
            try:
                figure.write_image(figures_dir / f"{name}.png", scale=2)
                figure.write_image(figures_dir / f"{name}.svg")
            except Exception as error:  # Plotly delegates static export to Kaleido.
                static_export_error = f"{type(error).__name__}: {error}"
                warnings.warn(
                    "Static Plotly export is unavailable; continuing with complete "
                    "interactive HTML figures.",
                    stacklevel=2,
                )

    shared_pca.save(reducers_dir / "shared_three_condition_pca.pkl")
    for subject, reducer in subject_pcas.items():
        reducer.save(reducers_dir / f"subject_{subject}_pca.pkl")
    for pair_name, reducer in focused_pcas.items():
        slug = pair_name.lower().replace(" vs ", "_").replace(" ", "_")
        reducer.save(reducers_dir / f"focused_{slug}_pca.pkl")

    focused_arrays: dict[str, np.ndarray] = {}
    for pair_name, result in focused_results.items():
        slug = pair_name.lower().replace(" vs ", "_").replace(" ", "_")
        for array_name in ("scores", "group", "sem", "separation"):
            focused_arrays[f"focused_{slug}_{array_name}"] = np.asarray(
                result[array_name], dtype=np.float32
            )

    np.savez_compressed(
        output / "shared_pca_trajectories.npz",
        scores=scores_baselined.astype(np.float32),
        times=times,
        labels=labels,
        subjects=trial_subjects,
        channels=channels,
        loadings=np.asarray(shared_pca.get_components()),
        explained_variance_ratio=explained_variance,
        subject_scores=(
            subject_scores_baselined.astype(np.float32)
            if subject_scores_baselined is not None
            else np.array([])
        ),
        group_trajectories=group_trajectories.astype(np.float32),
        group_sem=group_sem.astype(np.float32),
        **{f"null_{name}": values for name, values in null_export.items()},
        **focused_arrays,
    )

    # A self-contained report is required here: unlike the standalone figure
    # HTML files, its Plotly, Tailwind, and pako assets are embedded directly.
    report.add_summary_card(
        {
            "Participants": len(unique_subjects),
            "Trials": f"{X.shape[0]:,}",
            "Variance @ 3 PCs": f"{explained_variance[:3].sum():.1%}",
            "Sensors": f"{n_sensors} ({sensor_set})",
            "Permutations": n_perm,
        }
    )

    overview = Section(
        "Overview",
        icon="O",
        description="Face-processing trajectories in whitened MEG sensor space",
        metadata={
            "Participants": ", ".join(unique_subjects),
            "Trials x sensors x time": str(X.shape),
            "Window": f"{times[0]:+.3f} to {times[-1]:+.3f} s",
            "Sampling rate": f"{1 / np.diff(times).mean():.1f} Hz",
            "Sensor set": f"{sensor_set} ({n_sensors} sensors)",
            "PCA metric mode": str(metric_pca_mode),
            "Components": str(n_components),
            "Active window": f"{ACTIVE_WINDOW[0]:.1f}-{ACTIVE_WINDOW[1]:.1f} s",
            "Permutations": str(n_perm),
            "Whitening": str(container.meta.get("whitening", "not recorded")),
        },
    )
    overview.add_element(
        MarkdownElement(
            "This report mirrors every analysis and interpretation step in "
            "`tutorial_megfaces_main.ipynb`. It uses the **Wakeman-Henson "
            "multimodal face-processing dataset** to compare Famous faces, "
            "Unfamiliar faces, and Scrambled images in whitened MEG sensor space."
        )
    )
    overview.add_element(
        CalloutElement(
            "This is a descriptive sensor-space analysis, not source localisation.",
            kind="warning",
            title="Scope",
        )
    )
    report.add_section(overview)

    step0 = Section(
        "Step 0 - Setup and Analysis Choices",
        icon="0",
        description="Prepared derivatives in, every choice recorded in the manifests",
    )
    step0.add_element(
        MarkdownElement(
            "This workflow reads prepared MEG derivatives only; downloading and "
            "Maxwell/ICA preprocessing remain explicit upstream operations. The "
            "random seed, participant list, component count, active window, "
            "permutation count, and metric PCA mode are recorded in both manifests.\n\n"
            "`metric_pca_mode` selects `shared`, `subject`, or `both` for "
            "participant-level separation and speed. Group trajectory figures "
            "always use the shared PCA because participant-specific axes cannot be "
            "averaged directly."
        )
    )
    report.add_section(step0)

    step1 = Section(
        "Step 1 - Define the Neural State",
        icon="1",
        description="The whitened sensor pattern at each time point",
    )
    step1.add_element(
        MarkdownElement(
            f"At each time point the state is the pattern across **{n_sensors} "
            "retained MEG sensors**. Participant-specific empty-room covariance "
            "whitening places magnetometers and gradiometers on a common noise "
            "scale before PCA. Channels are not z-scored again because that would "
            "discard the measured noise scaling."
        )
    )
    step1.add_element(
        CalloutElement(
            "Sensor-space PCs mix already mixed neural sources. Interpret timing and "
            "trajectory geometry, not an individual PC as an anatomical generator.",
            kind="warning",
            title="What a PC is not",
        )
    )
    report.add_section(step1)

    step2 = Section(
        "Step 2 - Load and Inspect Participants",
        icon="2",
        description="Trial counts and a sensor-space RMS sanity check",
    )
    step2.add_element(
        MarkdownElement(
            "The loader returns `trial x sensor x time` and keeps condition, "
            "participant, and repetition metadata aligned with every trial. Trial "
            "counts expose the weighting of the pooled PCA fit. The RMS sensor-space "
            "view checks for stimulus-locked structure before dimensionality reduction."
        )
    )
    step2.add_element(PlotlyElement(fig_sensor, height="520px"))
    step2.add_element(
        TabsElement(
            {
                "By condition": InteractiveTableElement(
                    trial_counts, title="Trials by participant and condition"
                ),
                "By repetition": InteractiveTableElement(
                    repetition_counts, title="Trials by repetition type"
                ),
            }
        )
    )
    report.add_section(step2)

    step3 = Section(
        "Step 3 - Reshape into PCA Observations",
        icon="3",
        description="(trial, sensor, time) flattened to (trial x time, sensor)",
    )
    step3.add_element(
        MarkdownElement(
            "PCA receives `(trial, sensor, time) -> (trial x time, sensor)`. "
            f"The fitted matrix contains **{pooled.shape[0]:,} observations x "
            f"{pooled.shape[1]} sensors**. Labels are ignored during fitting. "
            "Participants with more retained trials contribute more observations to "
            "the PCA basis; later group summaries restore equal participant weight."
        )
    )
    report.add_section(step3)

    step4 = Section(
        "Step 4 - Fit One Shared PCA",
        icon="4",
        description="One unsupervised basis across every participant and condition",
    )
    step4.add_element(
        MarkdownElement(
            f"One unsupervised PCA is fitted across every participant and condition. "
            f"The first three PCs explain **{explained_variance[:3].sum():.1%}** "
            f"and all {n_components} fitted PCs explain **{explained_variance.sum():.1%}**."
        )
    )
    step4.add_columns(
        [
            StatCardElement(
                "Variance @ 3 PCs", f"{explained_variance[:3].sum():.1%}", color="blue"
            ),
            StatCardElement(
                f"Variance @ {n_components} PCs",
                f"{explained_variance.sum():.1%}",
                color="green",
            ),
            StatCardElement("PC1", f"{explained_variance[0]:.1%}", color="purple"),
        ]
    )
    step4.add_element(
        CalloutElement(
            "Three PCs are a display choice. A clear 3-D trajectory does not show "
            "that the response is intrinsically three-dimensional.",
            kind="warning",
            title="Dimensionality caveat",
        )
    )
    step4.add_element(PlotlyElement(fig_scree, height="520px"))
    step4.add_element(InteractiveTableElement(shared_variance, title="Shared PCA variance"))
    report.add_section(step4)

    step5 = Section(
        "Step 5 - Reconstruct and Baseline Trajectories",
        icon="5",
        description="Scores unstacked and anchored to the pre-stimulus baseline",
    )
    step5.add_element(
        MarkdownElement(
            "Scores are reshaped to `trial x time x component`, then each trial's "
            "mean pre-stimulus score is subtracted. The resulting mean absolute "
            f"baseline offset is **{baseline_offset:.3e}**. Participant-specific "
            "PCAs are also fitted when requested, but their coordinates are never "
            "averaged across participants. Distances and speeds are invariant to "
            "sign flips and rotations within a retained subspace; the important "
            "sensitivity is which subspace is retained."
        )
    )
    step5.add_element(PlotlyElement(fig_pc, height="760px"))
    step5.add_element(
        InteractiveTableElement(loading_table, title="Strongest displayed-PC sensor loadings")
    )
    if not subject_variance.empty:
        step5.add_element(
            InteractiveTableElement(subject_variance, title="Participant PCA variance")
        )
    report.add_section(step5)

    step6 = Section(
        "Step 6 - Plot All Conditions in One Space",
        icon="6",
        description="Participant-weighted condition means in the shared basis",
    )
    step6.add_element(
        MarkdownElement(
            "Trials are averaged within participant and condition before the "
            "participant trajectories are averaged. Every participant therefore "
            "has equal group weight despite unequal retained trial counts. Follow "
            "the path from baseline through the post-stimulus excursion; loops and "
            "bends are geometric descriptions, not evidence of oscillatory mechanisms."
        )
    )
    step6.add_element(
        TabsElement(
            {
                "2D": PlotlyElement(fig_2d, height="620px"),
                "3D": PlotlyElement(fig_3d, height="740px"),
            }
        )
    )
    report.add_section(step6)

    step7 = Section(
        "Step 7 - Evaluate Planned Contrasts",
        icon="7",
        description="Famous vs Unfamiliar and Faces vs Scrambled, per participant",
    )
    step7.add_element(
        MarkdownElement(
            "**Famous vs Unfamiliar** is the distance between those condition "
            "means. **Faces vs Scrambled** first gives Famous and Unfamiliar equal "
            "weight and then measures their distance from Scrambled. Each curve is "
            "computed per participant and made relative to that participant's mean "
            "pre-stimulus distance. Thin curves show participants; thick curves and "
            "bands show the group mean and SEM."
        )
    )
    step7.add_element(
        PlotlyElement(
            fig_contrasts, height=f"{390 * len(metric_trajectory_spaces)}px"
        )
    )
    step7.add_element(
        InteractiveTableElement(
            contrast_group_summary, title="Group planned-contrast summary"
        )
    )
    details = AccordionElement("Participant-level planned contrasts", open=False)
    details.add_element(InteractiveTableElement(contrast_summary, title="Participant metrics"))
    step7.add_element(details)
    report.add_section(step7)

    step8 = Section(
        "Step 8 - Compare with a Within-Participant Null",
        icon="8",
        description="Labels shuffled inside each participant, PCA axes held fixed",
    )
    step8.add_element(
        MarkdownElement(
            "PCA axes stay fixed while condition labels are shuffled separately "
            "within each participant. Every shuffle preserves participant identity, "
            "trial count, and preprocessing history, then recomputes both contrasts. "
            f"With **{n_perm} permutations**, the smallest attainable empirical "
            f"p-value is **{1 / (n_perm + 1):.4f}**. The p-values test condition "
            "assignment under this exchangeability scheme; they do not correct a "
            "larger family of exploratory choices."
        )
    )
    step8.add_element(PlotlyElement(fig_null, height=f"{360 * len(metric_trajectory_spaces)}px"))
    step8.add_element(InteractiveTableElement(inference, title="Permutation inference"))
    report.add_section(step8)

    step9 = Section(
        "Step 9 - Describe Trajectory Speed",
        icon="9",
        description="Norm of the time derivative of each condition-mean path",
    )
    step9.add_element(
        MarkdownElement(
            "Speed is the norm of the numerical time derivative of each participant's "
            "condition-mean trajectory. It can reveal a shared state transition but "
            "is not automatically condition-specific. Derivatives amplify noise, so "
            "broad changes are more trustworthy than isolated samples. Raw speed "
            "should not be compared between shared and participant spaces as if the "
            "coordinate scales were identical."
        )
    )
    step9.add_element(PlotlyElement(fig_speed, height=f"{360 * len(metric_trajectory_spaces)}px"))
    step9.add_element(InteractiveTableElement(speed_summary, title="Participant speed summary"))
    report.add_section(step9)

    step10 = Section(
        "Step 10 - Check Component Sensitivity",
        icon="10",
        description="2, 3, and 5 PCs compared by curve correlation and peak timing",
    )
    step10.add_element(
        MarkdownElement(
            "Distances grow as orthogonal dimensions are added, so raw AUC values "
            "from different component counts are not compared directly. Instead, "
            "the shared-space 2-, 3-, and 5-PC contrast time courses are compared by "
            "their correlation with the 3-PC reference and by peak timing."
        )
    )
    step10.add_element(
        InteractiveTableElement(sensitivity, title="Shared-PCA component sensitivity")
    )
    report.add_section(step10)

    step11 = Section(
        "Step 11 - Fit Focused Two-Condition PCAs",
        icon="11",
        description="Pair-specific bases as a sensitivity check on the shared PCA",
    )
    step11.add_element(
        MarkdownElement(
            "The three-condition PCA remains primary. Two additional shared PCAs "
            "are fitted to Famous-Unfamiliar and Famous-Scrambled trials separately. "
            "They can prioritize pair-relevant variance that the third condition "
            "otherwise dominates. Axes and raw distances cannot be compared across "
            "these models; compare divergence timing and participant consistency."
        )
    )
    step11.add_element(
        TabsElement(
            {
                name: PlotlyElement(figure, height="680px")
                for name, figure in focused_trajectory_figures.items()
            }
        )
    )
    step11.add_element(PlotlyElement(fig_focused_sep, height="520px"))
    step11.add_element(
        TabsElement(
            {
                "Focused PCA variance": InteractiveTableElement(
                    focused_variance, title="Focused PCA variance"
                ),
                "Contrast summary": InteractiveTableElement(
                    focused_group_summary, title="Focused contrast summary"
                ),
            }
        )
    )
    focused_details = AccordionElement("Participant-level focused contrasts", open=False)
    focused_details.add_element(
        InteractiveTableElement(focused_summary, title="Participant metrics")
    )
    step11.add_element(focused_details)
    report.add_section(step11)

    step12 = Section(
        "Step 12 - Export, Reproduce, and Interpret",
        icon="12",
        description="What the bundle contains and what the numbers do not say",
    )
    static_status = (
        "PNG and SVG exports completed."
        if static_export_error is None
        else "Static export was unavailable; all interactive HTML figures are complete."
    )
    step12.add_element(
        MarkdownElement(
            "The export preserves every table, interactive figure, fitted PCA, "
            "trajectory array, permutation null, and environment manifest. "
            f"**{static_status}**\n\n"
            "The complementary conclusions are:\n\n"
            "- one three-condition shared PCA supplies a common group space;\n"
            "- participant PCAs test metric sensitivity to the shared basis;\n"
            "- focused PCAs ask whether a selected pair is clearer without the third condition."
        )
    )
    step12.add_element(
        CalloutElement(
            "Group trajectory coordinates require shared axes. Statistical metrics "
            "are always computed per participant before group summarisation. The "
            "analysis is descriptive and sensor-space based; it does not localize "
            "sources.",
            kind="tip",
            title="Main takeaway",
        )
    )
    all_tables = AccordionElement("Appendix: all exported tables", open=False)
    for name, table in tables.items():
        if not table.empty:
            all_tables.add_element(
                InteractiveTableElement(table, title=name.replace("_", " ").title())
            )
    step12.add_element(all_tables)
    report.add_section(step12)
    report.save(output / "report.html")

    manifest = {
        "subjects_requested": subjects,
        "subjects_analyzed": unique_subjects.tolist(),
        "derivatives_root": str(derivatives_root),
        "conditions": list(CONDITIONS),
        "condition_names": [LABEL_NAMES[condition] for condition in CONDITIONS],
        "metric_pca_mode": metric_pca_mode,
        "sensor_set": sensor_set,
        "sensor_names": channels.tolist(),
        "n_components": n_components,
        "display_components": N_DISPLAY_COMPONENTS,
        "active_window": list(ACTIVE_WINDOW),
        "n_permutations": n_perm,
        "random_state": seed,
        "shape": list(X.shape),
        "whitening": container.meta.get("whitening", "not recorded"),
        "report_asset_mode": report.asset_mode,
        "static_figure_exports_complete": static_export_error is None,
        "static_figure_export_error": static_export_error,
    }
    write_manifest(output / "analysis_manifest.json", manifest, status="complete")
    print(f"Saved MEG Faces main analysis -> {output}")
    return {
        "manifest": manifest,
        "tables": tables,
        "figures": figures,
        "shared_scores": scores_baselined,
        "subject_scores": subject_scores_baselined,
        "focused_results": focused_results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=["01", "02", "03", "04", "05", "06"],
        help="Prepared participant IDs (default: 01 through 06).",
    )
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=(
            Path.home()
            / "mne_data"
            / "ds000117"
            / "derivatives"
            / "pca_trajectories"
        ),
        help="Root containing preprocessed subject-wise whitened epochs.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/megfaces_main")
    )
    parser.add_argument(
        "--sensor-set",
        choices=tuple(MEG_SENSOR_SETS),
        default="all_sensors",
    )
    parser.add_argument(
        "--metric-pca-mode",
        choices=("shared", "subject", "both"),
        default="both",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the first two requested participants and at most five permutations.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Re-run even if this output already contains a completed manifest.",
    )
    args = parser.parse_args(argv)

    subjects = args.subjects[:2] if args.smoke else args.subjects
    n_perm = min(args.n_perm, 5) if args.smoke else args.n_perm
    manifest_path = args.output / args.sensor_set / "run_manifest.json"
    if args.resume and _completed(manifest_path):
        print(f"Completed run found at {manifest_path}; nothing to do.")
        return

    settings = {**vars(args), "subjects": subjects, "n_perm": n_perm}
    write_manifest(manifest_path, settings, status="running")
    try:
        result = run_megfaces_main_analysis(
            subjects=subjects,
            derivatives_root=args.derivatives_root,
            output=args.output,
            metric_pca_mode=args.metric_pca_mode,
            n_components=args.n_components,
            n_perm=n_perm,
            seed=args.seed,
            sensor_set=args.sensor_set,
        )
    except Exception as error:
        write_manifest(
            manifest_path,
            settings,
            status="failed",
            extra={"error": f"{type(error).__name__}: {error}"},
        )
        raise
    write_manifest(
        manifest_path,
        settings,
        status="complete",
        extra={"analysis": result["manifest"]},
    )


if __name__ == "__main__":
    main()
