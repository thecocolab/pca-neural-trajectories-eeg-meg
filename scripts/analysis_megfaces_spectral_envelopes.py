"""Run and report the complete MEG Faces spectral-envelope analysis.

Headless counterpart to ``tutorials/tutorial_megfaces_spectral_envelopes.ipynb``.
The script uses the same padded preprocessing, all retained trials, Hilbert
power transform, band-specific shared and participant PCAs, planned
contrasts, speeds, focused spaces, cross-band descriptors, and family-corrected
permutation test. It saves all arrays, tables, reducers, figures, manifests, and
a fully self-contained HTML report.

Prepared long epochs are loaded by default. Pass ``--prepare`` explicitly to
create missing derivatives from already-downloaded raw data.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from coco_pipe.dim_reduction import DimReduction
from coco_pipe.report import PlotlyElement, Report, Section
from coco_pipe.report.elements import (
    AccordionElement,
    CalloutElement,
    ContainerElement,
    Element,
    InteractiveTableElement,
    MarkdownElement,
    TabsElement,
)
from coco_pipe.viz.interactive import (
    plot_scree,
    plot_timecourses,
    plot_trajectory,
    plot_trajectory_metric_series,
)
from scipy.ndimage import gaussian_filter1d

from pca_neural_trajectories import facet_figures, overlay_figures, write_manifest
from pca_neural_trajectories.spectral import SPECTRAL_BANDS
from pca_neural_trajectories.wakeman_henson import (
    LABEL_NAMES,
    MEG_SENSOR_SETS,
    _load_wakeman_henson_container,
    _preprocess_subject,
    epochs_path,
    preprocessing_config,
)

CONDITIONS = (1, 2, 3)
FOCUSED_PAIRS = {"Famous vs Unfamiliar": (1, 2), "Famous vs Scrambled": (1, 3)}
FINAL_WINDOW = (-0.2, 0.8)
BASELINE_WINDOW = (-0.2, 0.0)
ACTIVE_WINDOW = (0.0, 0.6)
ENVELOPE_SFREQ = 62.5
SMOOTHING_S = 0.04
CONDITION_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
CONDITION_FILLS = {
    1: "rgba(0,114,178,0.15)",
    2: "rgba(213,94,0,0.15)",
    3: "rgba(0,158,115,0.15)",
}
CONTRAST_COLORS = {
    "Faces vs Scrambled": "#7B2CBF",
    "Famous vs Unfamiliar": "#D55E00",
    "Famous vs Scrambled": "#0072B2",
}


def _contrast_curves(
    trajectories: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    selected_subjects: list[str],
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    curves: dict[str, list[np.ndarray]] = {
        "Faces vs Scrambled": [],
        "Famous vs Unfamiliar": [],
    }
    baseline = times < 0
    for subject in selected_subjects:
        means = {
            condition: trajectories[(subjects == subject) & (labels == condition)].mean(axis=0)
            for condition in CONDITIONS
        }
        distances = {
            "Faces vs Scrambled": np.linalg.norm(0.5 * (means[1] + means[2]) - means[3], axis=1),
            "Famous vs Unfamiliar": np.linalg.norm(means[1] - means[2], axis=1),
        }
        for name, values in distances.items():
            curves[name].append(values - values[baseline].mean())
    return {name: np.asarray(values) for name, values in curves.items()}


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def run_spectral_analysis(
    *,
    subjects: list[str],
    raw_root: Path,
    derivatives_root: Path,
    output: Path,
    metric_pca_mode: str = "both",
    n_components: int = 10,
    n_perm: int = 200,
    seed: int = 42,
    prepare: bool = False,
    sensor_set: str = "all_sensors",
) -> dict[str, object]:
    """Run the notebook-equivalent analysis and write the complete report bundle."""
    if metric_pca_mode not in {"shared", "subject", "both"}:
        raise ValueError("metric_pca_mode must be 'shared', 'subject', or 'both'.")
    if n_components < 3:
        raise ValueError("n_components must be at least 3.")
    if n_perm < 0:
        raise ValueError("n_perm cannot be negative.")
    subjects = [str(subject).removeprefix("sub-").zfill(2) for subject in subjects]
    if len(subjects) < 2:
        raise ValueError("At least two participants are required for group SEMs.")

    output = Path(output) / sensor_set
    figures_dir = output / "figures"
    reducers_dir = output / "reducers"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reducers_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if prepare:
        config = preprocessing_config(
            l_freq=0.5,
            h_freq=90.0,
            sfreq=250.0,
            tmin=-1.2,
            tmax=1.8,
            baseline=None,
            random_state=seed,
        )
        for subject in subjects:
            _preprocess_subject(
                raw_root,
                subject,
                derivatives_root=derivatives_root,
                config=config,
                overwrite=False,
            )
    missing = [
        subject for subject in subjects if not epochs_path(derivatives_root, subject).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing long spectral epochs for {missing}. Re-run with --prepare."
        )

    report = Report(
        title=(
            "MEG Faces Spectral Envelopes - "
            f"{sensor_set} ({len(subjects)} participants)"
        ),
        asset_urls="inline",
    )
    print("=== MEG Faces spectral-envelope analysis ===")
    print(f"subjects: {', '.join(subjects)}")
    print(f"derivatives: {derivatives_root}")

    filter_rows = []
    source_sfreq = 250.0
    source_n_times = int(3.0 * source_sfreq) + 1
    for band, (low, high) in SPECTRAL_BANDS.items():
        kernel = mne.filter.create_filter(
            np.empty((1, source_n_times)),
            sfreq=source_sfreq,
            l_freq=low,
            h_freq=high,
            fir_design="firwin",
            verbose=False,
        )
        filter_rows.append(
            {
                "band": band,
                "low_hz": low,
                "high_hz": high,
                "filter_length_s": len(kernel) / source_sfreq,
                "half_support_s": (len(kernel) - 1) / (2 * source_sfreq),
            }
        )
    filter_support = pd.DataFrame(filter_rows)

    container = _load_wakeman_henson_container(
        derivatives_root,
        subjects=subjects,
        conditions=CONDITIONS,
        notch_freq=50.0,
        sensor_set=sensor_set,
    )
    source_X = np.asarray(container.X, dtype=np.float32)
    source_times = np.asarray(container.coords["time"], dtype=float)
    source_channels = np.asarray(container.coords["channel"]).astype(str)
    source_subjects = np.asarray(container.coords["subject"]).astype(str)
    source_labels = np.asarray(container.y, dtype=int)
    source_repetitions = np.asarray(container.coords["repetition"]).astype(str)
    source_info = mne.create_info(
        source_channels.tolist(),
        sfreq=1 / np.diff(source_times).mean(),
        ch_types="misc",
    )
    sources = {}
    for subject in subjects:
        rows = source_subjects == subject
        sources[subject] = mne.EpochsArray(
            source_X[rows],
            source_info,
            tmin=source_times[0],
            metadata=pd.DataFrame(
                {
                    "condition_id": source_labels[rows],
                    "repetition": source_repetitions[rows],
                }
            ),
            baseline=None,
            verbose=False,
        )
    demo = sources[subjects[0]][0:1]
    demo_times = demo.times
    signed = demo.get_data(copy=True)[0, 0]
    filtered_epochs = demo.copy().filter(*SPECTRAL_BANDS["alpha"], picks="all", verbose=False)
    filtered = filtered_epochs.get_data(copy=True)[0, 0]
    envelope_epochs = filtered_epochs.copy().apply_hilbert(
        picks="all", envelope=True, verbose=False
    )
    power = envelope_epochs.get_data(copy=True)[0, 0] ** 2
    smoothed = gaussian_filter1d(power, sigma=SMOOTHING_S * demo.info["sfreq"], mode="reflect")
    demo_baseline = (demo_times >= BASELINE_WINDOW[0]) & (demo_times <= BASELINE_WINDOW[1])
    demo_db = 10 * np.log10(
        np.maximum(smoothed, np.finfo(float).tiny) / smoothed[demo_baseline].mean()
    )
    # Four stages of the same demo channel, stacked as one panel each.
    fig_transform = plot_timecourses(
        np.stack([signed, filtered, power, demo_db])[np.newaxis, :, :],
        times=demo_times,
        channel_names=[
            "Whitened signed MEG",
            "Alpha-filtered signal",
            "Hilbert power",
            "Smoothed baseline-relative power",
        ],
        n_cols=1,
        error_style=None,
        xlabel="Time (s)",
        ylabel="Amplitude",
        showlegend=False,
    )
    fig_transform.add_vline(x=0, line_dash="dash", line_color="black")
    fig_transform.update_layout(
        height=760,
        title="Alpha-power transformation: one trial, one sensor",
        template="plotly_white",
    )

    parts: dict[str, list[dict[str, np.ndarray]]] = {band: [] for band in SPECTRAL_BANDS}
    count_rows = []
    for subject in subjects:
        source = sources[subject]
        source_labels = source.metadata["condition_id"].to_numpy(int)
        for condition in CONDITIONS:
            count_rows.append(
                {
                    "subject": subject,
                    "condition": LABEL_NAMES[condition],
                    "trials": int(np.sum(source_labels == condition)),
                }
            )
        for band, (low, high) in SPECTRAL_BANDS.items():
            epochs = source.copy().filter(low, high, picks="all", verbose=False)
            epochs.apply_hilbert(picks="all", envelope=True, verbose=False)
            values = epochs.get_data(copy=True) ** 2
            values = gaussian_filter1d(
                values,
                sigma=SMOOTHING_S * source.info["sfreq"],
                axis=-1,
                mode="reflect",
            )
            source_times = source.times
            baseline = (source_times >= BASELINE_WINDOW[0]) & (source_times <= BASELINE_WINDOW[1])
            baseline_power = values[..., baseline].mean(axis=-1, keepdims=True)
            values = 10 * np.log10(
                np.maximum(values, np.finfo(float).tiny)
                / np.maximum(baseline_power, np.finfo(float).tiny)
            )
            power_epochs = mne.EpochsArray(
                values,
                source.info,
                events=source.events,
                event_id=source.event_id,
                tmin=source.tmin,
                metadata=source.metadata,
                baseline=None,
                verbose=False,
            )
            power_epochs.crop(*FINAL_WINDOW).resample(ENVELOPE_SFREQ, npad="auto", verbose=False)
            parts[band].append(
                {
                    "X": power_epochs.get_data(copy=True).astype(np.float32),
                    "times": power_epochs.times.copy(),
                    "channels": np.asarray(power_epochs.ch_names),
                    "labels": power_epochs.metadata["condition_id"].to_numpy(int),
                    "repetitions": power_epochs.metadata["repetition"].astype(str).to_numpy(),
                    "subjects": np.repeat(subject, len(power_epochs)),
                }
            )
    trial_counts = pd.DataFrame(count_rows)
    band_data = {
        band: {
            "X": np.concatenate([part["X"] for part in band_parts]),
            "times": band_parts[0]["times"],
            "channels": band_parts[0]["channels"],
            "labels": np.concatenate([part["labels"] for part in band_parts]),
            "repetitions": np.concatenate([part["repetitions"] for part in band_parts]),
            "subjects": np.concatenate([part["subjects"] for part in band_parts]),
        }
        for band, band_parts in parts.items()
    }
    del sources, container, source_X

    figures: dict[str, go.Figure] = {"power_transformation": fig_transform}
    # One panel per band; participants grouped by condition give mean + SEM.
    sensor_panels = {}
    for band, data in band_data.items():
        curves, groups = [], []
        for condition in CONDITIONS:
            for subject in subjects:
                curves.append(
                    data["X"][
                        (data["subjects"] == subject) & (data["labels"] == condition)
                    ].mean(axis=(0, 1))
                )
                groups.append(LABEL_NAMES[condition])
        sensor_panels[band] = plot_timecourses(
            np.asarray(curves)[:, np.newaxis, :],
            times=data["times"],
            channel_names=[band],
            group_labels=groups,
            palette={LABEL_NAMES[c]: CONDITION_COLORS[c] for c in CONDITIONS},
            error_style="band",
            xlabel="Time (s)",
            ylabel="Power (dB re baseline)",
            title=band,
        )
    fig_sensor = facet_figures(
        sensor_panels,
        n_cols=1,
        title="Sensor-space band power",
        row_height=320,
        shared_yaxes=False,
    )
    fig_sensor.add_vline(x=0, line_dash="dash", line_color="black")
    fig_sensor.update_layout(
        height=820, title="Whitened sensor-space band power", template="plotly_white"
    )
    figures["sensor_space_power"] = fig_sensor

    results: dict[str, dict[str, object]] = {}
    pca_rows = []
    subject_variance_rows = []
    loading_rows = []
    metric_rows = []
    metric_timeseries_rows = []
    speed_rows = []
    speed_timeseries_rows = []
    for band, data in band_data.items():
        X = data["X"]
        n_trials, n_sensors, n_times = X.shape
        pooled = X.transpose(0, 2, 1).reshape(n_trials * n_times, n_sensors)
        pca = DimReduction(method="PCA", n_components=n_components, random_state=seed)
        scores = pca.fit_transform(pooled).reshape(n_trials, n_times, n_components)
        baseline = data["times"] < 0
        scores -= scores[:, baseline].mean(axis=1, keepdims=True)
        evr = np.asarray(pca.get_diagnostics()["explained_variance_ratio_"])
        pca_rows.append(
            {
                "band": band,
                "variance_3pc": evr[:3].sum(),
                "variance_fitted": evr.sum(),
                "participation_ratio_fitted": evr.sum() ** 2 / np.square(evr).sum(),
            }
        )
        subject_scores = None
        subject_pcas: dict[str, DimReduction] = {}
        if metric_pca_mode in {"subject", "both"}:
            subject_scores = np.empty_like(scores)
            for subject in subjects:
                rows = data["subjects"] == subject
                X_subject = X[rows]
                matrix = X_subject.transpose(0, 2, 1).reshape(-1, n_sensors)
                reducer = DimReduction(method="PCA", n_components=n_components, random_state=seed)
                transformed = reducer.fit_transform(matrix).reshape(
                    len(X_subject), n_times, n_components
                )
                transformed -= transformed[:, baseline].mean(axis=1, keepdims=True)
                subject_scores[rows] = transformed
                subject_pcas[subject] = reducer
                subject_evr = np.asarray(reducer.get_diagnostics()["explained_variance_ratio_"])
                subject_variance_rows.append(
                    {"band": band, "subject": subject, "variance_3pc": subject_evr[:3].sum()}
                )
        spaces: dict[str, np.ndarray] = {}
        if metric_pca_mode in {"shared", "both"}:
            spaces["Shared PCA"] = scores[..., :3]
        if metric_pca_mode in {"subject", "both"}:
            assert subject_scores is not None
            spaces["Subject PCA"] = subject_scores[..., :3]
        contrasts = {
            space: _contrast_curves(
                trajectories,
                data["labels"],
                data["subjects"],
                subjects,
                data["times"],
            )
            for space, trajectories in spaces.items()
        }
        active = (data["times"] >= ACTIVE_WINDOW[0]) & (data["times"] <= ACTIVE_WINDOW[1])
        for space, space_curves in contrasts.items():
            for contrast, curves in space_curves.items():
                for subject, curve in zip(subjects, curves, strict=True):
                    metric_timeseries_rows.extend(
                        {
                            "band": band,
                            "pca_space": space,
                            "subject": subject,
                            "contrast": contrast,
                            "time_s": time,
                            "separation": value,
                        }
                        for time, value in zip(data["times"], curve, strict=True)
                    )
                    active_curve = curve[active]
                    peak = int(np.argmax(active_curve))
                    metric_rows.append(
                        {
                            "band": band,
                            "pca_space": space,
                            "subject": subject,
                            "contrast": contrast,
                            "auc_0_600ms": np.trapezoid(active_curve, data["times"][active]),
                            "peak_separation": active_curve[peak],
                            "peak_time_s": data["times"][active][peak],
                        }
                    )

        group = []
        group_sem = []
        for condition in CONDITIONS:
            means = np.stack(
                [
                    scores[(data["subjects"] == subject) & (data["labels"] == condition)].mean(
                        axis=0
                    )
                    for subject in subjects
                ]
            )
            group.append(means.mean(axis=0))
            group_sem.append(means.std(axis=0, ddof=1) / np.sqrt(len(subjects)))
        group = np.asarray(group)
        group_sem = np.asarray(group_sem)
        results[band] = {
            **data,
            "shared_pca": pca,
            "shared_scores": scores,
            "subject_scores": subject_scores,
            "subject_pcas": subject_pcas,
            "evr": evr,
            "metric_spaces": spaces,
            "contrast_curves": contrasts,
            "group": group,
            "group_sem": group_sem,
        }
        figures[f"{band}_scree"] = plot_scree(evr)
        figures[f"{band}_scree"].update_layout(title=f"{band}: shared PCA spectrum")

        pc_curves, pc_groups = [], []
        for condition in CONDITIONS:
            for subject in subjects:
                subject_mean = scores[
                    (data["subjects"] == subject) & (data["labels"] == condition)
                ].mean(axis=0)
                pc_curves.append(subject_mean[:, :3].T)
                pc_groups.append(LABEL_NAMES[condition])
        fig_pc = plot_timecourses(
            np.stack(pc_curves),
            times=data["times"],
            channel_names=[f"PC{index + 1}" for index in range(3)],
            group_labels=pc_groups,
            palette={LABEL_NAMES[c]: CONDITION_COLORS[c] for c in CONDITIONS},
            n_cols=1,
            error_style="band",
            xlabel="Time (s)",
            ylabel="Score (a.u.)",
            title=f"{band}: shared-PC timecourses",
        )
        fig_pc.add_vline(x=0, line_dash="dash", line_color="black")
        figures[f"{band}_pc_timecourses"] = fig_pc
        for pc, weights in enumerate(np.asarray(pca.get_components())[:3], start=1):
            for index in np.argsort(np.abs(weights))[-5:][::-1]:
                loading_rows.append(
                    {
                        "band": band,
                        "component": f"PC{pc}",
                        "sensor": data["channels"][index],
                        "loading": weights[index],
                    }
                )
        labels_plot = np.array([LABEL_NAMES[c] for c in CONDITIONS])
        colors = {LABEL_NAMES[c]: CONDITION_COLORS[c] for c in CONDITIONS}
        figures[f"{band}_trajectories_2d"] = plot_trajectory(
            X=group[..., :2],
            times=data["times"],
            labels=labels_plot,
            sem=group_sem[..., :2],
            color_map=colors,
            title=f"{band}: equal-participant PC1-PC2 trajectories",
            dimensions=2,
            show_markers=True,
            add_start_end_markers=True,
        )
        figures[f"{band}_trajectories_3d"] = plot_trajectory(
            X=group[..., :3],
            times=data["times"],
            labels=labels_plot,
            color_map=colors,
            title=f"{band}: equal-participant PC1-PC2-PC3 trajectories",
            dimensions=3,
            show_markers=True,
            add_start_end_markers=True,
            height=650,
        )

        contrast_panels = {}
        for space, space_curves in contrasts.items():
            for contrast in ("Faces vs Scrambled", "Famous vs Unfamiliar"):
                curves = np.asarray(space_curves[contrast])
                color = CONTRAST_COLORS[contrast]
                participants = plot_trajectory_metric_series(
                    curves,
                    times=data["times"],
                    labels=np.array([f"sub-{subject}" for subject in subjects]),
                    color_map={f"sub-{subject}": color for subject in subjects},
                    title=f"{space}: {contrast}",
                    ylabel="Baseline-relative distance",
                )
                participants.update_traces(line={"width": 1}, showlegend=False)
                group = plot_timecourses(
                    curves[:, np.newaxis, :],
                    times=data["times"],
                    channel_names=[contrast],
                    group_labels=[contrast] * len(curves),
                    palette={contrast: color},
                    error_style="band",
                    xlabel="Time (s)",
                    ylabel="Baseline-relative distance",
                    title=f"{space}: {contrast}",
                )
                contrast_panels[f"{space}: {contrast}"] = overlay_figures(
                    [participants, group], opacities=[0.28, None]
                )
        fig_contrast = facet_figures(
            contrast_panels,
            n_cols=2,
            title=f"{band}: planned contrasts",
            row_height=360,
        )
        figures[f"{band}_planned_contrasts"] = fig_contrast

        speed_panels = {}
        for space, trajectories in spaces.items():
            curves, groups = [], []
            for condition in CONDITIONS:
                for subject in subjects:
                    mean = trajectories[
                        (data["subjects"] == subject) & (data["labels"] == condition)
                    ].mean(axis=0)
                    speed = np.linalg.norm(
                        np.gradient(mean, data["times"], axis=0), axis=1
                    )
                    curves.append(speed)
                    groups.append(LABEL_NAMES[condition])
                    speed_timeseries_rows.extend(
                        {
                            "band": band,
                            "pca_space": space,
                            "subject": subject,
                            "condition": LABEL_NAMES[condition],
                            "time_s": time,
                            "speed": value,
                        }
                        for time, value in zip(data["times"], speed, strict=True)
                    )
                    speed_rows.append(
                        {
                            "band": band,
                            "pca_space": space,
                            "subject": subject,
                            "condition": LABEL_NAMES[condition],
                            "mean_speed_0_600ms": speed[active].mean(),
                            "peak_speed_0_600ms": speed[active].max(),
                        }
                    )
            stacked = np.asarray(curves)
            participants = plot_trajectory_metric_series(
                stacked,
                times=data["times"],
                labels=np.array(groups),
                color_map={LABEL_NAMES[c]: CONDITION_COLORS[c] for c in CONDITIONS},
                title=space,
                ylabel="Speed (a.u./s)",
            )
            participants.update_traces(line={"width": 1}, showlegend=False)
            group = plot_timecourses(
                stacked[:, np.newaxis, :],
                times=data["times"],
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
            title=f"{band}: trajectory speed",
            row_height=340,
            shared_yaxes=False,
        )
        figures[f"{band}_trajectory_speed"] = fig_speed

    pca_diagnostics = pd.DataFrame(pca_rows)
    subject_variance = pd.DataFrame(subject_variance_rows)
    loadings = pd.DataFrame(loading_rows)
    metric_summary = pd.DataFrame(metric_rows)
    metric_group = metric_summary.groupby(["band", "pca_space", "contrast"], as_index=False).agg(
        mean_auc=("auc_0_600ms", "mean"),
        sem_auc=("auc_0_600ms", "sem"),
        mean_peak_time_s=("peak_time_s", "mean"),
    )
    metric_timeseries = pd.DataFrame(metric_timeseries_rows)
    speed_summary = pd.DataFrame(speed_rows)
    speed_timeseries = pd.DataFrame(speed_timeseries_rows)

    focused_results: dict[tuple[str, str], dict[str, object]] = {}
    focused_rows = []
    focused_timeseries_rows = []
    for band, data in band_data.items():
        for pair_name, pair in FOCUSED_PAIRS.items():
            keep = np.isin(data["labels"], pair)
            X_pair = data["X"][keep]
            labels_pair = data["labels"][keep]
            subjects_pair = data["subjects"][keep]
            pca = DimReduction(method="PCA", n_components=n_components, random_state=seed)
            scores = pca.fit_transform(
                X_pair.transpose(0, 2, 1).reshape(-1, X_pair.shape[1])
            ).reshape(len(X_pair), X_pair.shape[2], n_components)
            baseline = data["times"] < 0
            scores -= scores[:, baseline].mean(axis=1, keepdims=True)
            means = {
                (subject, condition): scores[
                    (subjects_pair == subject) & (labels_pair == condition)
                ].mean(axis=0)
                for subject in subjects
                for condition in pair
            }
            group = np.stack(
                [
                    np.stack([means[subject, condition] for subject in subjects]).mean(axis=0)
                    for condition in pair
                ]
            )
            sem = np.stack(
                [
                    np.stack([means[subject, condition] for subject in subjects]).std(
                        axis=0, ddof=1
                    )
                    / np.sqrt(len(subjects))
                    for condition in pair
                ]
            )
            separation = np.stack(
                [
                    np.linalg.norm(
                        means[subject, pair[0]][:, :3] - means[subject, pair[1]][:, :3], axis=1
                    )
                    for subject in subjects
                ]
            )
            separation -= separation[:, baseline].mean(axis=1, keepdims=True)
            focused_results[band, pair_name] = {
                "pca": pca,
                "scores": scores,
                "group": group,
                "sem": sem,
                "separation": separation,
                "pair": pair,
            }
            active = (data["times"] >= ACTIVE_WINDOW[0]) & (data["times"] <= ACTIVE_WINDOW[1])
            for subject, curve in zip(subjects, separation, strict=True):
                focused_timeseries_rows.extend(
                    {
                        "band": band,
                        "focused_space": pair_name,
                        "subject": subject,
                        "time_s": time,
                        "separation": value,
                    }
                    for time, value in zip(data["times"], curve, strict=True)
                )
                active_curve = curve[active]
                peak = int(np.argmax(active_curve))
                focused_rows.append(
                    {
                        "band": band,
                        "focused_space": pair_name,
                        "subject": subject,
                        "auc_0_600ms": np.trapezoid(active_curve, data["times"][active]),
                        "peak_separation": active_curve[peak],
                        "peak_time_s": data["times"][active][peak],
                    }
                )
            figures[
                f"{band}_focused_{pair_name.lower().replace(' vs ', '_').replace(' ', '_')}"
            ] = plot_trajectory(
                X=group[..., :2],
                times=data["times"],
                labels=np.array([LABEL_NAMES[c] for c in pair]),
                sem=sem[..., :2],
                color_map={LABEL_NAMES[c]: CONDITION_COLORS[c] for c in pair},
                title=f"{band} focused PCA: {pair_name}",
                dimensions=2,
                show_markers=True,
                add_start_end_markers=True,
            )
    focused_summary = pd.DataFrame(focused_rows)
    focused_group = focused_summary.groupby(["band", "focused_space"], as_index=False).agg(
        mean_auc=("auc_0_600ms", "mean"),
        sem_auc=("auc_0_600ms", "sem"),
        mean_peak_time_s=("peak_time_s", "mean"),
    )
    focused_timeseries = pd.DataFrame(focused_timeseries_rows)

    cross_band_rows = []
    for band, result in results.items():
        first_space = next(iter(result["contrast_curves"]))
        active = (result["times"] >= ACTIVE_WINDOW[0]) & (result["times"] <= ACTIVE_WINDOW[1])
        for contrast, curves in result["contrast_curves"][first_space].items():
            mean = curves.mean(axis=0)
            cross_band_rows.append(
                {
                    "band": band,
                    "descriptor": contrast,
                    "peak_time_s": result["times"][active][np.argmax(mean[active])],
                    "variance_3pc": result["evr"][:3].sum(),
                }
            )
        for index, condition in enumerate(CONDITIONS):
            trajectory = result["group"][index, :, :3]
            roughness = np.linalg.norm(np.diff(trajectory, n=2, axis=0), axis=1).sum() / max(
                np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum(), np.finfo(float).eps
            )
            cross_band_rows.append(
                {
                    "band": band,
                    "descriptor": f"{LABEL_NAMES[condition]} trajectory",
                    "normalised_roughness": roughness,
                }
            )
    cross_band_summary = pd.DataFrame(cross_band_rows)

    spectral_inference = pd.DataFrame()
    family_null = np.array([])
    if n_perm:
        observed = {}
        for band, result in results.items():
            active = (result["times"] >= ACTIVE_WINDOW[0]) & (result["times"] <= ACTIVE_WINDOW[1])
            curves = _contrast_curves(
                result["shared_scores"][..., :3],
                result["labels"],
                result["subjects"],
                subjects,
                result["times"],
            )
            for contrast, values in curves.items():
                observed[band, contrast] = np.trapezoid(
                    values.mean(axis=0)[active], result["times"][active]
                )
        nulls = []
        for _ in range(n_perm):
            null_values = []
            for band, result in results.items():
                shuffled = result["labels"].copy()
                for subject in subjects:
                    for repetition in np.unique(
                        result["repetitions"][result["subjects"] == subject]
                    ):
                        rows = np.flatnonzero(
                            (result["subjects"] == subject) & (result["repetitions"] == repetition)
                        )
                        shuffled[rows] = rng.permutation(shuffled[rows])
                active = (result["times"] >= ACTIVE_WINDOW[0]) & (
                    result["times"] <= ACTIVE_WINDOW[1]
                )
                curves = _contrast_curves(
                    result["shared_scores"][..., :3],
                    shuffled,
                    result["subjects"],
                    subjects,
                    result["times"],
                )
                null_values.extend(
                    np.trapezoid(values.mean(axis=0)[active], result["times"][active])
                    for values in curves.values()
                )
            nulls.append(max(null_values))
        family_null = np.asarray(nulls)
        spectral_inference = pd.DataFrame(
            [
                {
                    "band": band,
                    "contrast": contrast,
                    "observed_auc": value,
                    "p_family_corrected": (1 + np.sum(family_null >= value)) / (n_perm + 1),
                    "n_permutations": n_perm,
                }
                for (band, contrast), value in observed.items()
            ]
        )
        fig_null = go.Figure(go.Histogram(x=family_null, nbinsx=25, marker_color="#7B2CBF"))
        for value in observed.values():
            fig_null.add_vline(x=value, line_width=2, line_color="black")
        fig_null.update_layout(
            title="Maximum-statistic family null",
            xaxis_title="Maximum separation AUC",
            template="plotly_white",
        )
        figures["family_corrected_null"] = fig_null

    tables = {
        "filter_support": filter_support,
        "trial_counts": trial_counts,
        "pca_diagnostics": pca_diagnostics,
        "subject_pca_variance": subject_variance,
        "sensor_loadings": loadings,
        "planned_contrasts": metric_summary,
        "planned_contrasts_group": metric_group,
        "planned_contrast_timeseries": metric_timeseries,
        "speed_summary": speed_summary,
        "speed_timeseries": speed_timeseries,
        "focused_contrasts": focused_summary,
        "focused_contrasts_group": focused_group,
        "focused_contrast_timeseries": focused_timeseries,
        "cross_band_summary": cross_band_summary,
        "family_corrected_inference": spectral_inference,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    static_error = None
    for name, figure in figures.items():
        figure.write_html(figures_dir / f"{name}.html", include_plotlyjs="cdn")
        if static_error is None:
            try:
                figure.write_image(figures_dir / f"{name}.png", scale=2)
                figure.write_image(figures_dir / f"{name}.svg")
            except Exception as error:
                static_error = f"{type(error).__name__}: {error}"
                warnings.warn(
                    "Static Plotly export unavailable; HTML figures are complete.", stacklevel=2
                )
    arrays: dict[str, np.ndarray] = {"family_null": family_null}
    for band, result in results.items():
        result["shared_pca"].save(reducers_dir / f"{band}_shared_pca.pkl")
        for subject, reducer in result["subject_pcas"].items():
            reducer.save(reducers_dir / f"{band}_subject_{subject}_pca.pkl")
        for name in (
            "X",
            "times",
            "labels",
            "subjects",
            "repetitions",
            "shared_scores",
            "evr",
            "group",
            "group_sem",
        ):
            arrays[f"{band}_{name}"] = np.asarray(result[name])
        arrays[f"{band}_subject_scores"] = (
            np.asarray(result["subject_scores"])
            if result["subject_scores"] is not None
            else np.array([])
        )
    for (band, pair_name), result in focused_results.items():
        slug = pair_name.lower().replace(" vs ", "_").replace(" ", "_")
        result["pca"].save(reducers_dir / f"{band}_focused_{slug}_pca.pkl")
        for name in ("scores", "group", "sem", "separation"):
            arrays[f"{band}_focused_{slug}_{name}"] = np.asarray(result[name])
    np.savez_compressed(output / "spectral_analysis_arrays.npz", **arrays)

    report.add_summary_card(
        {
            "Participants": len(subjects),
            "Bands": len(SPECTRAL_BANDS),
            "Sensors": f"{len(source_channels)} ({sensor_set})",
            "PCA mode": str(metric_pca_mode),
            "Permutations": n_perm,
        }
    )

    def _stack(*elements: Element) -> ContainerElement:
        """Bundle several elements into one tab panel."""
        box = ContainerElement()
        for element in elements:
            box.add_element(element)
        return box

    overview = Section(
        "Overview",
        icon="O",
        description="Independent alpha, beta and low-gamma PCA analyses",
        metadata={
            "Participants": ", ".join(subjects),
            "Sensor set": f"{sensor_set} ({len(source_channels)} sensors)",
            "Bands": "8-12, 13-30, and 30-45 Hz",
            "Final window": "-0.2 to 0.8 s",
            "PCA mode": str(metric_pca_mode),
            "Family permutations": str(n_perm),
        },
    )
    overview.add_element(
        MarkdownElement(
            "This report mirrors `tutorial_megfaces_spectral_envelopes.ipynb`. "
            "Padded filtering and Hilbert power are followed by independent alpha, "
            "beta, and low-gamma PCA analyses."
        )
    )
    overview.add_element(
        CalloutElement(
            "The analysis describes noise-normalised sensor-space power, not "
            "physical sensor power or source-localized oscillations.",
            kind="warning",
            title="Scope",
        )
    )
    report.add_section(overview)
    sections = [
        (
            "Step 0 - Setup and Analysis Contract",
            "All choices are declared once. Long prepared epochs are reused unless "
            "preparation is explicitly requested.",
            [filter_support],
            [],
        ),
        (
            "Step 1 - Validate Padding",
            "Filtering occurs on -1.2 to 1.8 s epochs before the -0.2 to 0.8 s "
            "crop. Measured FIR support verifies that boundaries remain outside "
            "the analysis window.",
            [filter_support],
            [],
        ),
        (
            "Step 2 - Load Whitened Trials",
            "All retained trials are loaded with aligned channels and metadata. "
            "Participant-specific empty-room whitening places sensors on a common "
            "noise scale, and a 50-Hz notch suppresses line noise. Participants "
            "remain the group unit.",
            [trial_counts],
            [],
        ),
        (
            "Step 3 - Construct Band Power",
            "MNE band-pass and Hilbert amplitude are squared, smoothed over 40 ms, "
            "converted to dB relative to -0.2 to 0 s, cropped, and resampled. "
            "Smoothness is partly imposed by this representation.",
            [],
            [fig_transform],
        ),
        (
            "Step 4 - Inspect Sensor-Space Power",
            "The sensor average checks baseline, scale, and post-onset timing before "
            "multivariate reduction; it is quality control rather than inference.",
            [],
            [fig_sensor],
        ),
        (
            "Step 5 - Fit One Shared PCA per Band",
            "Every band receives an independent shared PCA. Axes and raw distances "
            "are comparable within a band only.",
            [pca_diagnostics],
            {band: figures[f"{band}_scree"] for band in SPECTRAL_BANDS},
        ),
        (
            "Step 6 - Fit Participant PCAs",
            "Participant PCAs test subspace sensitivity for rotation-invariant "
            "within-participant metrics. Their coordinates are never averaged.",
            [subject_variance],
            [],
        ),
        (
            "Step 7 - Inspect PCs and Loadings",
            "PC signs are arbitrary. Timing and geometry are interpretable; strong "
            "sensor loadings are not source localization.",
            [loadings],
            {band: figures[f"{band}_pc_timecourses"] for band in SPECTRAL_BANDS},
        ),
        (
            "Step 8 - Reconstruct Shared Trajectories",
            "Trials are averaged within participant and condition before "
            "equal-participant group means. Follow time within panels; do not "
            "equate axes across bands.",
            [],
            {
                band: _stack(
                    PlotlyElement(figures[f"{band}_trajectories_2d"], height="620px"),
                    PlotlyElement(figures[f"{band}_trajectories_3d"], height="680px"),
                )
                for band in SPECTRAL_BANDS
            },
        ),
        (
            "Step 9 - Planned Contrasts and Speed",
            "Famous-Unfamiliar and equal-weight Faces-Scrambled distances are "
            "baseline-relative and computed per participant. Speed is descriptive "
            "and scale-specific.",
            [metric_group, speed_summary],
            {
                band: _stack(
                    PlotlyElement(figures[f"{band}_planned_contrasts"], height="620px"),
                    PlotlyElement(figures[f"{band}_trajectory_speed"], height="620px"),
                )
                for band in SPECTRAL_BANDS
            },
        ),
        (
            "Step 10 - Focused Two-Condition PCAs",
            "Focused models can prioritize pair-relevant variance. Compare "
            "divergence timing, not axes or raw distances between fitted models.",
            [focused_group],
            {
                name.replace("_", " "): figure
                for name, figure in figures.items()
                if "_focused_" in name
            },
        ),
        (
            "Step 11 - Compare Bands Carefully",
            "Cross-band summaries use peak timing, variance structure, and normalized "
            "roughness. They do not put band-specific PCA spaces on one ruler.",
            [cross_band_summary],
            [],
        ),
        (
            "Step 12 - Family-Corrected Inference",
            "Labels are shuffled within participant and repetition. The maximum AUC "
            "across three bands and two contrasts controls the six-test family. "
            "This remains conditional on the displayed participants.",
            [spectral_inference],
            [figures["family_corrected_null"]] if n_perm else [],
        ),
    ]
    for index, (title, text_value, section_tables, section_figures) in enumerate(sections):
        section = Section(title, icon=str(index))
        section.add_element(MarkdownElement(text_value))
        for table in section_tables:
            if not table.empty:
                section.add_element(InteractiveTableElement(table, title="Results"))
        if isinstance(section_figures, dict):
            # Band-wise figure sets are tabbed: three bands stacked vertically
            # would bury the cross-band comparison the section is making.
            section.add_element(
                TabsElement(
                    {
                        name: figure
                        if isinstance(figure, Element)
                        else PlotlyElement(figure, height="680px")
                        for name, figure in section_figures.items()
                    }
                )
            )
        else:
            for figure in section_figures:
                section.add_element(PlotlyElement(figure, height="680px"))
        report.add_section(section)
    export = Section("Step 13 - Export and Interpretation", icon="13")
    export.add_element(
        MarkdownElement(
            "Every processed tensor, score array, reducer, time-resolved table, "
            "scalar summary, figure, null distribution, and manifest is saved. "
            "The HTML report embeds all JavaScript assets and opens offline.\n\n"
            "Alpha, beta, and low gamma answer separate representation-specific "
            "questions. Low gamma remains especially vulnerable to residual muscle "
            "and line-noise artifacts, and none of the sensor-space results "
            "localize a generator."
        )
    )
    appendix = AccordionElement("Appendix: all exported tables", open=False)
    for name, table in tables.items():
        if not table.empty:
            appendix.add_element(
                InteractiveTableElement(table, title=name.replace("_", " ").title())
            )
    export.add_element(appendix)
    report.add_section(export)
    report.save(output / "report.html")

    manifest = {
        "subjects": subjects,
        "raw_root": str(raw_root),
        "derivatives_root": str(derivatives_root),
        "sensor_set": sensor_set,
        "sensor_names": source_channels.tolist(),
        "bands": {name: list(values) for name, values in SPECTRAL_BANDS.items()},
        "final_window": list(FINAL_WINDOW),
        "baseline_window": list(BASELINE_WINDOW),
        "active_window": list(ACTIVE_WINDOW),
        "envelope_sfreq": ENVELOPE_SFREQ,
        "smoothing_s": SMOOTHING_S,
        "metric_pca_mode": metric_pca_mode,
        "n_components": n_components,
        "n_permutations": n_perm,
        "random_state": seed,
        "report_asset_mode": report.asset_mode,
        "static_figure_exports_complete": static_error is None,
        "static_figure_export_error": static_error,
    }
    write_manifest(output / "analysis_manifest.json", manifest, status="complete")
    print(f"Saved MEG Faces spectral analysis -> {output}")
    return {"manifest": manifest, "tables": tables, "figures": figures, "results": results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=["01", "02", "03"])
    parser.add_argument("--raw-root", type=Path, default=Path.home() / "mne_data" / "ds000117")
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=Path.home() / "mne_data" / "ds000117" / "derivatives" / "pca_trajectories_power",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/megfaces_spectral_envelopes"))
    parser.add_argument(
        "--sensor-set",
        choices=tuple(MEG_SENSOR_SETS),
        default="all_sensors",
    )
    parser.add_argument("--metric-pca-mode", choices=("shared", "subject", "both"), default="both")
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument(
        "--n-perm", type=int, default=200, help="Use 0 to skip the optional family test."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create missing long derivatives from local raw data.",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Use two participants and at most five permutations."
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
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
        result = run_spectral_analysis(
            subjects=subjects,
            raw_root=args.raw_root,
            derivatives_root=args.derivatives_root,
            output=args.output,
            metric_pca_mode=args.metric_pca_mode,
            n_components=args.n_components,
            n_perm=n_perm,
            seed=args.seed,
            prepare=args.prepare,
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
        manifest_path, settings, status="complete", extra={"analysis": result["manifest"]}
    )


if __name__ == "__main__":
    main()
