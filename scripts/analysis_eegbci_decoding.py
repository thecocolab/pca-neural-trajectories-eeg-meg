"""Run and save the complete cross-participant EEGBCI decoding analysis.

This is the headless companion to
``tutorials/tutorial_eegbci_decoding.ipynb``. It mirrors the notebook's
data selection, leave-one-subject-out splits, fold-local scaling, sliding
logistic regression, transductive temporal alignment, summaries, figures, and
structured ten-step report.

Examples
--------
python scripts/analysis_eegbci_decoding.py
python scripts/analysis_eegbci_decoding.py --subjects 1 2 3 --n-jobs 1
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from coco_pipe.decoding import (
    CVConfig,
    Experiment,
    ExperimentConfig,
    TemporalAlignmentConfig,
    TemporalDecoderConfig,
)
from coco_pipe.decoding.configs import ClassicalModelConfig
from coco_pipe.report import PlotlyElement, Report, Section
from coco_pipe.report.elements import (
    AccordionElement,
    InteractiveTableElement,
    MarkdownElement,
)
from coco_pipe.viz.interactive.decoding import plot_temporal_score_curve
from coco_pipe.viz.theme import set_coco_theme
from plotly.subplots import make_subplots

from pca_neural_trajectories import (
    LABEL_NAMES,
    load_eegbci_container,
    setup_data_bids,
    write_manifest,
)

SEED = 42
CONDITIONS = (3, 4)
ANALYSIS_WINDOW = (-0.2, 1.0)
CHANCE_LEVEL = 0.5
EXCLUDED_SUBJECTS = {88, 92, 100}
REPRESENTATION_COLORS = {
    "Sensors": "#1b9e77",
    "Aligned PCA": "#2a78d6",
}


def build_decoding_report(
    output: Path,
    *,
    context: dict[str, object],
    tables: dict[str, pd.DataFrame],
    figures: dict[str, go.Figure],
) -> str:
    """Build the notebook-equivalent structured decoding report."""
    title = (
        "EEGBCI — Cross-Participant Temporal Decoding "
        f"({context['n_subjects']} participants)"
    )
    try:
        report = Report(title=title, asset_urls="inline")
        asset_mode = "inline"
    except OSError:
        warnings.warn(
            "Inline report assets are not cached and could not be downloaded; "
            "falling back to CDN-linked report assets.",
            stacklevel=2,
        )
        report = Report(title=title)
        asset_mode = "cdn"

    overview = Section("Overview", icon="O")
    overview.add_element(
        MarkdownElement(
            "This report follows the complete **10-step cross-participant EEGBCI "
            "decoding workflow** from the tutorial notebook. It asks whether "
            "left- versus right-hand execution can be decoded in a participant who "
            "contributed no labeled trials to model training.\n\n"
            "| Parameter | Value |\n"
            "|---|---|\n"
            f"| Participants | {context['n_subjects']} |\n"
            f"| Trials | {context['n_trials']} |\n"
            f"| Channels × time samples | {context['n_channels']} × "
            f"{context['n_times']} |\n"
            f"| Analysis window | {context['time_start']:.3f}–"
            f"{context['time_stop']:.3f} s |\n"
            "| Target | Left-hand versus right-hand execution |\n"
            "| Cross-validation | Leave one participant out |\n"
            "| Classifier | Sliding logistic regression |\n"
            "| Metric | Balanced accuracy |\n"
            f"| Chance level | {context['chance_level']:.2f} |\n"
            f"| Alignment components | {context['n_components']} |\n\n"
            "> The two representations use identical trials, folds, classifier, "
            "scaling, metric, and time axis. Only fold-local temporal alignment differs."
        )
    )
    report.add_section(overview)

    step1 = Section("Step 1 — Define the Predictive Question", icon="1")
    step1.add_element(
        MarkdownElement(
            "The predictive question is deliberately narrower than visual trajectory "
            "separation: **can a classifier trained on other participants identify "
            "left- versus right-hand execution in one entirely held-out participant?**\n\n"
            "A trial is an observation, but the participant is the inferential unit. "
            "Repeated trials from one participant are correlated and cannot be split "
            "independently across training and test sets.\n\n"
            "> **Generalization target:** Performance estimates apply to new "
            "participants drawn under a similar acquisition and preprocessing protocol, "
            "not merely to unseen trials from participants already represented in training."
        )
    )
    report.add_section(step1)

    step2 = Section("Step 2 — Load the Sensor-Time Representation", icon="2")
    step2.add_element(
        MarkdownElement(
            "The analysis loads preprocessed EEGBCI epochs directly as "
            "`(trial, channel, time)`. No local binning helper is required because "
            "`coco-pipe` accepts the native three-dimensional temporal array.\n\n"
            "The crop and baseline match the notebook: −0.2–1.0 s around the movement "
            "cue, with the pre-cue −0.2–0.0 s interval used as the sensor baseline. "
            "Labels 3 and 4 correspond to left- and right-hand execution."
        )
    )
    step2.add_element(
        InteractiveTableElement(
            tables["trial_counts"],
            title="Trials available per participant and class",
            selector_columns=["subject", "condition_name"],
        )
    )
    report.add_section(step2)

    step3 = Section("Step 3 — Verify the Target and Class Balance", icon="3")
    step3.add_element(
        MarkdownElement(
            "The target is encoded as `0 = left hand` and `1 = right hand`. Balanced "
            "accuracy is used because it averages sensitivity across classes and is "
            "therefore robust to modest trial-count differences. The logistic "
            "regression also receives `class_weight='balanced'`, fitted using the "
            "training fold only.\n\n"
            "> Class balance is documented rather than repaired by resampling. No test "
            "trial is duplicated, removed, or used to determine a training-fold weight."
        )
    )
    report.add_section(step3)

    step4 = Section("Step 4 — Declare Leave-One-Participant-Out CV", icon="4")
    step4.add_element(
        MarkdownElement(
            "Each outer fold holds out every trial from one participant and trains on "
            "all remaining participants. Scaling, alignment, and classifier fitting "
            "occur independently inside that fold.\n\n"
            "The audit table below is derived from `ExperimentResult.get_splits()`. "
            "A valid run has one held-out participant per fold and an empty train/test "
            "participant intersection in every row."
        )
    )
    step4.add_element(
        InteractiveTableElement(
            tables["split_audit"],
            title="Outer-fold participant separation audit",
            selector_columns=["held_out_subject"],
        )
    )
    report.add_section(step4)

    step5 = Section("Step 5 — Configure Fold-Local Temporal Decoding", icon="5")
    step5.add_element(
        MarkdownElement(
            "At every retained latency, a logistic-regression classifier is fitted to "
            "the training participants and evaluated on the held-out participant. "
            "`wrapper='sliding'` repeats this independently across time.\n\n"
            "Standardization is owned by `coco-pipe` and fitted inside each training "
            "fold. The test participant is transformed with training-fold scaling "
            "parameters. This prevents the global-standardization leakage that would "
            "occur if all participants were scaled before LOSO splitting.\n\n"
            "> **What the curve means:** Each time point is a separate predictive "
            "model. A peak is descriptive and is not automatically a corrected "
            "significance result across the many tested latencies."
        )
    )
    report.add_section(step5)

    step6 = Section("Step 6 — Compare Sensors with Aligned PCA", icon="6")
    step6.add_element(
        MarkdownElement(
            "The **Sensors** experiment decodes directly from channels. The "
            "**Aligned PCA** experiment enables `TemporalProcrustesAlignment` within "
            "each outer fold while leaving every other setting unchanged.\n\n"
            "For alignment, the shared PCA reference and temporal template are learned "
            "from training participants only. The unseen participant's unlabeled trials "
            "are used to estimate that participant's PCA and orthogonal rotation into "
            "the training reference. Labels from the held-out participant are never "
            "used during adaptation.\n\n"
            "> **Transductive scope:** Aligned performance answers what is possible "
            "after collecting an unlabeled calibration batch from the new participant. "
            "It is not zero-calibration inductive decoding and should not be presented "
            "as such."
        )
    )
    report.add_section(step6)

    step7 = Section("Step 7 — Run and Audit Every Fold", icon="7")
    step7.add_element(
        MarkdownElement(
            "Both experiments receive the same native epochs, binary labels, "
            "participant groups, trial identifiers, and scientific time axis. The raw "
            "`ExperimentResult` for each representation is preserved, including "
            "predictions, splits, fold scores, and fit diagnostics.\n\n"
            "Operational warnings or unusually long fold times can reveal convergence "
            "or configuration problems that are invisible in the mean accuracy curve."
        )
    )
    diagnostics_appendix = AccordionElement(
        "Details: fold fit and runtime diagnostics",
        open=False,
    )
    diagnostics_appendix.add_element(
        InteractiveTableElement(
            tables["fit_diagnostics"],
            title="Fold-level fit diagnostics",
            selector_columns=["Representation", "Fold"],
        )
    )
    step7.add_element(diagnostics_appendix)
    report.add_section(step7)

    step8 = Section("Step 8 — Inspect Time-Resolved Generalization", icon="8")
    step8.add_element(
        MarkdownElement(
            "The curve is the mean balanced accuracy across held-out participants; "
            "the ribbon is the fold-level standard deviation returned by `coco-pipe`. "
            "The horizontal line marks binary chance (0.5), and time zero marks the "
            "movement cue.\n\n"
            "Peak latency and accuracy summarize the plotted curve but do not provide "
            "a multiple-comparison-corrected inferential test. Broad, sustained "
            "above-chance periods are generally more stable to sampling noise than a "
            "single isolated maximum."
        )
    )
    step8.add_element(PlotlyElement(figures["temporal_decoding"], height="600px"))
    step8.add_element(
        InteractiveTableElement(
            tables["peak_summary"],
            title="Descriptive peak of the across-participant curve",
            selector_columns=["Representation"],
        )
    )
    report.add_section(step8)

    step9 = Section("Step 9 — Examine Held-Out-Participant Variability", icon="9")
    step9.add_element(
        MarkdownElement(
            "A mean curve can hide participants with very different decoding profiles. "
            "The heatmaps retain one row per LOSO fold, labelled by the held-out "
            "participant. The companion distribution plot summarizes two predeclared "
            "post-cue quantities per fold:\n\n"
            "- **Post-cue mean balanced accuracy:** average performance from 0–1 s.\n"
            "- **AUC above chance:** temporal integral of `(balanced accuracy − 0.5)` "
            "over the same interval.\n\n"
            "Aligned-minus-sensor differences are paired by outer fold. They are "
            "descriptive sensitivity summaries, not significance tests."
        )
    )
    step9.add_element(PlotlyElement(figures["fold_heatmaps"], height="720px"))
    step9.add_element(PlotlyElement(figures["fold_summaries"], height="520px"))
    step9.add_element(
        InteractiveTableElement(
            tables["representation_summary"],
            title="Across-fold descriptive summaries",
            selector_columns=["Representation"],
        )
    )
    step9.add_element(
        InteractiveTableElement(
            tables["paired_fold_differences"],
            title="Aligned PCA minus Sensors within each held-out participant",
            selector_columns=["held_out_subject"],
        )
    )
    report.add_section(step9)

    step10 = Section("Step 10 — Export, Reproduce & Interpret", icon="10")
    static_status = (
        "PNG and SVG exports completed."
        if context["static_exports_complete"]
        else "Static export was unavailable; interactive HTML figures are complete."
    )
    step10.add_element(
        MarkdownElement(
            "The output bundle preserves the summary tables, fold-level scores, split "
            "audit, fit diagnostics, raw `ExperimentResult` objects and tidy exports, "
            "figures, configuration/provenance manifest, and this report.\n\n"
            f"**Static export status:** {static_status}\n\n"
            "Interpretation remains bounded by four points:\n\n"
            "1. LOSO supports cross-participant rather than within-participant claims.\n"
            "2. Balanced accuracy peaks are descriptive without temporal multiplicity "
            "correction.\n"
            "3. Sensor and aligned spaces answer different calibration questions.\n"
            "4. Transductive alignment uses unlabeled held-out-participant data and must "
            "not be described as zero-calibration decoding.\n\n"
            "> **Main takeaway:** Compare the two representations through paired "
            "held-out-participant behavior and sustained temporal performance, not "
            "through a single maximum of the group-mean curve."
        )
    )
    appendix = AccordionElement("Appendix: all exported summary tables", open=False)
    for name, table in tables.items():
        appendix.add_element(
            InteractiveTableElement(table, title=name.replace("_", " ").title())
        )
    step10.add_element(appendix)
    report.add_section(step10)

    report.save(str(output / "report.html"))
    return asset_mode


def run_decoding_analysis(
    *,
    subjects_requested: list[int],
    bids_root: Path,
    output: Path,
    n_components: int = 30,
    n_jobs: int = -1,
    seed: int = SEED,
    prepare: bool = False,
) -> dict[str, object]:
    """Run the notebook-equivalent decoding analysis and save every output."""
    if len(subjects_requested) < 2:
        raise ValueError("Cross-participant decoding requires at least two subjects.")
    invalid = sorted(set(subjects_requested) & EXCLUDED_SUBJECTS)
    if invalid:
        raise ValueError(f"Subjects with incompatible sampling rates requested: {invalid}.")
    if n_components < 1:
        raise ValueError("n_components must be positive.")

    set_coco_theme(mode="paper", colorblind=True)
    output = Path(output)
    figures_dir = output / "figures"
    results_dir = output / "experiment_results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if prepare:
        setup_data_bids(
            subjects=subjects_requested,
            runs=list(range(3, 15)),
            root=bids_root,
        )
    if not bids_root.exists():
        raise FileNotFoundError(
            f"EEGBCI BIDS data were not found at {bids_root}. "
            "Run with --prepare or complete the introductory tutorial first."
        )

    requested_ids = tuple(f"{subject:03d}" for subject in subjects_requested)
    container = load_eegbci_container(
        bids_root,
        subjects=requested_ids,
        runs=tuple(range(3, 15)),
        conditions=CONDITIONS,
        tmin=ANALYSIS_WINDOW[0],
        tmax=ANALYSIS_WINDOW[1],
        baseline=(-0.2, 0.0),
    )
    X = np.asarray(container.X, dtype=np.float32)
    times = np.asarray(container.coords["time"], dtype=float)
    condition = np.asarray(container.y, dtype=int)
    subject_ids = np.asarray(container.coords["subject"]).astype(str)
    trial_ids = np.asarray(container.ids).astype(str)
    y = np.where(condition == CONDITIONS[0], 0, 1)

    if n_components > X.shape[1]:
        raise ValueError(
            f"n_components={n_components} exceeds the {X.shape[1]} input channels."
        )
    analyzed_subjects = sorted(np.unique(subject_ids).tolist())
    if len(analyzed_subjects) < 2:
        raise RuntimeError("Fewer than two participants were loaded successfully.")

    trial_counts = (
        pd.DataFrame(
            {
                "subject": subject_ids,
                "condition": condition,
                "condition_name": [LABEL_NAMES[value] for value in condition],
            }
        )
        .groupby(["subject", "condition", "condition_name"], as_index=False)
        .size()
        .rename(columns={"size": "n_trials"})
    )

    decoder = TemporalDecoderConfig(
        wrapper="sliding",
        base=ClassicalModelConfig(
            estimator="LogisticRegression",
            params={"class_weight": "balanced", "max_iter": 2000},
        ),
        n_jobs=1,
        verbose=False,
    )
    sensor_config = ExperimentConfig(
        task="classification",
        models={"Logistic regression": decoder},
        metrics=["balanced_accuracy"],
        cv=CVConfig(strategy="leave_one_group_out", shuffle=False),
        use_scaler=True,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
    )
    aligned_config = sensor_config.model_copy(deep=True)
    aligned_config.temporal_alignment = TemporalAlignmentConfig(
        enabled=True,
        n_components=n_components,
        adaptation="transductive",
    )
    experiments = {
        "Sensors": sensor_config,
        "Aligned PCA": aligned_config,
    }

    results = {}
    temporal_frames = []
    fold_frames = []
    diagnostic_frames = []
    for representation, config in experiments.items():
        result = Experiment(config).run(
            X,
            y,
            groups=subject_ids,
            sample_ids=trial_ids,
            observation_level="epoch",
            inferential_unit="subject",
            time_axis=times,
        )
        results[representation] = result

        temporal = result.get_temporal_score_summary()
        temporal["Estimator"] = temporal["Model"]
        temporal["Representation"] = representation
        temporal["Model"] = representation
        temporal_frames.append(temporal)

        folds = result.get_detailed_scores()
        folds["Estimator"] = folds["Model"]
        folds["Representation"] = representation
        fold_frames.append(folds)

        diagnostics = result.get_fit_diagnostics()
        diagnostics["Estimator"] = diagnostics["Model"]
        diagnostics["Representation"] = representation
        diagnostic_frames.append(diagnostics)

        result.export(
            results_dir / representation.lower().replace(" ", "_"),
            config=config.model_dump(),
            formats=("csv",),
        )

    temporal_scores = pd.concat(temporal_frames, ignore_index=True)
    fold_scores = pd.concat(fold_frames, ignore_index=True)
    fit_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)

    split_rows = results["Sensors"].get_splits()
    audit_records = []
    for fold in sorted(split_rows["Fold"].unique()):
        fold_rows = split_rows[split_rows["Fold"] == fold]
        train_rows = fold_rows[fold_rows["Set"] == "train"]
        test_rows = fold_rows[fold_rows["Set"] == "test"]
        train_subjects = sorted(train_rows["Group"].astype(str).unique())
        test_subjects = sorted(test_rows["Group"].astype(str).unique())
        overlap = sorted(set(train_subjects) & set(test_subjects))
        audit_records.append(
            {
                "Fold": int(fold),
                "held_out_subject": ", ".join(test_subjects),
                "n_train_subjects": len(train_subjects),
                "n_test_subjects": len(test_subjects),
                "n_train_trials": len(train_rows),
                "n_test_trials": len(test_rows),
                "subject_overlap": ", ".join(overlap),
                "leakage_free": len(overlap) == 0 and len(test_subjects) == 1,
            }
        )
    split_audit = pd.DataFrame(audit_records)
    if not split_audit["leakage_free"].all():
        raise RuntimeError("The outer-fold audit found participant overlap.")

    peak_summary = temporal_scores.loc[
        temporal_scores.groupby("Representation")["Mean"].idxmax(),
        ["Representation", "Time", "Mean", "Std"],
    ].sort_values("Mean", ascending=False)
    peak_summary = peak_summary.rename(
        columns={
            "Time": "peak_time_s",
            "Mean": "peak_balanced_accuracy",
            "Std": "peak_fold_std",
        }
    )

    fold_summary_records = []
    metric_rows = fold_scores[
        (fold_scores["Metric"] == "balanced_accuracy")
        & fold_scores["Time"].notna()
    ]
    held_out_by_fold = split_audit.set_index("Fold")["held_out_subject"].to_dict()
    for (representation, fold), rows in metric_rows.groupby(
        ["Representation", "Fold"]
    ):
        rows = rows.sort_values("Time")
        active = rows["Time"] >= 0
        active_rows = rows[active]
        peak_index = rows["Value"].idxmax()
        fold_summary_records.append(
            {
                "Representation": representation,
                "Fold": int(fold),
                "held_out_subject": held_out_by_fold[int(fold)],
                "peak_time_s": float(rows.loc[peak_index, "Time"]),
                "peak_balanced_accuracy": float(rows.loc[peak_index, "Value"]),
                "postcue_mean_balanced_accuracy": float(active_rows["Value"].mean()),
                "postcue_auc_above_chance": float(
                    np.trapezoid(
                        active_rows["Value"] - CHANCE_LEVEL,
                        active_rows["Time"],
                    )
                ),
            }
        )
    fold_summary = pd.DataFrame(fold_summary_records)
    representation_summary = (
        fold_summary.groupby("Representation")
        .agg(
            n_folds=("Fold", "nunique"),
            peak_ba_mean=("peak_balanced_accuracy", "mean"),
            peak_ba_std=("peak_balanced_accuracy", "std"),
            postcue_ba_mean=("postcue_mean_balanced_accuracy", "mean"),
            postcue_ba_std=("postcue_mean_balanced_accuracy", "std"),
            postcue_auc_mean=("postcue_auc_above_chance", "mean"),
            postcue_auc_std=("postcue_auc_above_chance", "std"),
        )
        .reset_index()
    )

    paired = fold_summary.pivot(
        index=["Fold", "held_out_subject"],
        columns="Representation",
        values=[
            "peak_balanced_accuracy",
            "postcue_mean_balanced_accuracy",
            "postcue_auc_above_chance",
        ],
    )
    paired_fold_differences = paired.index.to_frame(index=False)
    for metric in (
        "peak_balanced_accuracy",
        "postcue_mean_balanced_accuracy",
        "postcue_auc_above_chance",
    ):
        paired_fold_differences[f"{metric}_aligned_minus_sensors"] = (
            paired[(metric, "Aligned PCA")].to_numpy()
            - paired[(metric, "Sensors")].to_numpy()
        )

    temporal_figure = plot_temporal_score_curve(
        temporal_scores,
        metric="balanced_accuracy",
        title="Left- versus right-hand execution: LOSO decoding",
        colors=REPRESENTATION_COLORS,
    )
    temporal_figure.add_hline(
        y=CHANCE_LEVEL,
        line_dash="dot",
        line_color="#777777",
    )
    temporal_figure.add_vline(x=0, line_color="#999999")
    temporal_figure.update_yaxes(title_text="balanced accuracy")
    temporal_figure.update_xaxes(title_text="time from movement cue (s)")

    fold_heatmaps = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Sensors", "Aligned PCA"),
        horizontal_spacing=0.12,
    )
    for column, representation in enumerate(("Sensors", "Aligned PCA"), start=1):
        rows = metric_rows[metric_rows["Representation"] == representation]
        matrix = rows.pivot(index="Fold", columns="Time", values="Value").sort_index()
        fold_labels = [
            f"sub-{held_out_by_fold[int(fold)]}" for fold in matrix.index
        ]
        fold_heatmaps.add_trace(
            go.Heatmap(
                z=matrix.to_numpy(),
                x=matrix.columns.to_numpy(dtype=float),
                y=fold_labels,
                zmin=0,
                zmax=1,
                colorscale="Viridis",
                colorbar={"title": "BA"} if column == 2 else None,
                showscale=column == 2,
            ),
            row=1,
            col=column,
        )
    fold_heatmaps.update_xaxes(title_text="time from movement cue (s)")
    fold_heatmaps.update_yaxes(title_text="held-out participant", row=1, col=1)
    fold_heatmaps.update_layout(
        title="Balanced accuracy for every held-out participant",
        height=max(520, 32 * len(analyzed_subjects) + 250),
        width=1050,
    )

    fold_summary_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Post-cue mean balanced accuracy", "Post-cue AUC above chance"),
    )
    for column, metric in enumerate(
        ("postcue_mean_balanced_accuracy", "postcue_auc_above_chance"),
        start=1,
    ):
        for representation in ("Sensors", "Aligned PCA"):
            rows = fold_summary[fold_summary["Representation"] == representation]
            fold_summary_figure.add_trace(
                go.Box(
                    x=[representation] * len(rows),
                    y=rows[metric],
                    name=representation,
                    marker_color=REPRESENTATION_COLORS[representation],
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    showlegend=column == 1,
                    legendgroup=representation,
                ),
                row=1,
                col=column,
            )
    fold_summary_figure.add_hline(
        y=CHANCE_LEVEL,
        line_dash="dot",
        line_color="#777777",
        row=1,
        col=1,
    )
    fold_summary_figure.add_hline(
        y=0,
        line_dash="dot",
        line_color="#777777",
        row=1,
        col=2,
    )
    fold_summary_figure.update_layout(
        title="Held-out-participant post-cue summaries",
        height=500,
        width=1000,
    )

    tables = {
        "trial_counts": trial_counts,
        "split_audit": split_audit,
        "temporal_scores": temporal_scores,
        "fold_scores": fold_scores,
        "peak_summary": peak_summary,
        "fold_summary": fold_summary,
        "representation_summary": representation_summary,
        "paired_fold_differences": paired_fold_differences,
        "fit_diagnostics": fit_diagnostics,
    }
    figures = {
        "temporal_decoding": temporal_figure,
        "fold_heatmaps": fold_heatmaps,
        "fold_summaries": fold_summary_figure,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    static_export_error = None
    for name, figure in figures.items():
        figure.write_html(figures_dir / f"{name}.html", include_plotlyjs="cdn")
        if static_export_error is None:
            try:
                figure.write_image(figures_dir / f"{name}.png", scale=2)
                figure.write_image(figures_dir / f"{name}.svg")
            except Exception as error:  # Plotly delegates to an external browser.
                static_export_error = f"{type(error).__name__}: {error}"
                warnings.warn(
                    "Static Plotly export is unavailable; continuing with HTML "
                    "figures. Check the Kaleido browser installation.",
                    stacklevel=2,
                )

    np.savez_compressed(
        output / "analysis_metadata.npz",
        times=times,
        condition=condition,
        target=y,
        subjects=subject_ids,
        trial_ids=trial_ids,
    )
    context = {
        "n_subjects": len(analyzed_subjects),
        "n_trials": len(X),
        "n_channels": X.shape[1],
        "n_times": X.shape[2],
        "time_start": float(times[0]),
        "time_stop": float(times[-1]),
        "chance_level": CHANCE_LEVEL,
        "n_components": n_components,
        "static_exports_complete": static_export_error is None,
    }
    report_asset_mode = build_decoding_report(
        output,
        context=context,
        tables=tables,
        figures=figures,
    )

    manifest = {
        "subjects_requested": subjects_requested,
        "subjects_analyzed": analyzed_subjects,
        "bids_root": str(bids_root),
        "conditions": list(CONDITIONS),
        "analysis_window": list(ANALYSIS_WINDOW),
        "target_mapping": {"0": LABEL_NAMES[3], "1": LABEL_NAMES[4]},
        "shape": list(X.shape),
        "cv": "leave_one_group_out",
        "metric": "balanced_accuracy",
        "chance_level": CHANCE_LEVEL,
        "n_components": n_components,
        "alignment_adaptation": "transductive",
        "random_state": seed,
        "n_jobs": n_jobs,
        "static_figure_exports_complete": static_export_error is None,
        "static_figure_export_error": static_export_error,
        "report_asset_mode": report_asset_mode,
    }
    write_manifest(
        output / "analysis_manifest.json",
        manifest,
        status="complete",
    )

    print(f"Saved EEGBCI decoding analysis → {output}")
    return {
        "manifest": manifest,
        "tables": tables,
        "figures": figures,
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", type=int)
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--bids-root", type=Path, default=Path("PhysioNet_EEGBCI/BIDS"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tutorial_eegbci_decoding"),
    )
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Explicitly download/preprocess the requested EEGBCI participants.",
    )
    args = parser.parse_args(argv)

    if args.subjects:
        subjects_requested = args.subjects
    else:
        available = [
            subject
            for subject in range(1, 110)
            if subject not in EXCLUDED_SUBJECTS
        ]
        subjects_requested = available[: args.n_subjects]

    run_decoding_analysis(
        subjects_requested=subjects_requested,
        bids_root=args.bids_root,
        output=args.output,
        n_components=args.n_components,
        n_jobs=args.n_jobs,
        seed=args.seed,
        prepare=args.prepare,
    )


if __name__ == "__main__":
    main()
