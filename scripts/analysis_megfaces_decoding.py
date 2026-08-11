"""Run and save the complete cross-participant MEG Faces decoding analysis.

This is the headless companion to ``tutorials/tutorial_megfaces_decoding.ipynb``.
It mirrors the notebook's Famous-versus-Scrambled target, leave-one-participant-
out folds, fold-local scaling, sliding logistic regression, transductive temporal
alignment, fold audits, held-out-participant summaries, and interpretation.

The output contains the raw ``ExperimentResult`` exports, tidy tables, figures,
analysis arrays, provenance manifests, and a fully self-contained HTML report.
"""

from __future__ import annotations

import argparse
import json
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
from plotly.subplots import make_subplots

from pca_neural_trajectories import write_manifest
from pca_neural_trajectories.wakeman_henson import (
    LABEL_NAMES,
    MEG_SENSOR_SETS,
    _load_wakeman_henson_container,
)

SEED = 42
CONDITIONS = (1, 3)
CHANCE_LEVEL = 0.5
REPRESENTATION_COLORS = {"Sensors": "#1b9e77", "Aligned PCA": "#2a78d6"}


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _build_report(
    output: Path,
    *,
    context: dict[str, object],
    tables: dict[str, pd.DataFrame],
    figures: dict[str, go.Figure],
) -> str:
    report = Report(
        title=(
            "MEG Faces - Cross-Participant Temporal Decoding "
            f"- {context['sensor_set']} ({context['n_subjects']} participants)"
        ),
        asset_urls="inline",
    )

    overview = Section("Overview", icon="O")
    overview.add_element(
        MarkdownElement(
            "This report follows the complete MEG Faces decoding workflow. It asks "
            "whether Famous and Scrambled images can be distinguished in a participant "
            "whose labeled trials were excluded from training.\n\n"
            "| Parameter | Value |\n"
            "|---|---|\n"
            f"| Participants | {context['n_subjects']} |\n"
            f"| Trials | {context['n_trials']} |\n"
            f"| Sensor set | {context['sensor_set']} |\n"
            f"| Sensors x time samples | {context['n_channels']} x "
            f"{context['n_times']} |\n"
            f"| Analysis window | {context['time_start']:.3f} to "
            f"{context['time_stop']:.3f} s |\n"
            "| Target | Famous versus Scrambled |\n"
            "| Cross-validation | Leave one participant out |\n"
            "| Classifier | Sliding logistic regression |\n"
            "| Metric | Balanced accuracy |\n"
            f"| Alignment components | {context['n_components']} |\n\n"
            "> Both experiments use identical trials, folds, scaling, classifier, "
            "metric, and time axis. Only fold-local temporal alignment differs."
        )
    )
    report.add_section(overview)

    step1 = Section("Step 1 - Define the Predictive Question", icon="1")
    step1.add_element(
        MarkdownElement(
            "The prediction target is image category: `0 = Famous` and "
            "`1 = Scrambled`. A trial is an observation, while the participant is "
            "the cross-validation and inferential unit. Keeping every participant "
            "entirely within train or test prevents identity-specific sensor patterns "
            "from leaking across the split.\n\n"
            "> LOSO evaluates generalization to participants acquired under a similar "
            "protocol, not merely to new trials from known participants."
        )
    )
    report.add_section(step1)

    step2 = Section("Step 2 - Load the Whitened Sensor-Time Data", icon="2")
    step2.add_element(
        MarkdownElement(
            "Prepared epochs are loaded as `trial x sensor x time`. Each participant "
            "has already been whitened with their empty-room covariance, placing "
            "magnetometers and gradiometers on a common noise scale. No local time "
            "binning or global standardization is applied before `coco-pipe`."
        )
    )
    step2.add_element(
        InteractiveTableElement(
            tables["trial_counts"],
            title="Trials per participant and class",
            selector_columns=["subject", "condition_name"],
        )
    )
    report.add_section(step2)

    step3 = Section("Step 3 - Verify the Target and Class Counts", icon="3")
    step3.add_element(
        MarkdownElement(
            "Balanced accuracy averages class-specific recall, and logistic regression "
            "uses `class_weight='balanced'`. Those weights are estimated within each "
            "training fold. Trial counts are documented rather than altered by "
            "resampling."
        )
    )
    report.add_section(step3)

    step4 = Section("Step 4 - Declare and Audit LOSO", icon="4")
    step4.add_element(
        MarkdownElement(
            "Each fold holds out all trials from one participant. The split audit is "
            "derived from the saved `ExperimentResult`; a valid fold contains one test "
            "participant and no participant overlap between train and test."
        )
    )
    step4.add_element(
        InteractiveTableElement(
            tables["split_audit"],
            title="Outer-fold participant separation",
            selector_columns=["held_out_subject"],
        )
    )
    report.add_section(step4)

    step5 = Section("Step 5 - Configure Sliding Temporal Decoding", icon="5")
    step5.add_element(
        MarkdownElement(
            "At every latency, logistic regression is trained on the retained sensor "
            "features from the training participants and evaluated on the held-out "
            "participant. `coco-pipe` fits standardization independently inside every "
            "fold and repeats the classifier across the supplied scientific time axis.\n\n"
            "> Each latency is a separate model. A curve maximum is descriptive, not "
            "a multiple-comparison-corrected temporal significance test."
        )
    )
    report.add_section(step5)

    step6 = Section("Step 6 - Compare Sensors with Aligned PCA", icon="6")
    step6.add_element(
        MarkdownElement(
            "The Sensors experiment decodes directly from whitened channels. The "
            "Aligned PCA experiment enables temporal Procrustes alignment inside each "
            "outer fold while leaving all other settings unchanged. The shared PCA "
            "reference and temporal template use training participants only. The held-"
            "out participant contributes unlabeled trials to estimate its PCA and "
            "rotation.\n\n"
            "> Alignment is transductive: it assumes an unlabeled calibration batch "
            "from the new participant. It is not zero-calibration decoding."
        )
    )
    report.add_section(step6)

    step7 = Section("Step 7 - Run and Inspect Every Fold", icon="7")
    step7.add_element(
        MarkdownElement(
            "Both experiments receive the same epochs, labels, participant groups, "
            "trial identifiers, and time axis. Raw results, predictions, splits, fold "
            "scores, and fit diagnostics are exported for independent auditing."
        )
    )
    diagnostic_details = AccordionElement("Fold fit and runtime diagnostics", open=False)
    diagnostic_details.add_element(
        InteractiveTableElement(
            tables["fit_diagnostics"],
            title="Fit diagnostics",
            selector_columns=["Representation", "Fold"],
        )
    )
    step7.add_element(diagnostic_details)
    report.add_section(step7)

    step8 = Section("Step 8 - Inspect Time-Resolved Generalization", icon="8")
    step8.add_element(
        MarkdownElement(
            "The curve is mean balanced accuracy across held-out participants; the "
            "ribbon is the fold-level standard deviation returned by `coco-pipe`. "
            "Binary chance is 0.5 and time zero is image onset. Broad sustained "
            "periods are generally more stable than an isolated maximum."
        )
    )
    step8.add_element(PlotlyElement(figures["temporal_decoding"], height="620px"))
    step8.add_element(
        InteractiveTableElement(tables["peak_summary"], title="Descriptive curve peaks")
    )
    report.add_section(step8)

    step9 = Section("Step 9 - Retain Held-Out-Participant Variability", icon="9")
    step9.add_element(
        MarkdownElement(
            "Heatmaps preserve one time course per held-out participant. Post-onset "
            "mean balanced accuracy and AUC above chance summarize each fold from "
            "0 to 0.8 s. Aligned-minus-sensor differences remain paired by held-out "
            "participant and are descriptive sensitivity summaries."
        )
    )
    step9.add_element(PlotlyElement(figures["fold_heatmaps"], height="720px"))
    step9.add_element(PlotlyElement(figures["fold_summaries"], height="520px"))
    step9.add_element(
        InteractiveTableElement(tables["representation_summary"], title="Across-fold summaries")
    )
    step9.add_element(
        InteractiveTableElement(
            tables["paired_fold_differences"],
            title="Aligned PCA minus Sensors within participant",
        )
    )
    report.add_section(step9)

    step10 = Section("Step 10 - Export and Interpret", icon="10")
    static_status = (
        "PNG and SVG exports completed."
        if context["static_exports_complete"]
        else "Static export was unavailable; interactive HTML figures are complete."
    )
    step10.add_element(
        MarkdownElement(
            "The bundle preserves summary and fold-level tables, split audits, fit "
            "diagnostics, raw result exports, figures, analysis arrays, manifests, "
            f"and this offline report. **{static_status}**\n\n"
            "Interpret LOSO as cross-participant generalization, keep peak values "
            "descriptive without temporal correction, and distinguish transductive "
            "aligned decoding from direct sensor decoding."
        )
    )
    appendix = AccordionElement("Appendix: all exported tables", open=False)
    for name, table in tables.items():
        appendix.add_element(InteractiveTableElement(table, title=name.replace("_", " ").title()))
    step10.add_element(appendix)
    report.add_section(step10)
    report.save(output / "report.html")
    return report.asset_mode


def run_decoding_analysis(
    *,
    subjects_requested: list[str],
    derivatives_root: Path,
    output: Path,
    n_components: int = 30,
    n_jobs: int = 1,
    seed: int = SEED,
    sensor_set: str = "all_sensors",
) -> dict[str, object]:
    """Run the notebook-equivalent MEG decoding analysis and save every output."""
    if len(subjects_requested) < 3:
        raise ValueError(
            "Aligned LOSO decoding requires at least three participants so every "
            "training fold contains at least two participants."
        )
    if n_components < 1:
        raise ValueError("n_components must be positive.")
    subjects_requested = [
        str(subject).removeprefix("sub-").zfill(2) for subject in subjects_requested
    ]
    if not derivatives_root.exists():
        raise FileNotFoundError(f"Prepared MEG derivatives were not found at {derivatives_root}.")

    output = Path(output) / sensor_set
    figures_dir = output / "figures"
    results_dir = output / "experiment_results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve inline assets before the expensive decoding fits.
    Report(title="MEG Faces decoding asset check", asset_urls="inline")

    container = _load_wakeman_henson_container(
        derivatives_root,
        subjects=subjects_requested,
        conditions=CONDITIONS,
        sensor_set=sensor_set,
    )
    X = np.asarray(container.X, dtype=np.float32)
    times = np.asarray(container.coords["time"], dtype=float)
    condition = np.asarray(container.y, dtype=int)
    sensor_names = np.asarray(container.coords["channel"]).astype(str)
    subject_ids = np.asarray(container.coords["subject"]).astype(str)
    y = np.where(condition == CONDITIONS[0], 0, 1)
    if container.ids is None:
        trial_ids = np.asarray([f"trial-{index}" for index in range(len(X))])
    else:
        trial_ids = np.asarray(container.ids).astype(str)
    whitening = container.meta.get("whitening", "not recorded")
    del container

    if n_components > X.shape[1]:
        raise ValueError(f"n_components={n_components} exceeds the {X.shape[1]} sensors.")
    analyzed_subjects = sorted(np.unique(subject_ids).tolist())
    if len(analyzed_subjects) < 3:
        raise RuntimeError("Fewer than three participants were loaded successfully.")

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
    experiments = {"Sensors": sensor_config, "Aligned PCA": aligned_config}

    results = {}
    temporal_frames = []
    fold_frames = []
    diagnostic_frames = []
    for representation, config in experiments.items():
        print(f"Running {representation} decoding...")
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
        raise RuntimeError("The LOSO audit found participant overlap.")

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

    held_out_by_fold = split_audit.set_index("Fold")["held_out_subject"].to_dict()
    metric_scores = fold_scores[
        (fold_scores["Metric"] == "balanced_accuracy") & fold_scores["Time"].notna()
    ]
    fold_summary_records = []
    for (representation, fold), rows in metric_scores.groupby(["Representation", "Fold"]):
        rows = rows.sort_values("Time")
        active_rows = rows[(rows["Time"] >= 0) & (rows["Time"] <= 0.8)]
        peak_index = rows["Value"].idxmax()
        fold_summary_records.append(
            {
                "Representation": representation,
                "Fold": int(fold),
                "held_out_subject": held_out_by_fold[int(fold)],
                "peak_time_s": float(rows.loc[peak_index, "Time"]),
                "peak_balanced_accuracy": float(rows.loc[peak_index, "Value"]),
                "postonset_mean_balanced_accuracy": float(active_rows["Value"].mean()),
                "postonset_auc_above_chance": float(
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
            postonset_ba_mean=("postonset_mean_balanced_accuracy", "mean"),
            postonset_ba_std=("postonset_mean_balanced_accuracy", "std"),
            postonset_auc_mean=("postonset_auc_above_chance", "mean"),
            postonset_auc_std=("postonset_auc_above_chance", "std"),
        )
        .reset_index()
    )

    paired = fold_summary.pivot(
        index=["Fold", "held_out_subject"],
        columns="Representation",
        values=[
            "peak_balanced_accuracy",
            "postonset_mean_balanced_accuracy",
            "postonset_auc_above_chance",
        ],
    )
    paired_fold_differences = paired.index.to_frame(index=False)
    for metric in (
        "peak_balanced_accuracy",
        "postonset_mean_balanced_accuracy",
        "postonset_auc_above_chance",
    ):
        paired_fold_differences[f"{metric}_aligned_minus_sensors"] = (
            paired[(metric, "Aligned PCA")].to_numpy() - paired[(metric, "Sensors")].to_numpy()
        )

    temporal_figure = plot_temporal_score_curve(
        temporal_scores,
        metric="balanced_accuracy",
        title="Famous versus Scrambled: LOSO decoding",
        colors=REPRESENTATION_COLORS,
    )
    temporal_figure.add_hline(y=CHANCE_LEVEL, line_dash="dot", line_color="#777777")
    temporal_figure.add_vline(x=0, line_color="#999999")
    temporal_figure.update_yaxes(title_text="balanced accuracy")
    temporal_figure.update_xaxes(title_text="time from image onset (s)")

    fold_heatmaps = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Sensors", "Aligned PCA"),
        horizontal_spacing=0.12,
    )
    for column, representation in enumerate(("Sensors", "Aligned PCA"), start=1):
        rows = metric_scores[metric_scores["Representation"] == representation]
        matrix = rows.pivot(index="Fold", columns="Time", values="Value").sort_index()
        fold_labels = [f"sub-{held_out_by_fold[int(fold)]}" for fold in matrix.index]
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
    fold_heatmaps.update_xaxes(title_text="time from image onset (s)")
    fold_heatmaps.update_yaxes(title_text="held-out participant", row=1, col=1)
    fold_heatmaps.update_layout(
        title="Balanced accuracy for every held-out participant",
        height=max(520, 32 * len(analyzed_subjects) + 250),
        width=1050,
    )

    fold_summary_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Post-onset mean balanced accuracy",
            "Post-onset AUC above chance",
        ),
    )
    for column, metric in enumerate(
        ("postonset_mean_balanced_accuracy", "postonset_auc_above_chance"),
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
        title="Held-out-participant post-onset summaries",
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
            except Exception as error:
                static_export_error = f"{type(error).__name__}: {error}"
                warnings.warn(
                    "Static Plotly export is unavailable; HTML figures are complete.",
                    stacklevel=2,
                )

    np.savez_compressed(
        output / "analysis_metadata.npz",
        times=times,
        condition=condition,
        target=y,
        subjects=subject_ids,
        trial_ids=trial_ids,
        sensor_set=sensor_set,
        sensor_names=sensor_names,
    )
    context = {
        "n_subjects": len(analyzed_subjects),
        "n_trials": len(X),
        "n_channels": X.shape[1],
        "n_times": X.shape[2],
        "time_start": float(times[0]),
        "time_stop": float(times[-1]),
        "n_components": n_components,
        "sensor_set": sensor_set,
        "static_exports_complete": static_export_error is None,
    }
    report_asset_mode = _build_report(
        output,
        context=context,
        tables=tables,
        figures=figures,
    )

    manifest = {
        "subjects_requested": subjects_requested,
        "subjects_analyzed": analyzed_subjects,
        "derivatives_root": str(derivatives_root),
        "conditions": list(CONDITIONS),
        "target_mapping": {"0": "Famous", "1": "Scrambled"},
        "shape": list(X.shape),
        "sensor_set": sensor_set,
        "sensor_names": sensor_names.tolist(),
        "whitening": whitening,
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
    write_manifest(output / "analysis_manifest.json", manifest, status="complete")
    print(f"Saved MEG Faces decoding analysis -> {output}")
    return {"manifest": manifest, "tables": tables, "figures": figures, "results": results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=["01", "02", "03", "04", "05", "06"])
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=(Path.home() / "mne_data" / "ds000117" / "derivatives" / "pca_trajectories"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/megfaces_decoding"))
    parser.add_argument(
        "--sensor-set",
        choices=tuple(MEG_SENSOR_SETS),
        default="all_sensors",
    )
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the first three requested participants.",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args(argv)

    subjects = args.subjects[:3] if args.smoke else args.subjects
    manifest_path = args.output / args.sensor_set / "run_manifest.json"
    if args.resume and _completed(manifest_path):
        print(f"Completed run found at {manifest_path}; nothing to do.")
        return
    settings = {**vars(args), "subjects": subjects}
    write_manifest(manifest_path, settings, status="running")
    try:
        result = run_decoding_analysis(
            subjects_requested=subjects,
            derivatives_root=args.derivatives_root,
            output=args.output,
            n_components=args.n_components,
            n_jobs=args.n_jobs,
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
