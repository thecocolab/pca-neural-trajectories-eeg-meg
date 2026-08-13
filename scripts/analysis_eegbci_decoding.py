"""Run and save the complete cross-participant EEGBCI decoding analysis.

This is the headless companion to
``tutorials/tutorial_eegbci_decoding.ipynb``. It mirrors the MEG-faces
decoding script's design: leave-one-subject-out splits, fold-local scaling,
sliding logistic regression, a configurable target (``--conditions``; two ids
give binary decoding, more give multiclass), five representations (raw
sensors, a shared fold-local PCA with no per-subject rotation, an unaligned
per-participant PCA control, and temporally-aligned per-subject PCA at two
component counts), and three validations of any alignment gain (a
within-participant k-fold upper bound, the geometry of the rotation itself, and
a calibration-only alignment variant).

``--contrasts`` runs several targets in one sweep, each into its own
subdirectory, and reports them side by side: every sweep figure is a grid with
one panel per contrast, and every table carries a ``Contrast`` column. Contrasts
are checkpointed individually, so a sweep resumes where it stopped and
``--report-only`` rebuilds the sweep report from finished contrast directories
without decoding anything — which is how the four contrasts are combined when
they run as separate cluster jobs.

Examples
--------
python scripts/analysis_eegbci_decoding.py
python scripts/analysis_eegbci_decoding.py --subjects 1 2 3 --n-jobs 1
python scripts/analysis_eegbci_decoding.py --conditions 5 6  # left vs right, imagery
python scripts/analysis_eegbci_decoding.py --conditions 3 4 5 6  # 4-class
python scripts/analysis_eegbci_decoding.py --contrasts  # the four default contrasts
python scripts/analysis_eegbci_decoding.py --contrasts 3-4 5-6 7-9 3-4-5-6
python scripts/analysis_eegbci_decoding.py --report-only  # combine finished runs
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from coco_pipe.decoding import (
    ChanceAssessmentConfig,
    CVConfig,
    Experiment,
    ExperimentConfig,
    ReducerConfig,
    StatisticalAssessmentConfig,
    TemporalAlignmentConfig,
    TemporalDecoderConfig,
)
from coco_pipe.decoding.configs import ClassicalModelConfig
from coco_pipe.decoding.stats import run_paired_permutation_assessment
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
from coco_pipe.transforms import TemporalProcrustesAlignment
from coco_pipe.viz.interactive import (
    plot_distribution_groups,
    plot_group_scatter_with_mean,
    plot_heatmap,
)
from coco_pipe.viz.interactive.decoding import (
    plot_temporal_score_curve,
    plot_temporal_statistical_assessment,
)
from coco_pipe.viz.theme import set_coco_theme

from pca_neural_trajectories import (
    LABEL_NAMES,
    contrast_fold_metrics,
    contrast_label,
    contrast_slug,
    facet_figures,
    load_eegbci_container,
    parse_contrast,
    setup_data_bids,
    write_manifest,
)

SEED = 42
DEFAULT_CONDITIONS = (3, 4)
# The four contrasts FINDINGS.md reports for EEGBCI: the three documented
# binaries plus a 4-class target, mirroring the MEG sweep's shape.
DEFAULT_CONTRASTS: tuple[tuple[int, ...], ...] = ((3, 4), (5, 6), (7, 9), (3, 4, 5, 6))
# Faceted one panel per contrast in the sweep report. `fold_heatmaps` is
# excluded: it is already one panel per representation, so faceting it would
# put 20 heatmaps in a single figure — it becomes per-contrast tabs instead.
UNFACETED_FIGURES = frozenset({"fold_heatmaps"})
# Active interval for the two window summaries, matching Step 9's definition.
METRIC_WINDOW = (0.0, 1.0)
# Scalar summaries compared across contrasts, in report order.
CONTRAST_METRICS = {
    "peak_balanced_accuracy": "Peak balanced accuracy",
    "postcue_mean_balanced_accuracy": "Post-cue mean balanced accuracy",
    "postcue_auc_above_chance": "Post-cue AUC above chance",
}
METRIC_PREFIX = "postcue"
ANALYSIS_WINDOW = (-0.2, 1.0)
EXCLUDED_SUBJECTS = {88, 92, 100}
# Positional palette for the five representations, in the order they are built
# (Sensors, Shared PCA, Unaligned PCA, small Aligned PCA, large Aligned PCA) —
# the exact labels are dynamic (they embed the configured component counts).
REPRESENTATION_PALETTE = ("#1b9e77", "#e6ab02", "#d95f02", "#7570b3", "#2a78d6")
CALIBRATION_COLOR = "#c51b7d"
WITHIN_SUBJECT_COLOR = "#444444"
# Trials per participant used for the descriptive alignment-geometry table. The
# rotation is estimated from a grand-mean path, which is stable well below the
# full trial count, so this keeps an O(participants^2) diagnostic affordable.
GEOMETRY_MAX_TRIALS = 200


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _slug(name: str) -> str:
    """Filesystem-safe stem for a representation label."""
    return "".join(
        character if character.isalnum() else "_" for character in name.lower()
    ).strip("_")


def _within_subject_upper_bound(
    *,
    X: np.ndarray,
    y: np.ndarray,
    trial_ids: np.ndarray,
    subject_ids: np.ndarray,
    times: np.ndarray,
    base_config: ExperimentConfig,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode each participant against themselves with stratified k-fold CV.

    LOSO asks a harder question than the classifier's own capacity: it has to
    transfer across heads. Repeating the identical sliding decoder inside each
    participant separates "the signal is weak" from "the signal does not
    transfer", and gives every LOSO curve a ceiling to be read against.

    Returns the per-participant time courses and the group-mean curve.
    """
    config = base_config.model_copy(deep=True)
    config.cv = CVConfig(
        strategy="stratified", n_splits=n_splits, shuffle=True, random_state=seed
    )
    frames = []
    for subject in sorted(np.unique(subject_ids)):
        rows = subject_ids == subject
        _, counts = np.unique(y[rows], return_counts=True)
        if len(counts) < 2 or counts.min() < n_splits:
            warnings.warn(
                f"Participant {subject} has too few trials in one class for "
                f"{n_splits}-fold within-participant CV; skipping.",
                stacklevel=2,
            )
            continue
        result = Experiment(config).run(
            X[rows],
            y[rows],
            sample_ids=trial_ids[rows],
            observation_level="epoch",
            time_axis=times,
        )
        curve = result.get_temporal_score_summary()
        curve["subject"] = subject
        frames.append(curve)

    per_subject = pd.concat(frames, ignore_index=True)
    per_subject = per_subject[per_subject["Metric"] == "balanced_accuracy"]
    group_curve = (
        per_subject.groupby("Time")["Mean"].agg(["mean", "std", "count"]).reset_index()
    )
    group_curve["Model"] = "Within participant (sensors)"
    group_curve["Metric"] = "balanced_accuracy"
    group_curve["Mean"] = group_curve["mean"]
    group_curve["Std"] = group_curve["std"].fillna(0.0) / np.sqrt(
        group_curve["count"].clip(lower=1)
    )
    return per_subject, group_curve[["Model", "Metric", "Time", "Mean", "Std"]]


def _alignment_geometry(
    *,
    X: np.ndarray,
    subject_ids: np.ndarray,
    n_components: int,
    seed: int,
    max_trials: int = GEOMETRY_MAX_TRIALS,
) -> pd.DataFrame:
    """Describe how far each participant's mean path had to rotate to match.

    The decoder only reports whether alignment helped. This refits the alignment
    itself on every LOSO fold and reads out its geometry: how much the
    participant's grand-mean trajectory already resembled the training template,
    how much of that gap the Procrustes rotation closed, and how large a rotation
    that took. The held-out participant is the row of interest — their mapping is
    the one estimated without any labels.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for subject in np.unique(subject_ids):
        rows = np.flatnonzero(subject_ids == subject)
        if len(rows) > max_trials:
            rows = rng.choice(rows, size=max_trials, replace=False)
        keep.append(rows)
    keep = np.sort(np.concatenate(keep))
    X, subject_ids = X[keep], subject_ids[keep]

    records = []
    for held_out in sorted(np.unique(subject_ids)):
        train = subject_ids != held_out
        aligner = TemporalProcrustesAlignment(
            n_components=n_components, random_state=seed
        )
        aligner.fit(X[train], groups=subject_ids[train])
        aligner.transform(X[~train], groups=subject_ids[~train])
        for subject, diagnostics in aligner.alignment_diagnostics_.items():
            records.append(
                {
                    "held_out_subject": held_out,
                    "subject": subject,
                    "role": "held out" if subject == held_out else "training",
                    **diagnostics,
                }
            )
    return pd.DataFrame(records).drop(columns=["seen_in_training"])


def _stack(*elements: Element) -> ContainerElement:
    """Bundle several elements into one container (used as a tab panel)."""
    box = ContainerElement()
    for element in elements:
        box.add_element(element)
    return box


def _headline_metrics(
    tables: dict[str, pd.DataFrame],
    context: dict[str, object],
    contrasts: dict[str, float] | None,
) -> dict[str, object]:
    """The few numbers a reader wants before scrolling, as KPI cards.

    Peaks are taken across every contrast in a sweep, so the cards answer
    "how well did this work at best" rather than describing one panel.
    """
    peaks = tables["peak_summary"]
    best = peaks.loc[peaks["peak_balanced_accuracy"].idxmax()]
    sensors = peaks[peaks["Representation"] == "Sensors"]["peak_balanced_accuracy"].max()
    significant = tables["significance_summary"]["Significant"]
    metrics: dict[str, object] = {"Participants": context["n_subjects"]}
    if contrasts:
        metrics["Contrasts"] = len(contrasts)
    metrics["Best peak BA"] = f"{best['peak_balanced_accuracy']:.3f}"
    metrics["Sensors peak BA"] = f"{sensors:.3f}"
    metrics["Within-participant ceiling"] = (
        f"{tables['within_subject_peaks']['peak_balanced_accuracy'].max():.3f}"
    )
    metrics["Significant comparisons"] = f"{int(significant.sum())}/{len(significant)}"
    return metrics


def build_decoding_report(
    output: Path,
    *,
    context: dict[str, object],
    tables: dict[str, pd.DataFrame],
    figures: dict[str, go.Figure],
    target_names: list[str],
    representation_names: list[str],
    contrasts: dict[str, float] | None = None,
    contrast_figures: dict[str, dict[str, go.Figure]] | None = None,
) -> str:
    """Build the notebook-equivalent structured decoding report.

    With ``contrasts`` — a mapping of contrast name to its chance level — the
    same 12 steps describe a sweep: every figure is a grid of per-contrast
    panels, and the tables carry a ``Contrast`` column instead of covering a
    single target. ``contrast_figures`` supplies the figures shown as
    per-contrast tabs rather than faceted.
    """
    contrast_figures = contrast_figures or {}
    if contrasts:
        target_text = f"the {len(contrasts)} contrasts ({', '.join(contrasts)})"
        target_mapping_text = (
            "one target per contrast, class index `i` being that contrast's "
            "`i`-th condition"
        )
        chance_text = ", ".join(
            f"{name} {level:.3f}" for name, level in contrasts.items()
        )
    else:
        target_text = " versus ".join(target_names)
        target_mapping_text = ", ".join(
            f"`{index} = {name}`" for index, name in enumerate(target_names)
        )
        chance_text = f"{context['chance_level']:.3f}"

    def _selectors(columns: list[str]) -> list[str]:
        """Filter by contrast first when the tables span a sweep."""
        return ["Contrast", *columns] if contrasts else columns

    representations_text = ", ".join(representation_names)
    title = (
        "EEGBCI — Cross-Participant Temporal Decoding "
        f"({context['n_subjects']} participants"
        + (f", {len(contrasts)} contrasts)" if contrasts else ")")
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

    report.add_summary_card(_headline_metrics(tables, context, contrasts))

    overview = Section(
        "Overview",
        icon="O",
        description="Cross-participant decoding across five feature representations",
        metadata={
            "Participants": str(context["n_subjects"]),
            "Trials": str(context["n_trials"]),
            "Channels × time": f"{context['n_channels']} × {context['n_times']}",
            "Window": f"{context['time_start']:.3f}–{context['time_stop']:.3f} s",
            "Target": target_text,
            "Chance": chance_text,
            "CV": "Leave one participant out",
            "Classifier": "Sliding logistic regression",
            "Metric": "Balanced accuracy",
            "PCA components": f"{context['small_n_components']} and {context['n_components']}",
        },
    )
    overview.add_element(
        MarkdownElement(
            "This report follows the complete **12-step cross-participant EEGBCI "
            "decoding workflow** from the tutorial notebook. It asks whether "
            f"{target_text} can be decoded in a participant who "
            f"contributed no labeled trials to model training.\n\n"
            f"**Representations compared:** {representations_text}."
        )
    )
    overview.add_element(
        CalloutElement(
            "All representations use identical trials, folds, classifier, scaling, "
            "metric, and time axis. Only the fold-local feature space differs — so "
            "any difference between them is attributable to the representation.",
            kind="info",
            title="What is held constant",
        )
    )
    report.add_section(overview)

    step1 = Section(
        "Step 1 — Define the Predictive Question",
        icon="1",
        description="The participant, not the trial, is the inferential unit",
    )
    step1.add_element(
        MarkdownElement(
            "The predictive question is deliberately narrower than visual trajectory "
            f"separation: **can a classifier trained on other participants identify "
            f"{target_text} in one entirely held-out participant?**\n\n"
            "A trial is an observation, but the participant is the inferential unit. "
            "Repeated trials from one participant are correlated and cannot be split "
            "independently across training and test sets."
        )
    )
    step1.add_element(
        CalloutElement(
            "Performance estimates apply to new participants drawn under a similar "
            "acquisition and preprocessing protocol — not merely to unseen trials "
            "from participants already represented in training.",
            kind="info",
            title="Generalization target",
        )
    )
    report.add_section(step1)

    step2 = Section(
        "Step 2 — Load the Sensor-Time Representation",
        icon="2",
        description="Native (trial, channel, time) epochs, cropped and baselined",
    )
    step2.add_element(
        MarkdownElement(
            "The analysis loads preprocessed EEGBCI epochs directly as "
            "`(trial, channel, time)`. No local binning helper is required because "
            "`coco-pipe` accepts the native three-dimensional temporal array.\n\n"
            "The crop and baseline match the notebook: −0.2–1.0 s around the movement "
            "cue, with the pre-cue −0.2–0.0 s interval used as the sensor baseline. "
            f"The decoded target is {target_text}."
        )
    )
    step2.add_element(
        InteractiveTableElement(
            tables["trial_counts"],
            title="Trials available per participant and class",
            selector_columns=_selectors(["subject", "condition_name"]),
        )
    )
    report.add_section(step2)

    step3 = Section(
        "Step 3 — Verify the Target and Class Balance",
        icon="3",
        description="Balanced accuracy plus training-fold class weights",
    )
    step3.add_element(
        MarkdownElement(
            f"The target is encoded as {target_mapping_text}. Balanced "
            "accuracy is used because it averages sensitivity across classes and is "
            "therefore robust to modest trial-count differences. The logistic "
            "regression also receives `class_weight='balanced'`, fitted using the "
            "training fold only."
        )
    )
    step3.add_element(
        CalloutElement(
            "Class balance is documented rather than repaired by resampling. No test "
            "trial is duplicated, removed, or used to determine a training-fold weight.",
            kind="info",
            title="No resampling",
        )
    )
    report.add_section(step3)

    step4 = Section(
        "Step 4 — Declare Leave-One-Participant-Out CV",
        icon="4",
        description="One held-out participant per fold, audited for overlap",
    )
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
            selector_columns=_selectors(["held_out_subject"]),
        )
    )
    report.add_section(step4)

    step5 = Section(
        "Step 5 — Configure Fold-Local Temporal Decoding",
        icon="5",
        description="One independent classifier per latency, scaled inside the fold",
    )
    step5.add_element(
        MarkdownElement(
            "At every retained latency, a logistic-regression classifier is fitted to "
            "the training participants and evaluated on the held-out participant. "
            "`wrapper='sliding'` repeats this independently across time.\n\n"
            "Standardization is owned by `coco-pipe` and fitted inside each training "
            "fold. The test participant is transformed with training-fold scaling "
            "parameters. This prevents the global-standardization leakage that would "
            "occur if all participants were scaled before LOSO splitting."
        )
    )
    step5.add_element(
        CalloutElement(
            "Each time point is a separate predictive model. A peak is descriptive "
            "and is not automatically a corrected significance result across the many "
            "tested latencies.",
            kind="tip",
            title="What the curve means",
        )
    )
    report.add_section(step5)

    step6 = Section(
        "Step 6 — Compare Representations",
        icon="6",
        description="Sensors, shared PCA, and per-subject PCA with and without rotation",
    )
    step6.add_element(
        MarkdownElement(
            f"Five representations are compared ({representations_text}), all fed "
            "into the identical sliding logistic regression. **Sensors** decodes "
            "directly from channels. **Shared PCA** fits a single fold-local PCA on "
            "the pooled training participants (`coco-pipe`'s `ReducerConfig`) and "
            "applies it unchanged to every participant — no per-subject basis, no "
            "rotation. **Unaligned PCA** fits a separate PCA per participant and "
            "stops there. **Aligned PCA** (at two component counts) does the same and "
            "then Procrustes-rotates each participant's own basis onto a "
            "training-only shared temporal template.\n\n"
            "Unaligned PCA is the control that makes the comparison interpretable: it "
            "differs from Aligned PCA by the rotation and by nothing else, so any "
            "difference between them is attributable to the rotation rather than to "
            "the fact of having a participant-specific basis."
        )
    )
    step6.add_element(
        CalloutElement(
            "Aligned performance answers what is possible **after** collecting an "
            "unlabeled calibration batch from the new participant. It is not "
            "zero-calibration inductive decoding and should not be presented as such.",
            kind="warning",
            title="Transductive scope",
        )
    )
    report.add_section(step6)

    step7 = Section(
        "Step 7 — Run and Audit Every Fold",
        icon="7",
        description="Identical inputs to every representation, diagnostics preserved",
    )
    step7.add_element(
        MarkdownElement(
            "All four experiments receive the same native epochs, labels, "
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
            selector_columns=_selectors(["Representation", "Fold"]),
        )
    )
    step7.add_element(diagnostics_appendix)
    report.add_section(step7)

    step8 = Section(
        "Step 8 — Inspect Time-Resolved Generalization",
        icon="8",
        description="Where in the epoch each representation separates the classes",
    )
    step8.add_element(
        MarkdownElement(
            "The curve is the mean balanced accuracy across held-out participants; "
            "the ribbon is the fold-level standard deviation returned by `coco-pipe`. "
            f"The horizontal line in each panel marks chance ({chance_text}), and "
            "time zero marks the movement cue.\n\n"
            "Peak latency and accuracy summarize the plotted curve but do not provide "
            "a multiple-comparison-corrected inferential test. Broad, sustained "
            "above-chance periods are generally more stable to sampling noise than a "
            "single isolated maximum."
        )
    )
    step8.add_element(
        TabsElement(
            {
                "Decoding curve": _stack(
                    PlotlyElement(figures["temporal_decoding"], height="600px"),
                    InteractiveTableElement(
                        tables["peak_summary"],
                        title="Descriptive peak of the across-participant curve",
                        selector_columns=_selectors(["Representation"]),
                    ),
                ),
                "Gain over Sensors": _stack(
                    MarkdownElement(
                        "The same folds re-expressed as a paired difference against "
                        "Sensors at every latency — mean and SEM across held-out "
                        "participants of `(representation - Sensors)` balanced "
                        "accuracy. Above zero is a sustained enhancement over raw "
                        "sensors at that latency; below zero is a cost."
                    ),
                    PlotlyElement(figures["gain_through_time"], height="500px"),
                ),
            }
        )
    )
    report.add_section(step8)

    step9 = Section(
        "Step 9 — Examine Held-Out-Participant Variability",
        icon="9",
        description="Per-fold profiles behind the group-mean curve",
    )
    step9.add_element(
        MarkdownElement(
            "A mean curve can hide participants with very different decoding profiles. "
            "The heatmaps retain one row per LOSO fold, labelled by the held-out "
            "participant, one panel per representation. The companion distribution "
            "plot summarizes two predeclared post-cue quantities per fold:\n\n"
            "- **Post-cue mean balanced accuracy:** average performance from 0–1 s.\n"
            "- **AUC above chance:** temporal integral of `(balanced accuracy − "
            "chance)` over the same interval.\n\n"
            "Representation-minus-Sensors differences are paired by outer fold. They "
            "are descriptive sensitivity summaries, not significance tests."
        )
    )
    if "fold_heatmaps" in contrast_figures:
        # One 5-representation heatmap grid per contrast; tabbed rather than
        # faceted so the panels stay readable.
        heatmaps: Element = TabsElement(
            {
                name: PlotlyElement(figure, height="720px")
                for name, figure in contrast_figures["fold_heatmaps"].items()
            }
        )
    else:
        heatmaps = PlotlyElement(figures["fold_heatmaps"], height="720px")
    step9.add_element(
        TabsElement(
            {
                "Per-fold heatmaps": heatmaps,
                "Fold distributions": _stack(
                    PlotlyElement(figures["fold_summaries"], height="520px"),
                    InteractiveTableElement(
                        tables["representation_summary"],
                        title="Across-fold descriptive summaries",
                        selector_columns=_selectors(["Representation"]),
                    ),
                ),
                "Paired by participant": _stack(
                    PlotlyElement(figures["representation_comparison"], height="540px"),
                    InteractiveTableElement(
                        tables["paired_fold_differences"],
                        title=(
                            "Each representation minus Sensors, within each "
                            "held-out participant"
                        ),
                        selector_columns=_selectors(["held_out_subject"]),
                    ),
                ),
            }
        )
    )
    report.add_section(step9)

    if contrasts and any(key.startswith("contrast_") for key in figures):
        across = Section(
            "Across Contrasts — Scalar Comparison",
            icon="C",
            description="The same three summaries, every contrast on one axis",
        )
        across.add_element(
            MarkdownElement(
                "The panels above compare representations *within* a contrast. "
                "These compare the contrasts themselves on the three predeclared "
                "summaries, with one point per held-out participant and the mean "
                "with its SEM.\n\n"
                "The peak is read off the fold-averaged curve and every fold is "
                "then sampled at that one latency, so the points average exactly "
                "to the reported peak. Averaging each fold's own maximum instead "
                "would select a different latency per fold and bias the result "
                "upward."
            )
        )
        across.add_element(
            TabsElement(
                {
                    pretty: PlotlyElement(
                        figures[f"contrast_{column}"], height="520px"
                    )
                    for column, pretty in CONTRAST_METRICS.items()
                    if f"contrast_{column}" in figures
                }
            )
        )
        across.add_element(
            InteractiveTableElement(
                tables["contrast_summary"],
                title="Contrast x representation summary (fold means)",
                selector_columns=_selectors(["Representation"]),
            )
        )
        report.add_section(across)

    step10 = Section(
        "Step 10 — Statistical Significance",
        icon="10",
        description="Paired permutation tests, max-stat corrected across latencies",
    )
    step10.add_element(
        MarkdownElement(
            "Each representation is compared against Sensors, and the large "
            "Aligned PCA additionally against Shared PCA (does a "
            "participant-specific basis add anything?) and against Unaligned PCA "
            "(is it the rotation, or merely the basis?), with a **paired "
            f"permutation test** ({context['n_permutations']} shuffles): within "
            "each held-out participant, predictions from the two representations "
            "are randomly swapped, and the observed balanced-accuracy difference "
            "is compared to that null. `max_stat` multiple-comparison correction "
            "takes the maximum of the null distribution across all latencies, so "
            "a single corrected p-value protects the whole time course. This test "
            "costs no extra model fits: it reuses the already-fitted predictions "
            "above."
        )
    )
    step10.add_element(
        CalloutElement(
            "A black square marks a latency where the corrected p-value clears the "
            "significance threshold. The shaded band is the permutation null.",
            kind="tip",
            title="Reading the significance panels",
        )
    )
    step10.add_element(
        TabsElement(
            {
                key.removeprefix("significance_").replace("_", " "): PlotlyElement(
                    figure, height="450px"
                )
                for key, figure in figures.items()
                if key.startswith("significance_")
            }
        )
    )
    step10.add_element(
        InteractiveTableElement(
            tables["significance_summary"],
            title="Significance at each comparison's peak observed difference",
            selector_columns=_selectors(["Representation", "Baseline"]),
        )
    )
    report.add_section(step10)

    step11 = Section(
        "Step 11 — Bound the Alignment Result",
        icon="11",
        description="A ceiling, the rotation's geometry, and a transduction control",
    )
    step11.add_element(
        MarkdownElement("Three checks bound what any alignment gain can mean.")
    )
    ceiling = tables["within_subject_peaks"]["peak_balanced_accuracy"].max()
    loso_sensors = tables["peak_summary"].loc[
        tables["peak_summary"]["Representation"] == "Sensors", "peak_balanced_accuracy"
    ].max()
    step11.add_columns(
        [
            StatCardElement(
                "Within-participant ceiling", f"{ceiling:.3f}", color="green"
            ),
            StatCardElement("LOSO Sensors peak", f"{loso_sensors:.3f}", color="blue"),
            StatCardElement(
                "Cross-participant cost",
                f"{ceiling - loso_sensors:+.3f}",
                color="yellow",
                delta="ceiling − LOSO",
            ),
        ]
    )
    within_panel = _stack(
        MarkdownElement(
            "**The within-participant ceiling.** The same sliding decoder is "
            f"refitted inside each participant with {context['within_subject_splits']}-fold "
            "stratified cross-validation, so no head-to-head transfer is required. "
            "The distance between that curve and the LOSO Sensors curve is the cost "
            "of generalizing across participants rather than the weakness of the "
            "signal itself, and it is the ceiling any representation is working "
            "toward."
        ),
        PlotlyElement(figures["within_subject_upper_bound"], height="500px"),
        InteractiveTableElement(
            tables["within_subject_peaks"],
            title="Within-participant peak balanced accuracy",
        ),
    )
    geometry_panel = _stack(
        MarkdownElement(
            "**What the rotation does geometrically.** The alignment is refitted on "
            "every fold and read out directly, without a classifier. Each point is "
            "one held-out participant: the normalized agreement between their "
            "grand-mean trajectory and the training template, before and after the "
            "Procrustes rotation. A large rise means their principal axes pointed "
            "somewhere idiosyncratic and the rotation recovered the correspondence; "
            "a flat pair means alignment had little to work with. "
            f"Computed on up to {context['geometry_max_trials']} trials per "
            "participant, since the grand-mean path is stable well below the full "
            "trial count."
        ),
        PlotlyElement(figures["alignment_geometry"], height="480px"),
        InteractiveTableElement(
            tables["alignment_geometry"],
            title="Per-fold alignment geometry",
            selector_columns=_selectors(["held_out_subject", "role"]),
        ),
    )
    calibration_panel = _stack(
        MarkdownElement(
            "**Is the gain just transduction?** Transductive alignment estimates the "
            "held-out participant's rotation from all of their (unlabeled) trials — "
            "including the very trials it is then scored on. The calibration variant "
            "removes that: each half of the held-out participant's trials is mapped "
            "using the PCA and rotation estimated from the *other* half, so no trial "
            "informs the mapping applied to it. If the transductive and calibration "
            "curves agree, the gain is domain adaptation from unlabeled data, not "
            "leakage. If the calibration curve collapses to Shared PCA, it was the "
            "transduction."
        ),
        PlotlyElement(figures["calibration_alignment"], height="500px"),
    )
    step11.add_element(
        TabsElement(
            {
                "Within-participant ceiling": within_panel,
                "Rotation geometry": geometry_panel,
                "Calibration control": calibration_panel,
            }
        )
    )
    report.add_section(step11)

    step12 = Section(
        "Step 12 — Export, Reproduce & Interpret",
        icon="12",
        description="What the bundle contains and what the numbers do not say",
    )
    static_status = (
        "PNG and SVG exports completed."
        if context["static_exports_complete"]
        else "Static export was unavailable; interactive HTML figures are complete."
    )
    step12.add_element(
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
            "not be described as zero-calibration decoding."
        )
    )
    step12.add_element(
        CalloutElement(
            "Compare the representations through paired held-out-participant "
            "behavior and sustained temporal performance, not through a single "
            "maximum of the group-mean curve.",
            kind="tip",
            title="Main takeaway",
        )
    )
    appendix = AccordionElement("Appendix: all exported summary tables", open=False)
    for name, table in tables.items():
        appendix.add_element(
            InteractiveTableElement(table, title=name.replace("_", " ").title())
        )
    step12.add_element(appendix)
    report.add_section(step12)

    report.save(str(output / "report.html"))
    return asset_mode


def run_decoding_analysis(
    *,
    subjects_requested: list[int],
    bids_root: Path,
    output: Path,
    n_components: int = 30,
    small_n_components: int = 3,
    n_permutations: int = 200,
    within_subject_splits: int = 5,
    n_jobs: int = -1,
    seed: int = SEED,
    prepare: bool = False,
    conditions: tuple[int, ...] = DEFAULT_CONDITIONS,
) -> dict[str, object]:
    """Run the notebook-equivalent decoding analysis and save every output.

    ``conditions`` selects and orders the condition_ids to decode (from
    :data:`LABEL_NAMES`). Two conditions give binary decoding (as in the
    tutorial's left-vs-right execution default); more give multiclass. Class
    index ``i`` in the target always corresponds to ``conditions[i]``, and
    chance level is ``1 / len(conditions)``.

    Five representations are compared: raw ``Sensors``; ``Shared PCA``, a
    fold-local PCA fit once on the pooled training participants with no
    per-subject rotation (``coco_pipe``'s ``ReducerConfig``); ``Unaligned PCA``,
    a per-participant PCA with the Procrustes step switched off; and temporally
    Procrustes-aligned per-subject PCA (``TemporalAlignmentConfig``) at
    ``small_n_components`` and ``n_components``.

    Three validations follow: a within-participant ``within_subject_splits``-fold
    upper bound, the geometry of the rotation on every fold, and a
    calibration-half (non-transductive) alignment variant.
    """
    if len(subjects_requested) < 2:
        raise ValueError("Cross-participant decoding requires at least two subjects.")
    invalid = sorted(set(subjects_requested) & EXCLUDED_SUBJECTS)
    if invalid:
        raise ValueError(f"Subjects with incompatible sampling rates requested: {invalid}.")
    if n_components < 1 or small_n_components < 1:
        raise ValueError("n_components and small_n_components must be positive.")
    conditions = tuple(conditions)
    if len(set(conditions)) < 2:
        raise ValueError("conditions must name at least two distinct condition ids.")
    chance_level = 1.0 / len(conditions)
    target_names = [LABEL_NAMES[condition_id] for condition_id in conditions]
    target_mapping = {str(index): name for index, name in enumerate(target_names)}

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
        conditions=conditions,
        tmin=ANALYSIS_WINDOW[0],
        tmax=ANALYSIS_WINDOW[1],
        baseline=(-0.2, 0.0),
    )
    X = np.asarray(container.X, dtype=np.float32)
    times = np.asarray(container.coords["time"], dtype=float)
    condition = np.asarray(container.y, dtype=int)
    subject_ids = np.asarray(container.coords["subject"]).astype(str)
    trial_ids = np.asarray(container.ids).astype(str)
    condition_to_class = {condition_id: index for index, condition_id in enumerate(conditions)}
    y = np.array([condition_to_class[value] for value in condition])

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
    shared_pca_config = sensor_config.model_copy(deep=True)
    shared_pca_config.reducer = ReducerConfig(enabled=True, n_components=n_components)

    # The control that isolates the rotation: an identical per-participant PCA,
    # with the Procrustes step switched off.
    unaligned_config = sensor_config.model_copy(deep=True)
    unaligned_config.temporal_alignment = TemporalAlignmentConfig(
        enabled=True,
        n_components=n_components,
        adaptation="transductive",
        rotate=False,
    )

    aligned_small_config = sensor_config.model_copy(deep=True)
    aligned_small_config.temporal_alignment = TemporalAlignmentConfig(
        enabled=True,
        n_components=small_n_components,
        adaptation="transductive",
    )

    aligned_large_config = sensor_config.model_copy(deep=True)
    aligned_large_config.temporal_alignment = TemporalAlignmentConfig(
        enabled=True,
        n_components=n_components,
        adaptation="transductive",
    )
    shared_pca_name = f"Shared PCA ({n_components})"
    unaligned_name = f"Unaligned PCA ({n_components}C)"
    aligned_small_name = f"Aligned PCA ({small_n_components}C)"
    aligned_large_name = f"Aligned PCA ({n_components}C)"
    experiments = {
        "Sensors": sensor_config,
        shared_pca_name: shared_pca_config,
        unaligned_name: unaligned_config,
        aligned_small_name: aligned_small_config,
        aligned_large_name: aligned_large_config,
    }
    representation_colors = dict(zip(experiments.keys(), REPRESENTATION_PALETTE))

    # Validation variant, reported separately from the five representations
    # above: the same alignment, but each held-out participant's rotation is
    # estimated from one half of their trials and applied to the other half.
    calibration_name = f"Aligned PCA ({n_components}C, calibration)"
    calibration_config = sensor_config.model_copy(deep=True)
    calibration_config.temporal_alignment = TemporalAlignmentConfig(
        enabled=True,
        n_components=n_components,
        adaptation="calibration",
    )
    representation_colors[calibration_name] = CALIBRATION_COLOR


    results = {}
    temporal_frames = []
    fold_frames = []
    diagnostic_frames = []
    for representation, config in {**experiments, calibration_name: calibration_config}.items():
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
            results_dir / _slug(representation),
            config=config.model_dump(),
            formats=("csv",),
        )

    temporal_scores = pd.concat(temporal_frames, ignore_index=True)
    fold_scores = pd.concat(fold_frames, ignore_index=True)
    fit_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    representation_names = list(experiments)
    other_representations = [name for name in representation_names if name != "Sensors"]

    # Paired permutation significance, at every latency: is each representation's
    # balanced accuracy different from Sensors, and (for the headline aligned
    # variant) from Shared PCA — the two questions FINDINGS.md's original EEG
    # validation answered. This swaps whole held-out-participant predictions
    # between the two already-fitted experiments, so it costs no extra model
    # fits — cheap even at n_permutations=200.
    # `n_bootstraps` sets the confidence interval's resampling budget and
    # `n_permutations` the p-value's; tying them keeps a reduced smoke run cheap
    # on both, since the bootstrap otherwise dominates at 1000 resamples.
    stats_config = StatisticalAssessmentConfig(
        chance=ChanceAssessmentConfig(
            n_permutations=n_permutations,
            temporal_correction="max_stat",
        ),
        n_bootstraps=n_permutations,
        unit_of_inference="group_mean",
        random_state=seed,
        n_jobs=n_jobs,
    )
    comparison_pairs = [
        *((representation, "Sensors") for representation in other_representations),
        (aligned_large_name, shared_pca_name),
        (aligned_large_name, unaligned_name),
        (calibration_name, shared_pca_name),
    ]

    significance_frames = []
    significance_figures = {}
    for representation, baseline in comparison_pairs:
        comparison = run_paired_permutation_assessment(
            results[representation],
            results[baseline],
            "Logistic regression",
            "balanced_accuracy",
            stats_config,
        )
        comparison["Representation"] = representation
        comparison["Baseline"] = baseline
        significance_frames.append(comparison)
        figure_key = f"{representation} vs {baseline}"
        significance_figures[figure_key] = plot_temporal_statistical_assessment(
            comparison,
            title=(
                f"{figure_key}: paired permutation test "
                f"({n_permutations} shuffles, max-stat corrected)"
            ),
        )
    significance_assessment = pd.concat(significance_frames, ignore_index=True)
    significance_summary = (
        significance_assessment.loc[
            significance_assessment.groupby(["Representation", "Baseline"])["Observed"].idxmax()
        ]
        .reset_index(drop=True)
        .rename(columns={"Observed": "peak_observed_diff"})
    )

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

    # One peak latency per representation, taken from the fold-averaged curve.
    # Averaging each fold's own maximum instead would select a different latency
    # per fold and bias the result upward: the maximum of noisy estimates is not
    # an estimate of the maximum.
    group_peak_time = (
        temporal_scores.loc[temporal_scores.groupby("Representation")["Mean"].idxmax()]
        .set_index("Representation")["Time"]
        .to_dict()
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
        peak_time = float(group_peak_time[representation])
        at_peak = rows.loc[(rows["Time"] - peak_time).abs().idxmin()]
        own_peak_index = rows["Value"].idxmax()
        fold_summary_records.append(
            {
                "Representation": representation,
                "Fold": int(fold),
                "held_out_subject": held_out_by_fold[int(fold)],
                "peak_time_s": peak_time,
                "peak_balanced_accuracy": float(at_peak["Value"]),
                # Descriptive only: where this fold happened to peak. Never
                # averaged into a headline number.
                "own_peak_time_s": float(rows.loc[own_peak_index, "Time"]),
                "own_peak_balanced_accuracy": float(rows.loc[own_peak_index, "Value"]),
                "postcue_mean_balanced_accuracy": float(active_rows["Value"].mean()),
                "postcue_auc_above_chance": float(
                    np.trapezoid(
                        active_rows["Value"] - chance_level,
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
        for representation in other_representations:
            paired_fold_differences[f"{metric}_{_slug(representation)}_minus_sensors"] = (
                paired[(metric, representation)].to_numpy()
                - paired[(metric, "Sensors")].to_numpy()
            )

    temporal_figure = plot_temporal_score_curve(
        temporal_scores[temporal_scores["Representation"].isin(representation_names)],
        metric="balanced_accuracy",
        title=f"{' versus '.join(target_names)}: LOSO decoding",
        colors=representation_colors,
    )
    temporal_figure.add_hline(
        y=chance_level,
        line_dash="dot",
        line_color="#777777",
    )
    temporal_figure.add_vline(x=0, line_color="#999999")
    temporal_figure.update_yaxes(title_text="balanced accuracy")
    temporal_figure.update_xaxes(title_text="time from movement cue (s)")

    # Gain-through-time: (representation - Sensors) balanced accuracy, paired by
    # held-out fold at every timepoint, then averaged across folds. Unlike the
    # peak/post-cue point summaries above, this keeps the full time course of
    # the enhancement (or cost) relative to raw sensors.
    wide_scores = metric_rows.pivot_table(
        index=["Fold", "Time"], columns="Representation", values="Value"
    )
    gain_frames = []
    for representation in other_representations:
        gain = (wide_scores[representation] - wide_scores["Sensors"]).rename("Value")
        gain = gain.reset_index()
        summary = gain.groupby("Time")["Value"].agg(["mean", "std", "count"]).reset_index()
        summary["Model"] = representation
        summary["Metric"] = "balanced_accuracy_gain"
        summary["Mean"] = summary["mean"]
        summary["Std"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
        gain_frames.append(summary[["Model", "Metric", "Time", "Mean", "Std"]])
    gain_through_time = pd.concat(gain_frames, ignore_index=True)
    gain_through_time_figure = plot_temporal_score_curve(
        gain_through_time,
        metric="balanced_accuracy_gain",
        title="Gain through time: representation minus Sensors (paired by held-out participant)",
        colors=representation_colors,
    )
    gain_through_time_figure.add_hline(y=0.0, line_dash="dot", line_color="#777777")
    gain_through_time_figure.add_vline(x=0, line_color="#999999")
    gain_through_time_figure.update_yaxes(title_text="balanced accuracy gain over Sensors")
    gain_through_time_figure.update_xaxes(title_text="time from movement cue (s)")

    fold_panels = {}
    for representation in representation_names:
        rows = metric_rows[metric_rows["Representation"] == representation]
        matrix = rows.pivot(index="Fold", columns="Time", values="Value").sort_index()
        fold_panels[representation] = plot_heatmap(
            matrix.to_numpy(),
            x_labels=matrix.columns.to_numpy(dtype=float),
            y_labels=[f"sub-{held_out_by_fold[int(fold)]}" for fold in matrix.index],
            vmin=0,
            vmax=1,
            title=representation,
            xaxis_title="time from movement cue (s)",
            yaxis_title="held-out participant",
            colorbar_label="BA",
        )
    fold_heatmaps = facet_figures(
        fold_panels,
        n_cols=len(representation_names),
        title="Balanced accuracy for every held-out participant",
        row_height=max(420, 32 * len(analyzed_subjects) + 200),
    )

    distribution_panels = {}
    for metric, pretty in (
        ("postcue_mean_balanced_accuracy", "Post-cue mean balanced accuracy"),
        ("postcue_auc_above_chance", "Post-cue AUC above chance"),
    ):
        distribution_panels[pretty] = plot_distribution_groups(
            groups=[
                fold_summary.loc[
                    fold_summary["Representation"] == representation, metric
                ].to_numpy()
                for representation in representation_names
            ],
            labels=representation_names,
            color=[representation_colors[name] for name in representation_names],
            show_points=True,
            baseline=0.0 if metric.endswith("auc_above_chance") else chance_level,
            yaxis_title=pretty,
            title=pretty,
        )
    fold_summary_figure = facet_figures(
        distribution_panels,
        n_cols=2,
        title="Per-fold distributions by representation",
        row_height=460,
        shared_yaxes=False,
    )
    fold_summary_figure.update_layout(
        title="Held-out-participant post-cue summaries",
        height=500,
        width=1000,
    )

    representation_comparison_figure = plot_group_scatter_with_mean(
        [
            fold_summary.loc[
                fold_summary["Representation"] == representation, "peak_balanced_accuracy"
            ].to_numpy()
            for representation in representation_names
        ],
        representation_names,
        point_labels=[
            fold_summary.loc[
                fold_summary["Representation"] == representation, "held_out_subject"
            ].to_numpy()
            for representation in representation_names
        ],
        title="Peak balanced accuracy by representation, one point per held-out participant",
        yaxis_title="peak balanced accuracy",
        baseline=chance_level,
        baseline_label="chance",
        color=[representation_colors[name] for name in representation_names],
        height=520,
    )

    # --- Validation 1: the within-participant ceiling -----------------------
    print("Running within-participant upper bound...")
    within_subject_scores, within_subject_curve = _within_subject_upper_bound(
        X=X,
        y=y,
        trial_ids=trial_ids,
        subject_ids=subject_ids,
        times=times,
        base_config=sensor_config,
        n_splits=within_subject_splits,
        seed=seed,
    )
    upper_bound_frame = pd.concat(
        [
            temporal_scores.loc[
                temporal_scores["Representation"] == "Sensors",
                ["Model", "Metric", "Time", "Mean", "Std"],
            ],
            within_subject_curve,
        ],
        ignore_index=True,
    )
    upper_bound_figure = plot_temporal_score_curve(
        upper_bound_frame,
        metric="balanced_accuracy",
        title="Within-participant ceiling versus cross-participant transfer (sensors)",
        colors={
            "Sensors": representation_colors["Sensors"],
            "Within participant (sensors)": WITHIN_SUBJECT_COLOR,
        },
    )
    upper_bound_figure.add_hline(y=chance_level, line_dash="dot", line_color="#777777")
    upper_bound_figure.add_vline(x=0, line_color="#999999")
    upper_bound_figure.update_yaxes(title_text="balanced accuracy")
    upper_bound_figure.update_xaxes(title_text="time from movement cue (s)")

    within_subject_peaks = (
        within_subject_scores.loc[
            within_subject_scores.groupby("subject")["Mean"].idxmax(),
            ["subject", "Time", "Mean"],
        ]
        .rename(columns={"Time": "peak_time_s", "Mean": "peak_balanced_accuracy"})
        .sort_values("subject")
        .reset_index(drop=True)
    )

    # --- Validation 2: what the rotation actually does ----------------------
    print("Measuring alignment geometry...")
    alignment_geometry = _alignment_geometry(
        X=X,
        subject_ids=subject_ids,
        n_components=n_components,
        seed=seed,
    )
    held_out_geometry = alignment_geometry[alignment_geometry["role"] == "held out"]
    geometry_figure = plot_group_scatter_with_mean(
        [
            held_out_geometry["template_similarity_unrotated"].to_numpy(),
            held_out_geometry["template_similarity_rotated"].to_numpy(),
        ],
        ["before rotation", "after rotation"],
        point_labels=[
            held_out_geometry["subject"].to_numpy(),
            held_out_geometry["subject"].to_numpy(),
        ],
        title=(
            "Shape agreement between each held-out participant's mean path and "
            "the training template"
        ),
        yaxis_title="normalized similarity to template",
        baseline=0.0,
        color=[
            representation_colors[unaligned_name],
            representation_colors[aligned_large_name],
        ],
        height=460,
    )

    # --- Validation 3: is the gain just transduction? -----------------------
    calibration_frame = temporal_scores[
        temporal_scores["Representation"].isin(
            [shared_pca_name, aligned_large_name, calibration_name]
        )
    ]
    calibration_figure = plot_temporal_score_curve(
        calibration_frame,
        metric="balanced_accuracy",
        title="Transductive alignment versus calibration-half alignment",
        colors=representation_colors,
    )
    calibration_figure.add_hline(y=chance_level, line_dash="dot", line_color="#777777")
    calibration_figure.add_vline(x=0, line_color="#999999")
    calibration_figure.update_yaxes(title_text="balanced accuracy")
    calibration_figure.update_xaxes(title_text="time from movement cue (s)")

    tables = {
        "trial_counts": trial_counts,
        "split_audit": split_audit,
        "temporal_scores": temporal_scores,
        "fold_scores": fold_scores,
        "peak_summary": peak_summary,
        "fold_summary": fold_summary,
        "representation_summary": representation_summary,
        "paired_fold_differences": paired_fold_differences,
        "gain_through_time": gain_through_time,
        "significance_assessment": significance_assessment,
        "significance_summary": significance_summary,
        "within_subject_scores": within_subject_scores,
        "within_subject_peaks": within_subject_peaks,
        "alignment_geometry": alignment_geometry,
        "fit_diagnostics": fit_diagnostics,
    }
    figures = {
        "temporal_decoding": temporal_figure,
        "gain_through_time": gain_through_time_figure,
        "within_subject_upper_bound": upper_bound_figure,
        "alignment_geometry": geometry_figure,
        "calibration_alignment": calibration_figure,
        "fold_heatmaps": fold_heatmaps,
        "fold_summaries": fold_summary_figure,
        "representation_comparison": representation_comparison_figure,
        **{
            f"significance_{key.replace(' ', '_')}": figure
            for key, figure in significance_figures.items()
        },
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    static_export_error = None
    for name, figure in figures.items():
        figure.write_html(figures_dir / f"{name}.html", include_plotlyjs="cdn")
        # JSON round-trips losslessly back into a Figure, which is what lets a
        # contrast sweep re-facet an already-completed contrast without redoing
        # its decoding.
        (figures_dir / f"{name}.json").write_text(figure.to_json())
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
        "chance_level": chance_level,
        "n_components": n_components,
        "small_n_components": small_n_components,
        "n_permutations": n_permutations,
        "within_subject_splits": within_subject_splits,
        "geometry_max_trials": GEOMETRY_MAX_TRIALS,
        "static_exports_complete": static_export_error is None,
    }
    report_asset_mode = build_decoding_report(
        output,
        context=context,
        tables=tables,
        figures=figures,
        target_names=target_names,
        representation_names=representation_names,
    )

    manifest = {
        "subjects_requested": subjects_requested,
        "subjects_analyzed": analyzed_subjects,
        "bids_root": str(bids_root),
        "conditions": list(conditions),
        "analysis_window": list(ANALYSIS_WINDOW),
        "target_mapping": target_mapping,
        "shape": list(X.shape),
        "cv": "leave_one_group_out",
        "metric": "balanced_accuracy",
        "chance_level": chance_level,
        "representations": representation_names,
        "validation_representation": calibration_name,
        "n_components": n_components,
        "small_n_components": small_n_components,
        "n_permutations": n_permutations,
        "significance_comparisons": [
            {"representation": representation, "baseline": baseline}
            for representation, baseline in comparison_pairs
        ],
        "alignment_adaptation": "transductive",
        "within_subject_splits": within_subject_splits,
        "geometry_max_trials": GEOMETRY_MAX_TRIALS,
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


def _load_contrast_outputs(
    contrast_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, go.Figure]]:
    """Read one finished contrast's tables and figures back off disk.

    Fresh and reused contrasts both go through this, so a sweep that resumes
    three of four contrasts builds exactly the same report as one that ran all
    four in a single process.
    """
    tables = {
        path.stem: pd.read_csv(path) for path in sorted(contrast_dir.glob("*.csv"))
    }
    figures = {
        path.stem: pio.from_json(path.read_text())
        for path in sorted((contrast_dir / "figures").glob("*.json"))
    }
    if not tables or not figures:
        raise FileNotFoundError(
            f"{contrast_dir} has no decoding outputs to reuse. Rerun that "
            "contrast without --report-only."
        )
    return tables, figures


def run_contrast_sweep(
    *,
    contrasts: Sequence[Sequence[int]] = DEFAULT_CONTRASTS,
    output: Path,
    report_only: bool = False,
    resume: bool = True,
    **analysis_kwargs: object,
) -> dict[str, object]:
    """Decode several contrasts and report them side by side.

    Each contrast is a complete single-target analysis in its own
    ``<output>/<contrast_slug>/`` directory — same figures, same report, same
    manifest as a plain run. The sweep then rebuilds every figure as a grid of
    those per-contrast panels, so representations are compared *within* a
    contrast and contrasts *against* each other in one view.

    Contrasts are checkpointed individually: a completed one is reused rather
    than recomputed (unless ``resume=False``), and ``report_only`` rebuilds the
    sweep report from finished contrast directories without decoding anything.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contrasts = [parse_contrast(contrast) for contrast in contrasts]
    if not contrasts:
        raise ValueError("A sweep needs at least one contrast.")

    panels: dict[str, dict[str, go.Figure]] = {}
    frames: dict[str, list[pd.DataFrame]] = {}
    contrast_manifests: dict[str, object] = {}
    contrast_fold_frames: list[pd.DataFrame] = []

    for conditions in contrasts:
        # Short labels throughout: they title the facet panels, tag the
        # `Contrast` column, and drive the report's filters, so they have to
        # agree everywhere. The full names live in the sweep manifest.
        label = contrast_label(conditions, LABEL_NAMES, short=True)
        contrast_dir = output / contrast_slug(conditions, LABEL_NAMES)
        manifest_path = contrast_dir / "run_manifest.json"
        settings = {"conditions": list(conditions), **analysis_kwargs}

        if report_only or (resume and _completed(manifest_path)):
            if not _completed(manifest_path):
                raise FileNotFoundError(
                    f"No completed run for {label} at {contrast_dir}; "
                    "--report-only cannot rebuild it."
                )
            print(f"[{label}] reusing completed run → {contrast_dir}")
        else:
            print(f"[{label}] decoding {len(conditions)} conditions → {contrast_dir}")
            write_manifest(manifest_path, settings, status="running")
            try:
                result = run_decoding_analysis(
                    conditions=tuple(conditions),
                    output=contrast_dir,
                    **analysis_kwargs,
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

        tables, figures = _load_contrast_outputs(contrast_dir)
        contrast_manifests[label] = json.loads(manifest_path.read_text())
        per_fold = contrast_fold_metrics(
            tables["temporal_scores"],
            tables["fold_scores"],
            1.0 / len(conditions),
            window=METRIC_WINDOW,
            prefix=METRIC_PREFIX,
            representations=contrast_manifests[label]["extra"]["analysis"][
                "representations"
            ],
        )
        per_fold.insert(0, "Contrast", label)
        contrast_fold_frames.append(per_fold)
        for name, figure in figures.items():
            panels.setdefault(name, {})[label] = figure
        for name, table in tables.items():
            table.insert(0, "Contrast", label)
            frames.setdefault(name, []).append(table)

    labels = [
        contrast_label(conditions, LABEL_NAMES, short=True) for conditions in contrasts
    ]
    sweep_tables = {
        name: pd.concat(collected, ignore_index=True)
        for name, collected in frames.items()
    }
    first = contrast_manifests[labels[0]]["extra"]["analysis"]
    sweep_figures = {}
    contrast_figures: dict[str, dict[str, go.Figure]] = {}
    for name, per_contrast in panels.items():
        ordered = {label: per_contrast[label] for label in labels if label in per_contrast}
        if name in UNFACETED_FIGURES:
            contrast_figures[name] = ordered
            continue
        sweep_figures[name] = facet_figures(
            ordered,
            n_cols=2,
            title=(per_contrast[labels[0]].layout.title.text or name),
            row_height=400,
            # Chance level differs between binary and multiclass panels, so a
            # shared y-axis would misplace the reference line.
            shared_yaxes=len({len(c) for c in contrasts}) == 1,
        )

    # The scalar comparison across contrasts: one point per held-out fold,
    # drawn in the same style as the per-fold figures inside each contrast.
    contrast_folds = pd.concat(contrast_fold_frames, ignore_index=True)
    sweep_tables["contrast_fold_metrics"] = contrast_folds
    sweep_tables["contrast_summary"] = contrast_folds.groupby(
        ["Contrast", "Representation"], as_index=False
    )[["peak_time_s", *CONTRAST_METRICS]].mean()

    representation_order = first["representations"]
    representation_colors = dict(zip(representation_order, REPRESENTATION_PALETTE))
    for column, pretty in CONTRAST_METRICS.items():
        scalar_panels = {}
        for label, conditions in zip(labels, contrasts, strict=True):
            rows = contrast_folds[contrast_folds["Contrast"] == label]
            scalar_panels[label] = plot_group_scatter_with_mean(
                [
                    rows.loc[rows["Representation"] == name, column].to_numpy()
                    for name in representation_order
                ],
                representation_order,
                point_labels=[
                    rows.loc[rows["Representation"] == name, "Fold"].to_numpy()
                    for name in representation_order
                ],
                yaxis_title=pretty,
                baseline=(
                    0.0 if column.endswith("auc_above_chance") else 1.0 / len(conditions)
                ),
                color=[representation_colors[name] for name in representation_order],
            )
        sweep_figures[f"contrast_{column}"] = facet_figures(
            scalar_panels,
            n_cols=2,
            title=f"{pretty} across contrasts",
            shared_yaxes=False,
        )

    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, table in sweep_tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    for name, figure in sweep_figures.items():
        figure.write_html(figures_dir / f"{name}.html", include_plotlyjs="cdn")
        (figures_dir / f"{name}.json").write_text(figure.to_json())

    context = {
        "n_subjects": len(first["subjects_analyzed"]),
        "n_trials": int(sweep_tables["trial_counts"]["n_trials"].sum()),
        "n_channels": first["shape"][1],
        "n_times": first["shape"][2],
        "time_start": float(sweep_tables["temporal_scores"]["Time"].min()),
        "time_stop": float(sweep_tables["temporal_scores"]["Time"].max()),
        "chance_level": first["chance_level"],
        "n_components": first["n_components"],
        "small_n_components": first["small_n_components"],
        "n_permutations": first["n_permutations"],
        "within_subject_splits": first.get("within_subject_splits"),
        "geometry_max_trials": GEOMETRY_MAX_TRIALS,
        "static_exports_complete": True,
    }
    build_decoding_report(
        output,
        context=context,
        tables=sweep_tables,
        figures=sweep_figures,
        target_names=[],
        representation_names=first["representations"],
        contrasts={
            label: 1.0 / len(conditions)
            for label, conditions in zip(labels, contrasts, strict=True)
        },
        contrast_figures=contrast_figures,
    )

    manifest = {
        "contrasts": [list(conditions) for conditions in contrasts],
        "contrast_labels": labels,
        "contrast_names": [
            contrast_label(conditions, LABEL_NAMES) for conditions in contrasts
        ],
        "contrast_manifests": contrast_manifests,
        "representations": first["representations"],
    }
    write_manifest(output / "sweep_manifest.json", manifest, status="complete")
    print(f"Saved EEGBCI decoding sweep ({len(contrasts)} contrasts) → {output}")
    return {"manifest": manifest, "tables": sweep_tables, "figures": sweep_figures}


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
    parser.add_argument("--small-n-components", type=int, default=3)
    parser.add_argument(
        "--conditions",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Condition ids to decode, e.g. 3 4 (left/right execution, default), "
            "5 6 (left/right imagery), 7 9 (hands imagery vs execution), or "
            "3 4 5 6 for 4-class."
        ),
    )
    parser.add_argument(
        "--contrasts",
        nargs="*",
        default=None,
        help=(
            "Run a multi-contrast sweep instead of a single target, e.g. "
            "--contrasts 3-4 5-6 7-9 3-4-5-6 (the default set when the flag is "
            "given without values). Each contrast lands in its own subdirectory "
            "and every sweep figure shows one panel per contrast."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the sweep report from completed contrast directories.",
    )
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument(
        "--within-subject-splits",
        type=int,
        default=5,
        help="Folds for the within-participant upper-bound decoding.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Explicitly download/preprocess the requested EEGBCI participants.",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
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

    analysis_kwargs = {
        "subjects_requested": subjects_requested,
        "bids_root": args.bids_root,
        "n_components": args.n_components,
        "small_n_components": args.small_n_components,
        "n_permutations": args.n_permutations,
        "within_subject_splits": args.within_subject_splits,
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "prepare": args.prepare,
    }
    # A bare invocation sweeps every default contrast; --conditions asks
    # for one specific target instead.
    if args.conditions is None or args.contrasts is not None or args.report_only:
        run_contrast_sweep(
            contrasts=args.contrasts or DEFAULT_CONTRASTS,
            output=args.output,
            report_only=args.report_only,
            resume=args.resume,
            **analysis_kwargs,
        )
        return

    manifest_path = args.output / "run_manifest.json"
    if args.resume and _completed(manifest_path):
        print(f"Completed run found at {manifest_path}; nothing to do.")
        return
    settings = {**vars(args), "subjects": subjects_requested}
    write_manifest(manifest_path, settings, status="running")
    try:
        result = run_decoding_analysis(
            output=args.output,
            conditions=tuple(args.conditions),
            **analysis_kwargs,
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
