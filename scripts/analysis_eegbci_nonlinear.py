"""Run and save the EEGBCI nonlinear trajectory analysis.

This is the headless companion to
``tutorials/tutorial_eegbci_nonlinear.ipynb``. It uses the same
subjects, conditions, balancing rule, temporal decimation, reducers, quality
metrics, trial-safe velocity field, and alignment sensitivity analysis.

The output includes a structured standalone report whose overview and ten
numbered sections mirror the tutorial's methodological explanations, figures,
interactive tables, interpretation boundaries, and reproducibility checklist.

Examples
--------
python scripts/analysis_eegbci_nonlinear.py --skip-prepare
python scripts/analysis_eegbci_nonlinear.py --subjects 1 2 3 --trials-per-cell 2
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from coco_pipe.dim_reduction import DimReduction
from coco_pipe.dim_reduction.evaluation import MethodSelector, compute_velocity_fields
from coco_pipe.report import PlotlyElement, Report, Section
from coco_pipe.report.elements import (
    AccordionElement,
    InteractiveTableElement,
    MarkdownElement,
)
from coco_pipe.transforms import TemporalProcrustesAlignment
from coco_pipe.viz.theme import set_coco_theme
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score

from pca_neural_trajectories import (
    LABEL_NAMES,
    load_eegbci_container,
    setup_data_bids,
    write_manifest,
)

CONDITIONS = (3, 4, 5, 6)
ANALYSIS_WINDOW = (-0.2, 1.0)
NEIGHBORHOODS = (10, 20, 40)
METHODS = ("PCA", "UMAP", "PHATE", "Isomap")
CONDITION_COLORS = {
    3: "#0072B2",
    4: "#D55E00",
    5: "#0072B2",
    6: "#D55E00",
}
CONDITION_DASHES = {3: "solid", 4: "solid", 5: "dash", 6: "dash"}


def run_nonlinear_analysis(
    *,
    subjects: list[int],
    bids_root: Path,
    output: Path,
    trials_per_cell: int = 4,
    time_stride: int = 4,
    seed: int = 42,
    flow_method: str = "PHATE",
    prepare: bool = True,
) -> dict[str, object]:
    """Run the notebook-equivalent nonlinear analysis and save its full bundle."""
    if trials_per_cell < 1:
        raise ValueError("trials_per_cell must be at least 1.")
    if time_stride < 1:
        raise ValueError("time_stride must be at least 1.")
    if flow_method not in METHODS:
        raise ValueError(f"flow_method must be one of {METHODS}; got {flow_method!r}.")

    set_coco_theme(mode="paper", colorblind=True)
    output = Path(output)
    figures_dir = output / "figures"
    reducers_dir = output / "reducers"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reducers_dir.mkdir(parents=True, exist_ok=True)

    if prepare:
        setup_data_bids(
            subjects=subjects,
            runs=list(range(3, 15)),
            root=bids_root,
        )

    sensor = load_eegbci_container(
        bids_root,
        subjects=[f"{subject:03d}" for subject in subjects],
        conditions=CONDITIONS,
        runs=[f"{run:02d}" for run in range(3, 15)],
        tmin=ANALYSIS_WINDOW[0] - 1.0,
        tmax=ANALYSIS_WINDOW[1] + 1.5,
        baseline=(-1.0, 0.0),
    )
    full_times = np.asarray(sensor.coords["time"], dtype=float)
    analysis_mask = (full_times >= ANALYSIS_WINDOW[0] - 1e-8) & (
        full_times <= ANALYSIS_WINDOW[1] + 1e-8
    )
    sensor = sensor.isel(time=analysis_mask).zscore(
        dim=("obs", "time"), eps=1e-5
    )

    all_subjects = np.asarray(sensor.coords["subject"]).astype(str)
    all_labels = np.asarray(sensor.y).astype(int)
    complete_subjects = [
        subject
        for subject in np.unique(all_subjects)
        if all(
            np.any((all_subjects == subject) & (all_labels == condition))
            for condition in CONDITIONS
        )
    ]
    if not complete_subjects:
        raise RuntimeError("No subject has trials for every requested condition.")

    cells = {
        (subject, condition): np.flatnonzero(
            (all_subjects == subject) & (all_labels == condition)
        )
        for subject in complete_subjects
        for condition in CONDITIONS
    }
    available_per_cell = min(len(rows) for rows in cells.values())
    selected_per_cell = min(trials_per_cell, available_per_cell)
    rng = np.random.default_rng(seed)
    selected_rows: list[np.ndarray] = []
    balance_records: list[dict[str, object]] = []
    for (subject, condition), rows in sorted(cells.items()):
        chosen = np.sort(rng.choice(rows, selected_per_cell, replace=False))
        selected_rows.append(chosen)
        balance_records.append(
            {
                "subject": subject,
                "condition": condition,
                "condition_name": LABEL_NAMES[condition],
                "available_trials": len(rows),
                "selected_trials": len(chosen),
            }
        )
    balanced_idx = np.concatenate(selected_rows)
    time_idx = np.arange(0, sensor.X.shape[2], time_stride)
    sensor = sensor.isel(obs=balanced_idx.tolist(), time=time_idx.tolist())

    X = np.asarray(sensor.X, dtype=float)
    times = np.asarray(sensor.coords["time"], dtype=float)
    trial_subjects = np.asarray(sensor.coords["subject"]).astype(str)
    trial_labels = np.asarray(sensor.y).astype(int)
    trial_ids = np.asarray(sensor.ids).astype(str)
    samples = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    sample_subjects = np.repeat(trial_subjects, len(times))
    sample_labels = np.repeat(trial_labels, len(times))
    trial_sequence = np.repeat(np.arange(len(X)), len(times))
    within_trial_time = np.tile(times, len(X))

    valid_k = [k for k in NEIGHBORHOODS if k < len(samples)]
    if not valid_k:
        raise RuntimeError("The balanced sample is too small for neighborhood evaluation.")
    neighbor_count = min(20, len(samples) - 1)
    reducers = {
        "PCA": DimReduction("PCA", n_components=3, random_state=seed),
        "UMAP": DimReduction(
            "UMAP",
            n_components=3,
            n_neighbors=neighbor_count,
            random_state=seed,
        ),
        "PHATE": DimReduction("PHATE", n_components=3, random_state=seed),
        "Isomap": DimReduction(
            "Isomap", n_components=3, n_neighbors=neighbor_count
        ),
    }
    flat_embeddings: dict[str, np.ndarray] = {}
    trajectories: dict[str, np.ndarray] = {}
    for name, reducer in reducers.items():
        try:
            embedding = np.asarray(reducer.fit_transform(samples))
        except ImportError as error:
            raise ImportError(
                f"{name} requires its coco-pipe optional dependency. "
                "Install this project with the declared neighbor extras."
            ) from error
        reducer.score(
            embedding,
            X=samples,
            metrics=["trustworthiness", "continuity"],
            k_values=valid_k,
            max_eval_samples=min(3000, len(samples)),
        )
        flat_embeddings[name] = embedding
        trajectories[name] = embedding.reshape(len(X), len(times), 3)

    quality_records = MethodSelector(reducers).collect().to_frame()
    quality_summary = (
        quality_records.groupby(["method", "metric"], as_index=False)["value"]
        .mean()
        .sort_values(["metric", "value"], ascending=[True, False])
    )

    diagnostic_records = []
    silhouette_sample_size = min(2000, len(samples) - 1)
    for name, embedding in flat_embeddings.items():
        diagnostic_records.append(
            {
                "method": name,
                "subject_silhouette": silhouette_score(
                    embedding,
                    sample_subjects,
                    sample_size=silhouette_sample_size,
                    random_state=seed,
                ),
                "condition_silhouette": silhouette_score(
                    embedding,
                    sample_labels,
                    sample_size=silhouette_sample_size,
                    random_state=seed,
                ),
            }
        )
    embedding_diagnostics = pd.DataFrame(diagnostic_records)

    quality_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Trustworthiness", "Continuity"),
        shared_yaxes=True,
    )
    for column, metric in enumerate(("trustworthiness", "continuity"), start=1):
        for method in METHODS:
            rows = quality_records[
                (quality_records["method"] == method)
                & (quality_records["metric"] == metric)
            ].sort_values("scope_value")
            quality_figure.add_trace(
                go.Scatter(
                    x=rows["scope_value"],
                    y=rows["value"],
                    mode="lines+markers",
                    name=method,
                    legendgroup=method,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
    quality_figure.update_yaxes(range=[0, 1], title_text="score", row=1, col=1)
    quality_figure.update_xaxes(title_text="neighborhood size k")
    quality_figure.update_layout(
        title="Local geometry across neighborhood scales", height=480, width=980
    )

    diagnostic_figure = go.Figure()
    diagnostic_figure.add_bar(
        x=embedding_diagnostics["method"],
        y=embedding_diagnostics["subject_silhouette"],
        name="subject",
    )
    diagnostic_figure.add_bar(
        x=embedding_diagnostics["method"],
        y=embedding_diagnostics["condition_silhouette"],
        name="condition",
    )
    diagnostic_figure.update_layout(
        barmode="group",
        title="What structures each embedding: participant or condition?",
        yaxis_title="silhouette score",
    )

    trajectory_figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=METHODS,
        horizontal_spacing=0.1,
        vertical_spacing=0.14,
    )
    for panel, method in enumerate(METHODS):
        row, column = divmod(panel, 2)
        for condition in CONDITIONS:
            mean_trajectory = trajectories[method][trial_labels == condition].mean(
                axis=0
            )
            trajectory_figure.add_trace(
                go.Scatter(
                    x=mean_trajectory[:, 0],
                    y=mean_trajectory[:, 1],
                    mode="lines",
                    line={
                        "color": CONDITION_COLORS[condition],
                        "dash": CONDITION_DASHES[condition],
                        "width": 3,
                    },
                    name=LABEL_NAMES[condition],
                    legendgroup=str(condition),
                    showlegend=panel == 0,
                ),
                row=row + 1,
                col=column + 1,
            )
    trajectory_figure.update_xaxes(showticklabels=False, title_text="dimension 1")
    trajectory_figure.update_yaxes(showticklabels=False, title_text="dimension 2")
    trajectory_figure.update_layout(
        title="Condition-mean trajectories: compare shape, not coordinate scale",
        height=850,
        width=1000,
    )

    flow_embedding = flat_embeddings[flow_method]
    velocity = compute_velocity_fields(
        X=samples,
        X_emb=flow_embedding,
        groups=trial_sequence,
        times=within_trial_time,
        n_neighbors=neighbor_count,
    )
    velocity_norm = np.linalg.norm(velocity, axis=1)
    velocity_summary = pd.DataFrame(
        [
            {
                "condition": condition,
                "condition_name": LABEL_NAMES[condition],
                "median_velocity_norm": float(
                    np.median(velocity_norm[sample_labels == condition])
                ),
                "p95_velocity_norm": float(
                    np.percentile(velocity_norm[sample_labels == condition], 95)
                ),
                "zero_velocity_fraction": float(
                    np.mean(velocity_norm[sample_labels == condition] == 0)
                ),
            }
            for condition in CONDITIONS
        ]
    )
    nonzero = velocity_norm > 0
    plot_scale = 1.0
    if nonzero.any():
        span = np.ptp(flow_embedding[:, :2], axis=0)
        plot_scale = 0.04 * float(np.max(span)) / float(
            np.percentile(velocity_norm[nonzero], 90)
        )
    velocity_figure = go.Figure()
    shown = np.linspace(0, len(flow_embedding) - 1, min(180, len(flow_embedding)), dtype=int)
    for index in shown:
        if velocity_norm[index] == 0:
            continue
        condition = int(sample_labels[index])
        velocity_figure.add_trace(
            go.Scatter(
                x=[
                    flow_embedding[index, 0],
                    flow_embedding[index, 0] + velocity[index, 0] * plot_scale,
                ],
                y=[
                    flow_embedding[index, 1],
                    flow_embedding[index, 1] + velocity[index, 1] * plot_scale,
                ],
                mode="lines",
                line={"color": CONDITION_COLORS[condition], "width": 1},
                opacity=0.35,
                showlegend=False,
            )
        )
    velocity_figure.update_layout(
        title=f"Trial-safe descriptive flow in {flow_method} space",
        xaxis_title="dimension 1",
        yaxis_title="dimension 2",
    )

    alignment_components = min(10, X.shape[1])
    alignment = TemporalProcrustesAlignment(
        n_components=alignment_components,
        random_state=seed,
    )
    aligned = alignment.fit_transform(X, groups=trial_subjects)
    unaligned = np.empty_like(aligned)
    for subject in np.unique(trial_subjects):
        rows = trial_subjects == subject
        participant = X[rows]
        pooled = participant.transpose(0, 2, 1).reshape(-1, X.shape[1])
        scores = alignment.subject_pcas_[subject].transform(pooled)
        unaligned[rows] = scores.reshape(
            len(participant), len(times), alignment_components
        ).transpose(0, 2, 1)
    unaligned_trajectories = np.moveaxis(unaligned, 1, 2)
    aligned_trajectories = np.moveaxis(aligned, 1, 2)

    consistency_rows = []
    for representation, scores in (
        ("participant PCA", unaligned_trajectories),
        ("participant PCA + Procrustes", aligned_trajectories),
    ):
        for subject in np.unique(trial_subjects):
            own = scores[trial_subjects == subject].mean(axis=0).ravel()
            other = scores[trial_subjects != subject].mean(axis=0).ravel()
            consistency_rows.append(
                {
                    "representation": representation,
                    "subject": subject,
                    "correlation": float(np.corrcoef(own, other)[0, 1]),
                }
            )
    alignment_consistency = pd.DataFrame(consistency_rows)
    alignment_summary = (
        alignment_consistency.groupby("representation")["correlation"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    alignment_figure = go.Figure(
        go.Bar(
            x=alignment_summary["representation"],
            y=alignment_summary["mean"],
            error_y={"type": "data", "array": alignment_summary["sem"]},
        )
    )
    alignment_figure.update_layout(
        title="Alignment sensitivity: leave-one-subject trajectory consistency",
        yaxis_title="correlation",
    )

    aligned_trajectory_figure = go.Figure()
    for condition in CONDITIONS:
        mean_trajectory = aligned_trajectories[trial_labels == condition].mean(axis=0)
        aligned_trajectory_figure.add_trace(
            go.Scatter(
                x=mean_trajectory[:, 0],
                y=mean_trajectory[:, 1],
                mode="lines",
                line={
                    "color": CONDITION_COLORS[condition],
                    "dash": CONDITION_DASHES[condition],
                    "width": 3,
                },
                name=LABEL_NAMES[condition],
            )
        )
    aligned_trajectory_figure.update_layout(
        title="Condition trajectories after label-free temporal alignment",
        xaxis_title="aligned PC1",
        yaxis_title="aligned PC2",
    )

    balance_table = pd.DataFrame(balance_records)
    tables = {
        "balance": balance_table,
        "quality_records": quality_records,
        "quality_summary": quality_summary,
        "embedding_diagnostics": embedding_diagnostics,
        "velocity_summary": velocity_summary,
        "alignment_consistency": alignment_consistency,
        "alignment_summary": alignment_summary,
    }
    figures = {
        "quality_by_neighborhood": quality_figure,
        "embedding_diagnostics": diagnostic_figure,
        "method_trajectories": trajectory_figure,
        "velocity_field": velocity_figure,
        "alignment_consistency": alignment_figure,
        "aligned_trajectories": aligned_trajectory_figure,
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
    for name, reducer in reducers.items():
        reducer.save(reducers_dir / f"{name.lower()}.pkl")

    np.savez_compressed(
        output / "analysis_arrays.npz",
        sensor=X,
        times=times,
        subjects=trial_subjects,
        labels=trial_labels,
        trial_ids=trial_ids,
        unaligned=unaligned_trajectories,
        aligned=aligned_trajectories,
        velocity=velocity,
        **{f"embedding_{name.lower()}": values for name, values in trajectories.items()},
    )
    manifest = {
        "subjects_requested": subjects,
        "subjects_analyzed": complete_subjects,
        "bids_root": str(bids_root),
        "conditions": list(CONDITIONS),
        "analysis_window": list(ANALYSIS_WINDOW),
        "trials_per_cell_requested": trials_per_cell,
        "trials_per_cell_used": selected_per_cell,
        "time_stride": time_stride,
        "neighborhoods": valid_k,
        "flow_method": flow_method,
        "alignment_components": alignment_components,
        "random_state": seed,
        "shape": list(X.shape),
        "static_figure_exports_complete": static_export_error is None,
        "static_figure_export_error": static_export_error,
    }
    # ------------------------------------------------------------------ Report (10-step)
    report_title = (
        "EEGBCI — Nonlinear Neural Trajectories "
        f"({len(complete_subjects)} participants)"
    )
    try:
        report = Report(title=report_title, asset_urls="inline")
        report_asset_mode = "inline"
    except OSError:
        warnings.warn(
            "Inline report assets are not cached and could not be downloaded; "
            "falling back to CDN-linked report assets.",
            stacklevel=2,
        )
        report = Report(title=report_title)
        report_asset_mode = "cdn"

    # ── Overview ────────────────────────────────────────────────────────────
    overview = Section("Overview", icon="O")
    overview.add_element(
        MarkdownElement(
            "This companion report follows the complete **10-step nonlinear "
            "EEGBCI trajectory workflow** from the tutorial notebook. PCA, UMAP, "
            "PHATE, and Isomap are fit to exactly the same balanced sensor-time "
            "observations. Every visualization is paired with the diagnostic or "
            "assumption needed to interpret it.\n\n"
            "| Parameter | Value |\n"
            "|---|---|\n"
            f"| Participants | {len(complete_subjects)} analyzable / "
            f"{len(subjects)} requested |\n"
            f"| Conditions | {', '.join(LABEL_NAMES[c] for c in CONDITIONS)} |\n"
            f"| Trials per participant × condition | {selected_per_cell} |\n"
            f"| Balanced sensor tensor | {tuple(X.shape)} |\n"
            f"| Samples supplied to each reducer | {len(samples):,} |\n"
            f"| Analysis window | {times[0]:.3f}–{times[-1]:.3f} s |\n"
            f"| Temporal stride | every {time_stride} sample(s) |\n"
            f"| Neighborhood scales | {', '.join(map(str, valid_k))} |\n"
            f"| Velocity embedding | {flow_method} |\n"
            f"| Alignment components | {alignment_components} |\n\n"
            "> Each numbered section below corresponds directly to a numbered step "
            "in `tutorial_eegbci_nonlinear.ipynb`. The report and "
            "notebook use the same analysis choices and saved result tables."
        )
    )
    report.add_section(overview)

    # ── Step 1 — Representation ─────────────────────────────────────────────
    step1 = Section("Step 1 — Choose the Representation", icon="1")
    step1.add_element(
        MarkdownElement(
            "At each time point, the neural state is the simultaneous voltage "
            "pattern across the EEG sensors. A trial therefore traces a path through "
            f"a **{X.shape[1]}-dimensional sensor space**.\n\n"
            "PCA and every nonlinear reducer receive this same sensor vector. We do "
            "not feed PCA scores into UMAP, PHATE, or Isomap, because doing so would "
            "give PCA an unacknowledged filtering role and make the comparison "
            "asymmetric.\n\n"
            "> **Fair-comparison rule:** The trials, time samples, channels, crop, "
            "and normalization are fixed. Only the geometry-learning algorithm "
            "changes."
        )
    )
    report.add_section(step1)

    # ── Step 2 — Preprocess and balance ─────────────────────────────────────
    step2 = Section("Step 2 — Preprocess & Balance Whole Trials", icon="2")
    step2.add_element(
        MarkdownElement(
            "The BIDS conversion applies the cleaning pipeline described in the "
            "introductory EEGBCI analysis: bad-channel handling, common-average "
            "reference, artifact control, and 0.5–40 Hz filtering. Broad epochs are "
            "loaded before the data are cropped to the predeclared "
            f"**{ANALYSIS_WINDOW[0]:.1f}–{ANALYSIS_WINDOW[1]:.1f} s** task window "
            "and standardized per channel across observations and time.\n\n"
            "Nonlinear embeddings are density-sensitive. A participant or condition "
            "with extra trials would contribute extra time samples and exert more "
            "influence on every fitted space. We therefore retain participants with "
            "all four conditions and sample the same number of **whole trials** from "
            "every participant × condition cell before temporal decimation.\n\n"
            "> **Why whole trials?** Flattening first and sampling isolated time "
            "points would distort temporal coverage, fragment trials, and invalidate "
            "the velocity analysis."
        )
    )
    step2.add_element(
        InteractiveTableElement(
            balance_table,
            title="Participant × condition balance",
            selector_columns=["subject", "condition_name"],
        )
    )
    report.add_section(step2)

    # ── Step 3 — Reshape ────────────────────────────────────────────────────
    step3 = Section("Step 3 — Reshape into (Samples × Features)", icon="3")
    step3.add_element(
        MarkdownElement(
            "The balanced tensor is transposed and flattened as follows:\n\n"
            "`(trial, channel, time) → (trial, time, channel) → "
            "(trial × time, channel)`\n\n"
            f"This produces **{len(samples):,} rows × {samples.shape[1]} sensor "
            "features** for each reducer. Parallel vectors preserve participant, "
            "condition, trial identity, and within-trial time. These vectors allow "
            "lossless trajectory reconstruction and prevent false temporal "
            "connections between trials.\n\n"
            f"- Balanced trials: **{len(X):,}**\n"
            f"- Retained time samples per trial: **{len(times)}**\n"
            f"- Unique velocity groups: **{len(np.unique(trial_sequence)):,}**"
        )
    )
    report.add_section(step3)

    # ── Step 4 — Fit reducers ───────────────────────────────────────────────
    step4 = Section("Step 4 — Fit PCA, UMAP, PHATE & Isomap", icon="4")
    step4.add_element(
        MarkdownElement(
            "All methods produce three coordinates, but they optimize different "
            "geometries:\n\n"
            "| Method | Geometry emphasized | Main caution |\n"
            "|---|---|---|\n"
            "| PCA | Global linear variance | Cannot unfold curved structure |\n"
            "| UMAP | Local fuzzy neighborhoods | Global distance and density may "
            "be distorted |\n"
            "| PHATE | Diffusion geometry and continua | Axes and scale remain "
            "arbitrary |\n"
            "| Isomap | Geodesic distances on a neighbor graph | Sensitive to graph "
            "shortcuts or disconnections |\n\n"
            f"UMAP and Isomap use **{neighbor_count} neighbors**. Random seeds are "
            "fixed for stochastic methods, and every fitted reducer is saved under "
            "`reducers/`.\n\n"
            "> **No common coordinate ruler:** A distance of 1 in PHATE is not "
            "equivalent to a distance of 1 in PCA or UMAP. Cross-method comparisons "
            "must use dimensionless validation metrics rather than raw embedded "
            "distances."
        )
    )
    report.add_section(step4)

    # ── Step 5 — Geometry validation ────────────────────────────────────────
    step5 = Section("Step 5 — Validate Local Geometry", icon="5")
    step5.add_element(
        MarkdownElement(
            "A smooth-looking embedding can invent neighbors or tear apart sensor "
            "states that were originally close. Two complementary rank diagnostics "
            "are evaluated at every declared neighborhood scale:\n\n"
            "- **Trustworthiness** penalizes false neighbors: samples that appear "
            "close only after embedding.\n"
            "- **Continuity** penalizes missing neighbors: sensor-space neighbors "
            "that the embedding separates.\n\n"
            "Both range from 0 to 1, with larger values indicating better local "
            "preservation. Crossing curves indicate that the preferred method "
            "depends on neighborhood size.\n\n"
            "> High trustworthiness or continuity supports geometric fidelity. It "
            "does not establish a biological manifold, an attractor, predictive "
            "condition information, or generalization to unseen participants."
        )
    )
    step5.add_element(PlotlyElement(quality_figure, height="520px"))
    step5.add_element(
        InteractiveTableElement(
            quality_summary,
            title="Mean geometry-preservation scores",
            selector_columns=["method", "metric"],
        )
    )
    quality_appendix = AccordionElement(
        "Details: every method × metric × neighborhood score",
        open=False,
    )
    quality_appendix.add_element(
        InteractiveTableElement(
            quality_records,
            title="Neighborhood-level quality records",
            selector_columns=["method", "metric"],
        )
    )
    step5.add_element(quality_appendix)
    report.add_section(step5)

    # ── Step 6 — Participant/condition diagnostic ───────────────────────────
    step6 = Section("Step 6 — Diagnose Participant vs Condition Structure", icon="6")
    step6.add_element(
        MarkdownElement(
            "EEG sensor patterns differ across participants because of anatomy, "
            "electrode placement, impedance, and physiology. A visually separated "
            "embedding can therefore organize **who produced the data** rather than "
            "**which motor condition was performed**.\n\n"
            "Participant and condition silhouette scores diagnose these two sources "
            "of structure after unsupervised fitting. Values near 1 indicate compact, "
            "separated groups; values near 0 indicate overlap; negative values mean "
            "many samples are closer to another group.\n\n"
            "> **Descriptive, not inferential:** Repeated time samples from one trial "
            "are dependent. These silhouettes are not p-values, independent-sample "
            "effect estimates, or cross-validated decoding scores."
        )
    )
    step6.add_element(PlotlyElement(diagnostic_figure, height="500px"))
    step6.add_element(
        InteractiveTableElement(
            embedding_diagnostics,
            title="Embedding structure diagnostics",
            selector_columns=["method"],
        )
    )
    report.add_section(step6)

    # ── Step 7 — Trajectory reconstruction ─────────────────────────────────
    step7 = Section("Step 7 — Reconstruct Condition Trajectories", icon="7")
    step7.add_element(
        MarkdownElement(
            "Reducer fitting treated each sensor state as an unordered row. The "
            "embedding rows are now reshaped back to `(trial, time, dimension)` and "
            "averaged within condition. Execution is shown with solid lines, imagery "
            "with dashed lines, and color identifies left versus right hand.\n\n"
            "> **What can be compared:** Within a panel, inspect branching, loops, "
            "return paths, and relative condition separation. Across panels, compare "
            "qualitative topology only. PCA signs can flip, while nonlinear spaces "
            "can rotate, reflect, warp, or rescale without violating their intended "
            "geometry."
        )
    )
    step7.add_element(PlotlyElement(trajectory_figure, height="900px"))
    report.add_section(step7)

    # ── Step 8 — Trial-safe velocity ────────────────────────────────────────
    step8 = Section("Step 8 — Estimate Trial-Respecting Velocity", icon="8")
    step8.add_element(
        MarkdownElement(
            "Velocity is approximated from consecutive samples **within each trial**. "
            "Differencing the flattened matrix directly would connect the last state "
            "of one trial to the first state of the next and create a transition that "
            "never occurred.\n\n"
            "`compute_velocity_fields` receives both the trial identifier and the "
            "within-trial time coordinate. It computes valid temporal differences and "
            f"then pools local displacements in **{flow_method} space**. Trial "
            "endpoints naturally have no forward difference.\n\n"
            "> **Descriptive flow, not a fitted dynamical system:** Arrow lengths "
            "inherit the arbitrary scale of the selected embedding. The field does "
            "not estimate a differential equation, demonstrate an attractor, or "
            "support raw speed comparisons across reducers."
        )
    )
    step8.add_element(PlotlyElement(velocity_figure, height="620px"))
    step8.add_element(
        InteractiveTableElement(
            velocity_summary,
            title=f"Descriptive velocity summary in {flow_method} space",
            selector_columns=["condition_name"],
        )
    )
    report.add_section(step8)

    # ── Step 9 — Alignment sensitivity ──────────────────────────────────────
    step9 = Section("Step 9 — Evaluate Temporal Procrustes Alignment", icon="9")
    step9.add_element(
        MarkdownElement(
            "Participant structure can dominate a pooled EEG embedding. Temporal "
            "Procrustes alignment tests one specific explanation: participants may "
            "share a temporal pattern expressed in differently rotated "
            "participant-specific PCA spaces.\n\n"
            "The transform fits participant PCAs, builds a shared temporal reference, "
            "and estimates a label-free orthogonal rotation/reflection for each "
            "participant. To isolate alignment itself, the control is **participant "
            "PCA before rotation**, not raw sensor space.\n\n"
            "Each participant's mean temporal pattern is correlated with the mean of "
            "all other participants before and after rotation. An increase supports "
            "the shared-rotated-pattern assumption; it does not prove identical neural "
            "generators.\n\n"
            "> **Transductive-analysis warning:** All displayed participants contribute "
            "to this descriptive fit. Prediction for unseen participants requires a "
            "training-only reference and label-free test-participant calibration, as "
            "implemented in the decoding tutorial."
        )
    )
    step9.add_element(PlotlyElement(alignment_figure, height="500px"))
    step9.add_element(
        InteractiveTableElement(
            alignment_summary,
            title="Cross-participant consistency before and after alignment",
            selector_columns=["representation"],
        )
    )
    step9.add_element(PlotlyElement(aligned_trajectory_figure, height="620px"))
    alignment_appendix = AccordionElement(
        "Details: leave-one-participant consistency values",
        open=False,
    )
    alignment_appendix.add_element(
        InteractiveTableElement(
            alignment_consistency,
            title="Participant-level alignment sensitivity",
            selector_columns=["representation", "subject"],
        )
    )
    step9.add_element(alignment_appendix)
    report.add_section(step9)

    # ── Step 10 — Reproducibility and interpretation ────────────────────────
    step10 = Section("Step 10 — Export, Reproduce & Interpret", icon="10")
    static_status = (
        "PNG and SVG exports completed."
        if static_export_error is None
        else "Static export was unavailable; the interactive HTML figures are complete."
    )
    step10.add_element(
        MarkdownElement(
            "A reproducible analysis preserves the evidence behind the headline "
            "figure. This run saves:\n\n"
            "- all balance, geometry, silhouette, velocity, and alignment tables;\n"
            "- interactive HTML figures plus available PNG/SVG copies;\n"
            "- fitted PCA, UMAP, PHATE, and Isomap reducers;\n"
            "- sensor, embedding, velocity, and alignment arrays;\n"
            "- a manifest containing configuration, package versions, and Git commits;\n"
            "- this standalone structured report.\n\n"
            f"**Static export status:** {static_status}\n\n"
            "Before interpreting any nonlinear pattern, verify that:\n\n"
            "1. whole trials were balanced before flattening;\n"
            "2. trustworthiness and continuity support the relevant scale;\n"
            "3. participant identity does not silently dominate the condition story;\n"
            "4. comparisons remain within one embedding's coordinate system;\n"
            "5. temporal calculations respect trial boundaries; and\n"
            "6. alignment conclusions remain within their transductive scope.\n\n"
            "> **Main takeaway:** Nonlinear embeddings are complementary views of "
            "sensor-space geometry. Prefer conclusions that remain coherent across "
            "geometry validation, participant diagnostics, temporal reconstruction, "
            "and alignment sensitivity—not conclusions that depend on one visually "
            "striking panel."
        )
    )
    all_tables = AccordionElement("Appendix: all exported analysis tables", open=False)
    for name, table in tables.items():
        all_tables.add_element(
            InteractiveTableElement(
                table,
                title=name.replace("_", " ").title(),
            )
        )
    step10.add_element(all_tables)
    report.add_section(step10)

    report.save(str(output / "report.html"))
    manifest["report_asset_mode"] = report_asset_mode
    write_manifest(
        output / "analysis_manifest.json",
        manifest,
        status="complete",
    )

    print(f"Saved nonlinear analysis → {output}")
    return {
        "manifest": manifest,
        "tables": tables,
        "figures": figures,
        "embeddings": trajectories,
        "unaligned": unaligned_trajectories,
        "aligned": aligned_trajectories,
        "velocity": velocity,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", type=int, default=list(range(1, 11)))
    parser.add_argument("--bids-root", type=Path, default=Path("PhysioNet_EEGBCI/BIDS"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/tutorial_eegbci_nonlinear")
    )
    parser.add_argument("--trials-per-cell", type=int, default=4)
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow-method", choices=METHODS, default="PHATE")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Load an existing BIDS conversion without running setup_data_bids.",
    )
    args = parser.parse_args(argv)
    run_nonlinear_analysis(
        subjects=args.subjects,
        bids_root=args.bids_root,
        output=args.output,
        trials_per_cell=args.trials_per_cell,
        time_stride=args.time_stride,
        seed=args.seed,
        flow_method=args.flow_method,
        prepare=not args.skip_prepare,
    )


if __name__ == "__main__":
    main()
