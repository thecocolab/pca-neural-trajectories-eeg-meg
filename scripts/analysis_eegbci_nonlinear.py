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
from coco_pipe.dim_reduction import DimReduction
from coco_pipe.dim_reduction.evaluation import MethodSelector, compute_velocity_fields
from coco_pipe.report import PlotlyElement, Report, Section
from coco_pipe.report.elements import (
    AccordionElement,
    CalloutElement,
    ContainerElement,
    InteractiveTableElement,
    MarkdownElement,
    StatCardElement,
    TabsElement,
)
from coco_pipe.transforms import TemporalProcrustesAlignment
from coco_pipe.viz.interactive import (
    plot_group_scatter_with_mean,
    plot_grouped_bar,
    plot_scatter,
    plot_streamlines,
    plot_trajectory,
)
from coco_pipe.viz.theme import set_coco_theme
from sklearn.metrics import silhouette_score

from pca_neural_trajectories import (
    LABEL_NAMES,
    facet_figures,
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

    quality_figure = facet_figures(
        {
            metric.capitalize(): plot_scatter(
                quality_records[quality_records["metric"] == metric].sort_values(
                    "scope_value"
                ),
                x="scope_value",
                y="value",
                color="method",
                mode="lines+markers",
                xaxis_title="neighborhood size k",
                yaxis_title="score",
            )
            for metric in ("trustworthiness", "continuity")
        },
        n_cols=2,
        title="Local geometry across neighborhood scales",
        row_height=440,
    )
    quality_figure.update_yaxes(range=[0, 1])

    silhouettes = embedding_diagnostics.melt(
        id_vars="method",
        value_vars=["subject_silhouette", "condition_silhouette"],
        var_name="grouping",
        value_name="silhouette",
    )
    silhouettes["grouping"] = silhouettes["grouping"].str.removesuffix("_silhouette")
    diagnostic_figure = plot_grouped_bar(
        silhouettes,
        x="method",
        y="silhouette",
        group="grouping",
        title="What structures each embedding: participant or condition?",
        yaxis_title="silhouette score",
        baseline=0.0,
    )

    condition_names = np.array([LABEL_NAMES[condition] for condition in CONDITIONS])
    condition_color_map = {
        LABEL_NAMES[condition]: CONDITION_COLORS[condition] for condition in CONDITIONS
    }
    condition_dash_map = {
        LABEL_NAMES[condition]: CONDITION_DASHES[condition] for condition in CONDITIONS
    }

    def _condition_mean_paths(source: np.ndarray) -> np.ndarray:
        """Stack condition-mean paths into plot_trajectory's (traj, time, dim)."""
        return np.stack(
            [source[trial_labels == condition].mean(axis=0)[:, :2] for condition in CONDITIONS]
        )

    trajectory_figure = facet_figures(
        {
            method: plot_trajectory(
                _condition_mean_paths(trajectories[method]),
                labels=condition_names,
                color_map=condition_color_map,
                linestyle_map=condition_dash_map,
                dimensions=2,
                show_markers=False,
                axis_labels=["dimension 1", "dimension 2"],
                title=method,
            )
            for method in METHODS
        },
        n_cols=2,
        title="Condition-mean trajectories: compare shape, not coordinate scale",
        row_height=420,
        shared_yaxes=False,
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
    velocity_figure = plot_streamlines(
        flow_embedding[:, :2],
        velocity[:, :2],
        title=f"Trial-safe descriptive flow in {flow_method} space",
        random_state=seed,
    )
    velocity_figure.update_xaxes(title_text="dimension 1")
    velocity_figure.update_yaxes(title_text="dimension 2")

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
    # One point per participant rather than a bare bar: the spread is the
    # result here, since alignment is claimed to make participants agree.
    alignment_representations = list(alignment_summary["representation"])
    alignment_figure = plot_group_scatter_with_mean(
        [
            alignment_consistency.loc[
                alignment_consistency["representation"] == representation, "correlation"
            ].to_numpy()
            for representation in alignment_representations
        ],
        alignment_representations,
        point_labels=[
            alignment_consistency.loc[
                alignment_consistency["representation"] == representation, "subject"
            ].to_numpy()
            for representation in alignment_representations
        ],
        title="Alignment sensitivity: leave-one-subject trajectory consistency",
        yaxis_title="correlation",
        baseline=0.0,
    )

    aligned_trajectory_figure = plot_trajectory(
        _condition_mean_paths(aligned_trajectories),
        labels=condition_names,
        color_map=condition_color_map,
        linestyle_map=condition_dash_map,
        dimensions=2,
        show_markers=False,
        axis_labels=["aligned PC1", "aligned PC2"],
        title="Condition trajectories after label-free temporal alignment",
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
    report.add_summary_card(
        {
            "Participants": len(complete_subjects),
            "Methods": "PCA, UMAP, PHATE, Isomap",
            "Samples per reducer": f"{len(samples):,}",
            "Neighbors": neighbor_count,
            "Velocity embedding": flow_method,
        }
    )

    overview = Section(
        "Overview",
        icon="O",
        description="PCA, UMAP, PHATE and Isomap on identical balanced observations",
        metadata={
            "Participants": f"{len(complete_subjects)} analyzable / {len(subjects)} requested",
            "Conditions": ", ".join(LABEL_NAMES[c] for c in CONDITIONS),
            "Trials per cell": str(selected_per_cell),
            "Sensor tensor": str(tuple(X.shape)),
            "Samples per reducer": f"{len(samples):,}",
            "Window": f"{times[0]:.3f}–{times[-1]:.3f} s",
            "Temporal stride": f"every {time_stride} sample(s)",
            "Neighborhood scales": ", ".join(map(str, valid_k)),
            "Velocity embedding": str(flow_method),
            "Alignment components": str(alignment_components),
        },
    )
    overview.add_element(
        MarkdownElement(
            "This companion report follows the complete **10-step nonlinear "
            "EEGBCI trajectory workflow** from the tutorial notebook. PCA, UMAP, "
            "PHATE, and Isomap are fit to exactly the same balanced sensor-time "
            "observations. Every visualization is paired with the diagnostic or "
            "assumption needed to interpret it."
        )
    )
    overview.add_element(
        CalloutElement(
            "Each numbered section corresponds directly to a numbered step in "
            "`tutorial_eegbci_nonlinear.ipynb`. The report and notebook use the "
            "same analysis choices and saved result tables.",
            kind="info",
            title="Mirrors the notebook",
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
            "asymmetric."
        )
    )
    step1.add_element(
        CalloutElement(
            "The trials, time samples, channels, crop, and normalization are fixed. "
            "Only the geometry-learning algorithm changes.",
            kind="info",
            title="Fair-comparison rule",
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
            "every participant × condition cell before temporal decimation."
        )
    )
    step2.add_element(
        CalloutElement(
            "Flattening first and sampling isolated time points would distort temporal "
            "coverage, fragment trials, and invalidate the velocity analysis.",
            kind="info",
            title="Why whole trials",
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
            "`reducers/`."
        )
    )
    step4.add_element(
        CalloutElement(
            "A distance of 1 in PHATE is not equivalent to a distance of 1 in PCA or "
            "UMAP. Cross-method comparisons must use dimensionless validation metrics "
            "rather than raw embedded distances.",
            kind="warning",
            title="No common coordinate ruler",
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
            "depends on neighborhood size."
        )
    )
    step5.add_element(
        CalloutElement(
            "High trustworthiness or continuity supports geometric fidelity. It does "
            "not establish a biological manifold, an attractor, predictive condition "
            "information, or generalization to unseen participants.",
            kind="warning",
            title="What the scores do not show",
        )
    )
    best_geometry = quality_summary.loc[
        quality_summary.groupby("metric")["value"].idxmax()
    ]
    step5.add_columns(
        [
            StatCardElement(
                f"Best {row['metric']}",
                f"{row['value']:.3f}",
                delta=str(row["method"]),
                color="green" if row["value"] > 0.85 else "yellow",
            )
            for _, row in best_geometry.iterrows()
        ]
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
            "many samples are closer to another group."
        )
    )
    step6.add_element(
        CalloutElement(
            "Repeated time samples from one trial are dependent. These silhouettes are "
            "not p-values, independent-sample effect estimates, or cross-validated "
            "decoding scores.",
            kind="warning",
            title="Descriptive, not inferential",
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
            "with dashed lines, and color identifies left versus right hand."
        )
    )
    step7.add_element(
        CalloutElement(
            "Within a panel, inspect branching, loops, return paths, and relative "
            "condition separation. Across panels, compare qualitative topology only. "
            "PCA signs can flip, while nonlinear spaces can rotate, reflect, warp, or "
            "rescale without violating their intended geometry.",
            kind="tip",
            title="What can be compared",
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
            "endpoints naturally have no forward difference."
        )
    )
    step8.add_element(
        CalloutElement(
            "Arrow lengths inherit the arbitrary scale of the selected embedding. The "
            "field does not estimate a differential equation, demonstrate an attractor, "
            "or support raw speed comparisons across reducers.",
            kind="warning",
            title="Descriptive flow, not a fitted dynamical system",
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
            "generators."
        )
    )
    step9.add_element(
        CalloutElement(
            "All displayed participants contribute to this descriptive fit. Prediction "
            "for unseen participants requires a training-only reference and label-free "
            "test-participant calibration, as implemented in the decoding tutorial.",
            kind="warning",
            title="Transductive-analysis warning",
        )
    )
    consistency_panel = ContainerElement()
    consistency_panel.add_element(PlotlyElement(alignment_figure, height="500px"))
    consistency_panel.add_element(
        InteractiveTableElement(
            alignment_summary,
            title="Cross-participant consistency before and after alignment",
            selector_columns=["representation"],
        )
    )
    step9.add_element(
        TabsElement(
            {
                "Consistency before/after": consistency_panel,
                "Aligned trajectories": PlotlyElement(
                    aligned_trajectory_figure, height="620px"
                ),
            }
        )
    )
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
            "6. alignment conclusions remain within their transductive scope."
        )
    )
    step10.add_element(
        CalloutElement(
            "Nonlinear embeddings are complementary views of sensor-space geometry. "
            "Prefer conclusions that remain coherent across geometry validation, "
            "participant diagnostics, temporal reconstruction, and alignment "
            "sensitivity — not conclusions that depend on one visually striking panel.",
            kind="tip",
            title="Main takeaway",
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
