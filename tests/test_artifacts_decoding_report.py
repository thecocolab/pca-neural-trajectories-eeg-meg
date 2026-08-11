from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig, LogisticRegressionConfig
from coco_pipe.dim_reduction import DimReduction
from coco_pipe.io.structures import DataContainer
from coco_pipe.report import PlotlyElement, Report, Section
from coco_pipe.viz.interactive import plot_trajectory

from pca_neural_trajectories.artifacts import load_artifacts, save_artifacts


def test_artifact_round_trip(tmp_path):
    rng = np.random.default_rng(42)
    reducer = DimReduction("PCA", n_components=2, random_state=42)
    reducer.fit(rng.normal(size=(12, 4)))
    container = DataContainer(
        X=rng.normal(size=(4, 6, 2)),
        dims=("obs", "time", "component"),
        coords={
            "time": np.linspace(-0.2, 0.3, 6),
            "component": ["PC1", "PC2"],
            "subject": ["01", "01", "02", "02"],
        },
        y=np.array([3, 3, 4, 4]),
        ids=np.array(["trial-1", "trial-2", "trial-3", "trial-4"]),
    )
    scalar = pd.DataFrame({"metric": ["length"], "value": [1.25]})
    continuous = pd.DataFrame({"time": [0.0], "speed": [2.0]})
    pairs = pd.DataFrame({"pair": ["3_vs_4"], "value": [0.5]})

    save_artifacts(
        tmp_path,
        shared_reducer=reducer,
        trajectory_container=container,
        per_subject_reducers={"01": reducer},
        scalar_metrics=scalar,
        continuous_metrics=continuous,
        separation_pair_scalars=pairs,
    )
    restored = load_artifacts(tmp_path)

    assert restored["shared_reducer"].method == "PCA"
    assert restored["per_subject_reducers"].keys() == {"01"}
    assert np.allclose(restored["trajectory_container"].X, container.X)
    pd.testing.assert_frame_equal(restored["scalar_metrics"], scalar)


def test_load_artifacts_reports_all_missing_files(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="shared_reducer.pkl.*Re-run the tutorial"):
        load_artifacts(tmp_path)


def test_load_artifacts_rejects_empty_csv(tmp_path):
    rng = np.random.default_rng(42)
    reducer = DimReduction("PCA", n_components=2, random_state=42)
    reducer.fit(rng.normal(size=(12, 4)))
    container = DataContainer(
        X=rng.normal(size=(2, 5, 2)),
        dims=("obs", "time", "component"),
        coords={
            "time": np.linspace(-0.2, 0.2, 5),
            "component": ["PC1", "PC2"],
            "subject": ["01", "02"],
        },
        y=np.array([1, 2]),
        ids=np.array(["a", "b"]),
    )
    scalar = pd.DataFrame({"metric": ["length"], "value": [1.0]})
    continuous = pd.DataFrame({"time": [0.0], "speed": [1.0]})
    pairs = pd.DataFrame({"pair": ["1_vs_2"], "value": [1.0]})
    save_artifacts(tmp_path, reducer, container, {"01": reducer}, scalar, continuous, pairs)
    (tmp_path / "continuous_metrics.csv").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="continuous_metrics.csv.*Re-run the tutorial"):
        load_artifacts(tmp_path)


def test_save_artifacts_rejects_malformed_trajectory(tmp_path):
    rng = np.random.default_rng(42)
    reducer = DimReduction("PCA", n_components=2).fit(rng.normal(size=(12, 4)))
    stale = DataContainer(
        X=rng.normal(size=(2, 2, 5)),
        dims=("obs", "component", "time"),
        coords={"time": np.arange(5), "component": ["PC1", "PC2"]},
        y=np.array([1, 2]),
        ids=np.array(["a", "b"]),
    )
    table = pd.DataFrame({"metric": ["length"], "value": [1.0]})
    with pytest.raises(ValueError, match="dims"):
        save_artifacts(tmp_path, reducer, stale, {"01": reducer}, table, table, table)


def test_load_artifacts_rejects_malformed_saved_trajectory(tmp_path):
    rng = np.random.default_rng(42)
    reducer = DimReduction("PCA", n_components=2).fit(rng.normal(size=(12, 4)))
    valid = DataContainer(
        X=rng.normal(size=(2, 5, 2)),
        dims=("obs", "time", "component"),
        coords={
            "time": np.arange(5),
            "component": ["PC1", "PC2"],
            "subject": ["01", "02"],
        },
        y=np.array([1, 2]),
        ids=np.array(["a", "b"]),
    )
    scalar = pd.DataFrame({"metric": ["length"], "value": [1.0]})
    continuous = pd.DataFrame({"time": [0.0], "speed": [1.0]})
    pairs = pd.DataFrame({"pair": ["1_vs_2"], "value": [1.0]})
    save_artifacts(tmp_path, reducer, valid, {"01": reducer}, scalar, continuous, pairs)
    malformed = DataContainer(
        X=rng.normal(size=(2, 2, 5)),
        dims=("obs", "component", "time"),
        coords={"time": np.arange(5), "component": ["PC1", "PC2"]},
        y=np.array([1, 2]),
        ids=np.array(["a", "b"]),
    )
    malformed.save(tmp_path / "trajectory_pc_scores.pkl")

    with pytest.raises(ValueError, match="dims"):
        load_artifacts(tmp_path)


def test_group_aware_decoding_has_no_subject_overlap():
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(5), 4)
    y = np.tile([0, 1, 0, 1], 5)
    X = rng.normal(size=(20, 5))
    X[:, 0] += y * 1.5
    config = ExperimentConfig(
        task="classification",
        models={"logistic": LogisticRegressionConfig(max_iter=500)},
        metrics=["accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=5),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, groups=groups)

    assert result.raw["logistic"]["status"] == "success"
    for split in result.raw["logistic"]["splits"]:
        train_groups = set(groups[split["train_idx"]])
        test_groups = set(groups[split["test_idx"]])
        assert train_groups.isdisjoint(test_groups)


def test_plot_and_report_smoke(tmp_path):
    times = np.linspace(-0.2, 0.3, 6)
    trajectory = np.stack(
        [
            np.stack([times, times**2], axis=-1),
            np.stack([times + 0.2, times**2 + 0.1], axis=-1),
        ]
    )
    figure = plot_trajectory(
        X=trajectory,
        times=times,
        labels=np.array(["condition-a", "condition-b"]),
        dimensions=2,
        title="Synthetic trajectory",
    )
    report = Report(title="Trajectory smoke report")
    section = Section("Trajectory")
    section.add_element(PlotlyElement(figure))
    report.add_section(section)
    destination = tmp_path / "report.html"
    with pytest.warns(UserWarning, match="external CDN scripts"):
        report.save(destination)

    html = destination.read_text(encoding="utf-8")
    assert "Trajectory smoke report" in html
    assert "Trajectory" in html
    assert 'id="report-payload"' in html
