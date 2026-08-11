"""Save/load the bundle the main tutorial produces for the companion notebooks.

The main tutorial fits the shared and per-subject PCA reducers, builds the
trajectory container, and computes the metric tables. Companion analyses can
load those outputs instead of recomputing them, so this module is just the
write/read bridge between notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from coco_pipe.dim_reduction import DimReduction
from coco_pipe.io.structures import DataContainer

_CORE_FILES = (
    "shared_reducer.pkl",
    "trajectory_pc_scores.pkl",
    "scalar_metrics.csv",
    "continuous_metrics.csv",
    "separation_pair_scalars.csv",
)


def _producer_hint(src: Path) -> str:
    """Return an actionable, modality-neutral producer hint."""
    return (
        f"Re-run the tutorial that produces {src} from its first cell, then "
        "re-run this consumer notebook."
    )


def _read_required_csv(src: Path, name: str) -> pd.DataFrame:
    path = src / name
    if path.stat().st_size == 0:
        raise ValueError(f"Required artifact {path} is empty. {_producer_hint(src)}")
    try:
        table = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Required artifact {path} is not a readable CSV. {_producer_hint(src)}"
        ) from exc
    if table.empty or len(table.columns) == 0:
        raise ValueError(f"Required artifact {path} has no data rows. {_producer_hint(src)}")
    return table


def _validate_trajectory_container(container: DataContainer, src: Path) -> None:
    expected_dims = ("obs", "time", "component")
    if container.dims != expected_dims:
        raise ValueError(
            "Trajectory artifact must use dims "
            f"{expected_dims}, got {container.dims}. {_producer_hint(src)}"
        )
    if container.X.ndim != 3:
        raise ValueError(
            "Trajectory artifact must be a three-dimensional obs × time × component "
            f"array. {_producer_hint(src)}"
        )
    if min(container.X.shape) < 1:
        raise ValueError(f"Trajectory artifact contains an empty axis. {_producer_hint(src)}")


def save_artifacts(
    artifacts_dir: Path | str,
    shared_reducer: DimReduction,
    trajectory_container: DataContainer,
    per_subject_reducers: dict[Any, DimReduction],
    scalar_metrics: pd.DataFrame,
    continuous_metrics: pd.DataFrame,
    separation_pair_scalars: pd.DataFrame,
    extra_csvs: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Persist all main-tutorial outputs to ``artifacts_dir``.

    Writes the shared reducer, the trajectory container, one reducer per
    subject (``reducer_sub-<id>.pkl``), the three metric tables, and any
    ``extra_csvs``.
    """
    out = Path(artifacts_dir)
    _validate_trajectory_container(trajectory_container, out)
    required_tables = (scalar_metrics, continuous_metrics, separation_pair_scalars)
    if any(table.empty for table in required_tables):
        raise ValueError("Cannot save an empty tutorial artifact table.")
    if not per_subject_reducers:
        raise ValueError("Cannot save artifacts without at least one per-subject reducer.")
    out.mkdir(parents=True, exist_ok=True)

    shared_reducer.save(out / "shared_reducer.pkl")
    trajectory_container.save(out / "trajectory_pc_scores.pkl")
    for subject, reducer in per_subject_reducers.items():
        reducer.save(out / f"reducer_sub-{subject}.pkl")

    scalar_metrics.to_csv(out / "scalar_metrics.csv", index=False)
    continuous_metrics.to_csv(out / "continuous_metrics.csv", index=False)
    separation_pair_scalars.to_csv(out / "separation_pair_scalars.csv", index=False)
    for name, table in (extra_csvs or {}).items():
        table.to_csv(out / f"{name}.csv", index=False)

    print(f"Saved artifacts → {out}")
    return out


def load_artifacts(artifacts_dir: Path | str, method: str = "PCA") -> dict[str, Any]:
    """Inverse of :func:`save_artifacts`.

    Raises a clear ``FileNotFoundError`` if the artifacts directory is missing
    so the companion notebooks can tell the reader which notebook to run first.
    """
    src = Path(artifacts_dir)
    if not src.is_dir():
        raise FileNotFoundError(
            f"Artifacts directory {src} not found. {_producer_hint(src)}"
        )

    missing = [name for name in _CORE_FILES if not (src / name).is_file()]
    reducer_paths = sorted(src.glob("reducer_sub-*.pkl"))
    if not reducer_paths:
        missing.append("reducer_sub-<subject>.pkl")
    if missing:
        raise FileNotFoundError(
            f"Artifact bundle {src} is incomplete; missing: {', '.join(missing)}. "
            f"{_producer_hint(src)}"
        )

    trajectory_container = DataContainer.load(src / "trajectory_pc_scores.pkl")
    _validate_trajectory_container(trajectory_container, src)

    per_subject_reducers = {
        path.stem.removeprefix("reducer_sub-"): DimReduction.load(path, method=method)
        for path in reducer_paths
    }
    return {
        "shared_reducer": DimReduction.load(src / "shared_reducer.pkl", method=method),
        "trajectory_container": trajectory_container,
        "per_subject_reducers": per_subject_reducers,
        "scalar_metrics": _read_required_csv(src, "scalar_metrics.csv"),
        "continuous_metrics": _read_required_csv(src, "continuous_metrics.csv"),
        "separation_pair_scalars": _read_required_csv(
            src, "separation_pair_scalars.csv"
        ),
    }


__all__ = ["load_artifacts", "save_artifacts"]
