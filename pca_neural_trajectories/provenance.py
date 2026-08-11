"""Reproducibility manifests for long-running analyses."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _coco_pipe_root() -> Path | None:
    """Locate the installed ``coco_pipe`` source, or ``None`` if unavailable.

    Resolves the actually-installed package (the editable ``dev`` checkout on a
    developer machine) so the manifest records its commit wherever it lives,
    without a hardcoded path. Returns ``None`` for a released wheel, where the
    ``coco-pipe`` version string carries identity instead.
    """
    spec = importlib.util.find_spec("coco_pipe")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent.parent


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def git_commit(path: str | Path | None) -> str | None:
    """Return ``HEAD`` for a Git checkout, or ``None`` when unavailable."""
    if path is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def distribution_version(name: str) -> str | None:
    """Return an installed distribution version without importing it."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_manifest(
    config: Mapping[str, Any],
    *,
    status: str = "configured",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared environment and analysis manifest.

    ``config`` is the analysis settings to record, typically the parsed
    command-line arguments.
    """
    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "config": dict(config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: distribution_version(name)
                for name in (
                    "coco-pipe",
                    "pca-neural-trajectories",
                    "mne",
                    "mne-bids",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                )
            },
            "coco_pipe_commit": git_commit(_coco_pipe_root()),
            "project_commit": git_commit(Path(__file__).resolve().parents[1]),
        },
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_manifest(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    status: str = "configured",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a deterministic, human-readable JSON manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            build_manifest(config, status=status, extra=extra),
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "build_manifest",
    "distribution_version",
    "git_commit",
    "write_manifest",
]
