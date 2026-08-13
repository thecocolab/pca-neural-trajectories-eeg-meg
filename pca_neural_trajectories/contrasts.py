"""Multi-contrast decoding sweeps: specs, slugs, and figure faceting.

The decoding scripts run one classification target per call. A *sweep* runs
several targets — the four contrasts `FINDINGS.md` reports for each dataset —
and reports them side by side, so a claim like "Famous and Unfamiliar behave
identically against Scrambled" is one glance rather than four files.

Every sweep figure is a grid of the per-contrast figures the single-target run
already builds; :func:`pca_neural_trajectories.viz.facet_figures` does that
composition.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas is only needed for the metric helper's annotations
    import pandas as pd

__all__ = [
    "contrast_fold_metrics",
    "contrast_label",
    "contrast_slug",
    "parse_contrast",
]


def contrast_fold_metrics(
    curves: pd.DataFrame,
    folds: pd.DataFrame,
    chance: float,
    *,
    window: tuple[float, float],
    prefix: str,
    representations: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-fold values behind each reported summary, for one contrast.

    The peak is read off the **fold-averaged** curve, and every fold is then
    sampled at that one latency, so the per-fold values average exactly to the
    reported peak. Taking each fold's own maximum instead selects a different
    latency per fold and biases the mean upward — the maximum of noisy
    estimates is not an estimate of the maximum.

    The two window summaries are averages over a fixed interval, so averaging
    over folds first or over time first gives the same number; the fold level
    is kept for the spread the figures draw.

    Parameters
    ----------
    curves
        Fold-averaged time courses, with ``Representation``/``Time``/``Mean``.
    folds
        Fold-level scores, with ``Representation``/``Fold``/``Metric``/
        ``Time``/``Value``.
    chance
        Chance level for this contrast, subtracted before integrating.
    window
        ``(start, stop)`` of the active interval, in seconds.
    prefix
        Column-name prefix for the two window summaries (e.g. ``"postcue"``).
    representations
        Restrict to these representations, in this order. Defaults to all.
    """
    import numpy as np
    import pandas as pd

    rows = folds[(folds["Metric"] == "balanced_accuracy") & folds["Time"].notna()]
    if representations is not None:
        rows = rows[rows["Representation"].isin(representations)]
        curves = curves[curves["Representation"].isin(representations)]

    records = []
    for representation, group in rows.groupby("Representation"):
        curve = curves[curves["Representation"] == representation]
        peak_time = float(curve.loc[curve["Mean"].idxmax(), "Time"])
        for fold, fold_rows in group.groupby("Fold"):
            fold_rows = fold_rows.sort_values("Time")
            at_peak = fold_rows.loc[
                (fold_rows["Time"] - peak_time).abs().idxmin(), "Value"
            ]
            active = fold_rows[
                (fold_rows["Time"] >= window[0]) & (fold_rows["Time"] <= window[1])
            ]
            records.append(
                {
                    "Representation": representation,
                    "Fold": int(fold),
                    "peak_time_s": peak_time,
                    "peak_balanced_accuracy": float(at_peak),
                    f"{prefix}_mean_balanced_accuracy": float(active["Value"].mean()),
                    f"{prefix}_auc_above_chance": float(
                        np.trapezoid(active["Value"] - chance, active["Time"])
                    ),
                }
            )
    return pd.DataFrame(records)


def parse_contrast(spec: str | Sequence[int]) -> tuple[int, ...]:
    """Parse one contrast spec into an ordered tuple of condition ids.

    Accepts ``"3-4"``, ``"3,4"``, ``"3 4"``, or an already-parsed sequence.
    Class index ``i`` in the decoding target corresponds to element ``i``, so
    order is preserved.
    """
    if not isinstance(spec, str):
        conditions = tuple(int(value) for value in spec)
    else:
        parts = [part for part in re.split(r"[-,\s]+", spec.strip()) if part]
        if not parts:
            raise ValueError(f"Empty contrast spec: {spec!r}.")
        try:
            conditions = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError(
                f"Contrast spec {spec!r} is not a list of condition ids "
                "(expected e.g. '3-4' or '1-2-3')."
            ) from error
    if len(set(conditions)) < 2:
        raise ValueError(
            f"Contrast {spec!r} must name at least two distinct condition ids."
        )
    return conditions


def contrast_label(
    conditions: Sequence[int],
    label_names: Mapping[int, str],
    *,
    short: bool = False,
) -> str:
    """Human-readable contrast name, e.g. ``"Famous vs Scrambled"``.

    Three or more conditions are labelled as multiclass, matching how
    `FINDINGS.md` reports the three-class row alongside the binaries. Panel
    titles pass ``short=True`` to get the bare ``"3-class"`` rather than the
    full class list, which does not fit above a subplot.
    """
    names = [label_names[condition_id] for condition_id in conditions]
    if len(names) == 2:
        return " vs ".join(names)
    if short:
        return f"{len(names)}-class"
    return f"{len(names)}-class: {' / '.join(names)}"


def contrast_slug(conditions: Sequence[int], label_names: Mapping[int, str]) -> str:
    """Filesystem-safe directory name for one contrast's outputs.

    Binary contrasts read as names (``famous_vs_scrambled``); multiclass ones
    fall back to condition ids, since spelling every class out makes the
    directory name unusable.
    """
    if len(conditions) > 2:
        return f"{len(conditions)}class_" + "_".join(str(value) for value in conditions)
    slug = "_vs_".join(label_names[condition_id] for condition_id in conditions)
    return re.sub(r"[^0-9a-zA-Z]+", "_", slug).strip("_").lower()
