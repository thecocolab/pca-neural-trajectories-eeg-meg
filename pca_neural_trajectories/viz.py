"""Composing coco-pipe figures: subplot grids and layered overlays.

Both helpers work by copying traces between figures rather than reimplementing
any plot, so every mark still comes from a coco-pipe plotting function.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = ["facet_figures", "overlay_figures"]


def _axis_title(axis: object) -> str | None:
    """Read an axis title through Plotly's nested title object, if set."""
    title = getattr(axis, "title", None)
    return getattr(title, "text", None)


def facet_figures(
    figures: Mapping[str, go.Figure],
    *,
    n_cols: int = 2,
    title: str | None = None,
    row_height: int = 380,
    shared_yaxes: bool = True,
) -> go.Figure:
    """Lay per-contrast figures out as one subplot grid.

    Traces are deep-copied into the grid in source order, so overlapping
    mean/SEM band pairs keep their fill relationships. Legend entries are
    de-duplicated across panels by name: the first panel carries the visible
    legend and every later copy joins its ``legendgroup``, so toggling a
    representation toggles it in all panels at once.

    Reference lines added with ``add_hline`` / ``add_vline`` are re-added per
    panel rather than copied, since their original axis references do not
    survive the move into a grid.
    """
    if not figures:
        raise ValueError("facet_figures needs at least one figure.")

    panels = list(figures)
    n_cols = max(1, min(n_cols, len(panels)))
    n_rows = math.ceil(len(panels) / n_cols)
    grid = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=panels,
        shared_yaxes=shared_yaxes,
        horizontal_spacing=0.09,
        vertical_spacing=0.14 if n_rows > 1 else 0.0,
    )

    legend_seen: set[str] = set()
    for index, panel in enumerate(panels):
        row, col = divmod(index, n_cols)
        row, col = row + 1, col + 1
        source = figures[panel]

        for trace in source.data:
            copied = copy.deepcopy(trace)
            name = getattr(copied, "name", None)
            if name:
                copied.legendgroup = name
                copied.showlegend = name not in legend_seen
                legend_seen.add(name)
            else:
                copied.showlegend = False
            grid.add_trace(copied, row=row, col=col)

        # Reference lines are stored as layout shapes bound to the source
        # figure's single axis pair; re-add them against this panel's axes.
        for shape in source.layout.shapes or ():
            spec = {
                key: value
                for key, value in shape.to_plotly_json().items()
                if key not in {"xref", "yref"}
            }
            if spec.get("x0") == spec.get("x1"):
                grid.add_vline(x=spec["x0"], line=spec.get("line"), row=row, col=col)
            elif spec.get("y0") == spec.get("y1"):
                grid.add_hline(y=spec["y0"], line=spec.get("line"), row=row, col=col)

        # Categorical panels (one tick per group) carry their labels as an
        # explicit tick array; without it the copied traces would fall back to
        # the numeric positions the labels stand in for.
        x_axis = source.layout.xaxis
        ticks = {
            key: getattr(x_axis, key)
            for key in ("tickmode", "tickvals", "ticktext")
            if getattr(x_axis, key, None) is not None
        }
        grid.update_xaxes(
            title_text=_axis_title(x_axis), row=row, col=col, **ticks
        )
        if col == 1 or not shared_yaxes:
            grid.update_yaxes(
                title_text=_axis_title(source.layout.yaxis), row=row, col=col
            )

    grid.update_layout(
        title=title,
        height=row_height * n_rows + 140,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.12},
        margin={"t": 110},
    )
    return grid


def overlay_figures(
    figures: Sequence[go.Figure],
    *,
    title: str | None = None,
    legend_from: int = -1,
    opacities: Sequence[float | None] | None = None,
) -> go.Figure:
    """Layer several figures into one, in drawing order.

    Lets a figure combine plots that no single coco-pipe helper produces — for
    example per-participant lines from one call underneath a group mean and SEM
    band from another — without hand-building traces.

    Parameters
    ----------
    figures
        Figures to stack. The first is drawn first (furthest back).
    title
        Title for the combined figure. Defaults to the last figure's title.
    legend_from
        Index of the figure whose traces keep their legend entries; every other
        layer is silenced so the legend describes one thing. Pass ``None`` to
        keep whatever each source figure declared.
    opacities
        Optional per-figure opacity, aligned with *figures*. ``None`` entries
        leave a layer untouched — useful for fading a dense background layer.

    Returns
    -------
    plotly.graph_objects.Figure
        The combined figure, taking its layout from the last input.
    """
    if not figures:
        raise ValueError("overlay_figures needs at least one figure.")

    layers = list(figures)
    if legend_from is not None:
        legend_from %= len(layers)

    combined = go.Figure()
    for index, source in enumerate(layers):
        opacity = None if opacities is None else opacities[index]
        for trace in source.data:
            copied = copy.deepcopy(trace)
            if legend_from is not None:
                # An unset showlegend is None, which Plotly renders as visible;
                # only an explicit False means the source hid this trace.
                declared = getattr(copied, "showlegend", None)
                copied.showlegend = (declared is not False) and index == legend_from
            if opacity is not None:
                copied.opacity = opacity
            combined.add_trace(copied)

    # The last layer owns the axes: it is the one drawn against the reader.
    combined.update_layout(layers[-1].layout)
    combined.update_layout(title=title or layers[-1].layout.title.text)
    for source in layers:
        for shape in source.layout.shapes or ():
            combined.add_shape(shape)
    return combined
