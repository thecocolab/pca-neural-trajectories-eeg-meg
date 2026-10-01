#!/usr/bin/env python3
"""Build the results website from the analysis output tree.

Every ``report.html`` under ``--source`` is copied, unchanged, into
``website/reports/``. Two pages are written around them:

- ``index.html``, the landing page, from ``template.html``
- ``view.html``, which shows one report at a time under a navigation bar, from
  ``viewer.html``

The small plots on the landing page are redrawn from the figures stored inside
the reports, so the reports are the only input.

    python website/build.py                    # incremental copy + regenerate pages
    python website/build.py --clean            # drop reports/ first
    python website/build.py --source outputs/release

What appears where is the configuration below. A report that is not there yet is
left out; a report the configuration does not know about still gets a link,
named after its folder.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import re
import shutil
from datetime import date
from functools import cache
from pathlib import Path

# --- configuration -------------------------------------------------------

# Source directory name -> (dataset key, group key).
ANALYSES = {
    "eegbci_main": ("eegbci", "main"),
    "eegbci_decoding": ("eegbci", "decoding"),
    "eegbci_nonlinear": ("eegbci", "nonlinear"),
    "megfaces_main": ("megfaces", "main"),
    "megfaces_decoding": ("megfaces", "decoding"),
    "megfaces_spectral_envelopes": ("megfaces", "spectral"),
}

# Sensor selections, in the order the picker shows them. The first is the default.
SENSOR_SETS = {
    "sensors_right_occipital": "Right occipital",
    "sensors_right_occipito_temporal": "Right occipito-temporal",
    "sensors_right_temporal": "Right temporal",
    "sensors_occipital": "Occipital",
    "sensors_temporal": "Temporal",
    "sensors_occipito_temporal": "Occipito-temporal",
    "all_sensors": "All sensors",
}

# Names for the folders a report can sit in. "" is the analysis root.
LABELS = {
    "": "Full report",
    "best_models": "Best model for each representation",
    "4class_3_4_5_6": "All four conditions",
    "hands_exec_vs_hands_imag": "Executed vs imagined",
    "left_hand_exec_vs_right_hand_exec": "Left vs right, executed",
    "left_hand_imag_vs_right_hand_imag": "Left vs right, imagined",
    "3class_1_2_3": "All three image types",
    "famous_vs_scrambled": "Famous vs scrambled",
    "famous_vs_unfamiliar": "Famous vs unfamiliar",
    "unfamiliar_vs_scrambled": "Unfamiliar vs scrambled",
}
# The root report of a decoding sweep covers every contrast at once.
ROOT_LABELS = {"decoding": "All contrasts together"}

# Reports are listed in this order; anything else follows alphabetically.
ORDER = ["", "best_models", "4class_3_4_5_6", "3class_1_2_3"]

# The results shown in the paper, listed first on the landing page:
# (figure in the paper, heading, sentence, dataset, group, folders of the report,
#  title of the figure inside that report to redraw, what that plot shows)
PAPER = [
    ("Figure 2", "Motor execution and imagery in EEG",
     "Left and right hand, executed and imagined, in 106 people. From regional ERPs to a "
     "shared PCA space, the trajectories, how far apart the conditions are and how fast "
     "the state moves.",
     "eegbci", "main", (),
     "Step 7 — Subject-mean ERP trajectories (PC1–PC2–PC3)", "Trajectories, PC 1 against PC 2"),
    ("Figure 3a–b", "Face perception in MEG",
     "Famous, unfamiliar and scrambled faces in 16 people, on 36 right-occipital sensors. "
     "The sensor-level response, then the trajectories in a shared PCA space.",
     "megfaces", "main", ("sensors_right_occipital",),
     "Equal-participant mean trajectories: PC1-PC2", "Trajectories, PC 1 against PC 2"),
    ("Figure 3c–d", "The same task, seen through alpha power",
     "Alpha-band power (8 to 12 Hz) as the neural state, in its own PCA space. The report "
     "also covers beta and low gamma.",
     "megfaces", "spectral", ("sensors_right_occipital",),
     "alpha: equal-participant PC1-PC2-PC3 trajectories", "Alpha power, PC 1 against PC 2"),
    ("Figure 3e–f", "Decoding across participants",
     "Leave-one-participant-out decoding of the image type from sensors, shared PCA, "
     "trajectory descriptors and aligned PCA.",
     "megfaces", "decoding", ("best_models",),
     "3-class: Famous vs. Unfamiliar vs. Scrambled", "Balanced accuracy over time, three classes"),
]

# Each analysis in the full list: its group, a heading ("short" is the name used in
# the viewer's bar) and a sentence.
DATASETS = [
    {
        "key": "eegbci",
        "short": "EEG",
        "name": "EEG: moving a hand, or imagining it",
        "about": (
            "The PhysioNet EEG Motor Movement/Imagery dataset. 64-channel EEG from 106 people "
            "who opened and closed their left or right hand, or imagined doing so."
        ),
        "rows": [
            {"group": "main", "heading": "Trajectories",
             "text": "The main analysis. One PCA space for all four conditions, how much "
                     "variance it holds, and where the paths separate."},
            {"group": "nonlinear", "heading": "PCA next to nonlinear methods", "short": "Nonlinear",
             "text": "The same data through UMAP, PHATE and Isomap, to see what PCA keeps and "
                     "what it misses."},
            {"group": "decoding", "heading": "Decoding across participants", "short": "Decoding",
             "text": "Train on some people, test on someone new. Raw sensors compared with PCA "
                     "components aligned between participants."},
        ],
    },
    {
        "key": "megfaces",
        "short": "MEG",
        "name": "MEG: looking at faces",
        "about": (
            "The Wakeman and Henson dataset (OpenNeuro ds000117). 306-channel MEG from 16 "
            "people who looked at famous faces, unfamiliar faces and scrambled images."
        ),
        "picker_note": (
            "The paper uses the right-occipital sensors. These are groups of sensors by their "
            "position on the helmet. They are not source-localised brain regions."
        ),
        "rows": [
            {"group": "main", "heading": "Trajectories",
             "text": "The three image types in one shared PCA space, with the faces vs "
                     "scrambled and famous vs unfamiliar contrasts."},
            {"group": "spectral", "heading": "Frequency bands",
             "text": "The same analysis on alpha, beta and low-gamma (30 to 45 Hz) power "
                     "instead of the broadband signal."},
            {"group": "decoding", "heading": "Decoding across participants", "short": "Decoding",
             "text": "Train on 15 people, test on the 16th. Raw sensors compared with shared "
                     "and aligned PCA components."},
        ],
    },
]

# The figure you can play at the top of the landing page: PC scores over time, read
# from one report. (dataset, group, folders, title of the figure in it, caption)
HERO = (
    "megfaces", "main", ("sensors_right_occipital",), "Principal-component time courses",
    "The MEG response to famous faces, unfamiliar faces and scrambled images, averaged over "
    "16 people. These are the trajectories of Figure 3b, seen on the first two principal "
    "components. Drag the slider to move through time.",
)

PAYLOAD_RE = re.compile(r'<script type="application/json" id="report-payload">(.*?)</script>',
                        re.DOTALL)
# Trimmed from trace names in the small legends.
NAME_NOISE_RE = re.compile(r" / \w+$| \(peak [^)]*\)| \(raw[^)]*\)")


# --- finding and copying reports ------------------------------------------

def label_for(folders: tuple[str, ...], group: str) -> str:
    """Name of a report, from the folders it sits in (sensor folder removed)."""
    if not folders:
        return ROOT_LABELS.get(group, LABELS[""])
    return ", ".join(LABELS.get(f, f.replace("_", " ").capitalize()) for f in folders)


def order_key(folders: tuple[str, ...]) -> tuple:
    first = folders[0] if folders else ""
    return (ORDER.index(first) if first in ORDER else len(ORDER), folders)


def collect(source: Path) -> dict[tuple[str, str], list[dict]]:
    """Map (dataset, group) -> report records found under *source*."""
    found: dict[tuple[str, str], list[dict]] = {}
    for analysis, (dataset, group) in ANALYSES.items():
        analysis_dir = source / analysis
        if not analysis_dir.is_dir():
            continue
        for report in sorted(analysis_dir.rglob("report.html")):
            segments = report.relative_to(analysis_dir).parts[:-1]
            name = "/".join(segments) if segments else "overview"
            sensors = [s for s in segments if s in SENSOR_SETS]
            found.setdefault((dataset, group), []).append({
                "source": report,
                "dest": Path("reports") / dataset / group / f"{name}.html",
                # How the viewer addresses this report: view.html#<route>
                "route": f"{dataset}/{group}/{name}",
                "segments": tuple(segments),
                "folders": tuple(s for s in segments if s not in SENSOR_SETS),
                "sensor": sensors[0] if sensors else None,
            })
    return found


def copy_reports(records: list[dict], dest_root: Path) -> int:
    """Copy reports that are new or changed, byte for byte. Returns how many."""
    copied = 0
    for record in records:
        target = dest_root / record["dest"]
        source = record["source"].stat()
        stale = (not target.exists()
                 or target.stat().st_size != source.st_size
                 or target.stat().st_mtime < source.st_mtime)
        if stale:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["source"], target)
            copied += 1
    return copied


def sensors_of(found: dict[tuple[str, str], list[dict]], dataset: str) -> list[str]:
    present = {r["sensor"] for (ds, _), records in found.items() if ds == dataset
               for r in records if r["sensor"]}
    return [s for s in SENSOR_SETS if s in present]


def rows_of(found: dict[tuple[str, str], list[dict]], dataset: dict) -> list[dict]:
    """Configured analyses that have reports, then any group the configuration lacks."""
    key = dataset["key"]
    rows = list(dataset["rows"])
    known = {row["group"] for row in rows}
    rows += [{"group": group, "heading": group.replace("_", " ").capitalize()}
             for (ds, group) in sorted(found) if ds == key and group not in known]
    return [row for row in rows if found.get((key, row["group"]))]


def variants_of(records: list[dict]) -> list[tuple[tuple[str, ...], dict[str, str]]]:
    """Reports of one analysis as (folders, {sensor or "": route}), in display order."""
    by_folders: dict[tuple[str, ...], dict[str, str]] = {}
    for record in records:
        by_folders.setdefault(record["folders"], {})[record["sensor"] or ""] = record["route"]
    return [(folders, by_folders[folders]) for folders in sorted(by_folders, key=order_key)]


# --- reading figures out of a report ---------------------------------------

@cache
def report_figures(report: Path) -> dict[str, dict]:
    """The Plotly figures stored inside a report, by title."""
    match = PAYLOAD_RE.search(report.read_text(encoding="utf-8"))
    if not match:
        return {}
    payload = json.loads(gzip.decompress(base64.b64decode(match.group(1).strip())))
    figures: dict[str, dict] = {}
    for figure in payload.values():
        if isinstance(figure, dict) and "layout" in figure and "data" in figure:
            title = figure["layout"].get("title")
            title = title.get("text") if isinstance(title, dict) else title
            figures.setdefault(title, figure)
    return figures


def named_lines(figure: dict, axis: str = "x") -> list[dict]:
    """The labelled line traces of one panel, skipping bands and markers."""
    lines = []
    for trace in figure.get("data", []):
        x, y = trace.get("x"), trace.get("y")
        if (trace.get("name") and "lines" in (trace.get("mode") or "")
                and trace.get("xaxis", "x") == axis
                and isinstance(x, list) and isinstance(y, list) and len(x) > 3):
            lines.append(trace)
    return lines


def colour_of(trace: dict) -> str:
    colour = (trace.get("line") or {}).get("color") or "currentColor"
    return "var(--ink)" if colour.lower() in ("#000000", "#000", "black") else html.escape(colour)


def is_dashed(trace: dict) -> bool:
    return (trace.get("line") or {}).get("dash") not in (None, "solid")


class Frame:
    """Maps data coordinates into an SVG box."""

    def __init__(self, xs: list[float], ys: list[float], width: int, height: int, pad: int,
                 equal: bool) -> None:
        self.height, self.x0, self.y0 = height, min(xs), min(ys)
        span_x = (max(xs) - self.x0) or 1.0
        span_y = (max(ys) - self.y0) or 1.0
        self.sx, self.sy = (width - 2 * pad) / span_x, (height - 2 * pad) / span_y
        if equal:
            self.sx = self.sy = min(self.sx, self.sy)
        self.left = (width - self.sx * span_x) / 2
        self.bottom = (height - self.sy * span_y) / 2

    def x(self, value: float) -> float:
        return self.left + (value - self.x0) * self.sx

    def y(self, value: float) -> float:
        return self.height - self.bottom - (value - self.y0) * self.sy

    def point(self, x: float, y: float) -> str:
        return f"{self.x(x):.1f} {self.y(y):.1f}"


def small_plot(figure: dict) -> tuple[str, str]:
    """Redraw a report figure as a small SVG. Returns (svg, legend items)."""
    traces = named_lines(figure)
    if not traces:
        return "", ""
    width, height, pad = 320, 200, 14
    xs = [x for t in traces for x in t["x"]]
    ys = [y for t in traces for y in t["y"]]
    # A trace whose x only goes up is a time series; anything else is a trajectory.
    series = all(a <= b for t in traces for a, b in zip(t["x"], t["x"][1:]))
    frame = Frame(xs, ys, width, height, pad, equal=False)

    parts = []
    if series:
        # Reference lines: horizontal ones the report draws (chance level), and time zero.
        for shape in figure["layout"].get("shapes", []):
            level = shape.get("y0")
            if (shape.get("type") == "line" and isinstance(level, (int, float))
                    and level == shape.get("y1") and min(ys) < level < max(ys)):
                parts.append(f'<path class="ref" d="M{pad} {frame.y(level):.1f} H{width - pad}"/>')
        if min(xs) < 0 < max(xs):
            parts.append(f'<path class="ref" d="M{frame.x(0):.1f} {pad} V{height - pad}"/>')

    legend = []
    for trace in traces:
        points = " L".join(frame.point(x, y) for x, y in zip(trace["x"], trace["y"]))
        dashed = ' class="dashed"' if is_dashed(trace) else ""
        parts.append(f'<path{dashed} style="--c:{colour_of(trace)}" d="M{points}"/>')
        name = html.escape(NAME_NOISE_RE.sub("", trace["name"]))
        legend.append(f'<li{dashed} style="--c:{colour_of(trace)}">{name}</li>')

    svg = (f'<svg class="plot" viewBox="0 0 {width} {height}" aria-hidden="true">'
           + "".join(parts) + "</svg>")
    return svg, "".join(legend)


# --- landing page ----------------------------------------------------------

def render_hero(found: dict[tuple[str, str], list[dict]]) -> str:
    """The trajectory you can play: PC1 against PC2 over time. Empty if unavailable."""
    dataset, group, segments, title, caption = HERO
    match = [r for r in found.get((dataset, group), []) if r["segments"] == segments]
    figure = report_figures(match[0]["source"]).get(title) if match else None
    if not figure:
        return ""
    pc1 = {t["name"]: t for t in named_lines(figure, "x")}
    pc2 = {t["name"]: t for t in named_lines(figure, "x2")}
    names = [name for name in pc1 if name in pc2]
    if not names:
        return ""

    width, height, pad = 640, 440, 30
    frame = Frame([v for n in names for v in pc1[n]["y"]], [v for n in names for v in pc2[n]["y"]],
                  width, height, pad, equal=True)
    times = pc1[names[0]]["x"]
    data = {"ms": [round(t * 1000) for t in times], "paths": []}
    ghosts, lives, dots, key = [], [], [], []
    for name in names:
        points = [frame.point(x, y) for x, y in zip(pc1[name]["y"], pc2[name]["y"])]
        data["paths"].append(points)
        colour = colour_of(pc1[name])
        full = "M" + " L".join(points)
        ghosts.append(f'      <path class="ghost" style="--c:{colour}" d="{full}"/>')
        lives.append(f'      <path class="live" style="--c:{colour}" d="{full}"/>')
        cx, cy = points[-1].split()
        dots.append(f'      <circle style="--c:{colour}" cx="{cx}" cy="{cy}" r="5"/>')
        key.append(f'      <li style="--c:{colour}">{html.escape(name)}</li>')

    return (
        '<figure class="player">\n'
        f'    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Trajectories of the '
        'MEG response on the first two principal components, one per image type">\n'
        f'      <path class="ref" d="M{pad - 10} {pad - 10} V{height - pad + 10} '
        f'H{width - pad + 10}"/>\n'
        f'      <text x="{width - pad + 10}" y="{height - 4}" text-anchor="end">PC 1</text>\n'
        f'      <text x="{pad - 16}" y="{pad - 10}" text-anchor="end" '
        f'transform="rotate(-90 {pad - 16} {pad - 10})">PC 2</text>\n'
        + "\n".join(ghosts + lives + dots) + '\n'
        '    </svg>\n'
        '    <div class="controls">\n'
        '      <button type="button" class="play">Play</button>\n'
        f'      <input type="range" min="0" max="{len(times) - 1}" value="{len(times) - 1}" '
        'aria-label="Time">\n'
        '      <output></output>\n'
        '    </div>\n'
        '    <ul class="key">\n' + "\n".join(key) + '\n    </ul>\n'
        f'    <figcaption>{html.escape(caption)}</figcaption>\n'
        f'    <script type="application/json" class="player-data">{json.dumps(data)}</script>\n'
        '  </figure>'
    )


def render_paper(found: dict[tuple[str, str], list[dict]]) -> str:
    """The results shown in the paper: a plot, the figure it belongs to, a link."""
    tiles = []
    for label, heading, text, dataset, group, segments, title, shows in PAPER:
        match = [r for r in found.get((dataset, group), []) if r["segments"] == segments]
        if not match:
            continue
        figure = report_figures(match[0]["source"]).get(title)
        svg, legend = small_plot(figure) if figure else ("", "")
        plot = ""
        if svg:
            plot = (
                f'        {svg}\n'
                f'        <ul class="key">{legend}</ul>\n'
                f'        <p class="shows">{html.escape(shows)}</p>\n'
            )
        tiles.append(
            f'    <a class="result" href="view.html#{html.escape(match[0]["route"])}">\n'
            + plot +
            f'        <span class="figure-label">{html.escape(label)}</span>\n'
            f'        <h3>{html.escape(heading)}</h3>\n'
            f'        <p>{html.escape(text)}</p>\n'
            '        <span class="open">Open the report</span>\n'
            '    </a>'
        )
    return "\n".join(tiles)


def render_row(row: dict, records: list[dict], default_sensor: str | None) -> str:
    """One analysis: a heading, a sentence, and links into the viewer.

    Reports that exist once per sensor selection become a single link whose
    target follows the picker.
    """
    links = []
    for folders, routes in variants_of(records):
        label = html.escape(label_for(folders, row["group"]))
        if "" in routes:
            links.append(f'<li><a href="view.html#{html.escape(routes[""])}">{label}</a></li>')
            continue
        route = routes.get(default_sensor or "")
        by_sensor = html.escape(json.dumps({s: f"view.html#{r}" for s, r in routes.items()}),
                                quote=True)
        links.append('<li{hidden}><a href="{href}" data-by-sensor="{by_sensor}">{label}</a></li>'
                     .format(hidden="" if route else " hidden",
                             href=f"view.html#{html.escape(route)}" if route else "#",
                             by_sensor=by_sensor, label=label))

    text = f'          <p>{html.escape(row["text"])}</p>\n' if row.get("text") else ""
    return (
        '      <div class="row">\n'
        f'        <h4>{html.escape(row["heading"])}</h4>\n'
        '        <div>\n' + text +
        '          <ul class="links">\n            ' + "\n            ".join(links) + '\n'
        '          </ul>\n'
        '        </div>\n'
        '      </div>'
    )


def render_reports(found: dict[tuple[str, str], list[dict]]) -> str:
    blocks = []
    for dataset in DATASETS:
        key = dataset["key"]
        sensors = sensors_of(found, key)
        default_sensor = sensors[0] if sensors else None
        rendered = [render_row(row, found[(key, row["group"])], default_sensor)
                    for row in rows_of(found, dataset)]
        if not rendered:
            continue

        picker = ""
        if len(sensors) > 1:
            buttons = "\n".join(
                '      <button type="button" data-sensor="{key}" aria-pressed="{on}">'
                '{label}</button>'
                .format(key=s, on="true" if s == default_sensor else "false",
                        label=html.escape(SENSOR_SETS[s]))
                for s in sensors
            )
            picker = (
                '    <div class="picker">\n'
                '      <span class="ask">Sensors</span>\n' + buttons + '\n    </div>\n'
                f'    <p class="picker-note">{html.escape(dataset.get("picker_note", ""))}</p>\n'
            )

        blocks.append(
            f'<div class="dataset" id="{key}">\n'
            f'    <h3>{html.escape(dataset["name"])}</h3>\n'
            f'    <p class="about">{html.escape(dataset["about"])}</p>\n'
            + picker +
            '    <div class="rows">\n' + "\n".join(rendered) + '\n    </div>\n'
            '  </div>'
        )
    return "\n\n  ".join(blocks)


# --- viewer ----------------------------------------------------------------

def site_map(found: dict[tuple[str, str], list[dict]]) -> list[dict]:
    """Everything the viewer needs to move between reports."""
    site = []
    for dataset in DATASETS:
        key = dataset["key"]
        analyses = [
            {"name": row.get("short", row["heading"]),
             "reports": [{"name": label_for(folders, row["group"]), "routes": routes}
                         for folders, routes in variants_of(found[(key, row["group"])])]}
            for row in rows_of(found, dataset)
        ]
        if analyses:
            site.append({
                "name": dataset["short"],
                "sensors": [{"key": s, "name": SENSOR_SETS[s]} for s in sensors_of(found, key)],
                "analyses": analyses,
            })
    return site


# --- entry point ---------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=here.parent / "outputs" / "release",
                        help="analysis output tree to scan (default: outputs/release)")
    parser.add_argument("--clean", action="store_true",
                        help="delete website/reports/ before copying")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory not found: {source}")

    if args.clean:
        shutil.rmtree(here / "reports", ignore_errors=True)

    found = collect(source)
    if not found:
        raise SystemExit(f"no report.html files found under {source}")

    total = copied = 0
    for records in found.values():
        total += len(records)
        copied += copy_reports(records, here)

    today = date.today()
    index = (here / "template.html").read_text(encoding="utf-8")
    index = index.replace("<!-- HERO -->", render_hero(found))
    index = index.replace("<!-- PAPER -->", render_paper(found))
    index = index.replace("<!-- REPORTS -->", render_reports(found))
    index = index.replace("<!-- BUILT -->", f"{today.day} {today:%B %Y}")
    (here / "index.html").write_text(index, encoding="utf-8")

    viewer = (here / "viewer.html").read_text(encoding="utf-8")
    viewer = viewer.replace("/* SITE */", json.dumps(site_map(found)).replace("</", "<\\/"))
    (here / "view.html").write_text(viewer, encoding="utf-8")

    print(f"{total} reports linked ({copied} copied, {total - copied} already current)")
    for (dataset, group), records in sorted(found.items()):
        print(f"  {dataset}/{group}: {len(records)}")
    print(f"wrote {here / 'index.html'} and {here / 'view.html'}")


if __name__ == "__main__":
    main()
