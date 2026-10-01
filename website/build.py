#!/usr/bin/env python3
"""Build the results website from the analysis output tree.

Every ``report.html`` under ``--source`` is copied into ``website/reports/`` and
linked from a generated ``index.html``.

    python website/build.py                    # incremental copy + regenerate index
    python website/build.py --clean            # drop reports/ first
    python website/build.py --source outputs/release

The page itself (layout, wording, styling) is ``template.html``. What appears
where is the configuration below. A report that is not there yet is left out of
the page; a report the configuration does not know about still gets a link,
named after its folder.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import os
import re
import shutil
from datetime import date
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

# Link text for the folders a report can sit in. "" is the analysis root.
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

# Links are listed in this order; anything else follows alphabetically.
ORDER = ["", "best_models", "4class_3_4_5_6", "3class_1_2_3"]

DATASETS = [
    {
        "key": "eegbci",
        "name": "EEG: moving a hand, or imagining it",
        "about": (
            "The PhysioNet EEG Motor Movement/Imagery dataset. 64-channel EEG from 106 people "
            "who opened and closed their left or right hand, or imagined doing so."
        ),
        "rows": [
            ("main", "Trajectories",
             "The main analysis. One PCA space for all four conditions, how much variance it "
             "holds, and where the paths separate."),
            ("nonlinear", "PCA next to nonlinear methods",
             "The same data through UMAP, PHATE and Isomap, to see what PCA keeps and what it "
             "misses."),
            ("decoding", "Decoding across participants",
             "Train on some people, test on someone new. Raw sensors compared with PCA "
             "components aligned between participants."),
        ],
    },
    {
        "key": "megfaces",
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
            ("main", "Trajectories",
             "The three image types in one shared PCA space, with the faces vs scrambled and "
             "famous vs unfamiliar contrasts."),
            ("spectral", "Frequency bands",
             "The same analysis on alpha, beta and low-gamma (30 to 45 Hz) power instead of "
             "the broadband signal."),
            ("decoding", "Decoding across participants",
             "Train on 15 people, test on the 16th. Raw sensors compared with shared and "
             "aligned PCA components."),
        ],
    },
]

# The reports shown as cards under "Start here": (dataset, group, folders, label, title, text).
FEATURED = [
    ("eegbci", "main", (), "EEG, 106 people", "Hand movement as a trajectory",
     "Left and right hand, executed and imagined. The best place to start if the method is "
     "new to you."),
    ("megfaces", "main", ("sensors_right_occipital",), "MEG, 16 people",
     "Faces in one shared space",
     "Famous, unfamiliar and scrambled images, and the moment the face conditions pull away."),
    ("megfaces", "spectral", ("sensors_right_occipital",), "MEG, 16 people",
     "The same question in alpha, beta and gamma",
     "Trajectories built from band power instead of the evoked signal."),
    ("megfaces", "decoding", ("best_models",), "MEG, 16 people",
     "Can a classifier use these components?",
     "Decoding the image type in a new participant, from sensors and from PCA components."),
]

# The drawing at the top of the page: one figure read out of one report.
# (dataset, group, folders, title of the figure inside that report, caption)
HERO = (
    "megfaces", "main", ("sensors_right_occipital",),
    "Equal-participant mean trajectories: PC1-PC2",
    "The MEG response to three kinds of image, averaged over 16 people and drawn on the first "
    "two principal components. The dots mark the start of the epoch. The two face conditions "
    "take a wide loop that the scrambled images do not.",
)

PAYLOAD_RE = re.compile(r'<script type="application/json" id="report-payload">(.*?)</script>',
                        re.DOTALL)
BODY_RE = re.compile(r"<body[^>]*>", re.IGNORECASE)

# Added to the top of every copied report so there is a way back to the index.
BACK_LINK = (
    '\n<a href="{home}" style="display:block;padding:6px 16px;background:#1c1b19;'
    'color:#fdfcfa;font:14px -apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,'
    'sans-serif;text-decoration:none">&larr; All reports</a>\n'
)


# --- helpers -------------------------------------------------------------

def label_for(folders: tuple[str, ...], group: str) -> str:
    """Link text for a report, from the folders it sits in (sensor folder removed)."""
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
                "segments": tuple(segments),
                "folders": tuple(s for s in segments if s not in SENSOR_SETS),
                "sensor": sensors[0] if sensors else None,
            })
    return found


def copy_reports(records: list[dict], dest_root: Path) -> int:
    """Copy reports that are new or changed, adding the link back to the index.

    Returns how many were written.
    """
    copied = 0
    for record in records:
        target = dest_root / record["dest"]
        source_stat = record["source"].stat()
        if target.exists() and target.stat().st_mtime == source_stat.st_mtime:
            continue
        home = "../" * (len(record["dest"].parts) - 1) + "index.html"
        text = record["source"].read_text(encoding="utf-8")
        text = BODY_RE.sub(lambda m: m.group(0) + BACK_LINK.format(home=home), text, count=1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        os.utime(target, (source_stat.st_atime, source_stat.st_mtime))
        copied += 1
    return copied


def report_figure(report: Path, title: str) -> dict | None:
    """The Plotly figure with this title from a report's embedded data, if any."""
    match = PAYLOAD_RE.search(report.read_text(encoding="utf-8"))
    if not match:
        return None
    payload = json.loads(gzip.decompress(base64.b64decode(match.group(1).strip())))
    for figure in payload.values():
        if not isinstance(figure, dict) or "layout" not in figure:
            continue
        found = figure["layout"].get("title")
        if (found.get("text") if isinstance(found, dict) else found) == title:
            return figure
    return None


def render_hero(found: dict[tuple[str, str], list[dict]]) -> str:
    """Redraw the report's PC1-PC2 trajectories as a small SVG. Empty if unavailable."""
    dataset, group, segments, title, caption = HERO
    match = [r for r in found.get((dataset, group), []) if r["segments"] == segments]
    figure = report_figure(match[0]["source"], title) if match else None
    traces = [t for t in (figure or {}).get("data", [])
              if isinstance(t.get("x"), list) and isinstance(t.get("y"), list) and t.get("name")]
    if not traces:
        return ""

    width, height, pad = 460, 380, 34
    xs = [x for t in traces for x in t["x"]]
    ys = [y for t in traces for y in t["y"]]
    scale = min((width - 2 * pad) / (max(xs) - min(xs)), (height - 2 * pad) / (max(ys) - min(ys)))
    left = (width - scale * (max(xs) - min(xs))) / 2
    bottom = (height - scale * (max(ys) - min(ys))) / 2

    def place(x: float, y: float) -> str:
        return f"{left + (x - min(xs)) * scale:.1f} {height - bottom - (y - min(ys)) * scale:.1f}"

    paths, starts, key = [], [], []
    for trace in traces:
        colour = html.escape(trace.get("line", {}).get("color", "currentColor"))
        points = [place(x, y) for x, y in zip(trace["x"], trace["y"])]
        paths.append(f'      <path pathLength="1" style="--c:{colour}" d="M{" L".join(points)}"/>')
        cx, cy = points[0].split()
        starts.append(f'      <circle class="start" cx="{cx}" cy="{cy}" r="4"/>')
        key.append(f'      <li style="--c:{colour}">{html.escape(trace["name"])}</li>')

    return (
        '  <figure>\n'
        f'    <svg class="traj" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Trajectories on the first two principal components, one per condition">\n'
        f'      <path class="axis" d="M24 20 V{height - 18} H{width - 14}"/>\n'
        f'      <text x="{width - 14}" y="{height - 4}" text-anchor="end">PC 1</text>\n'
        '      <text x="14" y="20" text-anchor="end" transform="rotate(-90 14 20)">PC 2</text>\n'
        + "\n".join(paths + starts) + '\n'
        '    </svg>\n'
        '    <ul class="key">\n' + "\n".join(key) + '\n    </ul>\n'
        f'    <figcaption>{html.escape(caption)}</figcaption>\n'
        '  </figure>'
    )


def render_featured(found: dict[tuple[str, str], list[dict]]) -> str:
    cards = []
    for dataset, group, segments, where, title, text in FEATURED:
        match = [r for r in found.get((dataset, group), []) if r["segments"] == segments]
        if not match:
            continue
        cards.append(
            '    <a class="card" href="{href}">\n'
            '      <span class="where">{where}</span>\n'
            '      <h3>{title}</h3>\n'
            '      <p>{text}</p>\n'
            '      <span class="go">Open the report &rarr;</span>\n'
            '    </a>'.format(
                href=html.escape(match[0]["dest"].as_posix()),
                where=html.escape(where), title=html.escape(title), text=html.escape(text),
            )
        )
    return "\n".join(cards)


def render_row(heading: str, text: str, group: str, records: list[dict],
               default_sensor: str | None) -> str:
    """One analysis: a heading, a sentence, and its links.

    Reports that exist once per sensor selection become a single link whose
    target follows the picker.
    """
    by_folders: dict[tuple[str, ...], dict[str | None, str]] = {}
    for record in records:
        by_folders.setdefault(record["folders"], {})[record["sensor"]] = record["dest"].as_posix()

    links = []
    for folders in sorted(by_folders, key=order_key):
        targets = by_folders[folders]
        label = html.escape(label_for(folders, group))
        if None in targets:
            links.append(f'<li><a href="{html.escape(targets[None])}">{label}</a></li>')
            continue
        href = targets.get(default_sensor)
        by_sensor = html.escape(json.dumps(targets), quote=True)
        links.append('<li{hidden}><a href="{href}" data-by-sensor="{by_sensor}">{label}</a></li>'
                     .format(hidden="" if href else " hidden", href=html.escape(href or "#"),
                             by_sensor=by_sensor, label=label))

    return (
        '      <div class="row">\n'
        f'        <h4>{html.escape(heading)}</h4>\n'
        '        <div>\n'
        f'          <p>{html.escape(text)}</p>\n'
        '          <ul>\n            ' + "\n            ".join(links) + '\n          </ul>\n'
        '        </div>\n'
        '      </div>'
    )


def render_reports(found: dict[tuple[str, str], list[dict]]) -> str:
    blocks = []
    for dataset in DATASETS:
        key = dataset["key"]
        rows = list(dataset["rows"])
        known = {group for group, _, _ in rows}
        # Groups found on disk that the configuration does not describe.
        rows += [(group, group.replace("_", " ").capitalize(), "")
                 for (ds, group) in sorted(found) if ds == key and group not in known]

        present = {r["sensor"] for (ds, _), records in found.items() if ds == key
                   for r in records if r["sensor"]}
        sensors = [s for s in SENSOR_SETS if s in present]
        default_sensor = sensors[0] if sensors else None

        rendered = [render_row(heading, text, group, found[(key, group)], default_sensor)
                    for group, heading, text in rows if found.get((key, group))]
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
                '      <span class="ask">Sensors:</span>\n' + buttons + '\n    </div>\n'
                f'    <p class="picker-note">{html.escape(dataset.get("picker_note", ""))}</p>\n'
            )

        blocks.append(
            '  <div class="dataset">\n'
            f'    <h3>{html.escape(dataset["name"])}</h3>\n'
            f'    <p class="about">{html.escape(dataset["about"])}</p>\n'
            + picker +
            '    <div class="rows">\n' + "\n".join(rendered) + '\n    </div>\n'
            '  </div>'
        )
    return "\n\n".join(blocks)


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
    index = index.replace("<!-- FEATURED -->", render_featured(found))
    index = index.replace("<!-- REPORTS -->", render_reports(found))
    index = index.replace("<!-- BUILT -->", f"{today.day} {today:%B %Y}")
    (here / "index.html").write_text(index, encoding="utf-8")

    print(f"{total} reports linked ({copied} copied, {total - copied} already current)")
    for (dataset, group), records in sorted(found.items()):
        print(f"  {dataset}/{group}: {len(records)}")
    print(f"wrote {here / 'index.html'}")


if __name__ == "__main__":
    main()
