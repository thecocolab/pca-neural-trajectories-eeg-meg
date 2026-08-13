#!/usr/bin/env python3
"""Build the one-page results website from the analysis output tree.

Every ``report.html`` under ``--source`` is copied into ``website/reports/`` under a
path that spells out its analysis, and linked from a generated ``index.html``.

    python website/build.py                    # incremental copy + regenerate index
    python website/build.py --clean            # drop reports/ first
    python website/build.py --source outputs/v3

Page design lives in ``template.html``; what appears where is the configuration
below. Adding an analysis means adding one line to ANALYSES (plus a GROUPS entry
if it belongs to a new family). Reports that have not arrived yet are simply
absent — re-run the script once they land.
"""

from __future__ import annotations

import argparse
import html
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

DATASETS = [
    {
        "key": "eegbci",
        "name": "EEG BCI",
        "tagline": "PhysioNet motor movement/imagery, 106 participants",
        "blurb": (
            "The PhysioNet EEG Motor Movement/Imagery Database: 64-channel EEG recorded while "
            "participants executed or imagined left-hand, right-hand, and both-hands movements "
            "cued by a visual target. The full cohort analysed here is 106 participants."
        ),
    },
    {
        "key": "megfaces",
        "name": "MEG Faces",
        "tagline": "Wakeman–Henson ds000117, 16 participants",
        "blurb": (
            "The Wakeman–Henson multimodal face dataset (OpenNeuro ds000117): 306-channel MEG "
            "recorded while participants viewed famous, unfamiliar, and scrambled faces. Each "
            "analysis is repeated over four sensor selections, which are helmet-position subsets "
            "rather than source-localized regions."
        ),
    },
]

# Group key -> (dataset key, heading, description).
GROUPS = [
    ("main", "eegbci", "Main trajectory analysis",
     "One shared PCA basis for all conditions: trajectories, variance accounted for, "
     "geometric descriptors, and group-level contrasts."),
    ("decoding", "eegbci", "Decoding",
     "Subject-disjoint temporal decoding from raw sensors versus fold-local Procrustes-aligned "
     "components. One sweep report covers all contrasts; the rest break out a single contrast."),
    ("nonlinear", "eegbci", "Nonlinear embeddings",
     "PCA compared against UMAP, PHATE, and Isomap on the same observations, with "
     "trial-respecting velocity fields and label-free subject-space alignment."),
    ("main", "megfaces", "Main trajectory analysis",
     "One broadband PCA space shared by Famous, Unfamiliar, and Scrambled. Planned contrasts "
     "are derived views of that space, never refitted bases."),
    ("decoding", "megfaces", "Decoding",
     "Leakage-safe cross-participant temporal decoding on noise-whitened MEG, with sensors as "
     "the baseline representation and transductive alignment marked as such."),
    ("spectral", "megfaces", "Power bands (spectral envelopes)",
     "Alpha, beta, and 30–45 Hz low-gamma Hilbert amplitude envelopes, computed from the padded "
     "derivative set so filter and Hilbert edges stay outside the analysis window."),
]

# Path segments that deserve a nicer label than the automatic prettifier gives.
LABELS = {
    "all_sensors": "All sensors",
    "sensors_right_occipital": "Right occipital",
    "sensors_right_temporal": "Right temporal",
    "sensors_right_occipito_temporal": "Right occipito-temporal",
    "4class_3_4_5_6": "4-class (runs 3/4/5/6)",
    "hands_exec_vs_hands_imag": "Hands: execution vs imagery",
    "left_hand_exec_vs_right_hand_exec": "Left vs right hand (execution)",
    "left_hand_imag_vs_right_hand_imag": "Left vs right hand (imagery)",
    "3class_1_2_3": "3-class (famous / unfamiliar / scrambled)",
    "famous_vs_scrambled": "Famous vs scrambled",
    "famous_vs_unfamiliar": "Famous vs unfamiliar",
    "unfamiliar_vs_scrambled": "Unfamiliar vs scrambled",
}

# Sort earlier when a segment appears; anything unlisted sorts alphabetically after.
PRIORITY = ["all_sensors", "sensors_right_occipito_temporal",
            "sensors_right_occipital", "sensors_right_temporal"]

# Label for a report sitting at the root of its analysis directory, by group.
ROOT_LABELS = {"decoding": "All contrasts (full sweep)"}
ROOT_LABEL_DEFAULT = "Full report"

TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL | re.IGNORECASE)


# --- helpers -------------------------------------------------------------

def pretty(segment: str) -> str:
    """Human label for one path segment."""
    return LABELS.get(segment, segment.replace("_", " ").capitalize())


def report_title(path: Path) -> str:
    """The <title> of a report, used as its one-line description."""
    match = TITLE_RE.search(path.read_text(encoding="utf-8", errors="replace")[:8192])
    return " ".join(match.group(1).split()) if match else ""


def human_size(num_bytes: int) -> str:
    mb = num_bytes / 1e6
    return f"{mb:.1f} MB" if mb >= 1 else f"{num_bytes / 1e3:.0f} kB"


def sort_key(segments: list[str]) -> tuple:
    """Root reports first, then priority segments, then alphabetical."""
    head = segments[0] if segments else ""
    rank = PRIORITY.index(head) if head in PRIORITY else len(PRIORITY)
    return (len(segments), rank, segments)


def collect(source: Path) -> dict[tuple[str, str], list[dict]]:
    """Map (dataset, group) -> report records found under *source*."""
    found: dict[tuple[str, str], list[dict]] = {}
    for analysis, (dataset, group) in ANALYSES.items():
        analysis_dir = source / analysis
        if not analysis_dir.is_dir():
            continue
        for report in sorted(analysis_dir.rglob("report.html")):
            segments = list(report.relative_to(analysis_dir).parts[:-1])
            name = "/".join(segments) if segments else "overview"
            found.setdefault((dataset, group), []).append({
                "source": report,
                "dest": Path("reports") / dataset / group / f"{name}.html",
                "label": (" — ".join(pretty(s) for s in segments)
                          or ROOT_LABELS.get(group, ROOT_LABEL_DEFAULT)),
                "title": report_title(report),
                "size": report.stat().st_size,
                "sort": sort_key(segments),
            })
    for records in found.values():
        records.sort(key=lambda record: record["sort"])
    return found


def copy_reports(records: list[dict], dest_root: Path) -> int:
    """Copy reports that are new or changed. Returns how many were written."""
    copied = 0
    for record in records:
        target = dest_root / record["dest"]
        source = record["source"]
        stale = (not target.exists()
                 or target.stat().st_size != source.stat().st_size
                 or target.stat().st_mtime < source.stat().st_mtime)
        if stale:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied


def render(found: dict[tuple[str, str], list[dict]]) -> str:
    """Render the dataset tabs and group accordions."""
    tabs, panels = [], []

    for index, dataset in enumerate(DATASETS):
        key = dataset["key"]
        selected = "true" if index == 0 else "false"
        tabs.append(
            f'    <button class="tab" role="tab" aria-selected="{selected}" '
            f'aria-controls="panel-{key}" data-panel="panel-{key}">{html.escape(dataset["name"])}'
            f'<span class="tab-sub">{html.escape(dataset["tagline"])}</span></button>'
        )

        blocks = []
        open_group = True
        for group_key, dataset_key, heading, blurb in GROUPS:
            if dataset_key != key:
                continue
            records = found.get((key, group_key), [])
            if not records:
                continue
            items = []
            for record in records:
                items.append(
                    '        <li><a class="report" href="{href}">'
                    '<span class="label">{label}</span>'
                    '<span class="desc">{desc}</span>'
                    '<span class="size">{size}</span></a></li>'.format(
                        href=html.escape(record["dest"].as_posix()),
                        label=html.escape(record["label"]),
                        desc=html.escape(record["title"]),
                        size=human_size(record["size"]),
                    )
                )
            blocks.append(
                '    <details class="group"{open}>\n'
                '      <summary>{heading}<span class="count">{count} report{plural}</span></summary>\n'
                '      <p class="group-blurb">{blurb}</p>\n'
                '      <ul class="reports">\n{items}\n      </ul>\n'
                '    </details>'.format(
                    open=" open" if open_group else "",
                    heading=html.escape(heading),
                    count=len(records),
                    plural="" if len(records) == 1 else "s",
                    blurb=html.escape(blurb),
                    items="\n".join(items),
                )
            )
            open_group = False

        if not blocks:
            blocks = ['    <p class="empty">No reports available yet for this dataset.</p>']

        panels.append(
            '  <section class="dataset" id="panel-{key}" role="tabpanel"{hidden}>\n'
            '    <p class="blurb">{blurb}</p>\n'
            '    <input class="filter" type="search" placeholder="Filter reports…" '
            'aria-label="Filter reports">\n{blocks}\n  </section>'.format(
                key=key,
                hidden="" if index == 0 else " hidden",
                blurb=html.escape(dataset["blurb"]),
                blocks="\n".join(blocks),
            )
        )

    return ('<div class="tabs" role="tablist">\n' + "\n".join(tabs) + "\n</div>\n\n"
            + "\n\n".join(panels))


# --- entry point ---------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=here.parent / "outputs" / "v2",
                        help="analysis output tree to scan (default: outputs/v2)")
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

    index = (here / "template.html").read_text(encoding="utf-8")
    index = index.replace("<!-- CONTENT -->", render(found))
    index = index.replace("<!-- BUILT -->", date.today().isoformat())
    (here / "index.html").write_text(index, encoding="utf-8")

    print(f"{total} reports listed ({copied} copied, {total - copied} already current)")
    for (dataset, group), records in sorted(found.items()):
        print(f"  {dataset}/{group}: {len(records)}")
    print(f"wrote {here / 'index.html'}")


if __name__ == "__main__":
    main()
