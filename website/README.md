# Results website

A one-page index that links to every interactive analysis report. Static HTML, no
build tooling, no external assets — it works from `file://` and from any static host.

```
website/
  build.py        # scans the analysis output tree, copies reports, writes index.html
  template.html   # the page itself: layout, styling, tab/filter behaviour
  index.html      # generated — do not edit by hand
  reports/        # generated — copies of every report.html, one folder per analysis
```

## Rebuilding

```bash
python3 website/build.py                 # incremental: copies new/changed reports only
python3 website/build.py --clean         # rebuild reports/ from scratch
python3 website/build.py --source outputs/v3
```

Re-run it whenever more reports finish transferring. Reports that are not there yet are
simply absent from the page; nothing else needs changing.

Then open `website/index.html` in a browser.

## Changing things

- **Wording, colours, layout** — `template.html`. `<!-- CONTENT -->` is where the generated
  dataset tabs and group accordions are injected, `<!-- BUILT -->` is the build date.
  The repository link in the header is a placeholder; set it before publishing.
- **Which analyses appear, and where** — the configuration block at the top of `build.py`:
  `ANALYSES` maps an output directory to a dataset and group, `DATASETS` and `GROUPS`
  hold the headings and descriptions, `LABELS` gives nicer names to path segments,
  `PRIORITY` controls ordering.
- **A new analysis directory** — add one line to `ANALYSES`; add a `GROUPS` entry if it
  belongs to a new family.

Any `report.html` at any depth under a configured analysis directory is picked up, so
per-contrast and per-sensor-set sweeps need no extra configuration. Standalone plots in
`figures/` are not copied — they are already embedded in their report.

## Publishing

`reports/` is around 140 MB because each report embeds Plotly and its data. That is fine
for GitHub Pages (1 GB soft limit), but committing a rebuilt copy of every report on each
run grows repository history quickly. Options, in rough order of preference:

1. Publish from a dedicated branch or a separate repository that only holds `website/`.
2. Push to a static host (Netlify, Cloudflare Pages, an institutional web directory)
   without versioning the reports.
3. Commit `website/` directly, and re-run `build.py` sparingly.
