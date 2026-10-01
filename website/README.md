# Results website

The site is built from the `report.html` files that the analysis scripts write, and
from nothing else. Run the scripts, run `build.py`, and the site is up to date. The
reports are copied as they are. Nothing in them is changed.

```
website/
  build.py        # finds the reports, copies them, writes the two pages
  template.html   # the landing page: text, layout, styling
  viewer.html     # the page that shows one report under a navigation bar
  deploy.sh       # build, then publish to the gh-pages branch
  index.html      # generated from template.html
  view.html       # generated from viewer.html
  reports/        # generated: copies of the reports
```

## Build

```bash
python3 website/build.py                           # reads outputs/release
python3 website/build.py --source some/other/dir
python3 website/build.py --clean                   # recopy every report
```

Then open `website/index.html` in a browser. It works straight from disk.

`build.py` looks for `report.html` at any depth under each analysis folder
(`eegbci_main/`, `megfaces_decoding/`, and so on). A report that isn't there yet is
left off the site. Run the build again when it arrives.

## How the site is put together

- **Landing page** (`index.html`). It introduces the paper, lists the reports behind
  the paper's figures first (`PAPER` in `build.py`), then every report by dataset.
  The trajectory at the top and the small plots are redrawn from figures stored
  inside the reports, chosen by title in `build.py` (`HERO` and `PAPER`). If a figure
  is missing, the page is built without that plot.
- **Viewer** (`view.html`). Every link on the landing page opens a report inside the
  viewer, which keeps a bar on top for moving between datasets, analyses, contrasts
  and sensor selections. The address of a report is `view.html#` followed by its path
  under `reports/`, so links can be shared. "Open on its own" gives the bare report.

## Publish

```bash
./website/deploy.sh
```

This rebuilds the site and pushes it to `gh-pages` as a single commit that replaces
the previous one. The reports add up to a few hundred MB, and this keeps them out of
the repository history.

## Change things

- **Text (including the preprint link, authors and citation), colours, layout:** `template.html` for the landing page, `viewer.html` for
  the bar above the reports.
- **Which reports are listed as paper results, names and descriptions of the
  analyses, the order of reports:** the
  configuration at the top of `build.py`.
- **A new analysis folder:** add a line to `ANALYSES`, and a row to the dataset it
  belongs to in `DATASETS`.
- **A new contrast:** nothing to do. It is picked up from the folder name. Add an
  entry to `LABELS` if you want a nicer name.
- **A new sensor selection:** add it to `SENSOR_SETS` so it shows up in the pickers.
