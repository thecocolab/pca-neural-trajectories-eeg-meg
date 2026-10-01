# Results website

The site is built from the `report.html` files that the analysis scripts write, and
from nothing else. Run the scripts, run `build.py`, and the page is up to date.

```
website/
  build.py        # finds the reports, copies them, writes index.html
  template.html   # the page: text, layout, styling
  deploy.sh       # build, then publish to the gh-pages branch
  index.html      # generated
  reports/        # generated
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
left off the page. Run the build again when it arrives.

Two things are taken from the reports besides the links:

- The drawing at the top of the page is the PC1–PC2 trajectory figure from the MEG
  main report, read out of that report's embedded data. If the report or the figure
  is missing, the page is built without the drawing.
- Each copied report gets a small "All reports" link at the top, so there is a way
  back to the index.

## Publish

```bash
./website/deploy.sh
```

This rebuilds the site and pushes it to `gh-pages` as a single commit that replaces
the previous one. The reports add up to a few hundred MB, and this keeps them out of
the repository history.

## Change things

- **Text, colours, layout:** `template.html`.
- **Names and descriptions of the analyses, the four "Start here" cards, the order of
  links:** the configuration at the top of `build.py`.
- **A new analysis folder:** add a line to `ANALYSES`, and a row to the dataset it
  belongs to in `DATASETS`.
- **A new contrast:** nothing to do. It is picked up from the folder name. Add an
  entry to `LABELS` if you want a nicer name.
- **A new sensor selection:** add it to `SENSOR_SETS` so it shows up in the picker.
