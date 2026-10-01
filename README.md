# PCA trajectories for EEG and MEG

Code and notebooks for our tutorial paper, *A Primer on Low-Dimensional Neural
Dynamics: PCA-Based Trajectory Analysis for EEG and MEG*.

The idea is simple. An evoked response is a sensor pattern that changes over time.
If you project it onto a few principal components, you can draw that change as a
path and ask questions about its shape: when do two conditions separate, how fast
does the pattern move, does it look the same in another person. The notebooks show
how to do this on two open datasets, and where it stops working.

**Results:** <https://thecocolab.github.io/pca-neural-trajectories-eeg-meg/>

## Install

You need Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,meg]"
```

Most of the analysis code lives in [coco-pipe](https://github.com/BabaSanfour/coco-pipe),
which is installed as a dependency. This repository adds the data loading, the
notebooks and the scripts.

## The notebooks

Start `jupyter lab`, pick the `Python 3` kernel, and go through them in order. They
are saved without outputs.

| | Notebook | What you do in it |
|---|---|---|
| 1 | [`tutorial_eegbci_main`](tutorials/tutorial_eegbci_main.ipynb) | The core workflow on EEG: left and right hand movement, executed and imagined. |
| 2 | [`tutorial_eegbci_nonlinear`](tutorials/tutorial_eegbci_nonlinear.ipynb) | The same data through UMAP, PHATE and Isomap, compared with PCA. |
| 3 | [`tutorial_eegbci_decoding`](tutorials/tutorial_eegbci_decoding.ipynb) | Decoding in a new participant, from sensors and from PCA components aligned between people. |
| 4 | [`tutorial_megfaces_main`](tutorials/tutorial_megfaces_main.ipynb) | MEG: famous, unfamiliar and scrambled faces in one shared PCA space. |
| 5 | [`tutorial_megfaces_spectral_envelopes`](tutorials/tutorial_megfaces_spectral_envelopes.ipynb) | Trajectories of alpha, beta and low-gamma (30–45 Hz) power. |
| 6 | [`tutorial_megfaces_decoding`](tutorials/tutorial_megfaces_decoding.ipynb) | The decoding workflow again, on MEG. |

A few things to know before you run them:

- **Time.** The first EEG notebook takes 30 to 60 minutes once the data is
  preprocessed. The others take 10 to 40 minutes. MEG preprocessing is much slower
  (Maxwell filtering) and needs a lot of disk space.
- **Nothing is downloaded behind your back.** Downloading and preprocessing are
  explicit steps. The settings you can change are listed at the top of each notebook.
- **Permutation tests are off by default.** Set `RUN_PERMUTATIONS` in the notebook to
  turn them on.
- **The spectral notebook needs longer epochs** (−1.1 to 1.7 s) so that filtering and
  the Hilbert transform don't contaminate the window we analyse. It refuses to run on
  the usual short epochs.
- Outputs go to `outputs/tutorial_*/`. Data, derivatives and outputs are not tracked
  in Git.

## Notebooks and scripts

Each notebook has a script in `scripts/` that does the same analysis without Jupyter.
The only difference is the number of participants: the notebooks use a few so they
run on a laptop, and the scripts use everyone (106 for EEG, 16 for MEG). The numbers
in the paper and the reports on the website come from the scripts.

| Notebook | Script |
|---|---|
| `tutorial_eegbci_main` | `analysis_eegbci_main.py` |
| `tutorial_eegbci_nonlinear` | `analysis_eegbci_nonlinear.py` |
| `tutorial_eegbci_decoding` | `analysis_eegbci_decoding.py` |
| `tutorial_megfaces_main` | `analysis_megfaces_main.py` |
| `tutorial_megfaces_spectral_envelopes` | `analysis_megfaces_spectral_envelopes.py` |
| `tutorial_megfaces_decoding` | `analysis_megfaces_decoding.py` |

Every script writes its tables, figures and a single-file HTML report to
`outputs/<analysis>/` (MEG scripts add a `<sensor-set>/` folder). PNG and SVG copies
of the figures are written too when Kaleido can find a browser.

### EEG

```bash
python scripts/analysis_eegbci_main.py --smoke --output outputs/smoke   # quick check
python scripts/analysis_eegbci_main.py --output outputs/eegbci_main
python scripts/analysis_eegbci_nonlinear.py --skip-prepare
python scripts/analysis_eegbci_decoding.py
```

`analysis_eegbci_main.py` skips a run that already finished; pass `--no-resume` to
redo it. It also writes `separation_speed.{svg,png,csv}`. The sample size in a report
is the number of participants that actually loaded, which can be lower than what you
asked for (106 of 109 for the full cohort).

### MEG

```bash
python scripts/analysis_megfaces_main.py --smoke
python scripts/analysis_megfaces_main.py --n-perm 1000

python scripts/analysis_megfaces_spectral_envelopes.py --n-perm 1000
python scripts/analysis_megfaces_spectral_envelopes.py --prepare    # build the long epochs first

python scripts/analysis_megfaces_decoding.py                        # best models, then every contrast
python scripts/analysis_megfaces_decoding.py --best-models-only
```

The MEG scripts expect derivatives that are already prepared. If yours are somewhere
else, point to them with `--derivatives-root`.

They use the right-occipital sensors (`sensors_right_occipital`, 36 sensors) unless
you pass `--sensor-set`. Each PCA is fitted on −0.2 to 0.6 s and then applied to the
whole epoch.

## Getting the MEG data

One function downloads, preprocesses and loads the Wakeman–Henson dataset. The first
call needs `prepare=True`. It downloads about 5 GB and runs Maxwell filtering and ICA
for each participant, so expect it to take a while.

```python
from pca_neural_trajectories import load_wakeman_henson

meg = load_wakeman_henson(
    "data/wakeman_henson",
    subjects=("01",),
    prepare=True,
    sensor_set="all_sensors",
)
```

After that, leave `prepare=True` out. Pass `spectral=True` for the longer epochs used
by the spectral notebook.

`sensor_set` can be `all_sensors` (the default here), `sensors_occipital`,
`sensors_temporal`, `sensors_occipito_temporal`, or the right-hemisphere versions
`sensors_right_occipital`, `sensors_right_temporal` and
`sensors_right_occipito_temporal`. These are MNE's Neuromag VectorView helmet
selections. They group sensors by position and say nothing about which part of the
brain a signal comes from.

## The website

`website/build.py` builds the results site from the reports the scripts produce. See
[`website/README.md`](website/README.md).

## Tests

```bash
ruff check pca_neural_trajectories tests scripts
python -m pytest
```
