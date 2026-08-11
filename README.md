# PCA-Based Trajectory Analysis for EEG and MEG

Code and tutorials for *A Primer on Low-Dimensional Neural Dynamics:
PCA-Based Trajectory Analysis for EEG and MEG*. The repository presents a focused,
sensor-space tutorial series rather than an exhaustive analysis framework.

## Install

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -e ".[test,meg]"
```

The project depends on `coco-pipe[eeg,decoding,neighbor]`.

## Tutorial order

All notebooks are committed with outputs cleared. Launch `jupyter lab`, select the
portable `Python 3` kernel, and run them in this order:

1. [`tutorial_eegbci_main.ipynb`](tutorials/tutorial_eegbci_main.ipynb)
   — four left/right hand execution and imagery EEG trajectories with the core
   PCA workflow.
2. [`tutorial_eegbci_nonlinear.ipynb`](tutorials/tutorial_eegbci_nonlinear.ipynb)
   — PCA/UMAP/PHATE/Isomap comparison, trial-respecting velocity fields, and
   label-free subject-space alignment on the same EEGBCI observations.
3. [`tutorial_eegbci_decoding.ipynb`](tutorials/tutorial_eegbci_decoding.ipynb)
   — subject-disjoint EEG decoding from sensors versus fold-local temporal
   Procrustes alignment. The shared reference uses training participants, the
   held-out calibration is label-free, and channel normalization is fold-local.
4. [`tutorial_megfaces_main.ipynb`](tutorials/tutorial_megfaces_main.ipynb)
   — one broadband MEG space for Famous, Unfamiliar, and Scrambled images, with both
   planned contrasts derived without refitting.
5. [`tutorial_megfaces_spectral_envelopes.ipynb`](tutorials/tutorial_megfaces_spectral_envelopes.ipynb)
   — alpha, beta, and 30–45 Hz low-gamma Hilbert amplitude envelopes. This requires a
   distinct noise-whitened derivatives set (`-1.1..1.7 s`, providing 0.9 s around the
   `-0.2..0.8 s` crop); the usual short MEG epochs are deliberately rejected as unsafe
   for filtering/Hilbert edges.
6. [`tutorial_megfaces_decoding.ipynb`](tutorials/tutorial_megfaces_decoding.ipynb)
   — the same leakage-safe decoding and alignment workflow on whitened MEG,
   with contrast-specific spaces and explicit transductive-alignment caveats.

The EEG introduction is typically 30–60 minutes after preprocessing; nonlinear and
decoding notebooks take roughly 10–40 minutes depending on sample size. MEG
preprocessing (especially Maxwell filtering) is much slower and requires substantial
external storage. Notebooks never download or preprocess data implicitly. Expensive
permutation branches are opt-in through the visible `RUN_PERMUTATIONS` notebook setting.

Environment overrides are documented at the start of every notebook. The default
output roots are:

- `outputs/tutorial_eegbci/`
- `outputs/tutorial_eegbci_nonlinear/`
- `outputs/tutorial_eegbci_decoding/`
- `outputs/tutorial_megfaces_main/<sensor-set>/`
- `outputs/tutorial_megfaces_spectral_envelopes/<sensor-set>/`
- `outputs/tutorial_megfaces_decoding/<sensor-set>/`

Raw data, derivatives, executed notebook output, and large models stay outside Git.

## Loading Wakeman–Henson MEG

One function handles the MEG data hand-off. Preparation is explicit because it
downloads roughly 5 GB and runs expensive Maxwell/ICA processing per participant:

```python
from pca_neural_trajectories import load_wakeman_henson

meg = load_wakeman_henson(
    "data/wakeman_henson",
    subjects=("01",),
    prepare=True,
    sensor_set="all_sensors",
)
```

Later calls omit `prepare=True`. Pass `spectral=True` to prepare/load the separate
padded derivative set used by the Hilbert-envelope tutorial. `sensor_set` defaults
to `all_sensors`; the optional `sensors_occipital`, `sensors_temporal`, and
`sensors_occipito_temporal` choices use MNE's Neuromag VectorView helmet selections.
They are sensor-position subsets, not source-localized cortical ROIs.

## Full EEG analysis

The headless EEG workflow applies the notebook's four-condition execution/imagination
analysis to the full cohort and records run provenance:

```bash
python scripts/analysis_eegbci_main.py --smoke --output outputs/smoke
python scripts/analysis_eegbci_main.py --output outputs/eegbci_main
```

An existing completed run is skipped unless `--no-resume` is passed. Reported sample
size is always the set that loaded and validated, not the requested number.

The nonlinear EEGBCI tutorial has its own headless companion. It saves all tables,
interactive figures, fitted reducers, arrays, a manifest, and a standalone HTML
report organized into the same ten explained steps as the notebook. PNG/SVG copies
are also written when Kaleido's browser backend is available:

```bash
python scripts/analysis_eegbci_nonlinear.py --skip-prepare
```

The EEGBCI decoding notebook also has an equivalent headless workflow. It saves
the raw `ExperimentResult` objects, tidy fold/prediction/split exports, summary
tables, interactive and static figures, provenance, and the same self-contained
ten-step report as the notebook:

```bash
python scripts/analysis_eegbci_decoding.py
```

The main MEG Faces tutorial has the same kind of executable companion. It reads
the already-prepared, subject-wise whitened Wakeman–Henson derivatives and runs
the complete shared/participant/focused PCA workflow. Every table, time-resolved
metric, interactive figure, fitted reducer, array, permutation null, and manifest
is saved alongside a self-contained HTML report:

```bash
python scripts/analysis_megfaces_main.py --smoke
python scripts/analysis_megfaces_main.py --n-perm 1000
python scripts/analysis_megfaces_main.py --sensor-set sensors_occipital
```

The default output is `outputs/megfaces_main/<sensor-set>/`. The script never downloads or
preprocesses MEG data implicitly; use `--derivatives-root` when the prepared
derivatives are stored outside the documented MNE data location.

The spectral-envelope tutorial also has a complete report script. It uses the
prepared long epochs by default; preparation from local raw data remains an
explicit option because it is expensive:

```bash
python scripts/analysis_megfaces_spectral_envelopes.py --smoke
python scripts/analysis_megfaces_spectral_envelopes.py --n-perm 1000
python scripts/analysis_megfaces_spectral_envelopes.py --prepare
python scripts/analysis_megfaces_spectral_envelopes.py --sensor-set sensors_temporal
```

Outputs are written to `outputs/megfaces_spectral_envelopes/<sensor-set>/` and include the
offline HTML report, all figures and tables, processed arrays, PCA reducers,
the family-corrected null, and provenance manifests.

The MEG Faces decoding notebook has a matching report script for the direct
sensor and transductively aligned-PCA LOSO experiments:

```bash
python scripts/analysis_megfaces_decoding.py --smoke
python scripts/analysis_megfaces_decoding.py
python scripts/analysis_megfaces_decoding.py --sensor-set sensors_occipito_temporal
```

It saves the raw `ExperimentResult` exports, split audit, temporal and fold-level
scores, participant summaries, figures, analysis arrays, manifests, and offline
HTML report under `outputs/megfaces_decoding/<sensor-set>/`.

## Verification

```bash
.venv/bin/ruff check pca_neural_trajectories tests scripts
.venv/bin/python -m pytest
```
