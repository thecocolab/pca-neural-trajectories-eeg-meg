# PCA-Based Trajectory Analysis for EEG and MEG

This repository contains the codebase and tutorial accompanying the paper:
**"A Primer on Low-Dimensional Neural Dynamics: PCA-Based Trajectory Analysis for EEG and MEG"**.

Included here is a hands-on tutorial that walks through an end-to-end EEG trajectory analysis. The tutorial demonstrates how to extract and visualize low-dimensional EEG/MEG neural manifolds using PCA.

## Requirements

This project relies on `coco-pipe` for dimensionality reduction and data handling, along with standard scientific Python libraries (`mne`, `numpy`, `plotly`, etc.).

## Installation Guide

Follow these steps to set up the environment and install the required dependencies:

1. **Clone the repository and navigate into it:**
   ```bash
   git clone https://github.com/thecocolab/pca-neural-trajectories-eeg-meg
   cd pca-neural-trajectories-eeg-meg
   ```

2. **Create and activate a virtual environment (Python 3.11 recommended):**
   ```bash
   # On macOS/Linux
   python3.11 -m venv .venv
   source .venv/bin/activate
   
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the project in editable mode:**
   ```bash
   pip install -e .
   ```
   *Note: This command will automatically install all necessary packages defined in `pyproject.toml`, including the specific branch of `coco-pipe`, `jupyterlab`, and other dependencies.*

## Running the Tutorials

This repository contains a series of interactive Jupyter Notebooks for hands-on learning. The tutorials use the PhysioNet EEG BCI dataset, which will be downloaded automatically (approx. 2.5 GB) upon the first run.

**Tutorial Series Roadmap:**
1. **`tutorial_pca_trajectories_eegbci.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/pca-neural-trajectories-eeg-meg/blob/main/tutorial_pca_trajectories_eegbci.ipynb): Introductory tutorial covering the core 10-step PCA workflow.
2. **`tutorial_pca_trajectory_advanced.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/pca-neural-trajectories-eeg-meg/blob/main/tutorial_pca_trajectory_advanced.ipynb): Advanced tutorial diving deep into advanced trajectory metrics and analysis.
3. **`tutorial_pca_trajectory_decoding.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/pca-neural-trajectories-eeg-meg/blob/main/tutorial_pca_trajectory_decoding.ipynb): Decoding tutorial using trajectories for Brain-Computer Interface (BCI) decoding.
*(A future tutorial will cover applying these methods to MEG visual processing data).*

To run the tutorials:

1. The required Jupyter dependencies are already installed if you ran `pip install -e .`.

2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter lab
   ```

3. Open the notebook of your choice (we recommend starting with `tutorial_pca_trajectories_eegbci.ipynb`) in your browser and run the cells sequentially. 
*(Alternatively, you can open and run the notebooks directly inside VS Code or another compatible IDE by selecting the `.venv` kernel).*

## Outputs

All generated assets for the introductory tutorial are saved in the `outputs/tutorial_eegbci/` directory, organized as follows:
- `report_tutorial_eegbci.html`: The standalone interactive HTML dashboard.
- `figures/`: High-resolution static figures in SVG format.
- `artifacts/`: Saved PCA models and metadata required for the advanced tutorials.
