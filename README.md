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

## Running the Tutorial

The tutorial is available both as a standard Python script and as an interactive Jupyter Notebook. Both contain the same content and generate the same outputs. The tutorial uses the PhysioNet EEG BCI dataset, which will be downloaded automatically (approx. 2.5 GB) upon the first run.

### Option 1: Running the Python Script

To run the end-to-end analysis from the command line:

```bash
# Ensure your virtual environment is activated
source .venv/bin/activate

# Run the python script
python run_tutorial.py
```

The script will download the data, process it, and save all generated interactive Plotly figures and CSV reports into a newly created `outputs/tutorial_trajectories/` folder.

### Option 2: Running the Jupyter Notebook

For an interactive experience where you can view the code and plots cell-by-cell:

1. The required Jupyter dependencies are already installed if you ran `pip install -e .`.

2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter lab
   ```

3. Open the `run_tutorial.ipynb` file in your browser and run the cells sequentially. 
*(Alternatively, you can open and run the notebook directly inside VS Code or another compatible IDE by selecting the `.venv` kernel).*

## Outputs

All generated assets are saved in the `outputs/tutorial_trajectories/` directory, organized as follows:
- `figures/`: Interactive HTML files of the Plotly trajectories and embeddings (e.g., Exec vs. Imag hands, 3D PCA trajectories).
- `tables/`: CSV summaries of the conditions and trial metadata used in the analysis.
