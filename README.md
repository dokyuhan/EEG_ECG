# EEG ECG analysis Repository

This repository provides a comprehensive pipeline for analyzing EEG and ECG data using Canonical Correlation Analysis (CCA). The project is structured to support both trial-wise and session-level analyses, offering flexible tools for preprocessing, CCA computation, and post-hoc interpretation of neural–cardiac coupling.

## 🛠️ Environment Setup

Follow these instructions to set up the project environment using Conda and pip.

### 1. Create and activate the Conda environment

```bash
# Create a new Conda environment from the conda_env.yml file
conda env create -f conda_env.yml

# Activate the environment
conda activate myproject  # Replace 'myproject' with the actual environment name
```

### 2. Install additional pip packages

Some packages are installed via pip and are not managed by Conda. Install these with:

```bash
# Install pip packages from pip_packages.txt
pip install -r pip_packages.txt
```

## 📂 Environment Files

This project includes two environment definition files:

- `conda_env.yml`: Contains the Conda environment specifications
- `pip_packages.txt`: Contains additional pip packages needed

## 🧩 Troubleshooting

- If you encounter conflicts between Conda and pip packages, try installing the Conda packages first, then the pip packages.
- For version-specific issues, check the compatibility requirements in the environment files.
- Some packages may require additional system dependencies. Refer to their documentation for installation instructions.

## 📦 Project Dependencies

Key libraries used in this project include:

- Data processing: `numpy`, `pandas`
- Visualization: `matplotlib`, `seaborn`
- Analysis: `scikit-learn`
- File handling: `h5py`, `scipy`

Refer to the environment files for the full list of dependencies.


# 🧠 Code Structure and Execution Guide

The codebase is organized by analysis stage. Below is an overview of key components:

### 1. 🔄 Raw Data Preprocessing
Folder: **Raw Data Analysis/**

This folder contains the preprocessing scripts that extract and organize the raw EEG and ECG data from `.mat` files into `.csv` files for further analysis. There are two types of preprocessing available:

- **Trial-wise Analysis:** 
    
    * Each trial is processed individually (e.g., `trial1/subject_01.mat`) and saved as `trialX/subject_YY.csv`.

    **Use case:** Run CCA per trial across all subjects.

- **Session-level Analysis (Full Data Storage Folder):** 

    * Loads all trials and merges them into a single `.csv` file per subject.

    **Use case:** Analyze the entire session per subject instead of breaking it by trial.

    **Input format:** Folder path containing one `.csv` file per subject, each aggregating all trials.

### 2. 📊 CCA Analysis
- Group-level Analysis
    - File: **CCA_analysis.py**

        - Performs pooled CCA analysis for a single trial across multiple subjects.

        - Computes canonical correlations, Haufe-transformed weights, and stores results for interpretation.

                Input: CSV folder file organized by trial and subject (e.g., trial1/subject_01.csv, ..., subject_09.csv).

                Outputs: subject_results_csv/ with EEG/ECG weights, Haufe activations, and correlation values.
                
                * row_feature_contributions.csv — per-sample weighted contributions used for further analysis.

- Individual Subject-level Analysis

    - File: **Individual_CCA_analysis.py**

        - Computes CCA for a single subject.
        
        - Outputs subject-specific canonical correlations and EEG/ECG contributions.

                Input: EEG and ECG CSV for a single subject.

                Outputs: Canonical correlation results, subject_results_csv/ with weights, contributions, and canonical variates.


### 3. 📈 Post-CCA Analysis Tools

These scripts allow further exploration of EEG feature relevance and stability.

- **analyse_EEG_weights.py**
    
    - Loads EEG weights from `eeg_weights.csv`.

    - Sorts frequency bins by absolute contribution to CCA.

- **overall_band_weights.py**

    - Aggregates frequency bin weights into canonical EEG bands (Delta, Theta, Alpha, Beta, Gamma).

    - Computes both:

        - Total contribution per band.

        - Normalized contribution (per frequency bin).

- **regresion_analysis.py**

    - Performs regression between EEG band contributions and ECG canonical weights.

    - Provides:

        - Univariate regression (R², coefficients, p-values).

        - Multivariate model R².

        - Visualizes R² values using bar plots.

- **stability_eeg_weights_analysis.py**

    - Evaluates the stability of EEG frequency contributions across trials.

    - Computes stability score = mean /  standard deviation (std) for each bin.

    - Outputs the top n most stable EEG bins.

