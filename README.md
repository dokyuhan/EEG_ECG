# Project Environment Setup

Instructions for setting up the required environment for this project using both Conda and pip.

## Environment Setup

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

## Environment Files

This project includes two environment definition files:

- `conda_env.yml`: Contains the Conda environment specifications
- `pip_packages.txt`: Contains additional pip packages needed

## Troubleshooting

- If you encounter conflicts between Conda and pip packages, try installing the Conda packages first, then the pip packages.
- For version-specific issues, check the compatibility requirements in the environment files.
- Some packages may require additional system dependencies. Refer to their documentation for installation instructions.

## Project Dependencies

The main dependencies for this project include:
- Data processing: numpy, pandas
- Visualization: matplotlib, seaborn
- Analysis: scikit-learn
- File handling: h5py, scipy

For a complete list of dependencies, refer to the environment files.
