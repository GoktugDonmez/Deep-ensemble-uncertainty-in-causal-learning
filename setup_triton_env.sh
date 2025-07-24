#!/bin/bash
# Setup script for Triton environment following Triton best practices

echo "Setting up environment for DiBS Deep Ensembles on Triton..."

# Load mamba module (faster than conda, recommended by Triton)
module load mamba

# Configure conda to use work directory instead of home (for quota reasons)
echo "Configuring conda directories..."
mkdir -p $WRKDIR/.conda_pkgs
mkdir -p $WRKDIR/.conda_envs

conda config --append pkgs_dirs ~/.conda/pkgs
conda config --append envs_dirs ~/.conda/envs
conda config --prepend pkgs_dirs $WRKDIR/.conda_pkgs
conda config --prepend envs_dirs $WRKDIR/.conda_envs

# Create a new environment specifically for this project
# Using the new project-specific environment file
echo "Creating causal_experiments environment from root environment.yml..."
mamba env create -f environment.yml -n causal_experiments

# Activate the environment
source activate causal_experiments

# Install DiBS in development mode from the submodule
echo "Installing DiBS in development mode..."
cd dibs
pip install -e .
cd ..

echo "Environment setup complete!"
echo "To use this environment on Triton:"
echo "1. module load mamba"
echo "2. source activate causal_experiments"
echo "3. export PYTHONPATH=\"\${PYTHONPATH}:\${PWD}/dibs\"" 