#!/bin/bash
# Setup script for Triton environment

echo "Setting up environment for DiBS Deep Ensembles on Triton..."

# Load mamba module (faster than conda)
module load mamba

# Create a new environment specifically for this project
# Using the DiBS requirements but with a project-specific name
mamba env create -f dibs/environment.yml -n causal_experiments

# Activate the environment
source activate causal_experiments

# Install additional dependencies needed for your project
pip install pyyaml  # For YAML config files

# Install DiBS in development mode from the submodule
cd dibs
pip install -e .
cd ..

echo "Environment setup complete!"
echo "To use this environment on Triton:"
echo "1. module load mamba"
echo "2. conda activate causal_experiments"
echo "3. export PYTHONPATH=\"\${PYTHONPATH}:\${PWD}/dibs\"" 