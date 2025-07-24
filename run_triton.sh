#!/bin/bash
#SBATCH --job-name=causal_experiment
#SBATCH --output=logs/slurm_output_%j.txt
#SBATCH --error=logs/slurm_error_%j.txt
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --gpus=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Load the mamba module
module load mamba

# Activate your conda environment (created by setup_triton_env.sh)
source activate causal_experiments

# Set Python path to include the DiBS submodule (absolute path)
PROJECT_ROOT="${PWD}"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/dibs"

# Change to causal_experiments directory so results are saved there
cd causal_experiments

# Run the experiment
# Now using relative paths from within causal_experiments/
srun python run_experiment.py --config configs/dibs_vs_ensemble.yaml
