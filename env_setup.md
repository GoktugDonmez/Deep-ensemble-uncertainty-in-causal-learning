# Environment Setup for Triton

## Quick Setup (First Time Only)

### 1. Clone Repository
```bash
ssh your_username@triton.aalto.fi
cd $WRKDIR
git clone --recursive https://github.com/your_username/dibs_deep_ensembles.git
cd dibs_deep_ensembles
```

### 2. Setup Environment
```bash
chmod +x setup_triton_env.sh
./setup_triton_env.sh
```

### 3. Update Partition
Edit `run_triton.sh` and change:
```bash
#SBATCH --partition=batch-bdw  # or batch-csl, batch-skl, etc.
```

Check available partitions with: `sinfo`

## Running Experiments

### Submit Job
```bash
sbatch run_triton.sh
```

### Monitor
```bash
squeue --me                           # Check job status
tail -f logs/slurm_output_JOBID.txt   # Watch output
```

## Manual Testing (Optional)
```bash
module load mamba
source activate causal_experiments
PROJECT_ROOT="${PWD}"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/dibs"
cd causal_experiments
python run_experiment.py --config configs/dibs_vs_ensemble.yaml
```

## Environment Details

- **Environment**: `causal_experiments` (stored in `$WRKDIR/.conda_envs/`)
- **Python**: 3.8.3 with JAX 0.4.13
- **DiBS**: Installed in development mode
- **Results**: Saved to `causal_experiments/results/`
- **Logs**: Saved to `logs/slurm_*.txt`

## Troubleshooting

- **Environment not found**: Use `source activate causal_experiments` (no trailing slash)
- **Python not found**: Make sure environment is activated
- **Import errors**: Check `PYTHONPATH` includes DiBS directory
- **Partition errors**: Run `sinfo` to see available partitions

## Configuration Files

- `configs/dibs_vs_ensemble.yaml` - DiBS vs Deep Ensemble comparison
- `configs/heterogeneous_ensemble.yaml` - Configurable ensemble experiments

**Status**: ✅ Fully configured and tested