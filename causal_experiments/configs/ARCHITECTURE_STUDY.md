# Core Experiment Architecture Study

## Overview
This document defines the systematic parameter study for comparing DiBS SVGD vs. Deep Ensemble methods in causal discovery. The architecture explores different neural network configurations and training hyperparameters to understand their impact on causal structure learning.

## Experimental Design

### Parameter Dimensions
The study systematically varies the following parameters:

1. **Neural Network Architecture (3 variants)**:
   - `[5]`: Single hidden layer with 5 neurons
   - `[10]`: Single hidden layer with 10 neurons  
   - `[5, 5]`: Two hidden layers with 5 neurons each

2. **Activation Functions (2 variants)**:
   - `relu`: Rectified Linear Unit
   - `tanh`: Hyperbolic Tangent

3. **Signal Parameter (2 variants)**:
   - `1.0`: Standard Gaussian prior variance
   - `2.0`: Increased Gaussian prior variance

4. **Learning Rates (4 variants)**:
   - `5e-3`: High learning rate
   - `1e-3`: Standard learning rate
   - `1e-4`: Conservative learning rate
   - `2e-3`: Medium-high learning rate

5. **Optimizers (2 variants)**:
   - `adam`: Adaptive Moment Estimation
   - `rmsprop`: Root Mean Square Propagation

6. **Random Seeds (5 variants)**:
   - `42, 123, 456, 789, 999`: Different initialization seeds

7. **Runs per Configuration (4 runs)**:
   - Multiple runs per configuration for statistical robustness

### Total Experiment Count
**Total Configurations**: 3 × 2 × 2 × 4 × 2 × 5 × 4 = **1,920 experiments**

Each configuration tests both:
- DiBS SVGD (20 particles)
- Deep Ensemble (20 independent runs)

### Fixed Parameters
- **Number of variables**: 20
- **Training observations**: 100  
- **Hold-out observations**: 100
- **Intervention sets**: 3
- **Percentage intervened**: 10%
- **Training steps**: 2000
- **Observation noise**: 0.1

## Configuration Files

### Base Configuration
- `configs/base_config.yaml`: Defines parameter grids and fixed settings

### Generated Configurations  
- `configs/generated/`: Individual experiment configurations
- Each file named as `config_XXXX_run_Y.yaml` where:
  - `XXXX`: 4-digit configuration ID (0000-1919)
  - `Y`: Run number (0-3)

### Configuration Generator
- `utils/config_generator.py`: Script to generate all experimental configurations
- Usage: `python3 utils/config_generator.py`

## Evaluation Metrics
Each experiment measures:
- **E-SHD**: Expected Structural Hamming Distance
- **AUROC**: Area Under ROC Curve  
- **NegLL (Observational)**: Negative Log-Likelihood on observational data
- **NegLL (Interventional)**: Negative Log-Likelihood on interventional data

## Expected Outcomes
This study will provide insights into:
1. Optimal neural network architectures for causal discovery
2. Impact of activation functions on structure learning
3. Learning rate sensitivity across methods
4. Optimizer performance comparison
5. Variance across random initializations
6. Relative performance of SVGD vs. Deep Ensemble approaches

## Usage

### Generate Configurations
```bash
cd causal_experiments
python3 utils/config_generator.py --base-config configs/base_config.yaml --output-dir configs/generated
```

### Run Single Experiment
```bash
python3 run_experiment.py configs/generated/config_0000_run_0.yaml
```

### Run All Experiments (Sequential)
```bash
# Generate batch script
python3 utils/config_generator.py --generate-batch

# Execute
./run_all_experiments.sh
```

### Run with SLURM (Parallel)
```bash
# Submit jobs to cluster
for config in configs/generated/*.yaml; do
    sbatch run_triton.sh $config
done
```

## File Structure
```
causal_experiments/
├── configs/
│   ├── base_config.yaml          # Parameter grid definition
│   ├── generated/                # Generated experiment configs
│   └── ARCHITECTURE_STUDY.md     # This documentation
├── utils/
│   └── config_generator.py       # Configuration generator
├── learners/                     # Learning algorithm implementations
├── run_experiment.py             # Main experiment runner
└── results/                      # Experiment outputs
```
