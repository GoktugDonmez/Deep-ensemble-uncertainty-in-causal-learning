# CSV Results Tracking Implementation

## Overview

This implementation provides a robust CSV tracking system for experiment results that meets all the acceptance criteria specified in Ticket 3. The system automatically saves and appends experiment results to a master CSV file, enabling comprehensive analysis and comparison of different learners and configurations.

## Key Features

✅ **Automatic CSV Creation**: Creates CSV with proper headers if file doesn't exist
✅ **Safe Appending**: Appends new results without overwriting existing data  
✅ **Comprehensive Metadata**: Captures all required fields including timestamps, configurations, and metrics
✅ **Timing Tracking**: Records training and evaluation times for performance analysis
✅ **Multiple Learner Support**: Works with DiBS SVGD, Deep Ensemble, and Configurable Ensemble learners
✅ **Analysis Tools**: Includes utilities for generating comparison tables and summaries

## Implementation Details

### Core Components

1. **CSVResultsTracker** (`causal_experiments/utils/csv_tracker.py`)
   - Main tracking class that handles CSV creation and data appending
   - Robust configuration extraction and data formatting
   - Support for nested configuration parameters

2. **BaseLearner Updates** (`causal_experiments/learners/base_learner.py`)
   - Added timing tracking to all learners
   - Integrated CSV saving functionality
   - Added helper methods for timing and result export

3. **Experiment Integration** (`causal_experiments/run_experiment.py`)
   - Automatic CSV saving after each learner completes
   - Results saved to experiment-specific directory

4. **Analysis Tools** (`causal_experiments/utils/analysis.py`)
   - ExperimentAnalyzer class for generating comparison tables
   - Support for various aggregation methods and filtering
   - Export functionality for different table formats

### CSV Schema

The CSV includes the following columns (28 total):

#### Identification
- `timestamp`: ISO format timestamp of when results were saved
- `run_id`: Unique identifier for this experiment run
- `experiment_name`: Name from configuration file
- `learner_name`: Name of the learner (e.g., 'SVGD', 'DeepEnsemble')

#### Configuration Parameters
- `n_particles`: Number of particles for SVGD methods
- `n_ensemble_runs`: Number of ensemble members for ensemble methods
- `learning_rate`: Learning rate from training config
- `optimizer`: Optimizer type from training config
- `net_depth`: Depth of neural network (derived from hidden_layers)
- `net_width`: Width of neural network (max from hidden_layers)
- `n_steps`: Number of training steps
- `obs_noise`: Observation noise parameter
- `sig_param`: Sigma parameter from model config

#### Data Parameters
- `n_vars`: Number of variables in the causal graph
- `n_observations`: Number of observational data points
- `n_ho_observations`: Number of held-out observations
- `n_intervention_sets`: Number of intervention sets
- `perc_intervened`: Percentage of variables intervened upon

#### Seeds and Reproducibility
- `inference_seed`: Seed used for inference (learner-specific or global)
- `random_seed`: Global random seed

#### Metrics
- `eshd`: Expected Structural Hamming Distance
- `auroc`: Area Under ROC Curve
- `negll_obs`: Negative log-likelihood on observational data
- `negll_intrv`: Negative log-likelihood on interventional data

#### Performance Metadata
- `training_time_seconds`: Time spent training
- `evaluation_time_seconds`: Time spent evaluating
- `total_time_seconds`: Total execution time

## Usage

### Basic Usage in Experiments

The CSV tracking is automatically integrated into the experiment runner. Simply run experiments as usual:

```bash
cd causal_experiments
python run_experiment.py --config configs/dibs_vs_ensemble.yaml
```

Results will be automatically saved to `{results_dir}/experiment_results.csv`.

### Manual CSV Tracking

```python
from utils.csv_tracker import get_tracker

# Initialize tracker
tracker = get_tracker("my_results.csv")

# Save results
tracker.save_results(
    config=experiment_config,
    learner_name="SVGD",
    metrics={'eshd': 3.5, 'auroc': 0.82, 'negll_obs': -2.1, 'negll_intrv': -1.8},
    training_time=45.2,
    evaluation_time=5.3
)
```

### Analysis and Comparison Tables

```python
from utils.analysis import ExperimentAnalyzer

# Load and analyze results
analyzer = ExperimentAnalyzer("experiment_results.csv")

# Print comprehensive summary
analyzer.print_summary_report()

# Get comparison tables
comparison_tables = analyzer.get_comparison_table()
for metric, table in comparison_tables.items():
    print(f"\n{metric.upper()} Comparison:")
    print(table)

# Export analysis results
analyzer.export_comparison_tables("analysis_output/")
```

### Generating Comparison Tables

The system generates several types of comparison tables:

1. **Summary by Learner**: Average metrics across all runs for each learner
2. **Configuration Comparison**: Performance across different parameter settings
3. **Detailed Results**: Complete results with all parameters and metrics

Example output format:
```
                    eshd    auroc  negll_obs  negll_intrv
learner_name                                           
DeepEnsemble        2.80     0.87      -2.40       -2.10
SVGD                3.50     0.82      -2.10       -1.80
```

## File Structure

```
causal_experiments/
├── utils/
│   ├── csv_tracker.py          # Main CSV tracking functionality
│   ├── analysis.py             # Analysis and comparison tools
│   └── __init__.py
├── learners/
│   ├── base_learner.py         # Updated with timing and CSV integration
│   ├── dibs_svgd_learner.py    # Updated with timing tracking
│   ├── deep_ensemble_learner.py    # Updated with timing tracking
│   └── configurable_ensemble_learner.py  # Updated with timing tracking
├── run_experiment.py           # Updated with automatic CSV saving
└── configs/
    └── *.yaml                  # Experiment configurations
```

## Example Workflow

1. **Run Experiments**: Execute experiments with different configurations
   ```bash
   python run_experiment.py --config configs/dibs_vs_ensemble.yaml
   python run_experiment.py --config configs/deep_ensemble_v1.yaml
   ```

2. **Analyze Results**: Generate comparison tables and summaries
   ```python
   from utils.analysis import analyze_experiments
   analyzer = analyze_experiments("results/*/experiment_results.csv")
   ```

3. **Export Tables**: Create publication-ready comparison tables
   ```python
   analyzer.export_comparison_tables("paper_tables/")
   ```

## Benefits

1. **Automated Tracking**: No manual intervention required
2. **Comprehensive Data**: Captures all relevant parameters and metrics
3. **Reproducibility**: Full parameter tracking enables exact reproduction
4. **Easy Analysis**: Built-in tools for generating comparison tables
5. **Flexible Export**: Multiple output formats for different use cases
6. **Robust Implementation**: Handles missing values and type conversions gracefully

## Testing

Basic functionality tested with mock data:
- CSV creation and header generation
- Data appending without overwriting
- Configuration parameter extraction
- Metric formatting and storage
- Analysis table generation

Run tests with:
```bash
python3 simple_csv_test.py  # Basic CSV functionality
```

## Future Enhancements

1. **Statistical Significance Testing**: Add p-values and confidence intervals
2. **Visualization Integration**: Automatic plot generation from CSV data
3. **Real-time Monitoring**: Live experiment tracking dashboard
4. **Advanced Filtering**: More sophisticated query capabilities
5. **Database Backend**: Optional database storage for large-scale experiments
