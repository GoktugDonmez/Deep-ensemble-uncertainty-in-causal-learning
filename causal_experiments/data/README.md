# Data Generation Module

This module contains the refactored data generation functionality for causal discovery experiments.

## Overview

The data generation logic has been extracted from `run_experiment.py` into this separate, self-contained module to improve:
- **Modularity**: Data generation is now cleanly separated from experiment execution
- **Reusability**: The data generation functions can be easily imported and used in different contexts
- **Maintainability**: Changes to data generation logic are isolated to this module
- **Reproducibility**: Detailed intervention tracking enables exact reproduction of generated data

## Key Functions

### `generate_synthetic_data(config, key)`

The main data generation function that creates synthetic causal datasets including:
- Ground truth causal graph and parameters
- Observational training and held-out data  
- Multiple interventional training datasets
- One held-out interventional dataset for evaluation

**Returns**: `SyntheticDataResult` containing all generated data and detailed metadata

### `get_data_generator_from_config(config)`

Factory function that returns the appropriate data generation function based on configuration.

## Usage

```python
from data import generate_synthetic_data, get_data_generator_from_config

# Factory pattern (recommended)
data_generator = get_data_generator_from_config(config)
data_result = data_generator(config, key)

# Access generated data
x_train = data_result.x_train
mask_train = data_result.mask_train
```

## Enhanced Features

- **Detailed Intervention Tracking**: Complete intervention metadata for reproducibility
- **Ground Truth Preservation**: All model components saved for later use
- **Configuration Validation**: Clear error messages for invalid configurations

## Migration Compatibility

The refactoring maintains full compatibility with existing experiment code while providing enhanced functionality and better modularity.


# Data Generation Refactoring Summary

## Overview
This document summarizes the refactoring of data generation logic from `run_experiment.py` into a separate, self-contained module.

## Changes Made

### ✅ New Files Created
- `causal_experiments/data/synthetic_data.py` - Main data generation module
- `causal_experiments/data/__init__.py` - Updated module exports
- `causal_experiments/data/README.md` - Module documentation
- `causal_experiments/test_data_generation.py` - Validation script

### ✅ Modified Files
- `causal_experiments/run_experiment.py` - Updated to use new data generation module

### ✅ Key Improvements

#### 1. **Modularity**
- Data generation logic extracted into dedicated module
- Clean separation of concerns between data generation and experiment execution
- Easy to import and reuse in different contexts

#### 2. **Enhanced Tracking**
- Detailed intervention metadata for complete reproducibility
- Tracks intervened nodes, intervention values, and number of interventions
- Preserves ground truth information (graph, parameters, models)

#### 3. **Better Organization**
- Self-contained module with clear interfaces
- Factory pattern for extensibility to different data types
- Comprehensive return structure with `SyntheticDataResult` named tuple

#### 4. **Reproducibility**
- Intervention details saved to `intervention_details.yaml`
- Ground truth saved to `ground_truth.yaml` and `ground_truth_theta.pkl`
- Same random seed produces identical results

## Compatibility

### ✅ Backward Compatibility
- Existing experiments produce identical results for same random seed
- No changes required to configuration files
- All existing learners work without modification

### ✅ Interface Preservation
The refactored code maintains the same data structures expected by learners:
- `x_train`, `mask_train` for training
- `x_ho`, `x_ho_intrv`, `mask_ho_intrv` for evaluation
- `data_details.g`, `data_details.theta` for ground truth
- `graph_model`, `generative_model` for model objects

## Usage

### Before (in run_experiment.py):
```python
# 40+ lines of data generation logic mixed with experiment code
graph_model = make_graph_model(...)
generative_model = DenseNonlinearGaussian(...)
data_details = make_synthetic_bayes_net(...)
# Manual data processing and concatenation...
```

### After:
```python
# Clean, modular approach
from data import get_data_generator_from_config
data_generator = get_data_generator_from_config(config)
data_result = data_generator(config, key)
x_train = data_result.x_train  # Ready to use
```

## Future Extensions Enabled

The new modular structure enables:
- Real data loading support
- Different synthetic data generation strategies  
- Custom intervention types
- Alternative graph priors and generative models
- Easier testing and validation of data generation

## Validation

Run the test script to verify the refactoring:
```bash
cd causal_experiments
python test_data_generation.py
```

## Files Structure
```
causal_experiments/
├── data/
│   ├── __init__.py           # Module exports
│   ├── synthetic_data.py     # Main data generation logic
│   └── README.md            # Module documentation
├── run_experiment.py        # Updated to use new module
├── test_data_generation.py  # Validation script
└── REFACTORING_SUMMARY.md   # This file
```