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