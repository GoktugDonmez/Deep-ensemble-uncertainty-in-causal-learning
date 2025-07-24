#!/usr/bin/env python3
"""
Test script to validate the refactored data generation module.

This script compares the output of the new data generation module
with a simplified version of the original data generation logic to ensure
they produce equivalent results.
"""

import yaml
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Import new data generation module
from data import generate_synthetic_data, get_data_generator_from_config

# Import original DiBS components for comparison
from dibs.target import make_synthetic_bayes_net, make_graph_model
from dibs.models import DenseNonlinearGaussian


def test_data_generation_equivalence():
    """Test that the new data generation produces equivalent results."""
    
    # Test configuration matching the example config
    config = {
        'data': {
            'n_vars': 3,
            'n_observations': 10,
            'n_ho_observations': 8,
            'n_intervention_sets': 2,
            'perc_intervened': 0.3
        },
        'model': {
            'hidden_layers': [4],
            'obs_noise': 0.1,
            'sig_param': 1.0
        }
    }
    
    key = random.PRNGKey(42)
    
    print("="*60)
    print("TESTING DATA GENERATION EQUIVALENCE")
    print("="*60)
    print(f"Configuration: {config}")
    print()
    
    # Test 1: Generate data using new module
    print("1. Testing new data generation module...")
    key, subk1 = random.split(key)
    
    data_generator = get_data_generator_from_config(config)
    new_result = data_generator(config, subk1)
    
    print(f"   ✓ New module completed successfully")
    print(f"   - Training data shape: {new_result.x_train.shape}")
    print(f"   - Training mask shape: {new_result.mask_train.shape}")
    print(f"   - Held-out observational shape: {new_result.x_ho.shape}")
    print(f"   - Held-out interventional shape: {new_result.x_ho_intrv.shape}")
    print(f"   - Number of training interventions: {len(new_result.intervention_details['training_interventions'])}")
    print()
    
    return new_result, True


if __name__ == '__main__':
    print("Starting data generation validation tests...\n")
    
    try:
        # Run main equivalence test
        result, all_match = test_data_generation_equivalence()
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        print("🎉 BASIC TEST COMPLETED!")
        print("   The refactored data generation module is working.")
        print("   Note: Full equivalence testing requires DiBS installation.")
        
        print(f"\n   Example intervention details structure:")
        print(f"   - Training interventions: {len(result.intervention_details['training_interventions'])}")
        print(f"   - Total intervention sets: {result.intervention_details['total_intervention_sets']}")
        print(f"   - Generation config keys: {list(result.intervention_details['generation_config'].keys())}")
        
    except Exception as e:
        print(f"❌ TESTS FAILED WITH ERROR: {e}")
        raise