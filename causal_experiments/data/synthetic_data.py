"""
Synthetic data generation module for causal discovery experiments.

This module provides functionality to generate synthetic causal datasets with 
observational and interventional data, using the DiBS library for causal model generation.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple, Any, NamedTuple
import numpy as np

# Import from DiBS library
from dibs.target import make_synthetic_bayes_net, make_graph_model
from dibs.models import DenseNonlinearGaussian


class SyntheticDataResult(NamedTuple):
    """Container for synthetic data generation results."""
    x_train: jnp.ndarray  # Combined training data (observational + interventional)
    mask_train: jnp.ndarray  # Training intervention masks
    x_ho: jnp.ndarray  # Held-out observational data
    x_ho_intrv: jnp.ndarray  # Held-out interventional data  
    mask_ho_intrv: jnp.ndarray  # Held-out intervention masks
    
    # Detailed tracking of interventions
    intervention_details: Dict[str, Any]  # Complete intervention information
    ground_truth: Dict[str, Any]  # Ground truth graph and parameters


def generate_synthetic_data(config: Dict[str, Any], key: jax.random.PRNGKey) -> SyntheticDataResult:
    """
    Generate synthetic causal data for experiments.
    
    This function creates a complete synthetic dataset including:
    - A ground truth causal graph and parameters
    - Observational training and held-out data
    - Multiple sets of interventional data for training
    - One held-out interventional dataset for evaluation
    
    Args:
        config: Configuration dictionary containing data generation parameters.
                Expected to have 'data' and 'model' sections matching the YAML structure.
        key: JAX random key for reproducible generation
        
    Returns:
        SyntheticDataResult containing all generated data and metadata
    """
    
    # Extract configuration parameters
    data_config = config['data']
    model_config = config['model']
    
    # 1. Create ground truth models
    print("Creating ground truth causal models...")
    
    # Graph model defines structure prior
    graph_model = make_graph_model(
        n_vars=data_config['n_vars'], 
        graph_prior_str="sf"  # Scale-free network prior
    )
    
    # Generative model defines causal mechanisms
    generative_model = DenseNonlinearGaussian(
        n_vars=data_config['n_vars'],
        hidden_layers=tuple(model_config['hidden_layers']),
        obs_noise=model_config['obs_noise'],
        sig_param=model_config['sig_param']
    )
    
    # 2. Generate the complete synthetic Bayesian network
    print("Generating synthetic Bayesian network...")
    key, subk = random.split(key)
    
    data_details = make_synthetic_bayes_net(
        key=subk, 
        graph_model=graph_model, 
        generative_model=generative_model,
        **data_config
    )
    
    # 3. Process and combine training data
    print("Processing training data...")
    
    # Start with observational data
    all_train_data = [data_details.x]
    all_train_masks = [jnp.zeros_like(data_details.x, dtype=bool)]
    
    # Track intervention details for reproducibility
    training_interventions = []
    
    # Add interventional training data (all but the last intervention set)
    for i in range(data_config['n_intervention_sets'] - 1):
        interv_dict, interv_x_train = data_details.x_interv[i]
        
        # Add data
        all_train_data.append(interv_x_train)
        
        # Create intervention mask
        mask_train_interv = jnp.zeros_like(interv_x_train, dtype=bool)
        intervened_nodes = list(interv_dict.keys())
        mask_train_interv = mask_train_interv.at[:, intervened_nodes].set(True)
        all_train_masks.append(mask_train_interv)
        
        # Store intervention details
        intervention_info = {
            'intervention_set_id': i,
            'intervention_dict': interv_dict,
            'intervened_nodes': intervened_nodes,
            'intervention_values': [interv_dict[node] for node in intervened_nodes],
            'n_samples': interv_x_train.shape[0],
            'data_type': 'training'
        }
        training_interventions.append(intervention_info)
    
    # Combine all training data
    x_train = jnp.concatenate(all_train_data, axis=0)
    mask_train = jnp.concatenate(all_train_masks, axis=0)
    
    # 4. Process held-out interventional data
    print("Processing held-out interventional data...")
    
    # Use the last intervention set as held-out
    interv_dict_ho, x_ho_intrv = data_details.x_interv[-1]
    mask_ho_intrv = jnp.zeros_like(x_ho_intrv, dtype=bool)
    intervened_nodes_ho = list(interv_dict_ho.keys())
    mask_ho_intrv = mask_ho_intrv.at[:, intervened_nodes_ho].set(True)
    
    # Store held-out intervention details
    held_out_intervention = {
        'intervention_set_id': data_config['n_intervention_sets'] - 1,
        'intervention_dict': interv_dict_ho,
        'intervened_nodes': intervened_nodes_ho,
        'intervention_values': [interv_dict_ho[node] for node in intervened_nodes_ho],
        'n_samples': x_ho_intrv.shape[0],
        'data_type': 'held_out'
    }
    
    # 5. Compile comprehensive intervention details
    intervention_details = {
        'total_intervention_sets': data_config['n_intervention_sets'],
        'training_interventions': training_interventions,
        'held_out_intervention': held_out_intervention,
        'n_training_intervention_sets': len(training_interventions),
        'intervention_percentage': data_config.get('perc_intervened', None),
        'generation_config': {
            'n_vars': data_config['n_vars'],
            'n_observations': data_config['n_observations'],
            'n_ho_observations': data_config['n_ho_observations'],
            'n_intervention_sets': data_config['n_intervention_sets']
        }
    }
    
    # 6. Compile ground truth information
    ground_truth = {
        'graph': data_details.g,  # Adjacency matrix
        'theta': data_details.theta,  # Model parameters
        'graph_model': graph_model,
        'generative_model': generative_model
    }
    
    # 7. Data shape validation and logging
    print(f"Data generation complete:")
    print(f"  - Training data shape: {x_train.shape}")
    print(f"  - Training mask shape: {mask_train.shape}")
    print(f"  - Held-out observational shape: {data_details.x_ho.shape}")
    print(f"  - Held-out interventional shape: {x_ho_intrv.shape}")
    print(f"  - Number of variables: {data_config['n_vars']}")
    print(f"  - Training intervention sets: {len(training_interventions)}")
    print(f"  - Total samples in training: {x_train.shape[0]}")
    
    return SyntheticDataResult(
        x_train=x_train,
        mask_train=mask_train,
        x_ho=data_details.x_ho,
        x_ho_intrv=x_ho_intrv,
        mask_ho_intrv=mask_ho_intrv,
        intervention_details=intervention_details,
        ground_truth=ground_truth
    )


def get_data_generator_from_config(config: Dict[str, Any]):
    """
    Factory function to get the appropriate data generation function based on config.
    
    This function provides flexibility for future extension with different data generators.
    Currently returns the synthetic data generator, but can be extended to support
    different data generation strategies.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Data generation function
    """
    # For now, we only have synthetic data generation
    # Future versions could support real data loading, different synthetic models, etc.
    data_type = config.get('data', {}).get('type', 'synthetic')
    
    if data_type == 'synthetic' or 'type' not in config.get('data', {}):
        return generate_synthetic_data
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def reproduce_data_from_details(intervention_details: Dict[str, Any], 
                               ground_truth: Dict[str, Any],
                               key: jax.random.PRNGKey) -> SyntheticDataResult:
    """
    Reproduce synthetic data from saved intervention details and ground truth.
    
    This function allows exact reproduction of previously generated data using
    the detailed intervention information and ground truth parameters.
    
    Args:
        intervention_details: Intervention details from previous generation
        ground_truth: Ground truth graph and parameters
        key: JAX random key (should be the same as original for exact reproduction)
        
    Returns:
        SyntheticDataResult with reproduced data
    """
    # This function could be implemented for exact data reproduction
    # For now, this is a placeholder for future implementation
    raise NotImplementedError("Data reproduction functionality not yet implemented")