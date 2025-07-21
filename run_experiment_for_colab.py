# %% markdown
# # DiBS vs Deep Ensemble Causal Discovery Experiment
# 
# This notebook runs the DiBS SVGD vs Deep Ensemble comparison experiment.
# **Make sure to enable GPU runtime for faster training!**

# %% code
# Install dependencies and setup
%pip install --quiet jax jaxlib numpy igraph imageio matplotlib scikit-learn tqdm pyyaml dibs-lib

import jax
print(f"JAX backend: {jax.default_backend()}")

# %% code
# Import libraries
import argparse
import yaml
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import importlib
import re
import sys
from jax import random, tree_util
import igraph as ig

# Add paths for imports
sys.path.append('.')
sys.path.append('causal_experiments')

# Import from DiBS and our modules
from dibs.target import make_synthetic_bayes_net, make_graph_model
from dibs.models import DenseNonlinearGaussian

# %% code
# Utility functions
def _pascal_to_snake(name):
    """Converts a PascalCase string to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_learner(learner_config, global_config):
    """Dynamically imports and instantiates a learner class based on the config."""
    class_name = learner_config['type']
    module_name = _pascal_to_snake(class_name)
    
    try:
        learner_module = importlib.import_module(f"learners.{module_name}")
        learner_class = getattr(learner_module, class_name)
        
        # Merge global config with learner-specific config
        full_config = {**global_config}
        full_config.update(learner_config)
        
        return learner_class(full_config)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find learner class {class_name} in module {module_name}.py") from e

# %% code
# Load configuration
config_path = 'causal_experiments/configs/dibs_vs_ensemble.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
print("--- Configuration Loaded ---")
print(yaml.dump(config, default_flow_style=False))

# %% code
# Setup environment
key = random.PRNGKey(config['random_seed'])
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join("results", f"{timestamp}_{config['experiment_name']}")
os.makedirs(results_dir, exist_ok=True)
print(f"\n--- Results will be saved to: {results_dir} ---")

# %% code
# Generate Data
print("\n--- Generating Data ---")
key, subk = random.split(key)
graph_model = make_graph_model(n_vars=config['data']['n_vars'], graph_prior_str="sf")
generative_model = DenseNonlinearGaussian(
    n_vars=config['data']['n_vars'],
    hidden_layers=tuple(config['model']['hidden_layers']),
    obs_noise=config['model']['obs_noise'],
    sig_param=config['model']['sig_param']
)
data_details = make_synthetic_bayes_net(key=subk, **config['data'], graph_model=graph_model, generative_model=generative_model)

print(f"✅ Data generated successfully!")
print(f"  - Variables: {data_details.x.shape[1]}")
print(f"  - Training observations: {data_details.x.shape[0]}")
print(f"  - Held-out observations: {data_details.x_ho.shape[0]}")
print(f"  - True graph edges: {np.sum(data_details.g)}")

# %% markdown
# ## Prepare Training Data

# %% code
# Prepare combined training data
all_train_data = [data_details.x]
all_train_masks = [jnp.zeros_like(data_details.x, dtype=bool)]
for i in range(config['data']['n_intervention_sets'] - 1):
    interv_dict, interv_x_train = data_details.x_interv[i]
    all_train_data.append(interv_x_train)
    mask_train_interv = jnp.zeros_like(interv_x_train, dtype=bool)
    intervened_nodes = list(interv_dict.keys())
    mask_train_interv = mask_train_interv.at[:, intervened_nodes].set(True)
    all_train_masks.append(mask_train_interv)
x_train = jnp.concatenate(all_train_data, axis=0)
mask_train = jnp.concatenate(all_train_masks, axis=0)

# Prepare held-out interventional data
interv_dict_ho, x_ho_intrv = data_details.x_interv[-1]
mask_ho_intrv = jnp.zeros_like(x_ho_intrv, dtype=bool)
intervened_nodes_ho = list(interv_dict_ho.keys())
mask_ho_intrv = mask_ho_intrv.at[:, intervened_nodes_ho].set(True)

print(f"Training data shape: {x_train.shape}")

# %% markdown
# ## Run All Learners

# %% code
# Run All Learners
results = {}
for learner_config in config['learners']:
    learner_name = learner_config['name']
    print(f"\n--- Preparing Learner: {learner_name} ---")
    
    # Create learner with merged config
    learner = get_learner(learner_config, config)
    
    # Train and evaluate
    particles = learner.train(x_train, mask_train, graph_model, generative_model)
    metrics = learner.evaluate(particles, data_details.x_ho, x_ho_intrv, mask_ho_intrv, data_details.g, graph_model, generative_model)
    
    results[learner_name] = {'metrics': metrics, 'particles': particles}
    print(f"✅ {learner_name} completed!")

# %% markdown
# ## Visualization & Results

# %% code
# Generate visualizations (if plotting utilities are available)
try:
    from utils.plotting import sample_posterior_predictive, plot_interventional_distributions
    
    print("\n--- Generating Visualizations ---")
    # Find strongest causal link for plotting
    g_true_igraph = ig.Graph.Adjacency(np.array(data_details.g).tolist())
    topological_order = g_true_igraph.topological_sorting()
    target_node_i = topological_order[-1]
    ancestors = g_true_igraph.subcomponent(target_node_i, mode='in')
    candidate_interv_nodes_j = [node for node in ancestors if node != target_node_i]
    parent_node_j = candidate_interv_nodes_j[0] if candidate_interv_nodes_j else 0
    interv_dict_plot = {parent_node_j: config['evaluation']['interv_value']}
    n_plot_samples = config['evaluation']['n_interv_samples']

    # Sample from ground truth
    key, subk = random.split(key)
    gt_obs_samples = generative_model.sample_obs(key=subk, n_samples=n_plot_samples, g=g_true_igraph, theta=data_details.theta, interv=None)
    key, subk = random.split(key)
    gt_interv_samples = generative_model.sample_obs(key=subk, n_samples=n_plot_samples, g=g_true_igraph, theta=data_details.theta, interv=interv_dict_plot)

    # Sample from learned models
    for name, result in results.items():
        gs, thetas = result['particles']
        n_particles = gs.shape[0]
        key, subk = random.split(key)
        result['obs_samples'] = sample_posterior_predictive(gs, thetas, generative_model, None, n_plot_samples // n_particles, target_node_i, subk)
        key, subk = random.split(key)
        result['interv_samples'] = sample_posterior_predictive(gs, thetas, generative_model, interv_dict_plot, n_plot_samples // n_particles, target_node_i, subk)

    # Plot and save
    plot_interventional_distributions(
        results,
        gt_samples=(gt_obs_samples, gt_interv_samples),
        interv_info=(parent_node_j, target_node_i, config['evaluation']['interv_value']),
        results_dir=results_dir
    )
    print("✅ Visualizations generated!")
    
except ImportError:
    print("⚠️ Plotting utilities not available - skipping visualizations")

# %% markdown
# ## Save Results

# %% code
# Save All Results
print("\n--- Saving All Results ---")
with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config, f)

# Save metrics for all learners
all_metrics = {name: res['metrics'] for name, res in results.items()}
with open(os.path.join(results_dir, 'metrics.yaml'), 'w') as f:
    yaml.dump(all_metrics, f)

# Save particles for each learner
for name, result in results.items():
    learner_dir = os.path.join(results_dir, name)
    os.makedirs(learner_dir, exist_ok=True)
    gs_np = np.array(result['particles'][0])
    thetas_np = tree_util.tree_map(lambda x: np.array(x), result['particles'][1])
    np.savez(os.path.join(learner_dir, 'particles.npz'), gs=gs_np)
    with open(os.path.join(learner_dir, 'thetas.pkl'), 'wb') as f:
        pickle.dump(thetas_np, f)

print("Results saved successfully.")
print(f"Check the results in: {results_dir}")

# %% markdown
# ## Display Results Summary

# %% code
# Display final results
print("\n" + "="*60)
print("EXPERIMENT RESULTS")
print("="*60)

for name, result in results.items():
    print(f"\n{name}:")
    print("-" * len(name))
    for metric_name, metric_value in result['metrics'].items():
        if isinstance(metric_value, (int, float)):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")

print("\n🎉 Experiment completed successfully!") 