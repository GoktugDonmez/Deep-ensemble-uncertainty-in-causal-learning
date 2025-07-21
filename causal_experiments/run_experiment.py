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
from jax import random, tree_util
import igraph as ig

# Import from our new structure
from dibs.target import make_synthetic_bayes_net, make_graph_model
from dibs.models import DenseNonlinearGaussian
from utils.plotting import sample_posterior_predictive, plot_interventional_distributions


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
        # The learner-specific config takes precedence
        full_config = {**global_config}
        full_config.update(learner_config)
        
        return learner_class(full_config)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find learner class {class_name} in module {module_name}.py") from e

def main(config_path):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("--- Configuration Loaded ---")

    # 2. Setup Environment
    key = random.PRNGKey(config['random_seed'])
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", f"{timestamp}_{config['experiment_name']}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n--- Results will be saved to: {results_dir} ---")

    # 3. Generate Data
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

    # 4. Run All Learners
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

    # 5. Visualization
    print("\n--- Generating Visualizations ---")
    # Find strongest causal link for plotting
    g_true_igraph = ig.Graph.Adjacency(np.array(data_details.g).tolist())
    topological_order = g_true_igraph.topological_sorting()
    target_node_i = topological_order[-1]
    ancestors = g_true_igraph.subcomponent(target_node_i, mode='in')
    candidate_interv_nodes_j = [node for node in ancestors if node != target_node_i]
    # For simplicity, we just pick the first ancestor as the intervention target
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

    # 6. Save Results
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a causal discovery experiment.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    main(args.config)