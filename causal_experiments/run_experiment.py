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
import mlflow

# Import from our new structure
from data import generate_synthetic_data, get_data_generator_from_config
from utils.plotting import sample_posterior_predictive, plot_interventional_distributions
from utils.mlflow_utils import log_config_as_params, log_artifacts_selectively


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

    # 2. Setup MLflow
    mlflow.set_experiment("dibs_vs_ensembles")
    
    # 3. Setup Environment
    key = random.PRNGKey(config['random_seed'])
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", f"{timestamp}_{config['experiment_name']}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n--- Results will be saved to: {results_dir} ---")

    # Start MLflow run for this experiment
    with mlflow.start_run():
        # Log experiment configuration as parameters
        log_config_as_params(config)
        
        # 4. Generate Data
        print("\n--- Generating Data ---")
        key, subk = random.split(key)

        
        # Use the new data generation module
        data_generator = get_data_generator_from_config(config)
        data_result = data_generator(config, subk)
        
        # Extract data components for compatibility with existing code
        x_train = data_result.x_train
        mask_train = data_result.mask_train
        x_ho_intrv = data_result.x_ho_intrv
        mask_ho_intrv = data_result.mask_ho_intrv
        
        # Extract ground truth for compatibility
        data_details = type('DataDetails', (), {
            'x_ho': data_result.x_ho,
            'g': data_result.ground_truth['graph'],
            'theta': data_result.ground_truth['theta']
        })()
        graph_model = data_result.ground_truth['graph_model']
        generative_model = data_result.ground_truth['generative_model']

        # 5. Run All Learners
        results = {}
        all_metrics = {}
        for learner_config in config['learners']:
            learner_name = learner_config['name']
            print(f"\n--- Preparing Learner: {learner_name} ---")
            
            # Create learner with merged config
            learner = get_learner(learner_config, config)
            
            # Train and evaluate
            particles = learner.train(x_train, mask_train, graph_model, generative_model)
            metrics = learner.evaluate(particles, data_details.x_ho, x_ho_intrv, mask_ho_intrv, data_details.g, graph_model, generative_model)
            
            # Log learner-specific parameters and metrics to MLflow
            log_config_as_params(learner_config, learner_name)
            
            # Log metrics with learner name prefix
            learner_metrics = {f"{learner_name}.{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(learner_metrics)
            
            # Keep CSV logging for backward compatibility (can be disabled later)
            csv_path = os.path.join(results_dir, "experiment_results.csv")
            learner.save_to_csv(config, learner_name, metrics, csv_path, run_id=f"{timestamp}_{learner_name}")
            
            results[learner_name] = {'metrics': metrics, 'particles': particles}
            all_metrics[learner_name] = metrics

        # 6. Visualization
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

        # 7. Save Results
        print("\n--- Saving All Results ---")
        with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        
        # Save detailed intervention information for reproducibility
        with open(os.path.join(results_dir, 'intervention_details.yaml'), 'w') as f:
            yaml.dump(data_result.intervention_details, f)
        
        # Save ground truth information (graph as numpy array for easy loading)
        ground_truth_save = {
            'graph': np.array(data_result.ground_truth['graph']).tolist(),
            # Note: theta and models are saved separately due to complex structure
        }
        with open(os.path.join(results_dir, 'ground_truth.yaml'), 'w') as f:
            yaml.dump(ground_truth_save, f)
        
        # Save ground truth theta parameters
        with open(os.path.join(results_dir, 'ground_truth_theta.pkl'), 'wb') as f:
            pickle.dump(data_result.ground_truth['theta'], f)
        
        # Save metrics for all learners
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
        
        # 8. Log artifacts to MLflow (excluding plots to keep tracking lightweight)
        print("\n--- Logging Artifacts to MLflow ---")
        log_artifacts_selectively(results_dir)
        
        # Log learner-specific artifacts
        for learner_name in results.keys():
            log_artifacts_selectively(results_dir, learner_name)
        
        print("Results saved successfully.")
        print(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a causal discovery experiment.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    main(args.config)