import jax
import jax.numpy as jnp
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import os

# This is the posterior predictive sampling function from the notebook
def sample_posterior_predictive(gs, thetas, likelihood_model, interv_dict, n_samples_per_particle, obs_node, key):
    all_samples = []
    n_particles = gs.shape[0]
    
    if n_particles == 0:
        print("Warning: No particles provided for posterior predictive sampling")
        return None
    
    # Ensure we have at least 1 sample per particle
    if n_samples_per_particle <= 0:
        print(f"Warning: n_samples_per_particle={n_samples_per_particle} <= 0. Setting to 1.")
        n_samples_per_particle = 1

    successful_particles = 0
    for p in range(n_particles):
        try:
            g_matrix = np.array(gs[p])
            particle_theta = jax.tree_util.tree_map(lambda x: x[p], thetas)
            g_igraph = ig.Graph.Adjacency(g_matrix.tolist())
            
            # Check if graph is a DAG
            if not g_igraph.is_dag():
                continue

            key, subk = jax.random.split(key)
            samples = likelihood_model.sample_obs(
                key=subk,
                n_samples=n_samples_per_particle,
                g=g_igraph,
                theta=particle_theta,
                interv=interv_dict
            )
            
            # Extract samples for the target node
            if samples is not None and samples.shape[0] > 0:
                all_samples.append(samples[:, obs_node])
                successful_particles += 1
            
        except Exception as e:
            print(f"Warning: Failed to sample from particle {p}: {e}")
            continue

    if not all_samples:
        print(f"Warning: No valid samples collected from {n_particles} particles. "
              f"Successful particles: {successful_particles}")
        # Return a dummy array to prevent plotting errors
        return np.array([0.0])  # Single dummy sample
    
    result_samples = np.concatenate(all_samples)
    print(f"Successfully collected {len(result_samples)} samples from {successful_particles}/{n_particles} particles")
    return result_samples


def plot_interventional_distributions(results, gt_samples, interv_info, results_dir):
    """
    Generates and saves plots comparing the interventional distributions of all models.
    """
    parent, child, interv_value = interv_info
    gt_obs_samples, gt_interv_samples = gt_samples

    # --- Individual Plots ---
    n_models = len(results)
    fig, axs = plt.subplots(2, n_models + 1, figsize=(6 * (n_models + 1), 10))

    # Ground Truth
    axs[0, 0].hist(gt_obs_samples[:, child], bins=50, density=True, alpha=0.7, color='green')
    axs[0, 0].set_title("Ground Truth\nObservational")
    axs[1, 0].hist(gt_interv_samples[:, child], bins=50, density=True, alpha=0.7, color='blue')
    axs[1, 0].set_title("Ground Truth\nInterventional")

    # Model Plots
    for i, (name, model_results) in enumerate(results.items()):
        # Check if samples exist and are valid
        obs_samples = model_results.get('obs_samples')
        interv_samples = model_results.get('interv_samples')
        
        if obs_samples is not None and len(obs_samples) > 0:
            axs[0, i+1].hist(obs_samples, bins=50, density=True, alpha=0.7)
        else:
            axs[0, i+1].text(0.5, 0.5, 'No valid\nsamples', 
                           horizontalalignment='center', verticalalignment='center')
        axs[0, i+1].set_title(f"{name}\nObservational")
        
        if interv_samples is not None and len(interv_samples) > 0:
            axs[1, i+1].hist(interv_samples, bins=50, density=True, alpha=0.7)
        else:
            axs[1, i+1].text(0.5, 0.5, 'No valid\nsamples', 
                           horizontalalignment='center', verticalalignment='center')
        axs[1, i+1].set_title(f"{name}\nInterventional")

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "interventional_dists_individual.png"))
    plt.close(fig)

    # --- Comparison Plots ---
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Observational Comparison
    axs[0].hist(gt_obs_samples[:, child], bins=50, density=True, alpha=0.5, label='Ground Truth', color='black')
    for name, model_results in results.items():
        obs_samples = model_results.get('obs_samples')
        if obs_samples is not None and len(obs_samples) > 0:
            axs[0].hist(obs_samples, bins=50, density=True, alpha=0.5, label=name)
    axs[0].set_title(f"Observational Distribution p(X_{child})")
    axs[0].legend()

    # Interventional Comparison
    axs[1].hist(gt_interv_samples[:, child], bins=50, density=True, alpha=0.5, label='Ground Truth', color='black')
    for name, model_results in results.items():
        interv_samples = model_results.get('interv_samples')
        if interv_samples is not None and len(interv_samples) > 0:
            axs[1].hist(interv_samples, bins=50, density=True, alpha=0.5, label=name)
    axs[1].set_title(f"Interventional Distribution p(X_{child} | do(X_{parent}={interv_value}))")
    axs[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "interventional_dists_comparison.png"))
    plt.close(fig)

    print(f"\nPlots saved to {results_dir}")
