import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import time

from .base_learner import BaseLearner
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood

class ConfigurableEnsembleLearner(BaseLearner):
    """
    A deep ensemble learner where each member shares the same architecture
    but can be configured with different training parameters (e.g., optimizers,
    learning rates, random seeds).

    Currently, it functions as a standard deep ensemble, with diversity
    arising only from different random seeds for initialization.
    """

    def __init__(self, config):
        """
        Initialize the Configurable Ensemble learner.
        
        Args:
            config (dict): Configuration containing all necessary parameters.
                          Should have 'n_ensemble_runs'.
        """
        super().__init__(config)
        if 'n_ensemble_runs' not in config:
            raise ValueError("ConfigurableEnsembleLearner requires 'n_ensemble_runs' in config")

    def train(self, x_train, mask_train, graph_model, likelihood_model):
        """
        Trains the deep ensemble by running DiBS multiple times with n_particles=1.
        Each run uses the same likelihood_model architecture but a different random seed.
        """
        n_ensemble_runs = self.config['n_ensemble_runs']
        n_steps = self.config.get('training', {}).get('n_steps', 1000)
        random_seed = self.config.get('random_seed', 42)
        
        print(f"\n--- Training Configurable Ensemble ({n_ensemble_runs} runs) ---")
        ensemble_gs = []
        ensemble_thetas = []
        start_time = time.time()

        key = jax.random.PRNGKey(random_seed)

        for i in range(n_ensemble_runs):
            print(f"Running ensemble member {i+1}/{n_ensemble_runs}...")
            key, subk = jax.random.split(key)
            
            # TODO: Implement logic to vary training configurations per member,
            # e.g., by selecting different optimizers or learning rates from the config.
            
            # Each ensemble member is a JointDiBS instance trained independently
            # with the same model architecture but a different seed.
            dibs_single = JointDiBS(
                x=x_train, 
                interv_mask=mask_train, 
                graph_model=graph_model, 
                likelihood_model=likelihood_model
            )
            
            gs, thetas = dibs_single.sample(
                key=subk, 
                n_particles=1, 
                steps=n_steps
            )

            ensemble_gs.append(gs)
            ensemble_thetas.append(thetas)

        self.training_time = time.time() - start_time
        print(f"Finished training in {self.training_time:.2f}s")

        # Combine the particles from all runs into a single set
        combined_gs = jnp.concatenate(ensemble_gs, axis=0)
        combined_thetas = jax.tree_util.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=0), *ensemble_thetas)
        
        # The 'particles' for an ensemble are the combined graphs and thetas
        return combined_gs, combined_thetas

    def evaluate(self, particles, x_ho_obs, x_ho_intrv, mask_ho_intrv, g_true, graph_model, likelihood_model):
        """
        Evaluates the ensemble using standard DiBS metrics.
        """
        print("\n--- Evaluating Configurable Ensemble ---")
        start_time = time.time()
        
        gs, thetas = particles

        # Create a dummy DiBS instance to get access to the mixture function and likelihood helpers.
        dibs_instance = JointDiBS(x=x_ho_obs, interv_mask=None, graph_model=graph_model, likelihood_model=likelihood_model)
        mixture_dist = dibs_instance.get_mixture(gs, thetas)

        # --- Standard Metrics ---
        eshd = expected_shd(dist=mixture_dist, g=g_true)
        auroc = threshold_metrics(dist=mixture_dist, g=g_true)['roc_auc']
        print(f"E-SHD: {eshd:.2f}, AUROC: {auroc:.3f}")

        # --- Observational Negative Log-Likelihood ---
        print("--- Evaluating on OBSERVATIONAL Held-Out Data ---")
        negll_obs = neg_ave_log_likelihood(
            dist=mixture_dist,
            eltwise_log_likelihood=dibs_instance.eltwise_log_likelihood_observ,
            x=x_ho_obs
        )
        print(f"NLL (Observational): {negll_obs:.2f}")

        # --- Interventional Negative Log-Likelihood ---
        print("\n--- Evaluating on INTERVENTIONAL Held-Out Data ---")
        negll_intrv = neg_ave_log_likelihood(
            dist=mixture_dist,
            eltwise_log_likelihood=lambda g, theta, x: dibs_instance.eltwise_log_likelihood_interv(g, theta, x, mask_ho_intrv),
            x=x_ho_intrv
        )
        print(f"NLL (Interventional): {negll_intrv:.2f}")

        self.evaluation_time = time.time() - start_time

        metrics = {
            'eshd': float(eshd),
            'auroc': float(auroc),
            'negll_obs': float(negll_obs),
            'negll_intrv': float(negll_intrv)
        }
        return metrics