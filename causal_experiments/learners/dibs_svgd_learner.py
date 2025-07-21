import jax
import jax.numpy as jnp
import time

from .base_learner import BaseLearner
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood

class DibsSVGDLearner(BaseLearner):
    """
    A causal learner that uses the standard DiBS with SVGD.
    """

    def __init__(self, config):
        """
        Initialize the DiBS SVGD learner.
        
        Args:
            config (dict): Configuration containing all necessary parameters.
                          Should have 'n_particles' for the learner config.
        """
        super().__init__(config)
        
        # Extract learner-specific parameters
        if 'n_particles' not in config:
            raise ValueError("DibsSVGDLearner requires 'n_particles' in config")

    def train(self, x_train, mask_train, graph_model, likelihood_model):
        """
        Trains DiBS with SVGD using a set of particles.
        """
        n_particles = self.config['n_particles']
        n_steps = self.config.get('training', {}).get('n_steps', 1000)
        random_seed = self.config.get('random_seed', 42)
        
        print(f"\n--- Training DiBS with SVGD ({n_particles} particles) ---")
        start_time = time.time()

        key = jax.random.PRNGKey(random_seed)
        
        dibs_svgd = JointDiBS(
            x=x_train, 
            interv_mask=mask_train, 
            graph_model=graph_model, 
            likelihood_model=likelihood_model
        )
        
        gs, thetas = dibs_svgd.sample(
            key=key, 
            n_particles=n_particles, 
            steps=n_steps
        )

        train_time = time.time() - start_time
        print(f"Finished training in {train_time:.2f}s")
        
        return gs, thetas

    def evaluate(self, particles, x_ho_obs, x_ho_intrv, mask_ho_intrv, g_true, graph_model, likelihood_model):
        """
        Evaluates the SVGD model using standard DiBS metrics.
        """
        print("\n--- Evaluating DiBS SVGD ---")
        gs, thetas = particles

        dibs_instance = JointDiBS(x=x_ho_obs, interv_mask=None, graph_model=graph_model, likelihood_model=likelihood_model)
        mixture_dist = dibs_instance.get_mixture(gs, thetas)

        # --- Standard Metrics ---
        eshd = expected_shd(dist=mixture_dist, g=g_true)
        auroc = threshold_metrics(dist=mixture_dist, g=g_true)['roc_auc']
        print(f"E-SHD: {eshd:.2f}, AUROC: {auroc:.3f}")

        # --- Observational Negative Log-Likelihood ---
        negll_obs = neg_ave_log_likelihood(
            dist=mixture_dist,
            eltwise_log_likelihood=dibs_instance.eltwise_log_likelihood_observ,
            x=x_ho_obs
        )
        print(f"NLL (Observational): {negll_obs:.2f}")

        # --- Interventional Negative Log-Likelihood ---
        negll_intrv = neg_ave_log_likelihood(
            dist=mixture_dist,
            eltwise_log_likelihood=lambda g, theta, x: dibs_instance.eltwise_log_likelihood_interv(g, theta, x, mask_ho_intrv),
            x=x_ho_intrv
        )
        print(f"NLL (Interventional): {negll_intrv:.2f}")

        metrics = {
            'eshd': float(eshd),
            'auroc': float(auroc),
            'negll_obs': float(negll_obs),
            'negll_intrv': float(negll_intrv)
        }
        return metrics