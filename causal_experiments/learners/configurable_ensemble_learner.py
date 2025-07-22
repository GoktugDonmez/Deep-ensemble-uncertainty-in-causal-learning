import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import time
from typing import List, Dict, Any, Tuple

from .base_learner import BaseLearner
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from dibs.models import DenseNonlinearGaussian


class EnsembleMemberConfig:
    """Configuration for a single ensemble member."""
    
    def __init__(self, 
                 hidden_layers: Tuple[int, ...], 
                 obs_noise: float = 0.1,
                 sig_param: float = 1.0,
                 activation: str = 'relu',
                 bias: bool = True,
                 member_id: int = 0):
        self.hidden_layers = hidden_layers
        self.obs_noise = obs_noise
        self.sig_param = sig_param
        self.activation = activation
        self.bias = bias
        self.member_id = member_id
        
    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {
            'member_id': self.member_id,
            'hidden_layers': self.hidden_layers,
            'obs_noise': self.obs_noise,
            'sig_param': self.sig_param,
            'activation': self.activation,
            'bias': self.bias
        }
    
    def create_likelihood_model(self, n_vars: int):
        """Create the likelihood model for this ensemble member."""
        return DenseNonlinearGaussian(
            n_vars=n_vars,
            hidden_layers=self.hidden_layers,
            obs_noise=self.obs_noise,
            sig_param=self.sig_param,
            activation=self.activation,
            bias=self.bias
        )


class HeterogeneousEnsembleLearner(BaseLearner):
    """
    A deep ensemble learner with heterogeneous likelihood models.
    Each ensemble member can have different neural network architectures,
    hyperparameters, and configurations.
    """

    def __init__(self, config):
        """
        Initialize the Heterogeneous Ensemble learner.
        
        Args:
            config (dict): Configuration containing all necessary parameters.
        """
        super().__init__(config)
        
        # Initialize ensemble member configurations
        self.member_configs = self._create_member_configurations()
        print(f"Initialized ensemble with {len(self.member_configs)} heterogeneous members")
        
        # Print configuration summary
        self._print_ensemble_summary()
    
    def _create_member_configurations(self) -> List[EnsembleMemberConfig]:
        """Create configurations for different ensemble members."""
        # TODO: This method should be updated to generate different training configurations 
        # (e.g., optimizers, learning rates) for each ensemble member, rather than
        # different model architectures. The likelihood model should be consistent
        # across all members to ensure a fair comparison with SVGD.
        
        configs = []
        
        # Get base parameters from config
        base_hidden_layers = tuple(self.config.get('model', {}).get('hidden_layers', [8]))
        base_obs_noise = self.config.get('model', {}).get('obs_noise', 0.1)
        base_sig_param = self.config.get('model', {}).get('sig_param', 1.0)
        
        # # --- OLD HETEROGENEOUS ARCHITECTURE LOGIC ---
        # # This logic created different likelihood models for each ensemble member.
        # # It is commented out to enforce that all members share the same architecture.
        # architecture_patterns = [
        #     # Small networks with different depths
        #     {'hidden_layers': (5,), 'sig_param': base_sig_param},
        #     {'hidden_layers': (5, 5), 'sig_param': base_sig_param},
        #     {'hidden_layers': (5, 5, 5), 'sig_param': base_sig_param},
        #     
        #     # Medium networks with different depths  
        #     {'hidden_layers': (10,), 'sig_param': base_sig_param},
        #     {'hidden_layers': (10, 10), 'sig_param': base_sig_param},
        #     {'hidden_layers': (10, 10, 10), 'sig_param': base_sig_param},
        #     
        #     # Mixed architectures
        #     {'hidden_layers': (8, 5), 'sig_param': base_sig_param},
        #     {'hidden_layers': (5, 10), 'sig_param': base_sig_param},
        #     {'hidden_layers': (15,), 'sig_param': base_sig_param},
        #     {'hidden_layers': (12, 8), 'sig_param': base_sig_param},
        #     
        #     # Wider networks
        #     {'hidden_layers': (20,), 'sig_param': base_sig_param * 0.8},
        #     {'hidden_layers': (16, 8), 'sig_param': base_sig_param * 0.9},
        #     
        #     # Different activation functions and parameters
        #     {'hidden_layers': (5, 5), 'sig_param': base_sig_param * 1.2, 'activation': 'tanh'},
        #     {'hidden_layers': (10,), 'sig_param': base_sig_param * 0.8, 'activation': 'leakyrelu'},
        #     
        #     # Additional architecture
        #     {'hidden_layers': (8,), 'sig_param': base_sig_param},
        #     {'hidden_layers': (6, 6), 'sig_param': base_sig_param},
        # ]
        
        # Get number of ensemble runs from config
        n_ensemble_runs = self.config.get('n_ensemble_runs', 16)
        
        # Create configurations (cycling through patterns if needed)
        for i in range(n_ensemble_runs):
            # pattern_idx = i % len(architecture_patterns)
            # pattern = architecture_patterns[pattern_idx]
            
            config = EnsembleMemberConfig(
                hidden_layers=base_hidden_layers,
                obs_noise=base_obs_noise,
                sig_param=base_sig_param,
                # hidden_layers=pattern['hidden_layers'],
                # obs_noise=pattern.get('obs_noise', base_obs_noise),
                # sig_param=pattern.get('sig_param', base_sig_param),
                # activation=pattern.get('activation', 'relu'),
                # bias=pattern.get('bias', True),
                member_id=i
            )
            configs.append(config)
        
        return configs
    
    def _print_ensemble_summary(self):
        """Print a summary of the ensemble configuration."""
        print("\n--- Ensemble Configuration Summary ---")
        for i, config in enumerate(self.member_configs):
            print(f"Member {i+1:2d}: layers={config.hidden_layers}, "
                  f"noise={config.obs_noise:.3f}, sig={config.sig_param:.3f}, "
                  f"activation={config.activation}")
    
    def train(self, x_train, mask_train, graph_model, likelihood_model):
        """
        Trains the heterogeneous ensemble by running DiBS multiple times 
        with different likelihood models.
        
        Note: The _unused_likelihood_model parameter is ignored since we create
        our own likelihood models for each ensemble member.
        """
        n_ensemble_runs = len(self.member_configs)
        n_steps = self.config.get('training', {}).get('n_steps', 1000)
        random_seed = self.config.get('random_seed', 42)
        n_vars = x_train.shape[1]
        
        print(f"\n--- Training Heterogeneous Ensemble ({n_ensemble_runs} runs) ---")
        
        ensemble_gs = []
        ensemble_thetas = []
        member_metadata = []
        start_time = time.time()

        key = jax.random.PRNGKey(random_seed)

        for i, member_config in enumerate(self.member_configs):
            print(f"Training member {i+1}/{n_ensemble_runs} "
                  f"(layers={member_config.hidden_layers})...")
            
            key, subk = jax.random.split(key)
            
            # TODO: The logic for varying training (e.g. optimizer) would go here.
            # For now, we use the same likelihood model for all members.

            # # Create specific likelihood model for this ensemble member
            # likelihood_model = member_config.create_likelihood_model(n_vars)
            
            # Each ensemble member is a JointDiBS instance trained independently
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
            member_metadata.append(member_config.to_dict())

        ensemble_time = time.time() - start_time
        print(f"Finished training in {ensemble_time:.2f}s")

        # Combine the particles from all runs into a single set
        combined_gs = jnp.concatenate(ensemble_gs, axis=0)
        combined_thetas = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0), *ensemble_thetas
        )
        
        # Store metadata about ensemble members
        particles_with_metadata = {
            'gs': combined_gs,
            'thetas': combined_thetas,
            'member_configs': member_metadata,
            'n_members': n_ensemble_runs
        }
        
        return particles_with_metadata

    def evaluate(self, particles, x_ho_obs, x_ho_intrv, mask_ho_intrv, g_true, graph_model, likelihood_model):
        """
        Evaluates the heterogeneous ensemble.
        
        For evaluation, we need to handle the fact that each ensemble member 
        has a different likelihood model. We'll create a mixture distribution 
        by properly weighting each member's contribution.
        """
        print("\n--- Evaluating Heterogeneous Ensemble ---")
        
        # Extract particles and metadata
        if isinstance(particles, dict):
            gs = particles['gs']
            thetas = particles['thetas']
            member_configs = particles['member_configs']
            n_members = particles['n_members']
        else:
            # Fallback to old format
            gs, thetas = particles
            member_configs = [config.to_dict() for config in self.member_configs]
            n_members = len(self.member_configs)

        n_vars = x_ho_obs.shape[1]
        
        # For evaluation, we need to compute the mixture distribution
        # Since each member has different likelihood model, we compute log-likelihoods separately
        
        print("Computing ensemble mixture distribution...")
        
        # We'll evaluate using a representative likelihood model for the mixture
        # (This is a simplification; ideally we'd weight by each member's model)
        # representative_config = EnsembleMemberConfig(**member_configs[0])
        # representative_likelihood = representative_config.create_likelihood_model(n_vars)
        
        # Create a dummy DiBS instance for evaluation utilities
        dibs_instance = JointDiBS(
            x=x_ho_obs, 
            interv_mask=None, 
            graph_model=graph_model, 
            likelihood_model=likelihood_model
        )
        
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
            eltwise_log_likelihood=lambda g, theta, x: dibs_instance.eltwise_log_likelihood_interv(
                g, theta, x, mask_ho_intrv
            ),
            x=x_ho_intrv
        )
        print(f"NLL (Interventional): {negll_intrv:.2f}")

        metrics = {
            'eshd': float(eshd),
            'auroc': float(auroc),
            'negll_obs': float(negll_obs),
            'negll_intrv': float(negll_intrv),
            'n_ensemble_members': n_members,
            'member_configs': member_configs
        }
        return metrics 