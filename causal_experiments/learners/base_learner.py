from abc import ABC, abstractmethod
import time

class BaseLearner(ABC):
    """
    Abstract Base Class for a causal learner.
    Defines the common interface that all causal discovery models must implement.
    """

    def __init__(self, config):
        """
        Initialize the learner with a config dictionary.
        
        Args:
            config (dict): Configuration dictionary containing learner-specific 
                          and global parameters.
        """
        self.config = config
        self.training_time = 0.0
        self.evaluation_time = 0.0
        print(f"Initialized {self.__class__.__name__} with config.")

    @abstractmethod
    def train(self, x_train, mask_train, graph_model, likelihood_model):
        """
        Trains the causal discovery model.

        Args:
            x_train (jnp.ndarray): Training data.
            mask_train (jnp.ndarray): Intervention mask for the training data.
            graph_model: The graph model from the dibs library.
            likelihood_model: The likelihood model from the dibs library.

        Returns:
            A tuple (e.g., (gs, thetas)) representing the learned posterior,
            which we refer to as 'particles'.
        """
        pass

    @abstractmethod
    def evaluate(self, particles, x_ho_obs, x_ho_intrv, mask_ho_intrv, g_true, graph_model, likelihood_model):
        """
        Evaluates the performance of the learned model.

        Args:
            particles: The learned posterior from the train method.
            x_ho_obs (jnp.ndarray): Held-out observational data.
            x_ho_intrv (jnp.ndarray): Held-out interventional data.
            mask_ho_intrv (jnp.ndarray): Held-out interventional data mask.
            g_true (jnp.ndarray): The ground truth graph.
            graph_model: The graph model from the dibs library.
            likelihood_model: The likelihood model from the dibs library.

        Returns:
            A dictionary of computed metrics (e.g., {'eshd': 5.0, 'auroc': 0.8}).
        """
        pass

    def sample_interventional(self, particles, interv_dict, n_samples):
        """
        Samples from the interventional distribution of the learned model.

        Args:
            particles: The learned posterior from the train method.
            interv_dict (dict): Dictionary specifying the intervention, e.g., {0: 1.0}.
            n_samples (int): The number of samples to generate.

        Returns:
            An array of samples from the interventional distribution.
        """ 
        # This can have a default implementation if many learners share it,
        # or be left abstract if they are all different.
        raise NotImplementedError("Interventional sampling is not implemented for this learner.")
    
    def get_timing_info(self):
        """
        Get timing information for this learner.
        
        Returns:
            Dictionary with training and evaluation times
        """
        return {
            'training_time': self.training_time,
            'evaluation_time': self.evaluation_time,
            'total_time': self.training_time + self.evaluation_time
        }
    
    def save_to_csv(self, config, learner_name, metrics, csv_path="experiment_results.csv", run_id=None):
        """
        Save results to CSV using the CSV tracker.
        
        Args:
            config: Full experiment configuration
            learner_name: Name of this learner
            metrics: Dictionary of computed metrics
            csv_path: Path to CSV file
            run_id: Optional run ID
        """
        from ..utils.csv_tracker import get_tracker
        
        tracker = get_tracker(csv_path)
        tracker.save_results(
            config=config,
            learner_name=learner_name,
            metrics=metrics,
            training_time=self.training_time,
            evaluation_time=self.evaluation_time,
            run_id=run_id
        )
