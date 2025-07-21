from abc import ABC, abstractmethod

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
