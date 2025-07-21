# Research Plan: Deep Ensembles for Causal Discovery

This document outlines the research plan for investigating the effectiveness of deep ensembles in causal discovery, comparing them to existing Bayesian methods, and building a flexible codebase for future experiments.


## 21-07-2025:

# starting, with these review from last week, will need to implement these today 
- clean the code to be more modular use classes and better tracking of variables
- try different deep ensemble methods and model architectures
    - also adjust the sampling observeations correctly.  CORE
        - keep track of the specifications of each ensemble member more planned way,
            - for example keeping the likelihood model hyperparams
            - how to adjust the model while also using the dibs library functions (or what can be done addtional to their code, how), basically making the “*glue code*” better
    - look into how possible hyperparameter tuning libraries or doing ml-ii hyperparam search. see if it applicable to our case.


## 18-07-2025: DiBS Model Management & Interventional Distribution Analysis


### Summary outcomes 

- implement interventional data evaluation
    - one with sampling the weights of the non linear model
        - other one by ancestral topological ordering and getting the last element (check)
        - sample from the likelihood model defined for the particle theta graph pair. sample them each and concatanate the samples together
            - seperately sample for gt, svgd, ensemble
            - plot observed dist for selected node and intervened distribution of the selected node for 3 models. 6plots in total. also 2 comparative plots (obs and invtervened) on top of each model
    - added wasserstein distance evaluation (check)
- transfered code to notebook for colab
- set aalto vpn


### Model State Analysis
**Key Insight**: DiBS inference classes (`JointDiBS`, `MarginalDiBS`) store **training configuration and data** but NOT learned parameters. The learned information is in the particles `(gs, thetas)`.

**What JointDiBS stores**:
- Training data: `self.x`, `self.interv_mask`
- Model references: `self.likelihood_model`, `self.graph_model` 
- Evaluation functions: `self.eltwise_log_likelihood_observ` but can be directly accessed by likelihood model

**What JointDiBS does NOT store**:
- Learned graph structures (these are in `gs`)
- Learned neural network parameters (these are in `thetas`)
- Training progress or particle states

**Implication**: You can recreate the same DiBS instance with same config, but the learned knowledge is separate in the particles.

```python
# Save only particles + architecture info, not DiBS instances
experiment_data = {
    'ground_truth': {'g': data.g, 'theta': data.theta, 'x': data.x},
    'svgd_model': {'gs': gs_svgd, 'thetas': thetas_svgd},
    'ensemble_model': {'gs': combined_gs, 'thetas': combined_thetas},
    'architecture': {'hidden_layers': (5,), 'obs_noise': 0.1, 'sig_param': 1.0}
}
```

**Benefits**: 
- No retraining needed
- Small file sizes (only final particles)
- Easy model sharing and comparison
- Recreate exact same model by loading particles + recreating architecture



## High-Level Goals

### 1. Interventional Distribution Analysis
**Objective:** To visually and quantitatively compare the interventional distributions learned by different models (SVGD, Deep Ensembles) against the ground truth.

**Key Tasks:**
- Modify the existing experiment script (`ensemble_intrv_experiments.py`).
- Select a parent-child pair `(j -> i)` from the ground truth graph.
- Generate samples for `p(X_i | do(X_j = c))` from:
    - The ground truth model.
    - Each particle of the SVGD-learned posterior.
    - Each member of the deep ensemble.
- Plot histograms of these distributions to visually assess how well each method captures the true causal effect.
- (Optional) Compute quantitative metrics to compare the distributions (e.g., Wasserstein distance).

### 2. Deep Ensemble Hyperparameter Tuning and Architectural Exploration
**Objective:** To systematically explore how different aspects of the deep learning architecture and training process affect the performance of deep ensembles for causal discovery.

**Key Tasks:**
- **Optimizers and Learning Rates:** Experiment with different optimizers (e.g., Adam, SGD with momentum) and learning rate schedules.
- **Network Architecture:** Vary the depth and width of the neural networks used in the `DenseNonlinearGaussian` model (i.e., the `hidden_layers` parameter).
- **Regularization:** Introduce dropout into the neural network architecture to see its effect on generalization and uncertainty estimation. This will require modifying `dibs/models/nonlinearGaussian.py`.
- **Activation Functions:** Test different non-linearities (e.g., Tanh, LeakyReLU) to see their impact.

### 3. Generalizable and Abstracted Causal Discovery Framework
**Objective:** To refactor the codebase into a more abstract, modular, and extensible framework that allows for easy comparison of different causal discovery algorithms.

**Key Tasks:**
- **Define a `CausalLearner` Abstract Base Class:** This class will define a standard interface with methods like `train`, `evaluate`, `get_interventional_distribution`, etc.
- **Create Concrete Implementations:**
    - `DibsvLearner`: A wrapper for the existing DiBS implementation.
    - `EnsembleLearner`: A specific learner for the deep ensemble method.
    - Create wrappers for other external models (e.g., BCI-ARCO-GP) to conform to the interface.
- **Develop an Experiment Runner:** A main script that takes a configuration file (e.g., YAML) to define the experiment (model, hyperparameters, dataset) and runs the training and evaluation pipeline.
- **Standardize Output:** Ensure all learners produce results and metrics in a consistent format for easy analysis and plotting.
