# Codebase Summary: Deep Ensembles for Causal Discovery

This document provides a summary of the `causal_experiments` codebase, its structure, and the research implemented so far, based on the `plan_research.md` and the code itself.

## 1. Overall Project Goal

The primary goal of this research project is to investigate the effectiveness of deep ensembles for causal discovery. This involves comparing deep ensemble methods against established Bayesian inference techniques like Stein Variational Gradient Descent (SVGD) as implemented in the `dibs` library. The project aims to build a flexible and modular framework to conduct reproducible causal discovery experiments, explore various model architectures, and analyze the quality of learned interventional distributions.

## 2. Codebase Structure

The repository is organized into two main parts, following a clean separation of concerns:

-   `dibs/`: This directory is a Git submodule that points to the official `dibs` library. It serves as the core dependency for causal discovery algorithms. Using a submodule keeps the core library separate from the experimental code, allowing for independent versioning and maintenance.
-   `causal_experiments/`: This is the primary directory for this research project. It contains all the custom code for defining, running, and evaluating causal discovery experiments.
-  

## 3. The `causal_experiments` Framework

The code in this directory has been refactored from older scripts (now in `causal_experiments/legacy`) into a modular, configuration-driven framework.

### 3.1. Experiment Runner (`run_experiment.py`)

This is the main entry point for executing experiments. It is designed to:
1.  Load an experiment configuration from a YAML file.
2.  Set up the environment and results directory.
3.  Generate synthetic data based on the configuration.
4.  Dynamically instantiate and run one or more "learners" (causal discovery algorithms).
5.  Evaluate the trained learners using standard metrics and visualizations.
6.  Save all results, metrics, configurations, and learned model particles to a timestamped results directory.

### 3.2. Configuration Files (`configs/`)

All experiments are defined in YAML files, which control every aspect of the run:
-   **`deep_ensemble_v1.yaml`**: Defines a large-scale experiment to test a standard Deep Ensemble.
-   **`dibs_vs_ensemble.yaml`**: Defines a head-to-head comparison between DiBS with SVGD and a Deep Ensemble with an equal number of particles/members.
-   **`heterogeneous_ensemble.yaml`**: Defines an experiment to test a more advanced "Heterogeneous Ensemble," where each member has a different neural network architecture. This is compared against a standard ensemble and SVGD.

### 3.3. Modular Learner Architecture (`learners/`)

This is the core of the refactored codebase, designed for extensibility.

-   **`base_learner.py`**: Defines an abstract base class `BaseLearner` with a standard interface:
    -   `__init__(self, config)`: Initializes the learner with its configuration.
    -   `train(...)`: Trains the model and returns the learned particles (graphs and parameters).
    -   `evaluate(...)`: Computes performance metrics on held-out data. ()

-   **Implemented Learners**:
    -   **`dibs_svgd_learner.py`**: A wrapper for the standard DiBS algorithm using SVGD. It takes a number of `n_particles` to run in a single inference process.
    -   **`deep_ensemble_learner.py`**: Implements the "vanilla" deep ensemble method. It runs DiBS `n_ensemble_runs` times, each with a single particle, and combines the results to form the ensemble posterior.
    -   **`heterogeneous_ensemble_learner.py`**: Implements a more advanced ensemble where each member can have a different likelihood model (e.g., different neural network architectures, activations, or hyperparameters). This allows for exploring the impact of architectural diversity on performance. (THIS NEEDS TO BE CHANGED. WITHIN A SINGLE RUN THE MODELS SHOULD HAVE THE SAME LIKELIHOOD SIZE/ARCHITECTURE)

### 3.4. Utilities and Legacy Code

-   **`utils/plotting.py`**: Contains helper functions for visualizing the results, specifically for plotting and comparing the interventional distributions learned by the models against the ground truth.
-   **`legacy/`**: This directory contains the original Jupyter notebooks and Python scripts from which the current framework was refactored. They serve as a record of the initial exploratory work and contain valuable analyses and comparisons that informed the current, more structured design.