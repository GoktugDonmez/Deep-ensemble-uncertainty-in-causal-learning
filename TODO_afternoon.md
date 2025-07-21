# TODO for This Afternoon & Beyond

This file outlines the next steps for expanding the causal discovery framework.

## Immediate Next Step

- [ ] **Define `DibsSVGDLearner`**: Create the second concrete learner class (`dibs_svgd_learner.py`) to wrap the standard SVGD implementation from the DiBS library. This will allow for direct comparison with the `DeepEnsembleLearner` within the new framework.

## Future Tasks

- [ ] **Hyperparameter Optimization**: Integrate a hyperparameter tuning library like Optuna or Hyperopt to systematically search for the best model and training configurations.

- [ ] **Likelihood Model Exploration**: Generalize the framework to allow for easily swapping different likelihood models (e.g., for different data types or assumptions).

- [ ] **Cloud/HPC Integration**: Set up the project for testing and running experiments on cloud platforms like Google Colab and HPC systems with Triton.

- [ ] **Notebook Support**: Create example Jupyter notebooks that demonstrate how to use the new framework interactively for analysis, visualization, and debugging.
