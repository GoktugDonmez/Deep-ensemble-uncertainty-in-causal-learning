"""
MLflow utilities for experiment tracking.
"""
import mlflow
from typing import Dict, Any


def flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary to make it suitable for MLflow parameter logging.
    
    Args:
        config: Nested configuration dictionary
        parent_key: Parent key for recursive calls
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary with string keys
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to comma-separated strings for MLflow
            items.append((new_key, str(v)))
        else:
            # Convert all values to strings for MLflow compatibility
            items.append((new_key, str(v)))
    return dict(items)


def log_config_as_params(config: Dict[str, Any], learner_name: str = None):
    """
    Log a configuration dictionary as MLflow parameters.
    
    Args:
        config: Configuration dictionary to log
        learner_name: Optional learner name to include in parameter names
    """
    flattened = flatten_config(config)
    
    # Add learner name to parameter names if provided
    if learner_name:
        flattened[f"learner.name"] = learner_name
    
    # MLflow has a limit on parameter value length, so truncate if needed
    params_to_log = {}
    for key, value in flattened.items():
        str_value = str(value)
        if len(str_value) > 250:  # MLflow limit is 500, but we'll be conservative
            str_value = str_value[:247] + "..."
        params_to_log[key] = str_value
    
    mlflow.log_params(params_to_log)


def log_artifacts_selectively(results_dir: str, learner_name: str = None):
    """
    Log specific artifacts to MLflow, excluding plots to keep tracking lightweight.
    
    Args:
        results_dir: Directory containing experiment results
        learner_name: Optional learner name for learner-specific artifacts
    """
    import os
    
    # Log general experiment artifacts
    artifacts_to_log = [
        'config.yaml',
        'ground_truth.yaml', 
        'ground_truth_theta.pkl',
        'metrics.yaml',
        'intervention_details.yaml'
    ]
    
    for artifact in artifacts_to_log:
        artifact_path = os.path.join(results_dir, artifact)
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path)
    
    # Log learner-specific artifacts if learner_name is provided
    if learner_name:
        learner_dir = os.path.join(results_dir, learner_name)
        if os.path.exists(learner_dir):
            learner_artifacts = ['particles.npz', 'thetas.pkl']
            for artifact in learner_artifacts:
                artifact_path = os.path.join(learner_dir, artifact)
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_path=f"{learner_name}/")