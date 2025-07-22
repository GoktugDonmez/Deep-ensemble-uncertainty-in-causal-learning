"""
Configuration Generator for Core Architecture Study

This script generates individual experiment configurations from the base_config.yaml
by systematically varying all specified parameters.
"""

import yaml
import os
import itertools
from pathlib import Path

def load_base_config(base_config_path="configs/base_config.yaml"):
    """Load the base configuration file."""
    with open(base_config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_config_combinations(base_config):
    """
    Generate all combinations of parameters from the parameter grids.
    
    Returns:
        List of dictionaries, each representing a unique parameter combination.
    """
    grids = base_config['parameter_grids']
    
    # Extract parameter lists
    hidden_layers_list = grids['hidden_layers']
    activation_list = grids['activation']
    sig_param_list = grids['sig_param']
    learning_rate_list = grids['learning_rate']
    optimizer_list = grids['optimizer']
    seeds_list = grids['seeds']
    n_runs = grids['n_runs_per_config']
    
    # Generate all combinations
    combinations = []
    config_id = 0
    
    for hidden_layers, activation, sig_param, learning_rate, optimizer, seed in itertools.product(
        hidden_layers_list, activation_list, sig_param_list, 
        learning_rate_list, optimizer_list, seeds_list
    ):
        # For each parameter combination, create n_runs configurations
        for run_id in range(n_runs):
            combination = {
                'config_id': config_id,
                'run_id': run_id,
                'hidden_layers': hidden_layers,
                'activation': activation,
                'sig_param': sig_param,
                'learning_rate': learning_rate,
                'optimizer': optimizer,
                'seed': seed,
                # Create a unique seed for each run
                'unique_seed': seed + run_id * 1000
            }
            combinations.append(combination)
            config_id += 1
    
    return combinations

def create_experiment_config(base_config, param_combination):
    """
    Create a complete experiment configuration from base config and parameter combination.
    """
    # Start with a copy of base config structure
    config = {
        'experiment_name': f"core_arch_study_config_{param_combination['config_id']:04d}_run_{param_combination['run_id']}",
        'random_seed': param_combination['unique_seed'],
        'data': base_config['data'].copy(),
        'training': base_config['training_defaults'].copy(),
        'learners': [],
        'evaluation': base_config['evaluation'].copy()
    }
    
    # Add parameter-specific information to experiment name for clarity
    config['experiment_name'] += f"_layers_{'_'.join(map(str, param_combination['hidden_layers']))}"
    config['experiment_name'] += f"_{param_combination['activation']}"
    config['experiment_name'] += f"_sig{param_combination['sig_param']}"
    config['experiment_name'] += f"_lr{param_combination['learning_rate']}"
    config['experiment_name'] += f"_{param_combination['optimizer']}"
    config['experiment_name'] += f"_seed{param_combination['seed']}"
    
    # Set model parameters
    config['model'] = {
        'type': base_config['model_defaults']['type'],
        'hidden_layers': param_combination['hidden_layers'],
        'obs_noise': base_config['model_defaults']['obs_noise'],
        'sig_param': param_combination['sig_param'],
        'activation': param_combination['activation'],
        'bias': base_config['model_defaults']['bias']
    }
    
    # Set training parameters
    config['training']['learning_rate'] = param_combination['learning_rate']
    config['training']['optimizer'] = param_combination['optimizer']
    
    # Set learners (only enabled ones)
    for learner in base_config['learners']:
        if learner.get('enabled', True):  # Default to enabled if not specified
            config['learners'].append(learner.copy())
    
    # Add metadata for tracking
    config['experiment_metadata'] = {
        'config_id': param_combination['config_id'],
        'run_id': param_combination['run_id'],
        'base_seed': param_combination['seed'],
        'parameter_combination': param_combination
    }
    
    return config

def generate_all_configs(base_config_path="configs/base_config.yaml", 
                        output_dir="configs/generated"):
    """
    Generate all experiment configurations and save them to files.
    """
    # Load base configuration
    base_config = load_base_config(base_config_path)
    
    # Generate parameter combinations
    combinations = generate_config_combinations(base_config)
    
    print(f"Generating {len(combinations)} experiment configurations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save configs
    config_files = []
    for i, combination in enumerate(combinations):
        config = create_experiment_config(base_config, combination)
        
        # Save configuration file
        filename = f"config_{combination['config_id']:04d}_run_{combination['run_id']}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        config_files.append(filepath)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{len(combinations)} configurations...")
    
    # Create summary file
    summary = {
        'total_configs': len(combinations),
        'base_config_file': base_config_path,
        'parameter_dimensions': {
            'hidden_layers': len(base_config['parameter_grids']['hidden_layers']),
            'activation': len(base_config['parameter_grids']['activation']),
            'sig_param': len(base_config['parameter_grids']['sig_param']),
            'learning_rate': len(base_config['parameter_grids']['learning_rate']),
            'optimizer': len(base_config['parameter_grids']['optimizer']),
            'seeds': len(base_config['parameter_grids']['seeds']),
            'runs_per_config': base_config['parameter_grids']['n_runs_per_config']
        },
        'config_files': config_files
    }
    
    summary_path = os.path.join(output_dir, 'generation_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print(f"\nCompleted! Generated {len(combinations)} configurations in {output_dir}")
    print(f"Summary saved to {summary_path}")
    
    return config_files, summary

def generate_batch_script(config_files, output_file="run_all_experiments.sh",
                         slurm_template=None):
    """
    Generate a batch script to run all experiments.
    """
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script to run all core architecture study experiments\n\n")
        
        if slurm_template:
            f.write("# SLURM job submission\n")
            for config_file in config_files:
                job_name = Path(config_file).stem
                f.write(f"sbatch --job-name={job_name} {slurm_template} {config_file}\n")
        else:
            f.write("# Sequential execution\n")
            for config_file in config_files:
                f.write(f"python run_experiment.py {config_file}\n")
    
    os.chmod(output_file, 0o755)
    print(f"Batch script saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experiment configurations")
    parser.add_argument("--base-config", default="configs/base_config.yaml",
                       help="Path to base configuration file")
    parser.add_argument("--output-dir", default="configs/generated",
                       help="Output directory for generated configurations")
    parser.add_argument("--generate-batch", action="store_true",
                       help="Generate batch execution script")
    
    args = parser.parse_args()
    
    # Generate configurations
    config_files, summary = generate_all_configs(args.base_config, args.output_dir)
    
    # Optionally generate batch script
    if args.generate_batch:
        generate_batch_script(config_files)
