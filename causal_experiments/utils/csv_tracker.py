import csv
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


class CSVResultsTracker:
    """
    A robust CSV tracking system for experiment results.
    
    Handles creation and appending of experiment results to a master CSV file.
    Ensures data integrity and proper formatting for analysis.
    """
    
    def __init__(self, csv_path: str = "experiment_results.csv"):
        """
        Initialize the CSV tracker.
        
        Args:
            csv_path: Path to the master CSV file
        """
        self.csv_path = csv_path
        self.headers = [
            # Identification
            'timestamp',
            'run_id',
            'experiment_name',
            'learner_name',
            
            # Configuration parameters
            'n_particles',
            'n_ensemble_runs',
            'learning_rate',
            'optimizer',
            'net_depth',
            'net_width', 
            'n_steps',
            'obs_noise',
            'sig_param',
            
            # Data parameters
            'n_vars',
            'n_observations',
            'n_ho_observations',
            'n_intervention_sets',
            'perc_intervened',
            
            # Seeds and reproducibility
            'inference_seed',
            'random_seed',
            
            # Metrics
            'eshd',
            'auroc',
            'negll_obs',
            'negll_intrv',
            
            # Additional metadata
            'training_time_seconds',
            'evaluation_time_seconds',
            'total_time_seconds'
        ]
        
        # Create CSV with headers if it doesn't exist
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            print(f"Creating new CSV file: {self.csv_path}")
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        else:
            print(f"Using existing CSV file: {self.csv_path}")
    
    def _extract_config_value(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Extract a configuration value, handling nested dictionaries.
        
        Args:
            config: Configuration dictionary
            key: Key to extract (supports dot notation for nested access)
            default: Default value if key not found
            
        Returns:
            The value or default if not found
        """
        if '.' in key:
            keys = key.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        else:
            return config.get(key, default)
    
    def _extract_learner_config(self, config: Dict[str, Any], learner_name: str) -> Dict[str, Any]:
        """
        Extract learner-specific configuration.
        
        Args:
            config: Full experiment configuration
            learner_name: Name of the learner
            
        Returns:
            Learner-specific configuration dictionary
        """
        learner_config = {}
        
        # Find the specific learner config
        for learner in config.get('learners', []):
            if learner.get('name') == learner_name:
                learner_config = learner
                break
        
        return learner_config
    
    def save_results(self, 
                    config: Dict[str, Any],
                    learner_name: str,
                    metrics: Dict[str, float],
                    training_time: float = 0.0,
                    evaluation_time: float = 0.0,
                    run_id: Optional[str] = None) -> None:
        """
        Save experiment results to the CSV file.
        
        Args:
            config: Full experiment configuration dictionary
            learner_name: Name of the learner (e.g., 'SVGD', 'Ensemble')
            metrics: Dictionary of computed metrics
            training_time: Time spent training in seconds
            evaluation_time: Time spent evaluating in seconds
            run_id: Optional custom run ID, defaults to timestamp-based ID
        """
        
        # Generate timestamp and run ID
        timestamp = datetime.now().isoformat()
        if run_id is None:
            run_id = f"{learner_name}_{int(time.time())}"
        
        # Extract learner-specific config
        learner_config = self._extract_learner_config(config, learner_name)
        
        # Calculate total time
        total_time = training_time + evaluation_time
        
        # Prepare row data
        row_data = {
            # Identification
            'timestamp': timestamp,
            'run_id': run_id,
            'experiment_name': config.get('experiment_name', ''),
            'learner_name': learner_name,
            
            # Configuration parameters
            'n_particles': learner_config.get('n_particles', ''),
            'n_ensemble_runs': learner_config.get('n_ensemble_runs', ''),
            'learning_rate': self._extract_config_value(config, 'training.learning_rate', ''),
            'optimizer': self._extract_config_value(config, 'training.optimizer', ''),
            'net_depth': len(self._extract_config_value(config, 'model.hidden_layers', [])),
            'net_width': max(self._extract_config_value(config, 'model.hidden_layers', [0])) if self._extract_config_value(config, 'model.hidden_layers') else '',
            'n_steps': self._extract_config_value(config, 'training.n_steps', ''),
            'obs_noise': self._extract_config_value(config, 'model.obs_noise', ''),
            'sig_param': self._extract_config_value(config, 'model.sig_param', ''),
            
            # Data parameters
            'n_vars': self._extract_config_value(config, 'data.n_vars', ''),
            'n_observations': self._extract_config_value(config, 'data.n_observations', ''),
            'n_ho_observations': self._extract_config_value(config, 'data.n_ho_observations', ''),
            'n_intervention_sets': self._extract_config_value(config, 'data.n_intervention_sets', ''),
            'perc_intervened': self._extract_config_value(config, 'data.perc_intervened', ''),
            
            # Seeds and reproducibility
            'inference_seed': learner_config.get('inference_seed', config.get('random_seed', '')),
            'random_seed': config.get('random_seed', ''),
            
            # Metrics
            'eshd': metrics.get('eshd', ''),
            'auroc': metrics.get('auroc', ''),
            'negll_obs': metrics.get('negll_obs', ''),
            'negll_intrv': metrics.get('negll_intrv', ''),
            
            # Additional metadata
            'training_time_seconds': training_time,
            'evaluation_time_seconds': evaluation_time,
            'total_time_seconds': total_time
        }
        
        # Write to CSV
        self._append_to_csv(row_data)
        
        print(f"✓ Results saved to CSV: {self.csv_path}")
        print(f"  Run ID: {run_id}")
        print(f"  Learner: {learner_name}")
        print(f"  Metrics: {metrics}")
    
    def _append_to_csv(self, row_data: Dict[str, Any]) -> None:
        """
        Append a row of data to the CSV file.
        
        Args:
            row_data: Dictionary mapping header names to values
        """
        # Ensure we have all headers and in the right order
        row_values = []
        for header in self.headers:
            value = row_data.get(header, '')
            # Convert numpy types to Python types for CSV compatibility
            if isinstance(value, (np.floating, np.integer)):
                value = float(value) if np.isfinite(value) else ''
            row_values.append(value)
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_values)
    
    def load_results(self) -> List[Dict[str, Any]]:
        """
        Load all results from the CSV file.
        
        Returns:
            List of dictionaries, each representing a row of results
        """
        results = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
        return results
    
    def get_summary_table(self, group_by: List[str] = None) -> str:
        """
        Generate a summary table of results.
        
        Args:
            group_by: List of columns to group by for summary
            
        Returns:
            Formatted string table of results
        """
        results = self.load_results()
        if not results:
            return "No results found in CSV file."
        
        if group_by is None:
            group_by = ['learner_name']
        
        # Basic summary for now
        summary = f"Summary of {len(results)} experiments:\n"
        summary += f"Learners: {set(r.get('learner_name', '') for r in results)}\n"
        summary += f"Metrics tracked: eshd, auroc, negll_obs, negll_intrv\n"
        
        return summary


def get_tracker(csv_path: str = "experiment_results.csv") -> CSVResultsTracker:
    """
    Factory function to get a CSV tracker instance.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        CSVResultsTracker instance
    """
    return CSVResultsTracker(csv_path)
