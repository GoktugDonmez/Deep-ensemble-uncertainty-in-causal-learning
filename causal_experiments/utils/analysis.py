import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import os


class ExperimentAnalyzer:
    """
    Utility class for analyzing experiment results from CSV files.
    
    Provides functionality to generate comparison tables and statistical summaries
    of experiment results across different learners and configurations.
    """
    
    def __init__(self, csv_path: str = "experiment_results.csv"):
        """
        Initialize the analyzer.
        
        Args:
            csv_path: Path to the CSV file containing experiment results
        """
        self.csv_path = csv_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load data from CSV file."""
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} experiments from {self.csv_path}")
        else:
            print(f"CSV file not found: {self.csv_path}")
            self.df = pd.DataFrame()
    
    def get_summary_table(self, 
                         group_by: List[str] = None,
                         metrics: List[str] = None,
                         aggregation: str = 'mean') -> pd.DataFrame:
        """
        Generate a summary table grouped by specified columns.
        
        Args:
            group_by: Columns to group by (default: ['learner_name'])
            metrics: Metrics to include (default: ['eshd', 'auroc', 'negll_obs', 'negll_intrv'])
            aggregation: Aggregation method ('mean', 'median', 'std', etc.)
            
        Returns:
            DataFrame with summary statistics
        """
        if self.df.empty:
            return pd.DataFrame()
        
        if group_by is None:
            group_by = ['learner_name']
        
        if metrics is None:
            metrics = ['eshd', 'auroc', 'negll_obs', 'negll_intrv']
        
        # Convert metric columns to numeric, errors='coerce' converts non-numeric to NaN
        for metric in metrics:
            if metric in self.df.columns:
                self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        
        # Group and aggregate
        if aggregation == 'mean':
            summary = self.df.groupby(group_by)[metrics].mean()
        elif aggregation == 'median':
            summary = self.df.groupby(group_by)[metrics].median()
        elif aggregation == 'std':
            summary = self.df.groupby(group_by)[metrics].std()
        elif aggregation == 'count':
            summary = self.df.groupby(group_by)[metrics].count()
        else:
            summary = self.df.groupby(group_by)[metrics].agg(aggregation)
        
        return summary
    
    def get_comparison_table(self, 
                           pivot_column: str = 'learner_name',
                           metrics: List[str] = None,
                           config_columns: List[str] = None) -> pd.DataFrame:
        """
        Generate a comparison table with learners as columns and configurations as rows.
        
        Args:
            pivot_column: Column to use as columns in pivot table
            metrics: Metrics to include in comparison
            config_columns: Configuration columns to include as row identifiers
            
        Returns:
            DataFrame with comparison table
        """
        if self.df.empty:
            return pd.DataFrame()
        
        if metrics is None:
            metrics = ['eshd', 'auroc', 'negll_obs', 'negll_intrv']
        
        if config_columns is None:
            config_columns = ['n_vars', 'n_observations', 'n_steps', 'net_depth', 'net_width']
        
        # Convert metric columns to numeric
        for metric in metrics:
            if metric in self.df.columns:
                self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        
        comparison_tables = {}
        
        for metric in metrics:
            if metric in self.df.columns:
                # Create a pivot table for this metric
                pivot_table = self.df.pivot_table(
                    index=config_columns,
                    columns=pivot_column,
                    values=metric,
                    aggfunc='mean'
                )
                comparison_tables[metric] = pivot_table
        
        return comparison_tables
    
    def get_detailed_comparison(self, 
                              config_filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Get a detailed comparison table filtering by configuration parameters.
        
        Args:
            config_filters: Dictionary of column: value filters to apply
            
        Returns:
            Filtered DataFrame with detailed results
        """
        if self.df.empty:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        if config_filters:
            for column, value in config_filters.items():
                if column in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[column] == value]
        
        # Select relevant columns for comparison
        comparison_columns = [
            'learner_name', 'experiment_name', 'n_particles', 'n_ensemble_runs',
            'n_steps', 'net_depth', 'net_width', 'eshd', 'auroc', 'negll_obs', 'negll_intrv',
            'training_time_seconds', 'evaluation_time_seconds', 'total_time_seconds'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in comparison_columns if col in filtered_df.columns]
        
        return filtered_df[available_columns]
    
    def print_summary_report(self):
        """Print a comprehensive summary report of all experiments."""
        if self.df.empty:
            print("No experiment data available.")
            return
        
        print("=" * 80)
        print("EXPERIMENT RESULTS SUMMARY REPORT")
        print("=" * 80)
        
        print(f"\nTotal experiments: {len(self.df)}")
        print(f"Unique learners: {self.df['learner_name'].nunique()}")
        print(f"Learners: {', '.join(self.df['learner_name'].unique())}")
        
        print(f"\nExperiment names: {', '.join(self.df['experiment_name'].unique())}")
        
        # Summary by learner
        print("\n" + "-" * 50)
        print("SUMMARY BY LEARNER")
        print("-" * 50)
        
        summary = self.get_summary_table()
        if not summary.empty:
            print(summary.round(4))
        
        # Configuration analysis
        print("\n" + "-" * 50)
        print("CONFIGURATION ANALYSIS")
        print("-" * 50)
        
        config_cols = ['n_particles', 'n_ensemble_runs', 'n_steps', 'net_depth', 'net_width']
        for col in config_cols:
            if col in self.df.columns and not self.df[col].isna().all():
                print(f"{col}: {self.df[col].value_counts().to_dict()}")
        
        # Performance analysis
        print("\n" + "-" * 50)
        print("PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        metrics = ['eshd', 'auroc', 'negll_obs', 'negll_intrv']
        for metric in metrics:
            if metric in self.df.columns:
                metric_data = pd.to_numeric(self.df[metric], errors='coerce')
                if not metric_data.isna().all():
                    print(f"\n{metric.upper()}:")
                    print(f"  Mean: {metric_data.mean():.4f}")
                    print(f"  Std:  {metric_data.std():.4f}")
                    print(f"  Min:  {metric_data.min():.4f}")
                    print(f"  Max:  {metric_data.max():.4f}")
        
        # Timing analysis
        print("\n" + "-" * 50)
        print("TIMING ANALYSIS")
        print("-" * 50)
        
        timing_cols = ['training_time_seconds', 'evaluation_time_seconds', 'total_time_seconds']
        for col in timing_cols:
            if col in self.df.columns:
                time_data = pd.to_numeric(self.df[col], errors='coerce')
                if not time_data.isna().all():
                    print(f"{col}: {time_data.mean():.2f}s (avg), {time_data.std():.2f}s (std)")
    
    def export_comparison_tables(self, output_dir: str = "analysis_results"):
        """
        Export comparison tables to CSV files.
        
        Args:
            output_dir: Directory to save the analysis results
        """
        if self.df.empty:
            print("No data to export.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export summary table
        summary = self.get_summary_table()
        if not summary.empty:
            summary_path = os.path.join(output_dir, "summary_by_learner.csv")
            summary.to_csv(summary_path)
            print(f"Summary table saved to: {summary_path}")
        
        # Export comparison tables for each metric
        comparison_tables = self.get_comparison_table()
        for metric, table in comparison_tables.items():
            if not table.empty:
                table_path = os.path.join(output_dir, f"comparison_{metric}.csv")
                table.to_csv(table_path)
                print(f"Comparison table for {metric} saved to: {table_path}")
        
        # Export detailed results
        detailed = self.get_detailed_comparison()
        if not detailed.empty:
            detailed_path = os.path.join(output_dir, "detailed_results.csv")
            detailed.to_csv(detailed_path, index=False)
            print(f"Detailed results saved to: {detailed_path}")


def analyze_experiments(csv_path: str = "experiment_results.csv"):
    """
    Convenience function to perform a complete analysis of experiment results.
    
    Args:
        csv_path: Path to the CSV file containing results
    """
    analyzer = ExperimentAnalyzer(csv_path)
    analyzer.print_summary_report()
    analyzer.export_comparison_tables()
    return analyzer


if __name__ == "__main__":
    # Example usage
    analyzer = analyze_experiments()
