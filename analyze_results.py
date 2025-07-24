#!/usr/bin/env python3
"""
Standalone script to analyze experiment results from CSV files.
Can be run independently to generate comparison tables and summaries.
"""

import sys
import os
import argparse
import glob

# Add causal_experiments to path
sys.path.append('causal_experiments')

def find_csv_files(directory="."):
    """Find all CSV files that might contain experiment results."""
    patterns = [
        os.path.join(directory, "experiment_results.csv"),
        os.path.join(directory, "**", "experiment_results.csv"),
        os.path.join(directory, "results", "**", "experiment_results.csv"),
    ]
    
    csv_files = []
    for pattern in patterns:
        csv_files.extend(glob.glob(pattern, recursive=True))
    
    return list(set(csv_files))  # Remove duplicates


def analyze_csv_files(csv_files, output_dir="analysis_results"):
    """Analyze multiple CSV files and generate comprehensive reports."""
    
    print(f"Found {len(csv_files)} CSV files to analyze:")
    for f in csv_files:
        print(f"  - {f}")
    
    if not csv_files:
        print("No CSV files found. Please run some experiments first.")
        return
    
    # Try to import analysis tools
    try:
        from utils.analysis import ExperimentAnalyzer
    except ImportError:
        print("Error: Cannot import analysis tools. Please ensure numpy and pandas are available.")
        print("Analysis tools require: numpy, pandas")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {csv_file}")
        print(f"{'='*60}")
        
        try:
            analyzer = ExperimentAnalyzer(csv_file)
            
            # Generate summary report
            analyzer.print_summary_report()
            
            # Export comparison tables
            file_output_dir = os.path.join(output_dir, os.path.basename(csv_file).replace('.csv', ''))
            analyzer.export_comparison_tables(file_output_dir)
            
        except Exception as e:
            print(f"Error analyzing {csv_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results from CSV files')
    parser.add_argument('--directory', '-d', default='.', 
                        help='Directory to search for CSV files (default: current directory)')
    parser.add_argument('--output', '-o', default='analysis_results',
                        help='Output directory for analysis results (default: analysis_results)')
    parser.add_argument('--csv-file', '-f', 
                        help='Specific CSV file to analyze (overrides directory search)')
    
    args = parser.parse_args()
    
    if args.csv_file:
        if os.path.exists(args.csv_file):
            csv_files = [args.csv_file]
        else:
            print(f"Error: CSV file not found: {args.csv_file}")
            return 1
    else:
        csv_files = find_csv_files(args.directory)
    
    analyze_csv_files(csv_files, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
