#!/usr/bin/env python3
"""
Simple Python to Jupyter Notebook Converter

This script converts a Python file to a Jupyter notebook by splitting on 
special comment markers.

Usage:
    python py_to_ipynb_converter.py input_file.py output_file.ipynb

Special markers:
    # %% markdown
    # %% [markdown]
    # %% code
    # %%

Author: Assistant
"""

import json
import sys
import re
from pathlib import Path

def create_cell(cell_type, source):
    """Create a notebook cell dictionary."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source
    }
    
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    
    return cell

def split_python_file(content):
    """Split Python file content into cells based on markers."""
    cells = []
    current_cell_type = "code"
    current_content = []
    
    lines = content.split('\n')
    
    for line in lines:
        # Check for cell markers
        if line.strip().startswith('# %%'):
            # Save current cell if it has content
            if current_content:
                # Remove trailing empty lines
                while current_content and not current_content[-1].strip():
                    current_content.pop()
                
                if current_content:
                    cells.append(create_cell(current_cell_type, current_content))
            
            # Determine new cell type
            if 'markdown' in line.lower():
                current_cell_type = "markdown"
            else:
                current_cell_type = "code"
            
            current_content = []
        else:
            current_content.append(line)
    
    # Add final cell
    if current_content:
        # Remove trailing empty lines
        while current_content and not current_content[-1].strip():
            current_content.pop()
        
        if current_content:
            cells.append(create_cell(current_cell_type, current_content))
    
    return cells

def convert_py_to_ipynb(py_file_path, ipynb_file_path=None):
    """Convert Python file to Jupyter notebook."""
    py_path = Path(py_file_path)
    
    if not py_path.exists():
        raise FileNotFoundError(f"Python file not found: {py_file_path}")
    
    if ipynb_file_path is None:
        ipynb_file_path = py_path.with_suffix('.ipynb')
    
    # Read Python file
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into cells
    cells = split_python_file(content)
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write notebook file
    with open(ipynb_file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    return ipynb_file_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python py_to_ipynb_converter.py input_file.py [output_file.ipynb]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result_path = convert_py_to_ipynb(input_file, output_file)
        print(f"✅ Successfully converted {input_file} to {result_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 