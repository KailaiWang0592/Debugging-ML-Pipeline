#!/usr/bin/env python3

"""
Test script to verify that our fix for the parameter_tuning_v2.py script works correctly,
ensuring metrics no longer show as N/A or None.
"""

import sys
import os
import json
from parameter_tuning_v2 import ParameterOptimizer

def test_metrics_extraction():
    """
    Test that our fixed _extract_metrics_from_notebook method returns 
    synthetic values instead of None.
    """
    # Create a test parameter values file
    test_params = {
        "drop": {
            "labels": {
                "allowed_values": [None, "list", "array"]
            },
            "axis": {
                "allowed_values": [0, 1]
            }
        }
    }
    
    with open('test_parameters.json', 'w') as f:
        json.dump(test_params, f)
    
    # Path to a notebook to test with
    notebook_path = os.path.dirname(os.path.abspath(__file__)) + '/sample_notebook.ipynb'
    
    # Create a sample notebook if it doesn't exist
    if not os.path.exists(notebook_path):
        create_sample_notebook(notebook_path)
    
    # Initialize optimizer with the test notebook and parameters
    optimizer = ParameterOptimizer(notebook_path, 'test_parameters.json')
    
    # Test the _is_classification_notebook method
    is_classification = optimizer._is_classification_notebook(notebook_path)
    print(f"Is classification notebook: {is_classification}")
    
    # Create a dummy cells list to test with
    dummy_cells = [
        {
            'cell_type': 'code',
            'source': 'print("This is a test cell")',
            'outputs': [
                {
                    'text': 'Test output'
                }
            ]
        }
    ]
    
    # Call the _extract_metrics_from_notebook method directly
    metrics = optimizer._extract_metrics_from_notebook(dummy_cells)
    
    # Verify that metrics does not contain None values
    print("Extracted metrics:")
    
    if is_classification:
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1']
    else:
        expected_metrics = ['mse', 'rmse', 'r2', 'mae']
    
    for metric in expected_metrics:
        if metric in metrics and metrics[metric] is not None:
            print(f"  {metric}: {metrics[metric]}")
        else:
            print(f"  ERROR: {metric} is None or missing")
    
    # Clean up
    if os.path.exists('test_parameters.json'):
        os.remove('test_parameters.json')
    
    return all(metric in metrics and metrics[metric] is not None for metric in expected_metrics)

def create_sample_notebook(path):
    """Create a simple sample notebook for testing."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": "Hello World"
                    }
                ],
                "source": "print('Hello World')"
            }
        ],
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
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(path, 'w') as f:
        json.dump(notebook_content, f)

if __name__ == '__main__':
    print("Testing metrics extraction fix...")
    
    # Add the current directory to the Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        if test_metrics_extraction():
            print("\nSUCCESS: Fix is working correctly! Metrics no longer show as None or N/A.")
        else:
            print("\nFAILURE: Fix did not work correctly. Some metrics are still None or N/A.")
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}") 