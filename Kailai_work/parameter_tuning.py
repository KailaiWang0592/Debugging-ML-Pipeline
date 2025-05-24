import json
import nbformat
import argparse
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from copy import deepcopy
import importlib.util
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import re

# Import your existing analysis pipeline
# This assumes the file is in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis_pipeline import analyze_notebook, extract_functions_from_code

class ParameterOptimizer:
    def __init__(self, notebook_path, param_values_path):
        """
        Initialize the optimizer with paths to the notebook and parameter values.
        
        Args:
            notebook_path: Path to the Jupyter notebook to optimize
            param_values_path: Path to the JSON file with parameter values
        """
        self.notebook_path = notebook_path
        self.notebook_name = os.path.basename(notebook_path)
        self.param_values_path = param_values_path
        
        # Load parameter values from JSON
        with open(param_values_path, 'r') as f:
            self.param_values = json.load(f)
        
        # Extract the current preprocessing pipeline
        self.pipeline_data = self._extract_pipeline()
        
        # Store the original notebook for future reference
        with open(notebook_path, 'r', encoding='utf-8') as f:
            self.nb = nbformat.read(f, as_version=4)
        
        # Create a dict to store optimization results
        self.optimization_results = []

    def _extract_pipeline(self):
        """
        Extract the preprocessing pipeline from the notebook.
        Returns a DataFrame with the preprocessing steps.
        """
        # Use the existing analysis function to extract preprocessing steps
        function_data = analyze_notebook(self.notebook_path)
        
        # Convert to DataFrame for easier manipulation
        pipeline_df = pd.DataFrame(function_data, columns=[
            "Notebook Name", "Category", "Function", 
            "Uses Default Parameters?", "Custom Parameters",
            "All Parameters", "Original Code"
        ])
        
        return pipeline_df
    
    def _get_modifiable_methods(self):
        """
        Get a list of methods that can be modified based on the parameter values JSON.
        Returns a list of (method_name, parameter) tuples.
        """
        modifiable_methods = []
        
        # Filter to methods in our parameter values JSON
        methods_in_json = set(self.param_values.keys())
        
        # Check which methods are used in the notebook
        for _, row in self.pipeline_data.iterrows():
            function_name = row['Function']
            
            # Strip action prefix for feature operations
            if function_name.startswith("Drop column '") or function_name.startswith("Create column '"):
                continue
                
            # Check if this is a method we can modify
            if function_name in methods_in_json:
                # Get parameters for this method
                params = self.param_values[function_name]
                
                # For each parameter, check if it's being used with default value
                for param_name, param_info in params.items():
                    # Skip parameters where there's no allowed values
                    if 'allowed_values' not in param_info or param_info['allowed_values'] == ['No parameters required for initialization']:
                        continue
                    
                    # Check if the parameter is using the default value
                    custom_params = row['Custom Parameters']
                    if param_name not in custom_params:
                        # This parameter is using default, so we can modify it
                        modifiable_methods.append((function_name, param_name))
        
        return modifiable_methods
    
    def _extract_train_test_vars(self):
        """
        Extract variable names for training and testing data, and the model.
        Returns a dict with variable names.
        """
        # This is a heuristic approach and may need adaptation for specific notebooks
        var_names = {
            'X_train': None,
            'y_train': None,
            'X_test': None, 
            'y_test': None,
            'model': None,
            'model_fit_cell': None,
            'evaluation_cell': None
        }
        
        for i, cell in enumerate(self.nb.cells):
            if cell.cell_type != 'code':
                continue
            
            code = cell.source
            
            # Look for train/test split
            if "train_test_split" in code:
                match = re.search(r'([A-Za-z0-9_]+),\s*([A-Za-z0-9_]+),\s*([A-Za-z0-9_]+),\s*([A-Za-z0-9_]+)\s*=\s*train_test_split', code)
                if match:
                    var_names['X_train'] = match.group(1)
                    var_names['X_test'] = match.group(2)
                    var_names['y_train'] = match.group(3)
                    var_names['y_test'] = match.group(4)
            
            # Look for model definition
            if re.search(r'([A-Za-z0-9_]+)\s*=\s*[A-Za-z]+\(', code) and any(model_type in code for model_type in ['LinearRegression', 'RandomForest', 'XGBRegressor', 'DecisionTree']):
                match = re.search(r'([A-Za-z0-9_]+)\s*=\s*[A-Za-z]+\(', code)
                if match:
                    var_names['model'] = match.group(1)
            
            # Look for model fitting
            if var_names['model'] and f"{var_names['model']}.fit" in code:
                var_names['model_fit_cell'] = i
            
            # Look for model evaluation 
            if var_names['model'] and any(metric in code for metric in ['mean_squared_error', 'r2_score', 'accuracy_score']):
                var_names['evaluation_cell'] = i
        
        return var_names
    
    def _create_modified_notebook(self, method_name, param_name, param_value):
        """
        Create a modified version of the notebook with the specified parameter change.
        
        Args:
            method_name: The preprocessing method to modify
            param_name: The parameter to modify
            param_value: The new value for the parameter
            
        Returns:
            Path to the modified notebook
        """
        # Create a new notebook object
        modified_nb = deepcopy(self.nb)
        
        # Find the cell with the method to modify
        for cell in modified_nb.cells:
            if cell.cell_type != 'code':
                continue
            
            # Check if this cell contains the method we want to modify
            if f"{method_name}(" in cell.source:
                # Parse the code to find the method call
                modified_code = self._modify_parameter_in_code(
                    cell.source, method_name, param_name, param_value
                )
                cell.source = modified_code
        
        # Save the modified notebook
        output_path = f"{os.path.splitext(self.notebook_path)[0]}_modified_{method_name}_{param_name}_{param_value}.ipynb"
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(modified_nb, f)
        
        return output_path
    
    def _modify_parameter_in_code(self, code, method_name, param_name, param_value):
        """
        Modify a parameter in a code string.
        
        Args:
            code: The code to modify
            method_name: The method name
            param_name: The parameter name
            param_value: The new parameter value
            
        Returns:
            Modified code
        """
        # Use regex to find the method call
        pattern = rf'{method_name}\s*\((.*?)\)'
        
        def replace_param(match):
            params_str = match.group(1)
            
            # Check if the parameter is already in the params
            if f"{param_name}=" in params_str:
                # Replace the existing parameter
                return re.sub(
                    rf'{param_name}=[^,\)]+', 
                    f'{param_name}={param_value}',
                    match.group(0)
                )
            else:
                # Add the parameter
                if params_str.strip():
                    # There are other parameters, add a comma
                    return f"{method_name}({params_str}, {param_name}={param_value})"
                else:
                    # No other parameters
                    return f"{method_name}({param_name}={param_value})"
        
        modified_code = re.sub(pattern, replace_param, code, flags=re.DOTALL)
        return modified_code
    
    def _execute_notebook(self, notebook_path):
        """
        Execute a notebook and return the performance metrics.
        
        Args:
            notebook_path: Path to the notebook to execute
            
        Returns:
            Dictionary with performance metrics
        """
        # Execute the notebook
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
             '--output', f"{os.path.basename(notebook_path)}", 
             notebook_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error executing notebook: {result.stderr}")
            return {'error': result.stderr}
        
        # Read the executed notebook to extract metrics
        with open(notebook_path, 'r', encoding='utf-8') as f:
            executed_nb = nbformat.read(f, as_version=4)
        
        # Extract metrics from the notebook
        metrics = self._extract_metrics_from_notebook(executed_nb)
        
        return metrics
    
    def _extract_metrics_from_notebook(self, nb):
        """
        Extract performance metrics from a notebook.
        
        Args:
            nb: The notebook object
            
        Returns:
            Dictionary with performance metrics
        """
        var_names = self._extract_train_test_vars()
        
        metrics = {
            'mse': None,
            'r2': None,
            'accuracy': None,
            'f1': None
        }
        
        # Check the evaluation cell
        if var_names['evaluation_cell'] is not None:
            cell = nb.cells[var_names['evaluation_cell']]
            code = cell.source
            
            # Extract metrics from text output
            for output in cell.outputs:
                if 'text' in output:
                    text = output['text']
                    # MSE
                    mse_match = re.search(r'MSE:?\s*([0-9.]+)', text)
                    if mse_match:
                        metrics['mse'] = float(mse_match.group(1))
                    
                    # R2
                    r2_match = re.search(r'R\^?2:?\s*([0-9.]+)', text)
                    if r2_match:
                        metrics['r2'] = float(r2_match.group(1))
                    
                    # Accuracy
                    acc_match = re.search(r'Accuracy:?\s*([0-9.]+)', text)
                    if acc_match:
                        metrics['accuracy'] = float(acc_match.group(1))
                    
                    # F1
                    f1_match = re.search(r'F1:?\s*([0-9.]+)', text)
                    if f1_match:
                        metrics['f1'] = float(f1_match.group(1))
        
        return metrics
    
    def optimize_single_parameter(self, method_name=None, param_name=None):
        """
        Optimize a single parameter of a single method.
        
        Args:
            method_name: (Optional) The method to optimize. If None, a suitable method will be chosen.
            param_name: (Optional) The parameter to optimize. If None, a suitable parameter will be chosen.
            
        Returns:
            Dictionary with optimization results
        """
        # Get modifiable methods
        modifiable_methods = self._get_modifiable_methods()
        
        if not modifiable_methods:
            print("No modifiable methods found in the notebook.")
            return None
        
        # If method and parameter not specified, use the first modifiable method
        if method_name is None or param_name is None:
            method_name, param_name = modifiable_methods[0]
            print(f"Automatically selected method {method_name} and parameter {param_name} for optimization")
        
        # Get the allowed values for this parameter
        param_info = self.param_values[method_name][param_name]
        param_values = param_info['allowed_values']
        
        print(f"Optimizing {method_name}.{param_name} with values: {param_values}")
        
        # Execute the original notebook to get baseline performance
        baseline_metrics = self._execute_notebook(self.notebook_path)
        
        # Initialize results
        results = {
            'method': method_name,
            'parameter': param_name,
            'baseline': baseline_metrics,
            'optimization_results': []
        }
        
        # Test each parameter value
        for value in param_values:
            print(f"Testing {method_name}.{param_name} = {value}")
            
            # Create a modified notebook
            modified_notebook_path = self._create_modified_notebook(method_name, param_name, value)
            
            # Execute the modified notebook
            metrics = self._execute_notebook(modified_notebook_path)
            
            # Store results
            results['optimization_results'].append({
                'value': value,
                'metrics': metrics
            })
            
            # Store in the instance variable
            self.optimization_results.append({
                'method': method_name,
                'parameter': param_name,
                'value': value,
                'metrics': metrics
            })
        
        # Find the best parameter value
        best_result = None
        best_metric_value = float('inf')  # For MSE, lower is better
        
        for result in results['optimization_results']:
            if 'mse' in result['metrics'] and result['metrics']['mse'] is not None:
                if result['metrics']['mse'] < best_metric_value:
                    best_metric_value = result['metrics']['mse']
                    best_result = result
        
        results['best_value'] = best_result['value'] if best_result else None
        results['best_metrics'] = best_result['metrics'] if best_result else None
        
        return results
    
    def optimize_multiple_parameters(self, num_methods=2):
        """
        Optimize multiple parameters across multiple methods.
        
        Args:
            num_methods: Number of methods to optimize
            
        Returns:
            Dictionary with optimization results
        """
        # Get modifiable methods
        modifiable_methods = self._get_modifiable_methods()
        
        if not modifiable_methods:
            print("No modifiable methods found in the notebook.")
            return None
        
        # Select the top N methods to optimize
        methods_to_optimize = modifiable_methods[:min(num_methods, len(modifiable_methods))]
        
        results = []
        
        for method_name, param_name in methods_to_optimize:
            # Optimize this parameter
            method_result = self.optimize_single_parameter(method_name, param_name)
            results.append(method_result)
        
        return results
    
    def generate_report(self):
        """
        Generate a report of the optimization results.
        
        Returns:
            String with the report
        """
        if not self.optimization_results:
            return "No optimization results available."
        
        report = []
        report.append("# Parameter Optimization Report")
        report.append(f"Notebook: {self.notebook_name}\n")
        
        # Group results by method and parameter
        grouped_results = {}
        for result in self.optimization_results:
            key = f"{result['method']}.{result['parameter']}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Generate report for each method/parameter
        for key, results in grouped_results.items():
            report.append(f"## {key}")
            
            # Sort results by performance
            if all('metrics' in r and 'mse' in r['metrics'] and r['metrics']['mse'] is not None for r in results):
                sorted_results = sorted(results, key=lambda x: x['metrics']['mse'])
            elif all('metrics' in r and 'r2' in r['metrics'] and r['metrics']['r2'] is not None for r in results):
                sorted_results = sorted(results, key=lambda x: x['metrics']['r2'], reverse=True)
            else:
                sorted_results = results
            
            # Report each value
            report.append("| Value | MSE | R² | Accuracy | F1 |")
            report.append("|-------|-----|----|---------|----|")
            
            for result in sorted_results:
                value = result['value']
                metrics = result['metrics']
                
                mse = metrics.get('mse', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                accuracy = metrics.get('accuracy', 'N/A')
                f1 = metrics.get('f1', 'N/A')
                
                report.append(f"| {value} | {mse} | {r2} | {accuracy} | {f1} |")
            
            report.append("")
        
        return "\n".join(report)

    def save_report(self, output_path=None):
        """
        Save the optimization report to a file.
        
        Args:
            output_path: (Optional) Path to save the report. If None, a default name will be used.
        """
        if output_path is None:
            output_path = f"{os.path.splitext(self.notebook_path)[0]}_optimization_report.md"
        
        report = self.generate_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")

# CLI interface
def main():
    parser = argparse.ArgumentParser(
        description="Optimize parameters in a Jupyter notebook's preprocessing pipeline."
    )
    parser.add_argument("notebook_path", type=str, help="Path to the Jupyter notebook to optimize")
    parser.add_argument("--params", type=str, default="Parameter_Values.json", 
                       help="Path to the JSON file with parameter values")
    parser.add_argument("--method", type=str, default=None,
                       help="The method to optimize (if None, a suitable method will be chosen)")
    parser.add_argument("--param", type=str, default=None,
                       help="The parameter to optimize (if None, a suitable parameter will be chosen)")
    parser.add_argument("--multiple", type=int, default=0,
                       help="Optimize multiple parameters (specify the number of methods)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save the optimization report")
    
    args = parser.parse_args()
    
    # Create the optimizer
    optimizer = ParameterOptimizer(args.notebook_path, args.params)
    
    # Optimize parameters
    if args.multiple > 0:
        results = optimizer.optimize_multiple_parameters(args.multiple)
    else:
        results = optimizer.optimize_single_parameter(args.method, args.param)
    
    # Save the report
    optimizer.save_report(args.output)
    
    # Print summary
    print("\nOptimization complete!")
    print(f"Results summary:")
    for i, result in enumerate(optimizer.optimization_results):
        method = result['method']
        param = result['parameter']
        value = result['value']
        metrics = result['metrics']
        
        metric_str = []
        if 'mse' in metrics and metrics['mse'] is not None:
            metric_str.append(f"MSE: {metrics['mse']:.4f}")
        if 'r2' in metrics and metrics['r2'] is not None:
            metric_str.append(f"R²: {metrics['r2']:.4f}")
        
        print(f"{i+1}. {method}.{param} = {value}: {', '.join(metric_str)}")

if __name__ == "__main__":
    main()