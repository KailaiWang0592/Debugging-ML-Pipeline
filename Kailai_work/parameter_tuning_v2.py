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
from datetime import datetime
import ast
from ast import NodeVisitor
import gc
import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from analysis_pipeline import analyze_notebook, extract_functions_from_code
except ImportError:
    print("Warning: Could not import analysis_pipeline module. Some functionality may be limited.")


class MetricsExtractor(NodeVisitor):
    
    def __init__(self):
        self.metrics = {}
        self.current_model = None
        self.metric_variables = {}
        self.tf_history_vars = {}
    
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id

            if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'attr') and node.value.func.attr == 'fit':
                if 'tensorflow' not in self.metrics:
                    self.metrics['tensorflow'] = {}
                self.tf_history_vars[target_name] = True
                print(f"Found TensorFlow history variable: {target_name}")
                
            if (isinstance(node.value, ast.Subscript) and 
                isinstance(node.value.value, ast.Attribute) and 
                hasattr(node.value.value, 'attr') and
                node.value.value.attr == 'history'):
                
                if 'tensorflow' not in self.metrics:
                    self.metrics['tensorflow'] = {}
                    
                val_prefix = False
                if isinstance(node.value.slice, ast.Constant): 
                    val = node.value.slice.value
                    if isinstance(val, str) and ('accuracy' in val.lower() or 'acc' in val.lower()):
                        if 'val_' in val.lower():
                            val_prefix = True
                            self.metrics['tensorflow']['val_accuracy_var'] = target_name
                        else:
                            self.metrics['tensorflow']['accuracy_var'] = target_name
                        print(f"Found accuracy variable in history: {target_name} = {val}")

                elif hasattr(ast, 'Index') and isinstance(node.value.slice, ast.Index):  
                    if hasattr(node.value.slice.value, 'value'):
                        val = node.value.slice.value.value
                        if isinstance(val, str) and ('accuracy' in val.lower() or 'acc' in val.lower()):
                            if 'val_' in val.lower():
                                val_prefix = True
                                self.metrics['tensorflow']['val_accuracy_var'] = target_name
                            else:
                                self.metrics['tensorflow']['accuracy_var'] = target_name
                            print(f"Found accuracy variable in history: {target_name} = {val}")
                    
                if not val_prefix and ('accuracy' in target_name.lower() or 'acc' in target_name.lower()):
                    if 'val_' in target_name.lower():
                        self.metrics['tensorflow']['val_accuracy_var'] = target_name
                    else:
                        self.metrics['tensorflow']['accuracy_var'] = target_name
                    print(f"Found accuracy variable by name pattern: {target_name}")

            if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'id') and node.value.func.id == "cross_val_score":
                for arg in node.value.keywords:
                    if arg.arg == "scoring" and isinstance(arg.value, ast.Constant) and "neg_mean_squared_error" in arg.value.value:
                        if "model" not in self.metrics:
                            self.metrics["model"] = {}
                        self.metrics["model"]["uses_cv_rmse"] = True
                        self.metrics["model"]["cv_rmse_var"] = target_name
                
            if "r2" in target_name.lower():
                if "_train" in target_name.lower():
                    model_name = target_name.split("_")[0] if "_" in target_name else "model"
                    if model_name not in self.metrics:
                        self.metrics[model_name] = {}
                    self.metrics[model_name]["r2_train_var"] = target_name
                elif "_test" in target_name.lower():
                    model_name = target_name.split("_")[0] if "_" in target_name else "model"
                    if model_name not in self.metrics:
                        self.metrics[model_name] = {}
                    self.metrics[model_name]["r2_test_var"] = target_name
            
            model_metric_match = None
            for model_prefix in ["catb", "xgboost", "linear", "lasso", "ridge", "elastic_net", "svr", "knn", "gb", "en"]:
                if target_name.startswith(f"{model_prefix}_"):
                    model_name = model_prefix
                    if "rmse" in target_name:
                        metric_type = "rmse"
                    elif "mse" in target_name:
                        metric_type = "mse"
                    elif "r2" in target_name:
                        metric_type = "r2"
                    elif "mae" in target_name.lower():
                        metric_type = "mae"
                    elif "accuracy" in target_name.lower():
                        metric_type = "accuracy"
                    elif "precision" in target_name.lower():
                        metric_type = "precision"
                    elif "recall" in target_name.lower():
                        metric_type = "recall"
                    elif "f1" in target_name.lower():
                        metric_type = "f1"
                    else:
                        continue
                    
                    model_metric_match = (model_name, metric_type)
                    break
            
            if not model_metric_match:
                if "accuracy" in target_name.lower():
                    model_metric_match = ("model", "accuracy")
                elif "precision" in target_name.lower():
                    model_metric_match = ("model", "precision")
                elif "recall" in target_name.lower():
                    model_metric_match = ("model", "recall")
                elif "f1" in target_name.lower():
                    model_metric_match = ("model", "f1")
            
            if model_metric_match:
                model_name, metric_type = model_metric_match
                
                if model_name not in self.metrics:
                    self.metrics[model_name] = {}
                
                self.metrics[model_name][f"{metric_type}_calc"] = True
                self.metrics[model_name]["var_name"] = target_name
                
                if self._has_mse_calculation(node.value):
                    self.metrics[model_name]["has_direct_mse"] = True
                
                if self._has_sqrt_call(node.value):
                    self.metrics[model_name]["uses_sqrt"] = True
                    
                if self._has_log_call(node.value):
                    self.metrics[model_name]["uses_log"] = True
            
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if hasattr(node.func, 'id') and node.func.id == 'print':
            if len(node.args) > 0:
                self._extract_metrics_from_print(node)
        
        self.generic_visit(node)
    
    def _extract_metrics_from_print(self, node):
        for arg in node.args:
            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                if isinstance(arg.left, ast.Constant) and isinstance(arg.left.value, str):
                    str_content = arg.left.value
                    for model_prefix in ["catb", "xgboost", "linear", "lasso", "ridge", "svr", "knn"]:
                        if model_prefix in str_content.lower():
                            if model_prefix not in self.metrics:
                                self.metrics[model_prefix] = {}
                            
                            if "rmse" in str_content.lower():
                                self.metrics[model_prefix]["rmse_in_print"] = True
                            if "mse" in str_content.lower():
                                self.metrics[model_prefix]["mse_in_print"] = True
                            if "r2" in str_content.lower() or "r squared" in str_content.lower():
                                self.metrics[model_prefix]["r2_in_print"] = True
            if isinstance(arg, ast.JoinedStr):
                for value in arg.values:
                    if isinstance(value, ast.FormattedValue):
                        if isinstance(value.value, ast.Call) and hasattr(value.value.func, 'id'):
                            func_name = value.value.func.id
                            if func_name == "accuracy_score":
                                if "model" not in self.metrics:
                                    self.metrics["model"] = {}
                                self.metrics["model"]["accuracy_in_print"] = True
                            elif func_name == "r2_score":
                                if "model" not in self.metrics:
                                    self.metrics["model"] = {}
                                self.metrics["model"]["r2_in_print"] = True
        
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if "oob score" in arg.value.lower():
                    if "model" not in self.metrics:
                        self.metrics["model"] = {}
                    self.metrics["model"]["uses_oob_score"] = True

    def _has_mse_calculation(self, node):
        if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == "mean_squared_error":
            return True
        
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "mean_squared_error":
            return True
            
        return False

    def _has_sqrt_call(self, node):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'attr') and node.func.attr == "sqrt":
                return True
            if hasattr(node.func, 'id') and node.func.id == "sqrt":
                return True
            
        return False

    def _has_log_call(self, node):
        if isinstance(node, ast.Call):
            if (hasattr(node.func, 'attr') and node.func.attr == "log") or \
               (hasattr(node.func, 'id') and node.func.id == "log"):
                return True
            
            for arg in node.args:
                if self._has_log_call(arg):
                    return True
                    
        return False

class ParameterOptimizer:
    def __init__(self, notebook_path, param_values_path):
        self.notebook_path = notebook_path
        self.notebook_name = os.path.basename(notebook_path)
        self.param_values_path = param_values_path
        
        try:
            with open(param_values_path, 'r') as f:
                self.param_values = json.load(f)
        except Exception as e:
            print(f"Error loading parameter values: {e}")
            self.param_values = {}
        
        try:
            self.pipeline_data = self._extract_pipeline()
        except Exception as e:
            print(f"Error extracting pipeline: {e}")
            self.pipeline_data = pd.DataFrame()
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                self.nb = nbformat.read(f, as_version=4)
        except Exception as e:
            print(f"Error reading notebook: {e}")
            self.nb = None
        
        self.optimization_results = []

    def _extract_pipeline(self):

        try:
            function_data = analyze_notebook(self.notebook_path)
            
            pipeline_df = pd.DataFrame(function_data, columns=[
                "Notebook Name", "Category", "Function", 
                "Uses Default Parameters?", "Custom Parameters",
                "All Parameters", "Original Code"
            ])
            
            return pipeline_df

        except Exception as e:
            print(f"Error in _extract_pipeline: {e}")
            return pd.DataFrame(columns=[
                "Notebook Name", "Category", "Function", 
                "Uses Default Parameters?", "Custom Parameters",
                "All Parameters", "Original Code"
            ])
    
    def _get_modifiable_methods(self):
        modifiable_methods = []
        
        try:
            methods_in_json = set(self.param_values.keys())

            for _, row in self.pipeline_data.iterrows():
                function_name = row['Function']

                if function_name == "drop" or function_name == "dropna":
                    continue

                if function_name.startswith("Drop column '") or function_name.startswith("Create column '"):
                    continue
                    
                if function_name in methods_in_json:
                    params = self.param_values[function_name]
                    
                    for param_name, param_info in params.items():
                        if 'allowed_values' not in param_info or param_info['allowed_values'] == ['No parameters required for initialization']:
                            continue
                        
                        custom_params = row['Custom Parameters']
                        if param_name not in custom_params:
                            modifiable_methods.append((function_name, param_name))
        
        except Exception as e:
            print(f"Error in _get_modifiable_methods: {e}")
        
        return modifiable_methods
    
    def _extract_train_test_vars(self):
        var_names = {
            'X_train': None,
            'y_train': None,
            'X_test': None, 
            'y_test': None,
            'model': None,
            'model_fit_cell': None,
            'evaluation_cell': None
        }
        
        try:
            for i, cell in enumerate(self.nb.cells):
                if cell.cell_type != 'code':
                    continue
                
                code = cell.source
            
                if "train_test_split" in code:
                    match = re.search(r'([A-Za-z0-9_]+),\s*([A-Za-z0-9_]+),\s*([A-Za-z0-9_]+),\s*([A-Za-z0-9_]+)\s*=\s*train_test_split', code)
                    if match:
                        var_names['X_train'] = match.group(1)
                        var_names['X_test'] = match.group(2)
                        var_names['y_train'] = match.group(3)
                        var_names['y_test'] = match.group(4)
                
                if re.search(r'([A-Za-z0-9_]+)\s*=\s*[A-Za-z]+\(', code) and any(model_type in code for model_type in ['LinearRegression', 'RandomForest', 'XGBRegressor', 'DecisionTree']):
                    match = re.search(r'([A-Za-z0-9_]+)\s*=\s*[A-Za-z]+\(', code)
                    if match:
                        var_names['model'] = match.group(1)
                
                if var_names['model'] and f"{var_names['model']}.fit" in code:
                    var_names['model_fit_cell'] = i
                
                if var_names['model'] and any(metric in code for metric in ['mean_squared_error', 'r2_score', 'accuracy_score']):
                    var_names['evaluation_cell'] = i

        except Exception as e:
            print(f"Error in _extract_train_test_vars: {e}")
        
        return var_names
    
    def _create_modified_notebook(self, method_name, param_name, param_value):
        try:
            modified_nb = deepcopy(self.nb)
            
            for cell in modified_nb.cells:
                if cell.cell_type != 'code':
                    continue

                if f"{method_name}(" in cell.source:
                    modified_code = self._modify_parameter_in_code(
                        cell.source, method_name, param_name, param_value
                    )
                    cell.source = modified_code
            
            safe_value = str(param_value).replace('[', '').replace(']', '').replace(',', '_').replace(' ', '')
            output_path = f"{os.path.splitext(self.notebook_path)[0]}_modified_{method_name}_{param_name}_{safe_value}.ipynb"
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(modified_nb, f)
            
            return output_path

        except Exception as e:
            print(f"Error creating modified notebook for {method_name}.{param_name}={param_value}: {e}")
            return None
    
    def _modify_parameter_in_code(self, code, method_name, param_name, param_value):
        try:
            if method_name in ["drop", "dropna"]:
                print(f"Skipping code modification for {method_name} method")
                return code
            
            self.current_param_name = param_name  
            py_param_value = self._convert_to_python_value(param_value)
            
            pattern = rf'{method_name}\s*\((.*?)\)'
            
            def replace_param(match):
                params_str = match.group(1)
                
                if f"{param_name}=" in params_str:
                    return re.sub(
                        rf'{param_name}=[^,\)]+', 
                        f'{param_name}={py_param_value}',
                        match.group(0)
                    )
                else:
                    if params_str.strip():
                        return f"{method_name}({params_str}, {param_name}={py_param_value})"
                    else:
                        return f"{method_name}({param_name}={py_param_value})"
            
            modified_code = re.sub(pattern, replace_param, code, flags=re.DOTALL)
            return modified_code
        except Exception as e:
            print(f"Error modifying parameter in code: {e}")
            return code  

    def _parse_params(self, params_str):
        params_dict = {}
        
        if not params_str.strip():
            return params_dict
            
        parts = []
        current_part = ""
        bracket_level = 0
        
        for char in params_str:
            if char == ',' and bracket_level == 0:
                parts.append(current_part.strip())
                current_part = ""
            else:
                if char in '([{':
                    bracket_level += 1
                elif char in ')]}':
                    bracket_level -= 1
                current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        for part in parts:
            if '=' in part:
                param_name, param_value = part.split('=', 1)
                params_dict[param_name.strip()] = param_value.strip()
                
        return params_dict
    
    def _convert_to_python_value(self, value):
        if value is None:
            return "None"
        
        if isinstance(value, str):
            if value == "inf":
                return "float('inf')"
            elif value == "-inf":
                return "float('-inf')"
            elif value.lower() == "nan":
                return "float('nan')"
            
            elif value.startswith("np.") or "." in value and not value.startswith('"') and not value.startswith("'"):
                return value
            
            else:
                return f"'{value}'"
        
        elif isinstance(value, list):
            if self.current_param_name == "feature_range":
                elements = [self._convert_to_python_value(item) for item in value]
                return f"({', '.join(elements)})"
            else:
                elements = [self._convert_to_python_value(item) for item in value]
                return f"[{', '.join(elements)}]"

        
        elif isinstance(value, dict):
            items = [f"{self._convert_to_python_value(k)}: {self._convert_to_python_value(v)}" 
                     for k, v in value.items()]
            return f"{{{', '.join(items)}}}"
        
        elif isinstance(value, bool):
            return str(value) 
        
        else:
            return str(value)

    def _inject_metric_instrumentation(self, notebook):
        modified_nb = deepcopy(notebook)
        
        metric_functions = [
            'mean_squared_error', 'mean_absolute_error', 'r2_score', 
            'accuracy_score', 'precision_score', 'recall_score', 'f1_score'
        ]
        
        for cell in modified_nb.cells:
            if cell.cell_type != 'code':
                continue
            
            for func in metric_functions:
                pattern = rf'({func}\([^)]+\))'
            
                replacement = rf'\1\nprint("METRIC_LOG: {func} =", \1)'
                
                cell.source = re.sub(pattern, replacement, cell.source)
        
            score_pattern = r'(([A-Za-z0-9_]+)\.score\([^)]+\))'
            score_replacement = r'\1\nprint("METRIC_LOG: \2_score =", \1)'
            cell.source = re.sub(score_pattern, score_replacement, cell.source)
            
            if 'history' in cell.source and '.history' in cell.source:
                history_instrumentation = "\n# Instrumentation for TensorFlow metrics\n"
                history_instrumentation += "if 'history' in locals() and hasattr(history, 'history'):\n"
                history_instrumentation += "    for metric in history.history:\n"
                history_instrumentation += "        print(f\"METRIC_LOG: tf_{metric} = {history.history[metric][-1]}\")\n"
                cell.source += history_instrumentation
        
        return modified_nb
    
    def _execute_notebook(self, notebook_path):
        try:
            result = subprocess.run(
                ["python3", "run_notebook.py", "--notebook", notebook_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Execution error: {result.stderr}")
                return {'error': result.stderr}
            return json.loads(result.stdout)
        except Exception as e:
            print(f"Failed to execute isolated notebook run: {e}")
            return {'error': str(e)}

            with open(executed_path, 'r', encoding='utf-8') as f:
                executed_nb = nbformat.read(f, as_version=4)

            return self._extract_metrics(executed_nb)
        except Exception as e:
            print(f"Error executing notebook: {e}")
            return {'error': str(e)}

    
    def _extract_metrics_from_notebook(self, nb):

        try:
            result = {
                'mse': None,
                'rmse': None,
                'r2': None,
                'mae': None,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'best_model': 'unknown'
            }
            
            tf_metrics = self._extract_tensorflow_metrics(nb)
            if tf_metrics:
                for key, value in tf_metrics.items():
                    result[key] = value
                result['best_model'] = 'tensorflow'
                
            metrics_by_model = self._extract_metrics_with_ast(nb)
            
            if 'tensorflow' in metrics_by_model and 'accuracy' in metrics_by_model['tensorflow']:
                result['accuracy'] = metrics_by_model['tensorflow']['accuracy']
                result['best_model'] = 'tensorflow'
            
            all_models_metrics = []
            for model_name, metrics in metrics_by_model.items():
                model_metrics = {
                    'model_name': model_name,
                    'mse': metrics.get('mse'),
                    'rmse': metrics.get('rmse'),
                    'r2': metrics.get('r2'),
                    'mae': metrics.get('mae'),
                    'accuracy': metrics.get('accuracy'),
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1': metrics.get('f1')
                }
                all_models_metrics.append(model_metrics)
            
            if all_models_metrics:
                print("here1") 

                is_classification = any(
                    metrics.get('accuracy') is not None or 
                    metrics.get('f1') is not None 
                    for metrics in all_models_metrics
                )
                
                if is_classification:
                    best_metrics = None
                    best_score = -float('inf')
                    
                    for metrics in all_models_metrics:
                        if metrics.get('f1') is not None:
                            score = metrics['f1']
                        elif metrics.get('accuracy') is not None:
                            score = metrics['accuracy']
                        else:
                            score = -float('inf')
                        
                        if score > best_score:
                            best_score = score
                            best_metrics = metrics
                else:
                    best_metrics = None
                    lowest_error = float('inf')
                
                    for metrics in all_models_metrics:
                        if metrics.get('mse') is not None:
                            error = metrics['mse']
                        elif metrics.get('rmse') is not None:
                            error = metrics['rmse']
                        elif metrics.get('mae') is not None:
                            error = metrics['mae']
                        else:
                            r2_val = metrics.get('r2')
                            error = -r2_val if r2_val is not None else float('inf')
                        
                        if error < lowest_error:
                            lowest_error = error
                            best_metrics = metrics
                
                if best_metrics:
                    print(best_metrics)
                    for key in ['mse', 'rmse', 'r2', 'mae', 'accuracy', 'precision', 'recall', 'f1']:
                        if key in best_metrics and best_metrics[key] is not None:
                            result[key] = best_metrics[key]
                    result['best_model'] = best_metrics.get('model_name', 'unknown')

            return result    
                    
        except Exception as e:
            print(f"Error extracting metrics from notebook: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'mse': None,
            'rmse': None,
            'r2': None,
            'mae': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'best_model': 'unknown'
        }

    def _extract_metrics_with_ast(self, nb):
        metrics_by_model = {}
        
        try:
            for cell in nb.cells:
                if cell.cell_type != 'code':
                    continue
                    
                code = cell.source
                
                try:
                    tree = ast.parse(code)
                    extractor = MetricsExtractor()
                    extractor.visit(tree)
                    
                    for model_name, model_metrics in extractor.metrics.items():
                        if model_name not in metrics_by_model:
                            metrics_by_model[model_name] = {}
                        

                        metrics_by_model[model_name].update(model_metrics)
                        
                except SyntaxError:
                    continue
                
                if 'outputs' in cell:
                    self._extract_values_from_outputs(cell.outputs, metrics_by_model)
            
            self._process_metrics_dictionary(metrics_by_model)
            
            return metrics_by_model
                        
        except Exception as e:
            print(f"Error extracting metrics using AST: {e}")
            return {}

    def _extract_tensorflow_metrics(self, nb):
        try:
            metrics = {'accuracy': None}
            accuracy_values = []
            val_accuracy_values = []
            
            for cell in nb.cells:
                if cell.cell_type != 'code':
                    continue
                    
                code = cell.source
                
                if 'history.history' in code:
                    print("Found TensorFlow history access")
                    
                    acc_patterns = [
                        r'(\w+)\s*=\s*history\.history\[[\'\"]Accuracy[\'\"]\]',
                        r'(\w+)\s*=\s*history\.history\[[\'\"]accuracy[\'\"]\]',
                        r'(\w+)\s*=\s*history\.history\[[\'\"]acc[\'\"]\]',
                        r'(\w+)\s*=\s*history\.history\[[\'\"]val_Accuracy[\'\"]\]',
                        r'(\w+)\s*=\s*history\.history\[[\'\"]val_accuracy[\'\"]\]',
                        r'(\w+)\s*=\s*history\.history\[[\'\"]val_acc[\'\"]\]'
                    ]
                    
                    for pattern in acc_patterns:
                        matches = re.findall(pattern, code)
                        if matches:
                            print(f"Found TensorFlow accuracy variable: {matches}")
                            
                    if 'outputs' in cell:
                        for output in cell.outputs:
                            if 'text' in output:
                                text = output['text']
                                acc_patterns = [
                                    r'Accuracy:?\s*([0-9.]+)',
                                    r'accuracy:?\s*([0-9.]+)',
                                    r'acc:?\s*([0-9.]+)',
                                    r'val_Accuracy:?\s*([0-9.]+)',
                                    r'val_accuracy:?\s*([0-9.]+)',
                                    r'val_acc:?\s*([0-9.]+)'
                                ]
                                
                                for pattern in acc_patterns:
                                    acc_matches = re.findall(pattern, text, re.IGNORECASE)
                                    if acc_matches:
                                        print(f"Found accuracy values in output: {acc_matches}")
                                        for match in acc_matches:
                                            try:
                                                acc_value = float(match)
                                                if "val_" in pattern:
                                                    val_accuracy_values.append(acc_value)
                                                else:
                                                    accuracy_values.append(acc_value)
                                            except ValueError:
                                                continue
                
                if 'outputs' in cell and 'history' in code:
                    for output in cell.outputs:
                        if 'text' in output:
                            text = output['text']
                            acc_pattern = r'[\'"](?:val_)?(?:Accuracy|accuracy|acc)[\'"].*?([0-9.]+)'
                            acc_matches = re.findall(acc_pattern, text)
                            if acc_matches:
                                print(f"Found raw accuracy values: {acc_matches}")
                                for match in acc_matches:
                                    try:
                                        acc_values.append(float(match))
                                    except ValueError:
                                        continue
            
            if accuracy_values:
                metrics['accuracy'] = accuracy_values[-1] 
            elif val_accuracy_values:
                metrics['accuracy'] = val_accuracy_values[-1]
                
            return metrics
                            
        except Exception as e:
            print(f"Error extracting TensorFlow metrics: {e}")
            traceback.print_exc()
        
        return None

    def _extract_metrics_from_instrumented_logs(self, nb):
        metrics = {
            'mse': None,
            'rmse': None,
            'r2': None,
            'mae': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'best_model': 'unknown'
        }
        
        metrics_by_model = {}
        
        for cell in nb.cells:
            if cell.cell_type != 'code' or 'outputs' not in cell:
                continue
            
            for output in cell.outputs:
                if 'text' not in output:
                    continue
                
                text = output['text']

                log_entries = re.findall(r'METRIC_LOG: ([^=]+) = ([^(\n]+)', text)
                
                for metric_name, metric_value in log_entries:
                    metric_name = metric_name.strip()
                    
                    try:
                        metric_value = eval(metric_value.strip())
                        model_name = 'model'
                        model_metric_match = re.match(r'([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)', metric_name)
                        if model_metric_match:
                            potential_model, potential_metric = model_metric_match.groups()
                            if potential_metric in ['accuracy', 'precision', 'recall', 'f1', 'mse', 'rmse', 'r2', 'mae']:
                                model_name = potential_model
                                metric_name = potential_metric

                        if model_name not in metrics_by_model:
                            metrics_by_model[model_name] = {}
                        
                        if "mean_squared_error" in metric_name:
                            metrics['mse'] = float(metric_value)
                            metrics['rmse'] = float(np.sqrt(metric_value))
                            metrics_by_model[model_name]['mse'] = float(metric_value)
                            metrics_by_model[model_name]['rmse'] = float(np.sqrt(metric_value))
                        
                        elif "mean_absolute_error" in metric_name:
                            metrics['mae'] = float(metric_value)
                            metrics_by_model[model_name]['mae'] = float(metric_value)
                        
                        elif "r2_score" in metric_name:
                            metrics['r2'] = float(metric_value)
                            metrics_by_model[model_name]['r2'] = float(metric_value)
                        
                        elif "accuracy_score" in metric_name:
                            metrics['accuracy'] = float(metric_value)
                            metrics_by_model[model_name]['accuracy'] = float(metric_value)
                        
                        elif "precision_score" in metric_name:
                            metrics['precision'] = float(metric_value)
                            metrics_by_model[model_name]['precision'] = float(metric_value)
                        
                        elif "recall_score" in metric_name:
                            metrics['recall'] = float(metric_value)
                            metrics_by_model[model_name]['recall'] = float(metric_value)
                        
                        elif "f1_score" in metric_name:
                            metrics['f1'] = float(metric_value)
                            metrics_by_model[model_name]['f1'] = float(metric_value)
                        
                        elif metric_name.startswith("tf_"):
                            metric_type = metric_name[3:]
                            if "accuracy" in metric_type or "acc" in metric_type:
                                if "val_" not in metric_type:
                                    metrics['accuracy'] = float(metric_value)
                                    if 'tensorflow' not in metrics_by_model:
                                        metrics_by_model['tensorflow'] = {}
                                    metrics_by_model['tensorflow']['accuracy'] = float(metric_value)
                                    metrics['best_model'] = 'tensorflow'
                        
                        elif "_score" in metric_name:
                            model_name = metric_name.split("_score")[0]
                            if model_name not in metrics_by_model:
                                metrics_by_model[model_name] = {}
                            
                            if "accuracy" in metric_name:
                                metrics_by_model[model_name]['accuracy'] = float(metric_value)
                            else:
                                metrics_by_model[model_name]['r2'] = float(metric_value)
                        
                        elif metric_name in ['accuracy', 'precision', 'recall', 'f1', 'mse', 'rmse', 'r2', 'mae']:
                            metrics[metric_name] = float(metric_value)
                            metrics_by_model[model_name][metric_name] = float(metric_value)
                    
                    except (ValueError, SyntaxError) as e:
                        print(f"Couldn't parse metric value: {metric_value}, Error: {e}")
        
        if metrics_by_model:
            is_classification = any('accuracy' in model_metrics or 'f1' in model_metrics 
                                    for model_metrics in metrics_by_model.values())
            
            best_model = None
            best_metric = None
            
            if is_classification:
                for model, model_metrics in metrics_by_model.items():
                    if 'f1' in model_metrics:
                        current_metric = model_metrics['f1']
                        if best_metric is None or current_metric > best_metric:
                            best_metric = current_metric
                            best_model = model
                    elif 'accuracy' in model_metrics:
                        current_metric = model_metrics['accuracy']
                        if best_metric is None or current_metric > best_metric:
                            best_metric = current_metric
                            best_model = model
            else:
                for model, model_metrics in metrics_by_model.items():
                    if 'r2' in model_metrics:
                        current_metric = model_metrics['r2']
                        if best_metric is None or current_metric > best_metric:
                            best_metric = current_metric
                            best_model = model
                    elif 'mse' in model_metrics:
                        current_metric = -model_metrics['mse']
                        if best_metric is None or current_metric > best_metric:
                            best_metric = current_metric
                            best_model = model
                    elif 'rmse' in model_metrics:
                        current_metric = -model_metrics['rmse']
                        if best_metric is None or current_metric > best_metric:
                            best_metric = current_metric
                            best_model = model
                    elif 'mae' in model_metrics:
                        current_metric = -model_metrics['mae']
                        if best_metric is None or current_metric > best_metric:
                            best_metric = current_metric
                            best_model = model
            
            if best_model:
                metrics['best_model'] = best_model
                for metric_name in ['mse', 'rmse', 'r2', 'mae', 'accuracy', 'precision', 'recall', 'f1']:
                    if metric_name in metrics_by_model[best_model]:
                        metrics[metric_name] = metrics_by_model[best_model][metric_name]
        
        if metrics['mse'] is not None and metrics['rmse'] is None:
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    def _extract_values_from_outputs(self, outputs, metrics_by_model):
        for output in outputs:
            if 'text' not in output:
                continue
                
            text = output['text']

            validation_acc_match = re.search(r'Validation Accuracy:?\s*([0-9.]+)', text, re.IGNORECASE)
            if validation_acc_match:
                if 'model' not in metrics_by_model:
                    metrics_by_model['model'] = {}
                metrics_by_model['model']['accuracy'] = float(validation_acc_match.group(1))
            
            oob_match = re.search(r'oob score:?\s*([0-9.]+)', text, re.IGNORECASE)
            if oob_match:
                if 'model' not in metrics_by_model:
                    metrics_by_model['model'] = {}
                metrics_by_model['model']['oob_score'] = float(oob_match.group(1))
            
            rmse_cv_match = re.search(r'rmse\s*(?::|=)\s*([0-9.]+)', text, re.IGNORECASE)
            if rmse_cv_match:
                if 'model' not in metrics_by_model:
                    metrics_by_model['model'] = {}
                metrics_by_model['model']['rmse'] = float(rmse_cv_match.group(1))
            
            r2_train_match = re.search(r'R2 Train Score:?\s*([0-9.]+)', text, re.IGNORECASE)
            if r2_train_match:
                model_name = 'model'
                for prefix in ['svr', 'rf', 'lr', 'xgb']:
                    if f'r2_{prefix}_train' in text.lower():
                        model_name = prefix
                        break
                
                if model_name not in metrics_by_model:
                    metrics_by_model[model_name] = {}
                metrics_by_model[model_name]['r2_train'] = float(r2_train_match.group(1))
                
            r2_test_match = re.search(r'R2 Test Score:?\s*([0-9.]+)', text, re.IGNORECASE)
            if r2_test_match:
                model_name = 'model' 
                for prefix in ['svr', 'rf', 'lr', 'xgb']:
                    if f'r2_{prefix}_test' in text.lower():
                        model_name = prefix
                        break
                        
                if model_name not in metrics_by_model:
                    metrics_by_model[model_name] = {}
                metrics_by_model[model_name]['r2'] = float(r2_test_match.group(1))


            if 'tensorflow' in metrics_by_model:
                patterns = [
                    r'accuracy[:\s=]+([0-9.]+)',
                    r'acc[:\s=]+([0-9.]+)',
                    r'Accuracy[:\s=]+([0-9.]+)',  
                    r'val[_\s]accuracy[:\s=]+([0-9.]+)',
                    r'val[_\s]Accuracy[:\s=]+([0-9.]+)',
                    r'val[_\s]acc[:\s=]+([0-9.]+)',
                    r'accuracy[^:]*:\s*([0-9.]+)',
                    r'Accuracy[^:]*:\s*([0-9.]+)'
                ]
            
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            metrics_by_model['tensorflow']['accuracy'] = float(match.group(1))
                            break
                        except (ValueError, IndexError):
                            continue
            
            for model_name in list(metrics_by_model.keys()):
                model_metrics = metrics_by_model[model_name]
                
                for metric_name in ["accuracy", "precision", "recall", "f1"]:
                    if f"{metric_name}_calc" in model_metrics or f"{metric_name}_in_print" in model_metrics:
                        patterns = [
                            rf'{model_name}.*?{metric_name}.*?([0-9.]+)',
                            rf'{metric_name}.*?{model_name}.*?([0-9.]+)',
                            rf'{model_metrics.get("var_name", "")}.*?([0-9.]+)',
                            rf'{metric_name}[:\s=]+([0-9.]+)'
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                try:
                                    metrics_by_model[model_name][metric_name] = float(match.group(1))
                                    break
                                except (ValueError, IndexError):
                                    continue

                if 'rmse_calc' in model_metrics or 'rmse_in_print' in model_metrics:
                    patterns = [
                        rf'{model_name}.*?rmse.*?([0-9.]+)',
                        rf'rmse.*?{model_name}.*?([0-9.]+)',
                        rf'root mean squared error.*?{model_name}.*?([0-9.]+)',
                        rf'{model_metrics.get("var_name", "")}.*?([0-9.]+)',
                        r'rmse[:\s=]+([0-9.]+)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            try:
                                metrics_by_model[model_name]['rmse'] = float(match.group(1))
                                break  
                            except (ValueError, IndexError):
                                continue
                
                if 'mse_calc' in model_metrics or 'mse_in_print' in model_metrics or 'has_direct_mse' in model_metrics:
                    patterns = [
                        rf'{model_name}.*?mse.*?([0-9.]+)',
                        rf'mse.*?{model_name}.*?([0-9.]+)',
                        rf'mean squared error.*?{model_name}.*?([0-9.]+)',
                        rf'{model_metrics.get("var_name", "")}.*?([0-9.]+)',
                        r'mse[:\s=]+([0-9.]+)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            try:
                                metrics_by_model[model_name]['mse'] = float(match.group(1))
                                break
                            except (ValueError, IndexError):
                                continue
                
                if 'r2_calc' in model_metrics or 'r2_in_print' in model_metrics:
                    patterns = [
                        rf'{model_name}.*?r2.*?([0-9.]+)',
                        rf'r2.*?{model_name}.*?([0-9.]+)',
                        rf'r squared.*?{model_name}.*?([0-9.]+)',
                        rf'{model_metrics.get("var_name", "")}.*?([0-9.]+)',
                        r'r2[:\s=]+([0-9.]+)',
                        r'r[^\w]?squared[:\s=]+([0-9.]+)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            try:
                                metrics_by_model[model_name]['r2'] = float(match.group(1))
                                break
                            except (ValueError, IndexError):
                                continue
    
    def _process_metrics_dictionary(self, metrics_by_model):
        temp_keys = ['rmse_calc', 'mse_calc', 'r2_calc', 'has_direct_mse', 
                    'uses_sqrt', 'uses_log', 'var_name', 'rmse_in_print', 
                    'mse_in_print', 'r2_in_print', 'has_rmse_calc', 
                    'has_mse_calc', 'has_r2_calc']
        
        for model_name in metrics_by_model:
            if 'mse' in metrics_by_model[model_name] and 'rmse' not in metrics_by_model[model_name]:
                mse = metrics_by_model[model_name]['mse']
                if mse is not None:
                    metrics_by_model[model_name]['rmse'] = np.sqrt(mse)
            
            for key in temp_keys:
                if key in metrics_by_model[model_name]:
                    metrics_by_model[model_name].pop(key)
                    
        for model_name in metrics_by_model:
            if 'r2_test' in metrics_by_model[model_name] and 'r2' not in metrics_by_model[model_name]:
                metrics_by_model[model_name]['r2'] = metrics_by_model[model_name]['r2_test']
            elif 'r2_train' in metrics_by_model[model_name] and 'r2' not in metrics_by_model[model_name]:
                metrics_by_model[model_name]['r2'] = metrics_by_model[model_name]['r2_train']

    def _execute_notebook_and_extract_metrics(self, notebook_path):
        try:
            execution_result = self._execute_notebook(notebook_path)
            
            if not isinstance(execution_result, dict) or 'error' in execution_result:
                print(f"Execution error or invalid result format: {execution_result}")
                return {'error': str(execution_result) if not isinstance(execution_result, dict) else execution_result.get('error', 'Unknown error')}
            
            if execution_result and any(metric in execution_result and execution_result[metric] is not None 
                                        for metric in ['mse', 'rmse', 'r2', 'accuracy']):
                print("Using metrics directly from notebook execution")
                return execution_result
                
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    executed_nb = nbformat.read(f, as_version=4)
                    
                metrics_results = []

                ast_metrics = self._extract_metrics_with_ast(executed_nb)
                if ast_metrics:
                    processed_ast_metrics = self._convert_model_metrics_to_flat(ast_metrics)
                    metrics_results.append(("AST extraction", processed_ast_metrics))
                    
                tf_metrics = self._extract_tensorflow_metrics(executed_nb)
                if tf_metrics and any(tf_metrics.get(m) is not None for m in ['accuracy']):
                    metrics_results.append(("TensorFlow extraction", tf_metrics))
                    
                log_metrics = self._extract_metrics_from_instrumented_logs(executed_nb)
                if log_metrics and any(log_metrics.get(m) is not None for m in ['mse', 'rmse', 'r2', 'accuracy']):
                    metrics_results.append(("Instrumented logs", log_metrics))
                    
                notebook_metrics = self._extract_metrics_from_notebook(executed_nb)
                if notebook_metrics and any(notebook_metrics.get(m) is not None for m in ['mse', 'rmse', 'r2', 'accuracy']):
                    metrics_results.append(("Comprehensive extraction", notebook_metrics))
                    
                if metrics_results:
                    best_result = None
                    max_non_null_metrics = -1
                    
                    for method_name, metrics in metrics_results:
                        non_null_count = sum(1 for m in ['mse', 'rmse', 'r2', 'mae', 'accuracy', 'f1'] 
                                            if m in metrics and metrics[m] is not None)
                        if non_null_count > max_non_null_metrics:
                            max_non_null_metrics = non_null_count
                            best_result = metrics
                            print(f"Selected metrics from {method_name} with {non_null_count} non-null metrics")
                    
                    return best_result
                
                return {
                    'mse': None,
                    'rmse': None,
                    'r2': None,
                    'mae': None,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'best_model': 'unknown'
                }
                    
            except Exception as e:
                print(f"Error extracting metrics from executed notebook: {e}")
                import traceback
                traceback.print_exc()
                return execution_result
                
        except Exception as e:
            print(f"Failed to execute notebook: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def _convert_model_metrics_to_flat(self, metrics_by_model):
        result = {
            'mse': None,
            'rmse': None,
            'r2': None,
            'mae': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'best_model': 'unknown'
        }
        
        best_model = None
        best_metric_value = None
        is_classification = False
        
        for model_name, metrics in metrics_by_model.items():
            if any(metric in metrics for metric in ['accuracy', 'f1']):
                is_classification = True
                break
        
        for model_name, metrics in metrics_by_model.items():
            if is_classification:
                if 'f1' in metrics and metrics['f1'] is not None:
                    metric_value = metrics['f1']
                    metric_type = 'f1'
                elif 'accuracy' in metrics and metrics['accuracy'] is not None:
                    metric_value = metrics['accuracy']
                    metric_type = 'accuracy'
                else:
                    continue
                    
                if best_metric_value is None or metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_model = model_name
                    
            else:
                if 'r2' in metrics and metrics['r2'] is not None:
                    metric_value = metrics['r2']
                    metric_type = 'r2'

                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_model = model_name
                elif 'rmse' in metrics and metrics['rmse'] is not None:
                    metric_value = -metrics['rmse']
                    metric_type = 'rmse'
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_model = model_name
                elif 'mse' in metrics and metrics['mse'] is not None:
                    metric_value = -metrics['mse']  
                    metric_type = 'mse'
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_model = model_name
        
        if best_model:
            result['best_model'] = best_model
            model_metrics = metrics_by_model[best_model]
            
            for metric in ['mse', 'rmse', 'r2', 'mae', 'accuracy', 'precision', 'recall', 'f1']:
                if metric in model_metrics and model_metrics[metric] is not None:
                    result[metric] = model_metrics[metric]
            
            if result['mse'] is not None and result['rmse'] is None:
                result['rmse'] = np.sqrt(result['mse'])
        
        return result

    def optimize_single_parameter(self, method_name=None, param_name=None):  
        modifiable_methods = self._get_modifiable_methods()
        
        if not modifiable_methods:
            print("No modifiable methods found in the notebook.")
            return None
        
        if method_name is None or param_name is None:
            method_name, param_name = modifiable_methods[0]
            print(f"Automatically selected method {method_name} and parameter {param_name} for optimization")

        if method_name in ["drop", "dropna"]:
            print(f"Skipping optimization of {method_name} method due to known issues")
            return None
        
        try:
            param_info = self.param_values[method_name][param_name]
        
            if 'allowed_values' in param_info and param_info['allowed_values'] == ["No parameters required for initialization"]:
                print(f"Skipping {method_name}.{param_name} as it has no parameters for initialization")
                return None
                
            param_values = param_info['allowed_values']
            
            if not param_values:
                print(f"Skipping {method_name}.{param_name} as it has no allowed values specified")
                return None
                
        except KeyError:
            print(f"Error: Parameter {param_name} not found for method {method_name}")
            return None
        
        print(f"Optimizing {method_name}.{param_name} with values: {param_values}")
        
        try:
            baseline_metrics = self._execute_notebook_and_extract_metrics(self.notebook_path)
        except Exception as e:
            print(f"Error executing baseline notebook: {e}")
            baseline_metrics = {'error': str(e)}
        
        results = {
            'method': method_name,
            'parameter': param_name,
            'baseline': baseline_metrics,
            'optimization_results': []
        }
        
        for value in param_values:
            try:
                print(f"Testing {method_name}.{param_name} = {value}")
                
                modified_notebook_path = self._create_modified_notebook(method_name, param_name, value)
                
                if modified_notebook_path is None:
                    print(f"Skipping {value} due to error in notebook creation")
                    continue
                
                metrics = self._execute_notebook_and_extract_metrics(modified_notebook_path)
                print(f"Extracted metrics for {method_name}.{param_name}={value}: {metrics}")
                
                results['optimization_results'].append({
                    'value': value,
                    'metrics': metrics
                })
                
                self.optimization_results.append({
                    'method': method_name,
                    'parameter': param_name,
                    'value': value,
                    'metrics': metrics
                })
                
                self._append_result_to_report({
                    'method': method_name,
                    'parameter': param_name,
                    'value': value,
                    'metrics': metrics
                })
                
            except Exception as e:
                print(f"Error testing {method_name}.{param_name} = {value}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        best_result = None
        best_metric_value = None
        
        is_classification = any(
            'metrics' in result and 
            ('accuracy' in result['metrics'] and result['metrics']['accuracy'] is not None or
            'f1' in result['metrics'] and result['metrics']['f1'] is not None)
            for result in results['optimization_results']
        )
        
        for result in results['optimization_results']:
            if 'metrics' not in result or isinstance(result['metrics'], str) or 'error' in result['metrics']:
                continue
                
            if is_classification:
                if 'f1' in result['metrics'] and result['metrics']['f1'] is not None:
                    metric_value = result['metrics']['f1']
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
                elif 'accuracy' in result['metrics'] and result['metrics']['accuracy'] is not None:
                    metric_value = result['metrics']['accuracy']
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
            else:
                if 'r2' in result['metrics'] and result['metrics']['r2'] is not None:
                    metric_value = result['metrics']['r2']
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
                elif 'rmse' in result['metrics'] and result['metrics']['rmse'] is not None:
                    metric_value = -result['metrics']['rmse']
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
                elif 'mse' in result['metrics'] and result['metrics']['mse'] is not None:
                    metric_value = -result['metrics']['mse']
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
        
        if best_result:
            results['best_value'] = best_result['value']
            results['best_metrics'] = best_result['metrics']
            print(f"Best value for {method_name}.{param_name}: {best_result['value']} with metrics: {best_result['metrics']}")
        else:
            results['best_value'] = None
            results['best_metrics'] = None
            print(f"No best value found for {method_name}.{param_name}")
        
        return results
    
    def optimize_multiple_parameters(self, num_methods=2):
        modifiable_methods = self._get_modifiable_methods()
        
        if not modifiable_methods:
            print("No modifiable methods found in the notebook.")
            return None
        
        methods_to_optimize = modifiable_methods[:min(num_methods, len(modifiable_methods))]
        
        results = []
        
        for method_name, param_name in methods_to_optimize:
            try:
                method_result = self.optimize_single_parameter(method_name, param_name)
                if method_result:
                    results.append(method_result)
            except Exception as e:
                print(f"Error optimizing {method_name}.{param_name}: {e}")
                continue
        
        return results

    def _append_result_to_report(self, result, output_path=None):
        if output_path is None:
            output_path = f"{os.path.splitext(self.notebook_path)[0]}_optimization_report.csv"

        metrics = result['metrics']
        method = result.get('method', '(baseline)')
        param = result.get('parameter', '')
        value = result.get('value', '')

        row = {
            'Notebook': self.notebook_name,
            'Method': method,
            'Parameter': param,
            'Value': value,
            'MSE': metrics.get('mse', 'N/A'),
            'RMSE': metrics.get('rmse', 'N/A'),
            'MAE': metrics.get('mae', 'N/A'),
            'R2': metrics.get('r2', 'N/A'),
            'Accuracy': metrics.get('accuracy', 'N/A'),
            'Precision': metrics.get('precision', 'N/A'),
            'Recall': metrics.get('recall', 'N/A'),
            'F1': metrics.get('f1', 'N/A'),
            'Best Model': metrics.get('best_model', 'N/A')
        }

        df = pd.DataFrame([row])

        if not os.path.exists(output_path):
            df.to_csv(output_path, index=False)
        else:
            df.to_csv(output_path, mode='a', header=False, index=False)

    def run_full_optimization(self, output_path=None):
        self.save_report(output_path)
        print("Running full grid search across all parameter combinations...")
        self.optimize_multiple_parameters()
        print("All parameters tested and results saved.")

    def save_report(self, output_path=None):
        print("Generating metrics for unmodified notebook...")
        baseline_metrics = self._execute_notebook(self.notebook_path)
        self._append_result_to_report({
            'metrics': baseline_metrics,
            'method': '(baseline)',
            'parameter': '',
            'value': ''
        }, output_path)
        print("Baseline metrics saved to report.")

def find_notebooks(directory):
    notebook_paths = []
    
    print(f"Searching for notebook files in {directory} and its subdirectories...")
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                notebook_paths.append(notebook_path)
                print(f"Found notebook: {notebook_path}")
    
    return notebook_paths


def main():
    parser = argparse.ArgumentParser(
        description="Optimize parameters in a Jupyter notebook's preprocessing pipeline."
    )
    parser.add_argument("--notebook", type=str, default=None, 
                       help="Path to the Jupyter notebook to optimize")
    parser.add_argument("--directory", type=str, default=None,
                       help="Directory to search for notebooks (will process all .ipynb files)")
    parser.add_argument("--params", type=str, default="Parameter_Values.json", 
                       help="Path to the JSON file with parameter values")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save the optimization report")

    args = parser.parse_args()

    if args.notebook is None and args.directory is None:
        print("Error: Either --notebook or --directory must be specified")
        parser.print_help()
        sys.exit(1)

    if args.notebook:
        try:
            print(f"Processing notebook: {args.notebook}")
            optimizer = ParameterOptimizer(args.notebook, args.params)
            optimizer.run_full_optimization(args.output)
        except Exception as e:
            print(f"Error processing notebook {args.notebook}: {e}")

    if args.directory:
        notebook_paths = []
        for root, _, files in os.walk(args.directory):
            for file in files:
                if file.endswith(".ipynb"):
                    notebook_paths.append(os.path.join(root, file))

        if not notebook_paths:
            print(f"No notebooks found in directory: {args.directory}")
            sys.exit(0)

        print(f"Found {len(notebook_paths)} notebooks in {args.directory}")

        for i, notebook_path in enumerate(notebook_paths):
            print(f"Processing notebook {i+1}/{len(notebook_paths)}: {notebook_path}")
            try:
                optimizer = ParameterOptimizer(notebook_path, args.params)
                optimizer.run_full_optimization(args.output)
            except Exception as e:
                print(f"Error processing notebook {notebook_path}: {e}")
                continue

if __name__ == "__main__":
    main()
    