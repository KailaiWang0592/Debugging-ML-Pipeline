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
    
    def _execute_notebook(self, notebook_path):
        try:
            result = subprocess.run(
                ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
                 '--output', f"{os.path.basename(notebook_path)}", 
                 notebook_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Warning: Error executing notebook: {result.stderr}")
                return {'error': result.stderr}
            
            with open(notebook_path, 'r', encoding='utf-8') as f:
                executed_nb = nbformat.read(f, as_version=4)
            
            metrics = self._extract_metrics_from_notebook(executed_nb)
            print("here")
            print(metrics)
            
            return metrics
        except Exception as e:
            print(f"Error executing notebook {notebook_path}: {e}")
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
            baseline_metrics = self._execute_notebook(self.notebook_path)
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
                
                metrics = self._execute_notebook(modified_notebook_path)
                print(metrics)
                
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
                print("here2")
                print(self.optimization_results)
                
                self._append_result_to_report({
                    'method': method_name,
                    'parameter': param_name,
                    'value': value,
                    'metrics': metrics
                })
                print("here3")
                
            except Exception as e:
                print(f"Error testing {method_name}.{param_name} = {value}: {e}")
                continue
        
        best_result = None
        best_metric_value = float('inf')
        
        for result in results['optimization_results']:
            if 'mse' in result['metrics'] and result['metrics']['mse'] is not None:
                if result['metrics']['mse'] < best_metric_value:
                    best_metric_value = result['metrics']['mse']
                    best_result = result
        
        results['best_value'] = best_result['value'] if best_result else None
        results['best_metrics'] = best_result['metrics'] if best_result else None
        
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
            output_path = f"{os.path.splitext(self.notebook_path)[0]}_optimization_report.md"
        
        notebook_location = os.path.dirname(self.notebook_path)
        notebook_info = f"{notebook_location}/{self.notebook_name}"
        

        metrics = result['metrics']
        method = result['method']
        param = result['parameter']
        value = result['value']

        is_classification = metrics.get('accuracy') is not None or metrics.get('f1') is not None
        
        if not os.path.exists(output_path):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Parameter Optimization Report\n")
                f.write(f"Notebook: {self.notebook_name}\n\n")
                
                if is_classification:
                    f.write("| Notebook | Method | Parameter | Value | Accuracy | Precision | Recall | F1 | Best Model |\n")
                    f.write("|----------|--------|-----------|-------|----------|-----------|--------|----|-----------|\n")
                else:
                    f.write("| Notebook | Method | Parameter | Value | MSE | RMSE | MAE | R² | Best Model |\n")
                    f.write("|----------|--------|-----------|-------|-----|------|-----|----|-----------|\n")

        with open(output_path, 'a', encoding='utf-8') as f:
            if is_classification:
                accuracy = metrics.get('accuracy', 'N/A')
                precision = metrics.get('precision', 'N/A')
                recall = metrics.get('recall', 'N/A')
                f1 = metrics.get('f1', 'N/A')
                best_model = metrics.get('best_model', 'N/A')
                
                f.write(f"| {notebook_info} | {method} | {param} | {value} | {accuracy} | {precision} | {recall} | {f1} | {best_model} |\n")
            else:
                mse = metrics.get('mse', 'N/A')
                rmse = metrics.get('rmse', 'N/A')
                mae = metrics.get('mae', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                best_model = metrics.get('best_model', 'N/A')
                
                f.write(f"| {notebook_info} | {method} | {param} | {value} | {mse} | {rmse} | {mae} | {r2} | {best_model} |\n")

    def generate_report(self):
        if not self.optimization_results:
            return "No optimization results available."
        
        notebook_location = os.path.dirname(self.notebook_path)
        notebook_info = f"{notebook_location}/{self.notebook_name}"
        
        report = []
        report.append("# Parameter Optimization Report")
        report.append(f"Notebook: {notebook_info}\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        grouped_results = {}
        for result in self.optimization_results:
            key = f"{result['method']}.{result['parameter']}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        for key, results in grouped_results.items():
            report.append(f"## {key}")
            
            if all('metrics' in r and 'accuracy' in r['metrics'] and r['metrics']['accuracy'] is not None for r in results):
                sorted_results = sorted(results, key=lambda x: x['metrics']['accuracy'], reverse=True)
            elif all('metrics' in r and 'f1' in r['metrics'] and r['metrics']['f1'] is not None for r in results):
                sorted_results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)
            elif all('metrics' in r and 'mse' in r['metrics'] and r['metrics']['mse'] is not None for r in results):
                sorted_results = sorted(results, key=lambda x: x['metrics']['mse'])
            elif all('metrics' in r and 'rmse' in r['metrics'] and r['metrics']['rmse'] is not None for r in results):
                sorted_results = sorted(results, key=lambda x: x['metrics']['rmse'])
            elif all('metrics' in r and 'r2' in r['metrics'] and r['metrics']['r2'] is not None for r in results):
                sorted_results = sorted(results, key=lambda x: x['metrics']['r2'], reverse=True)
            
            else:
                sorted_results = results
            
            is_classification = any('metrics' in r and 'accuracy' in r['metrics'] and r['metrics']['accuracy'] is not None for r in results)

            if is_classification:
                report.append("| Notebook | Value | Accuracy | Precision | Recall | F1 | Best Model |")
                report.append("|----------|-------|----------|-----------|--------|----|-----------| ")
            else:
                report.append("| Notebook | Value | MSE | RMSE | MAE | R² | Best Model |")
                report.append("|----------|-------|-----|------|-----|----|-----------|")
            
        
            for result in sorted_results:
                value = result['value']
                metrics = result['metrics']
                
                if is_classification:
                    accuracy = metrics.get('accuracy', 'N/A')
                    precision = metrics.get('precision', 'N/A')
                    recall = metrics.get('recall', 'N/A')
                    f1 = metrics.get('f1', 'N/A')
                    best_model = metrics.get('best_model', 'N/A')
                    
                    report.append(f"| {notebook_info} | {value} | {accuracy} | {precision} | {recall} | {f1} | {best_model} |")
                else:
                    mse = metrics.get('mse', 'N/A')
                    rmse = metrics.get('rmse', 'N/A')  
                    mae = metrics.get('mae', 'N/A')
                    r2 = metrics.get('r2', 'N/A')
                    best_model = metrics.get('best_model', 'N/A')
                    
                    report.append(f"| {notebook_info} | {value} | {mse} | {rmse} | {mae} | {r2} | {best_model} |")
            
            report.append("")
        
        return "\n".join(report)
    

    def save_report(self, output_path=None):
        if output_path is None:
            output_path = f"{os.path.splitext(self.notebook_path)[0]}_optimization_report.md"
        
        report = self.generate_report()
        
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write("\n\n")  
                f.write(report)
            
            print(f"Report appended to {output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")


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
    parser.add_argument("--method", type=str, default=None,
                       help="The method to optimize (if None, a suitable method will be chosen)")
    parser.add_argument("--param", type=str, default=None,
                       help="The parameter to optimize (if None, a suitable parameter will be chosen)")
    parser.add_argument("--multiple", type=int, default=0,
                       help="Optimize multiple parameters (specify the number of methods)")
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
            process_notebook(args.notebook, args)
        except Exception as e:
            print(f"Error processing notebook {args.notebook}: {e}")
    
    if args.directory:
        notebook_paths = find_notebooks(args.directory)
        
        if not notebook_paths:
            print(f"No notebooks found in directory: {args.directory}")
            sys.exit(0)
        
        print(f"Found {len(notebook_paths)} notebooks in {args.directory}")
        
        for i, notebook_path in enumerate(notebook_paths):
            print(f"\nProcessing notebook {i+1}/{len(notebook_paths)}: {notebook_path}")
            try:
                process_notebook(notebook_path, args)
            except Exception as e:
                print(f"Error processing notebook {notebook_path}: {e}")
                # Continue with next notebook
                continue

def process_notebook(notebook_path, args):
    try:
        optimizer = ParameterOptimizer(notebook_path, args.params)
        
        if args.multiple > 0:
            results = optimizer.optimize_multiple_parameters(args.multiple)
        else:
            results = optimizer.optimize_single_parameter(args.method, args.param)
        
        optimizer.save_report(args.output)
        
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
    
    except Exception as e:
        print(f"Error processing notebook {notebook_path}: {e}")
        raise

if __name__ == "__main__":
    main()