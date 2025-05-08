import nbformat
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Tuple, Optional
import logging
import json
import os
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotebookDataAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.train_var = None
        self.test_var = None
        self.target_col = None
        self.dataframes = {}
        
    def visit_Assign(self, node):
        # Look for dataframe assignments
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == 'read_csv':
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id.lower()
                            if 'train' in var_name:
                                self.train_var = target.id
                            elif 'test' in var_name:
                                self.test_var = target.id
            
        self.generic_visit(node)

class DataTransformer:
    def __init__(self):
        self.best_params = {}
        self.scalers = {}
        self.numeric_cols = None
        self.categorical_cols = None
        
    def analyze_notebook(self, notebook_path: str) -> Dict:
        """Analyze notebook to identify datasets and columns"""
        logger.info(f"Analyzing notebook: {notebook_path}")
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        analyzer = NotebookDataAnalyzer()
        notebook_env = {}
        
        # First pass: find DataFrame assignments
        for cell in nb.cells:
            if cell.cell_type == 'code':
                try:
                    # Look for data loading
                    if 'read_csv' in cell.source:
                        tree = ast.parse(cell.source)
                        analyzer.visit(tree)
                        exec(cell.source, globals(), notebook_env)
                except Exception as e:
                    logger.warning(f"Error in data loading cell: {e}")
                    continue
        
        # Get dataframes
        train_df = notebook_env.get(analyzer.train_var)
        test_df = notebook_env.get(analyzer.test_var)
        
        # If not found, try common names
        if train_df is None or test_df is None:
            common_train_names = ['train', 'df_train', 'train_df', 'X_train']
            common_test_names = ['test', 'df_test', 'test_df', 'X_test']
            
            for name in common_train_names:
                if name in notebook_env:
                    train_df = notebook_env[name]
                    break
            
            for name in common_test_names:
                if name in notebook_env:
                    test_df = notebook_env[name]
                    break
        
        if train_df is None or test_df is None:
            raise ValueError("Could not find train or test datasets in notebook")
        
        # Identify target column
        target_candidates = ['Transported', 'target', 'Target', 'label', 'Label', 'y', 'Y']
        self.target_col = next((col for col in target_candidates if col in train_df.columns), None)
        
        # Identify column types
        if self.target_col:
            train_features = train_df.drop(columns=[self.target_col])
        else:
            train_features = train_df
            
        self.numeric_cols = train_features.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = train_features.select_dtypes(include=['object', 'category', 'bool']).columns
        
        return {
            'train_df': train_df,
            'test_df': test_df,
            'target_col': self.target_col
        }
    
    def optimize_and_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize transformations and transform data"""
        logger.info("Optimizing transformations")
        
        transformed_train = train_df.copy()
        transformed_test = test_df.copy()
        
        # Handle numeric features
        if len(self.numeric_cols) > 0:
            scaler = self._find_best_scaler(train_df[self.numeric_cols])
            if scaler is not None:
                transformed_train[self.numeric_cols] = scaler.fit_transform(train_df[self.numeric_cols])
                transformed_test[self.numeric_cols] = scaler.transform(test_df[self.numeric_cols])
        
        # Handle categorical features
        if len(self.categorical_cols) > 0:
            for col in self.categorical_cols:
                if col in train_df.columns and col in test_df.columns:
                    transformed_train[col], transformed_test[col] = self._encode_categorical(
                        train_df[col], test_df[col]
                    )
        
        return transformed_train, transformed_test
    
    def _find_best_scaler(self, data: pd.DataFrame):
        """Find the best scaler for numeric data"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        best_scaler = None
        best_variance = float('-inf')
        
        for name, scaler in scalers.items():
            try:
                scaled_data = scaler.fit_transform(data.fillna(data.mean()))
                variance = np.var(scaled_data)
                
                if variance > best_variance:
                    best_variance = variance
                    best_scaler = scaler
                    self.best_params['numeric_scaler'] = name
            except Exception as e:
                logger.warning(f"Error with {name} scaler: {e}")
                continue
        
        return best_scaler
    
    def _encode_categorical(self, train_series: pd.Series, test_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Encode categorical data"""
        try:
            # Fill missing values
            train_series = train_series.fillna('missing')
            test_series = test_series.fillna('missing')
            
            if train_series.nunique() / len(train_series) < 0.05:  # Low cardinality
                # One-hot encoding
                combined = pd.get_dummies(pd.concat([train_series, test_series]))
                train_encoded = combined[:len(train_series)]
                test_encoded = combined[len(train_series):]
                self.best_params[f'encoding_{train_series.name}'] = 'onehot'
            else:
                # Label encoding
                unique_values = pd.concat([train_series, test_series]).unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                train_encoded = train_series.map(encoding_map)
                test_encoded = test_series.map(encoding_map)
                self.best_params[f'encoding_{train_series.name}'] = 'label'
            
            return train_encoded, test_encoded
        except Exception as e:
            logger.warning(f"Error encoding {train_series.name}: {e}")
            return train_series, test_series

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform datasets from Jupyter notebook')
    parser.add_argument('notebook_path', type=str, help='Path to input notebook')
    parser.add_argument('output_dir', type=str, help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize transformer
    transformer = DataTransformer()
    
    try:
        # Analyze notebook and get datasets
        data = transformer.analyze_notebook(args.notebook_path)
        
        if data['train_df'] is None or data['test_df'] is None:
            raise ValueError("Could not extract datasets from notebook")
        
        # Optimize and transform data
        transformed_train, transformed_test = transformer.optimize_and_transform(
            data['train_df'], 
            data['test_df']
        )
        
        # Save transformed datasets
        train_path = os.path.join(args.output_dir, 'train_transformed.csv')
        test_path = os.path.join(args.output_dir, 'test_transformed.csv')
        
        transformed_train.to_csv(train_path, index=False)
        transformed_test.to_csv(test_path, index=False)
        
        # Save transformation parameters
        params_path = os.path.join(args.output_dir, 'transformation_params.json')
        with open(params_path, 'w') as f:
            json.dump(transformer.best_params, f, indent=2)
        
        logger.info(f"Transformed datasets saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing notebook: {e}")
        raise

if __name__ == "__main__":
    main()