import nbformat
import sys
import json
import argparse
import re
import numpy as np
import subprocess


def extract_metrics(nb):
    metrics = {
        'mse': None, 'rmse': None, 'r2': None, 'mae': None,
        'accuracy': None, 'precision': None, 'recall': None, 'f1': None,
        'best_model': 'unknown'
    }

    for cell in nb.cells:
        if cell.cell_type != 'code' or 'outputs' not in cell:
            continue
        for output in cell.outputs:
            if 'text' not in output:
                continue
            text = output['text']
            patterns = {
                'mse': r'MSE[:=\s]+([0-9.]+)',
                'rmse': r'RMSE[:=\s]+([0-9.]+)',
                'r2': r'R2[:=\s]+([0-9.]+)',
                'mae': r'MAE[:=\s]+([0-9.]+)',
                'accuracy': r'Accuracy[:=\s]+([0-9.]+)',
                'precision': r'Precision[:=\s]+([0-9.]+)',
                'recall': r'Recall[:=\s]+([0-9.]+)',
                'f1': r'F1[:=\s]+([0-9.]+)'
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        metrics[key] = float(match.group(1))
                    except ValueError:
                        continue

    if metrics['mse'] is not None and metrics['rmse'] is None:
        metrics['rmse'] = np.sqrt(metrics['mse'])

    if metrics['accuracy'] or metrics['f1']:
        metrics['best_model'] = 'classification_model'
    elif metrics['r2'] or metrics['mse']:
        metrics['best_model'] = 'regression_model'

    return metrics


def run_and_extract(notebook_path):
    executed_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    subprocess.run([
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--output', executed_path, notebook_path
    ], check=True)

    with open(executed_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    metrics = extract_metrics(nb)
    print(json.dumps(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=str, required=True)
    args = parser.parse_args()

    try:
        run_and_extract(args.notebook)
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)