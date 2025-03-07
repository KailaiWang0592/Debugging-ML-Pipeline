import json
import nbformat
import ast
import argparse
import csv
import os

# Track existing columns for each notebook (global state per notebook)
existing_columns = set()

def extract_functions_from_code(code, notebook_name, cell_index):
    global existing_columns

    try:
        tree = ast.parse(code)
        functions = []
        feature_additions = []

        code_lines = code.strip().split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if (isinstance(node.targets[0], ast.Subscript) and
                    isinstance(node.targets[0].value, ast.Name) and
                    isinstance(node.targets[0].slice, ast.Constant)):
                    col_name = node.targets[0].slice.value
                    line_number = node.lineno - 1
                    line_content = code_lines[line_number].strip()

                    if col_name not in existing_columns:
                        if isinstance(node.value, ast.BinOp):
                            op_type = type(node.value.op).__name__
                            op_mapping = {"Add": "Addition", "Sub": "Subtraction", "Mult": "Multiplication", "Div": "Division"}
                            feature_additions.append(("Feature Addition", f"Create column '{col_name}' by {op_mapping.get(op_type, op_type)}", line_content))
                        else:
                            feature_additions.append(("Feature Addition", f"Create column '{col_name}'", line_content))
                        existing_columns.add(col_name)
                    else:
                        feature_additions.append(("Feature Engineering", f"Modify column '{col_name}'", line_content))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        func_name = node.func.attr

                if func_name:
                    line_number = node.lineno - 1
                    line_content = code_lines[line_number].strip()

                    args = {kw.arg: ast.unparse(kw.value) if hasattr(ast, 'unparse') else str(kw.value)
                            for kw in node.keywords}
                    functions.append((func_name, args, line_content))

                    if func_name == 'drop':
                        if 'columns' in args:
                            columns = ast.literal_eval(args['columns'])
                        elif len(node.args) > 0 and isinstance(node.args[0], ast.Constant):
                            columns = [node.args[0].value]
                        else:
                            columns = []

                        axis = args.get('axis', '1')
                        if axis == '1':
                            for col in columns:
                                if col in existing_columns:
                                    functions.append(("Feature Selection", {"action": f"Drop column '{col}'"}, line_content))
                                    existing_columns.remove(col)
        
        return functions + feature_additions

    except Exception as e:
        print(f"\n\ud83d\udea8 Error parsing cell {cell_index + 1} in notebook '{notebook_name}': {e}")
        print("---- Cell content ----")
        print(code)
        print("----------------------\n")
        return []

def categorize_function(func_name):
    if func_name in ["SelectKBest", "VarianceThreshold", "SelectFromModel", "RFE", "drop", "dropna"]:
        return "Feature Selection"
    elif func_name in ["SimpleImputer", "PolynomialFeatures", "StandardScaler", "MinMaxScaler",
                       "RobustScaler", "Normalizer", "OneHotEncoder", "LabelEncoder", "OrdinalEncoder",
                       "ColumnTransformer", "fillna", "replace", "map", "get_dummies", "log", "exp", "sqrt"]:
        return "Feature Engineering"
    elif func_name in ["Feature Addition", "Feature Engineering", "Feature Selection"]:
        return func_name
    return "Other"

def analyze_notebook(notebook_path):
    global existing_columns
    existing_columns = set()

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading notebook {notebook_path}: {e}")
        return

    function_data = []
    notebook_name = os.path.basename(notebook_path)

    for cell_index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = cell["source"]
            functions = extract_functions_from_code(code, notebook_name, cell_index)
            for func_name, args, line_content in functions:
                category = categorize_function(func_name)
                if category not in ["Other"] and func_name != "print":
                    uses_default = "No" if args else "Yes"
                    default_params = ", ".join([f"{k}={v}" for k, v in (args.items() if isinstance(args, dict) else [])])
                    display_name = func_name if isinstance(func_name, str) else args.get("action", "")

                    function_data.append([
                        notebook_name,
                        category,
                        display_name,
                        uses_default,
                        default_params,
                        line_content
                    ])

    return function_data

def main():
    parser = argparse.ArgumentParser(description="Analyze Jupyter Notebooks for preprocessing function usage and output results to CSV.")
    parser.add_argument("notebook_paths", nargs='+', type=str, help="Paths to the Jupyter Notebook files")
    parser.add_argument("--output", type=str, default="notebook_analysis.csv", help="Output CSV filename")
    args = parser.parse_args()

    all_function_data = []
    for notebook_path in args.notebook_paths:
        result = analyze_notebook(notebook_path)
        if result:
            all_function_data.extend(result)

    output_path = os.path.join(os.getcwd(), args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Notebook Name", "Category", "Function", "Uses Default Parameters?", "Default Parameters", "Other Parameters"])
        writer.writerows(all_function_data)

    print(f"\u2705 Analysis saved to {output_path}")

if __name__ == "__main__":
    main()




# python analysis_pipeline.py \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__1_aadarshvelu__spaceship-titanic-tensorflow-80/spaceship-titanic-tensorflow-80.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__2_aaron95629__spaceship-titanic/spaceship-titanic.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__3_abdullahalbunni__spaceship-titanic-with-randomforestclassifier/spaceship-titanic-with-randomforestclassifier.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__4_abraamsaid__spaceship-titanic-eda-predictions/spaceship-titanic-eda-predictions.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__5_adends__spaceship-titanic-competition-with-ensemble-models/spaceship-titanic-competition-with-ensemble-models.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__6_adhamad0__spaceship-titanic/spaceship-titanic.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__7_aestroe__spaceship-titanic-classification/spaceship-titanic-classification.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__8_ahmedanwar89__spaceship-titanic-ml/spaceship-titanic-ml.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__9_ahmedgaitani__spaceship-titanic-code/spaceship-titanic-code.ipynb \
# /Users/kailaiwang/Documents/Debugg_MEng_Project/kaggle_notebooks/kaggle_notebooks/notebooks/kaggle__10_akankshasolanki123__spaceship-titanic/spaceship-titanic.ipynb \
# --output analysis.csv
