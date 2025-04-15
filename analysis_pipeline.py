import json
import nbformat
import ast
import argparse
import csv
import os
import inspect

try:
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        Normalizer,
        OneHotEncoder,
        LabelEncoder,
        OrdinalEncoder,
        PolynomialFeatures
    )
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import (
        SelectKBest,
        VarianceThreshold,
        SelectFromModel,
        RFE
    )

except ImportError:
    pass


KNOWN_CLASSES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "Normalizer": Normalizer,
    "OneHotEncoder": OneHotEncoder,
    "LabelEncoder": LabelEncoder,
    "OrdinalEncoder": OrdinalEncoder,
    "PolynomialFeatures": PolynomialFeatures,
    "SimpleImputer": SimpleImputer,
    "SelectKBest": SelectKBest,
    "VarianceThreshold": VarianceThreshold,
    "SelectFromModel": SelectFromModel,
    "RFE": RFE
}


MANUAL_PARAMETERS = {
    "fillna": "value=None, method=None, axis=None, inplace=False, limit=None, downcast=None",
    "drop": "labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'",
    "dropna": "axis=0, how='any', thresh=None, subset=None, inplace=False",
    "replace": "to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'",
    "map": "arg, na_action=None",
    "get_dummies": "data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None",
    "LabelEncoder": "No parameters required for initialization"
}

existing_columns = set()
df_variable = None

magic_commands = [
    "%matplotlib", "%timeit", "%load_ext", "%run", "%debug", "%store", "%reset",
    "%who", "%who_ls", "%reset_selective", "%prun", "%time", "%pdb", "%paste", "%history"
]

def clean_code(code):
    cleaned_lines = []
    for line in code.strip().split("\n"):
        if not any(line.strip().startswith(magic) for magic in magic_commands):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def get_all_parameters(func_name):
    if func_name in ["Feature Addition", "Feature Engineering", "Feature Selection"]:
        return "N/A"
    
    if func_name in MANUAL_PARAMETERS:
        return MANUAL_PARAMETERS[func_name]
    
    if func_name in KNOWN_CLASSES:
        cls = KNOWN_CLASSES[func_name]
        try:
            sig = inspect.signature(cls)
            parts = []
            for param_name, param in sig.parameters.items():
                default_val = param.default
                if default_val is inspect.Parameter.empty:
                    parts.append(f"{param_name}=No Default")
                else:
                    parts.append(f"{param_name}={default_val}")
            return ", ".join(parts)
        except Exception as e:
            print(f"Error getting parameters for {func_name}: {e}")
            if func_name == "LabelEncoder":
                return MANUAL_PARAMETERS["LabelEncoder"]
            return "Unknown"
    
    return ""

def extract_functions_from_code(code, notebook_name, cell_index):
    global existing_columns, df_variable

    try:
        code = clean_code(code)
        tree = ast.parse(code)
        functions = []
        feature_additions = []

        code_lines = code.strip().split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if (
                    isinstance(node.targets[0], ast.Subscript) and 
                    isinstance(node.targets[0].value, ast.Name) and 
                    isinstance(node.targets[0].slice, ast.Constant)
                ):
                    col_name = node.targets[0].slice.value
                    line_number = node.lineno - 1

                    if 0 <= line_number < len(code_lines):
                        line_content = code_lines[line_number].strip()
                        if any(test_var in line_content for test_var in ["test_data", "X_test"]):
                            continue

                        if col_name not in existing_columns:
                            if isinstance(node.value, ast.BinOp):
                                op_type = type(node.value.op).__name__
                                op_mapping = {"Add": "Addition", "Sub": "Subtraction", 
                                              "Mult": "Multiplication", "Div": "Division"}
                                feature_additions.append(
                                    ("Feature Addition",
                                     f"Create column '{col_name}' by {op_mapping.get(op_type, op_type)}",
                                     line_content)
                                )
                            else:
                                feature_additions.append(
                                    ("Feature Addition",
                                     f"Create column '{col_name}'",
                                     line_content)
                                )
                            existing_columns.add(col_name)
                        else:
                            feature_additions.append(
                                ("Feature Engineering",
                                 f"Modify column '{col_name}'",
                                 line_content)
                            )

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
                    if 0 <= line_number < len(code_lines):
                        line_content = code_lines[line_number].strip()

                    args = {
                        kw.arg: (ast.unparse(kw.value) if hasattr(ast, 'unparse') 
                                 else str(kw.value))
                        for kw in node.keywords
                    }
                    if any(test_var in line_content for test_var in ["test_data", "X_test"]):
                        continue

                    functions.append((func_name, args, line_content))

                    if func_name == 'drop':
                        if 'columns' in args:
                            columns = args['columns']
                        elif len(node.args) > 0 and isinstance(node.args[0], ast.Constant):
                            columns = [node.args[0].value]
                        else:
                            columns = []

                        if isinstance(columns, str):
                            columns = columns.replace("[", "").replace("]", "").replace("'", "").strip()
                            columns = [c.strip() for c in columns.split(",") if c.strip()]

                        axis = args.get('axis', '1')
                        if axis == '1':
                            for col in columns:
                                if col in existing_columns:
                                    functions.append(
                                        ("Feature Selection", 
                                         {"action": f"Drop column '{col}'"}, 
                                         line_content)
                                    )
                                    existing_columns.remove(col)

                if func_name in ["fit", "transform", "fit_transform"]:
                    if df_variable is None:
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Name):
                            df_variable = node.args[0].id
        
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
    elif func_name in [
        "SimpleImputer", "PolynomialFeatures", "StandardScaler", "MinMaxScaler",
        "RobustScaler", "Normalizer", "OneHotEncoder", "LabelEncoder",
        "OrdinalEncoder", "ColumnTransformer", "fillna", "replace", "map",
        "get_dummies"
    ]:
        return "Feature Engineering"
    elif func_name in ["Feature Addition", "Feature Engineering", "Feature Selection"]:
        return func_name
    return "Other"

def analyze_notebook(notebook_path):
    global existing_columns, df_variable
    existing_columns = set()
    df_variable = None

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
            entries = extract_functions_from_code(code, notebook_name, cell_index)
            for func_name, args, line_content in entries:
                category = categorize_function(func_name)
                if category not in ["Other"] and func_name != "print":
                    uses_default = "No" if isinstance(args, dict) and len(args) > 0 else "Yes"

                    custom_params = ""
                    if isinstance(args, dict) and args:
                        custom_params = ", ".join([f"{k}={v}" for k, v in args.items()])

                    all_params = "N/A"
                    if not (category in ["Feature Addition", "Feature Engineering", "Feature Selection"] and 
                            isinstance(args, dict) and "action" in args):
                        all_params = get_all_parameters(func_name)

                    display_name = func_namey
                    if isinstance(args, dict) and "action" in args:
                        display_name = args["action"]

                    function_data.append([
                        notebook_name,
                        category,
                        display_name,
                        uses_default,
                        custom_params,
                        all_params,
                        line_content
                    ])

    return function_data

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Jupyter Notebooks for preprocessing function usage and output results to CSV."
    )
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
        writer.writerow([
            "Notebook Name",
            "Category",
            "Function",
            "Uses Default Parameters?",
            "Custom Parameters",  
            "All Parameters",     
            "Other Parameters"
        ])
        writer.writerows(all_function_data)

if __name__ == "__main__":
    main()