import json
import nbformat
import ast
import argparse
import csv
import os
import inspect

# For demonstration, we import some common scikit-learn classes:
# (You can add or remove imports based on your needs)
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
    # Add other imports as needed...
except ImportError:
    # If you don't have scikit-learn installed or want a fallback,
    # you can simply pass in a blank dictionary for KNOWN_CLASSES.
    pass

# ------------------------------------------------------------------------------
# 1) Dictionary mapping function/class names to actual objects for introspection
# ------------------------------------------------------------------------------
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

# Manually define parameters for common methods that might not be accessible through introspection
MANUAL_PARAMETERS = {
    # Pandas methods
    "fillna": "value=None, method=None, axis=None, inplace=False, limit=None, downcast=None",
    "drop": "labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'",
    "dropna": "axis=0, how='any', thresh=None, subset=None, inplace=False",
    "replace": "to_replace=None, value=<no_default>, *, inplace=False, limit=None, regex=False, method=<no_default>",
    "map": "arg, na_action=None",
    "get_dummies": "data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None",
    
    # Classes that might have issues with introspection
    "LabelEncoder": "No parameters required for initialization"
}

# Track existing columns for each notebook (global state per notebook)
existing_columns = set()
df_variable = None

magic_commands = [
    "%matplotlib", "%timeit", "%load_ext", "%run", "%debug", "%store", "%reset",
    "%who", "%who_ls", "%reset_selective", "%prun", "%time", "%pdb", "%paste", "%history"
]

def clean_code(code):
    """
    Remove any Jupyter magic commands from the code before AST parsing.
    """
    cleaned_lines = []
    for line in code.strip().split("\n"):
        if not any(line.strip().startswith(magic) for magic in magic_commands):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def get_all_parameters(func_name):
    """
    Return a string listing all parameters (with default values) for known functions/classes.
    For non-function operations (like direct column assignments), returns an empty string.
    """
    # For feature additions/modifications (not function calls)
    if func_name in ["Feature Addition", "Feature Engineering", "Feature Selection"]:
        return "N/A"
    
    # Check if we have manually defined parameters for this function
    if func_name in MANUAL_PARAMETERS:
        return MANUAL_PARAMETERS[func_name]
    
    # For functions in KNOWN_CLASSES, use introspection
    if func_name in KNOWN_CLASSES:
        cls = KNOWN_CLASSES[func_name]
        try:
            sig = inspect.signature(cls)
            parts = []
            for param_name, param in sig.parameters.items():
                default_val = param.default
                if default_val is inspect.Parameter.empty:
                    # If no default is provided, display as e.g., "param_name=No Default"
                    parts.append(f"{param_name}=No Default")
                else:
                    parts.append(f"{param_name}={default_val}")
            return ", ".join(parts)
        except Exception as e:
            print(f"Error getting parameters for {func_name}: {e}")
            # Try to handle special cases
            if func_name == "LabelEncoder":
                return MANUAL_PARAMETERS["LabelEncoder"]
            return "Unknown"
    
    # For functions not in our known lists
    return ""

def extract_functions_from_code(code, notebook_name, cell_index):
    """
    Parse the code in a notebook cell and return a list of:
    - (func_name, args_dict, original_line_of_code)
    - For columns, returns (type_of_change, description, original_line_of_code)
    """
    global existing_columns, df_variable

    try:
        code = clean_code(code)
        tree = ast.parse(code)
        functions = []
        feature_additions = []

        code_lines = code.strip().split("\n")

        for node in ast.walk(tree):
            # 1. Detect column assignment
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
                        # Skip test data references
                        if any(test_var in line_content for test_var in ["test_data", "X_test"]):
                            continue

                        # Distinguish new vs existing columns
                        if col_name not in existing_columns:
                            # Is it some arithmetic operation?
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

        # 2. Detect function calls (ast.Call)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None

                # function() or obj.function()
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        func_name = node.func.attr

                if func_name:
                    line_number = node.lineno - 1
                    if 0 <= line_number < len(code_lines):
                        line_content = code_lines[line_number].strip()

                    # Build args dictionary
                    args = {
                        kw.arg: (ast.unparse(kw.value) if hasattr(ast, 'unparse') 
                                 else str(kw.value))
                        for kw in node.keywords
                    }
                    # Skip references to test data
                    if any(test_var in line_content for test_var in ["test_data", "X_test"]):
                        continue

                    # Store the function call
                    functions.append((func_name, args, line_content))

                    # Detect dropping columns
                    if func_name == 'drop':
                        if 'columns' in args:
                            columns = args['columns']
                        elif len(node.args) > 0 and isinstance(node.args[0], ast.Constant):
                            columns = [node.args[0].value]
                        else:
                            columns = []

                        if isinstance(columns, str):
                            # e.g. "['col1', 'col2']" or "col1, col2"
                            # A naive approach: just remove brackets/quotes if present
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

                # Track if user calls fit, transform, etc.
                if func_name in ["fit", "transform", "fit_transform", "train_test_split"]:
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
    """
    Map common function names to categories.
    """
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
    """
    Process each cell of the notebook to extract function/column operations.
    """
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
                    # "Uses Default Parameters?" => if no args, "Yes", otherwise "No"
                    uses_default = "No" if isinstance(args, dict) and len(args) > 0 else "Yes"

                    # Build a "Custom Parameters" string from the AST
                    custom_params = ""
                    if isinstance(args, dict) and args:
                        custom_params = ", ".join([f"{k}={v}" for k, v in args.items()])

                    # For direct column operations, set "All Parameters" to "N/A"
                    all_params = "N/A"
                    if not (category in ["Feature Addition", "Feature Engineering", "Feature Selection"] and 
                            isinstance(args, dict) and "action" in args):
                        all_params = get_all_parameters(func_name)

                    display_name = func_name
                    # For drop column references stored as dict, handle slightly differently
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

    # Write the CSV with the updated column names
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Notebook Name",
            "Category",
            "Function",
            "Uses Default Parameters?",
            "Custom Parameters",  # Renamed from "Default Parameters"
            "All Parameters",     # New column
            "Other Parameters"
        ])
        writer.writerows(all_function_data)

if __name__ == "__main__":
    main()