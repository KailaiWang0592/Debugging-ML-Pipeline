import nbformat
import ast
import json
import re
import io
import contextlib
from typing import List, Dict, Any, Optional
import os

# === Configuration ===
# Only model types that constitute separate models.
MODEL_KEYWORDS = {
    "RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression",
    "SVC", "RandomForestRegressor", "GradientBoostingRegressor", "LinearRegression",
    "SVR", "Sequential", "XGBClassifier", "KNeighborsClassifier",
    "LGBMClassifier", "LazyClassifier"
}

LAYER_KEYWORDS = {"Dense"}

SCORING_KEYWORDS = {
    "score", "evaluate", "accuracy_score", "f1_score", "roc_auc_score",
    "log_loss", "mean_squared_error", "mean_absolute_error", "r2_score",
    "precision_score", "recall_score", "cross_val_score", "balanced_accuracy_score",    
    "classification_report"

}

# === Helper Functions ===
def extract_best_metric(lines: List[str], metric_method: str) -> Optional[str]:
    patterns = {
        "accuracy_score": r"(accuracy|acc)[\s:=]+[0-9]*\.?[0-9]+",
        "precision_score": r"(precision)[\s:=]+[0-9]*\.?[0-9]+",
        "recall_score": r"(recall)[\s:=]+[0-9]*\.?[0-9]+",
        "f1_score": r"(f1[\s_-]*score)[\s:=]+[0-9]*\.?[0-9]+",
        "score": r"(score)[\s:=]+[0-9]*\.?[0-9]+",
        "classification_report": r"classification\s*report",  
    }
    pattern = patterns.get(metric_method)
    if not pattern:
        return None
    regex = re.compile(pattern, re.IGNORECASE)
    last_val = None
    for line in lines:
        if regex.search(line):
            last_val = line.strip()
    return last_val

def find_last_float_line(lines: List[str]) -> Optional[str]:
    for line in reversed(lines):
        if re.search(r"\d\.\d{2,}", line):
            return line.strip()
    return None

# === Enhanced ModelTracker with Assignment Resolution ===
class ModelTracker(ast.NodeVisitor):
    """
    AST Node Visitor that detects model definitions and attempts to resolve
    literal parameter values.
    """
    def __init__(self) -> None:
        self.models: List[Dict[str, Any]] = []
        self.modelMapping: Dict[str, Dict[str, Any]] = {}  # variable name -> model record
        self.assignments: Dict[str, Any] = {}  # symbol table for literal assignments
        self._current_model: Optional[Dict[str, Any]] = None
        self._current_assignment = None

    def visit_Assign(self, node: ast.Assign) -> None:
        # Attempt to resolve literal assignments and store them.
        try:
            # Only handle single-target assignments where target is a Name.
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Try to evaluate node.value with literal_eval.
                    value = ast.literal_eval(node.value)
                    self.assignments[target.id] = value
        except Exception:
            pass
        self._current_assignment = node.targets
        self.generic_visit(node)
        self._current_assignment = None

    def visit_Call(self, node: ast.Call) -> None:
        try:
            # Determine function name.
            if isinstance(node.func, (ast.Name, ast.Attribute)):
                func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                # If it's a model constructor call.
                if func_name in MODEL_KEYWORDS:
                    self._capture_model(node, func_name)
                # For Sequential model, capture layers added via .add()
                if (self._current_model and self._current_model.get("type") == "Sequential" and
                        func_name == "add" and node.args):
                    layer_node = node.args[0]
                    if isinstance(layer_node, ast.Call):
                        self._capture_layer(layer_node)
            # Check for compile calls (for optimizer/metrics) to update the current model.
            if (isinstance(node.func, ast.Attribute) and
                node.func.attr == "compile" and self._current_model):
                self._capture_compile_params(node)
            # New part: if we see a call to .score() (e.g., model.score(X_train,y_train)),
            # try to resolve the model variable and update its score field.
            if (isinstance(node.func, ast.Attribute) and 
                node.func.attr == "score"):
                # Attempt to get the model variable name from the node (e.g., "model" in model.score(...))
                if isinstance(node.func.value, ast.Name):
                    var_name = node.func.value.id
                    if var_name in self.modelMapping:
                        # Record the scoring method.
                        self.modelMapping[var_name]["score"] = "score"
            # Also, if the call is directly a scoring function (e.g., score(...)), capture as before.
            if isinstance(node.func, ast.Name) and node.func.id in SCORING_KEYWORDS:
                self._capture_score(node)
        except Exception:
            pass
        self.generic_visit(node)

    def _capture_model(self, node: ast.Call, model_type: str) -> None:
        args = [self._evaluate_node(a) for a in node.args]
        kwargs = {kw.arg: self._evaluate_node(kw.value) for kw in node.keywords}
        if model_type == "Sequential":
            layers = []
            for a in node.args:
                if isinstance(a, ast.List):
                    layers.extend([self._safe_unparse(e) for e in a.elts])
            if layers:
                args = [f"[{', '.join(layers)}]"]
        self._current_model = {
            "type": model_type,
            "parameters": {"args": args, "kwargs": kwargs},
            "score": None,
        }
        self.models.append(self._current_model)
        # If the model was assigned to a variable, record it in our mapping.
        model_name = self._get_assignment_target()
        if model_name:
            self.modelMapping[model_name] = self._current_model

    def _capture_layer(self, layer_node: ast.Call) -> None:
        layer_type = layer_node.func.attr if isinstance(layer_node.func, ast.Attribute) else layer_node.func.id
        args = [self._evaluate_node(a) for a in layer_node.args]
        kwargs = {kw.arg: self._evaluate_node(kw.value) for kw in layer_node.keywords}
        layer_str = f"{layer_type}(" + ", ".join(
            [str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()]
        ) + ")"
        current_layers = self._current_model["parameters"]["args"]
        if not current_layers:
            self._current_model["parameters"]["args"] = [f"[{layer_str}]"]
        else:
            layer_list_str = current_layers[0]
            if layer_list_str.startswith("[") and layer_list_str.endswith("]"):
                content = layer_list_str[1:-1].strip()
                new_content = f"{content}, {layer_str}" if content else f"{layer_str}"
                self._current_model["parameters"]["args"][0] = f"[{new_content}]"
            else:
                self._current_model["parameters"]["args"][0] = f"[{layer_str}]"

    def _capture_compile_params(self, node: ast.Call) -> None:
        optimizer = None
        metrics = []
        for kw in node.keywords:
            if kw.arg == "optimizer":
                optimizer = self._evaluate_node(kw.value)
            elif kw.arg == "metrics":
                if isinstance(kw.value, ast.List):
                    metrics = [self._evaluate_node(e) for e in kw.value.elts]
                else:
                    metrics = [self._evaluate_node(kw.value)]
        if optimizer:
            self._current_model["parameters"].setdefault("kwargs", {})["optimizer"] = optimizer
        if metrics:
            self._current_model["parameters"].setdefault("kwargs", {})["metrics"] = metrics

    def _capture_score(self, node: ast.Call) -> None:
        # For calls like score(...)
        if self._current_model is not None and self._current_model["score"] is None:
            self._current_model["score"] = node.func.id

    def _evaluate_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            var_name = node.id
            if var_name in self.assignments:
                return self.assignments[var_name]
        try:
            return ast.literal_eval(node)
        except Exception:
            return self._safe_unparse(node)

    def _safe_unparse(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node).strip()
        except Exception:
            return "<expression>"

    def _get_assignment_target(self) -> Optional[str]:
        if self._current_assignment:
            for target in self._current_assignment:
                if isinstance(target, ast.Name):
                    return target.id
        return None

# === Notebook Execution Helper with Output Capture ===
def run_notebook(file_path: str) -> (Dict[str, Any], str):
    env: Dict[str, Any] = {}
    captured_output = io.StringIO()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cleaned_source = re.sub(r"^\s*[!%].*", "", cell.source, flags=re.MULTILINE)
                try:
                    with contextlib.redirect_stdout(captured_output):
                        exec(cleaned_source, env)
                except Exception as e:
                    error_log = os.path.join("logs", f"error_{os.path.basename(file_path)}.log")
                    with open(error_log, "a", encoding="utf-8") as errf:
                        errf.write(f"Error executing cell: {e}\n")
    except Exception as e:
        error_log = os.path.join("logs", f"error_{os.path.basename(file_path)}.log")
        with open(error_log, "a", encoding="utf-8") as errf:
            errf.write(f"Error reading notebook {file_path}: {e}\n")
    return env, captured_output.getvalue()

def extract_metrics_from_output(console_output: str) -> List[str]:
    metric_keywords = ["accuracy", "precision", "recall", "f1", "roc_auc", "loss", "score"]
    lines = console_output.splitlines()
    return [line for line in lines if any(kw in line.lower() for kw in metric_keywords)]

def update_notebook_data(notebook_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    simple_models = []
    for model in notebook_data.get("models", []):
        simple_models.append({
            "type": model.get("type"),
            "parameters": model.get("parameters", {}),
            "score": model.get("score")
        })
    return simple_models

def process_notebook(file_path: str) -> Dict[str, Any]:
    try:
        env, console_output = run_notebook(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        tracker = ModelTracker()
        for cell in nb.cells:
            if cell.cell_type == "code":
                cleaned_source = re.sub(r"^\s*[!%].*", "", cell.source, flags=re.MULTILINE)
                try:
                    tree = ast.parse(cleaned_source)
                    tracker.visit(tree)
                except Exception as e:
                    error_log = os.path.join("logs", f"error_{os.path.basename(file_path)}.log")
                    with open(error_log, "a", encoding="utf-8") as errf:
                        errf.write(f"Error parsing cell: {e}\n")
                    continue
        notebook_data = {
            "notebook": file_path,
            "models": tracker.models,
        }
        simple_models = update_notebook_data(notebook_data)
        return {"notebook": file_path, "models": simple_models}
    except Exception as e:
        return {"notebook": file_path, "error": str(e)}

def main():
    notebook_files = [
        "notebooks/spaceship-titanic.ipynb",
        "notebooks/spaceship-titanic-tensorflow-80.ipynb",
        "notebooks/spaceship-titanic-with-randomforestclassifier.ipynb",
        "notebooks/spaceship-titanic-competition-with-ensemble-models.ipynb",
        "notebooks/spaceship-titanic-classification.ipynb",
        "notebooks/spaceship-titanic-code.ipynb",
        "notebooks/spaceship-titanic-ml.ipynb",
        "notebooks/titanic-machine-learning-from-disaster-challenge.ipynb",
        "notebooks/titanic-passenger-survival-prediction.ipynb",
        "notebooks/titanic-prediction-using-logistic-regression.ipynb",
        "notebooks/titanic-spaceship-survival.ipynb",
        "notebooks/titanic.ipynb",
    ]
    results = []
    for nb_file in notebook_files:
        print(f"Processing notebook: {nb_file}")
        result = process_notebook(nb_file)
        results.append(result)
    with open("model_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("✅ Organized results saved to model_scores.json")


if __name__ == "__main__":
    main()
