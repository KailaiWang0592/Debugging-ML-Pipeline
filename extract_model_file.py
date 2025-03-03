import nbformat
import re

file_paths = [
    "spaceship-titanic.ipynb",
    "spaceship-titanic-tensorflow-80.ipynb"
]

model_keywords = [
    "RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", "SVC",
    "RandomForestRegressor", "GradientBoostingRegressor", "LinearRegression", "SVR",
    "Sequential", "Dense", "Adam" 
]

notebook_model_types = {}

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    
    model_types = set() 
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for keyword in model_keywords:
                if re.search(rf"\b{keyword}\b", cell.source):
                    model_types.add(keyword)
    
    notebook_model_types[file_path] = list(model_types) if model_types else ["No model detected"]

print(notebook_model_types)
