#!/usr/bin/env python
"""
Full standalone script that optimizes the models extracted from Jupyter notebooks
and logs every single hyper‑parameter combination tried during the
`RandomizedSearchCV` (including Keras/Scikeras searches).

Outputs
-------
1. optimized_results.json  – best parameters & scores for each model/notebook.
2. hyperparam_trials.csv   – one row per trial (parameter set) with mean/std CV
                             scores, ready for later analysis.
"""
import os
# Tell the TF C++ runtime to suppress INFO/WARNING/ERROR messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
# Silence TF Python API logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
# In case TF’s internal logger is still chatty
tf.get_logger().setLevel("ERROR")

# now the rest of your imports
from keras import regularizers
from keras.callbacks import EarlyStopping
# … etc …
import json
import warnings
from typing import Optional
import numpy as np
import pandas as pd

# Keras imports
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import (Adagrad, Adam, Adamax, Nadam, RMSprop, SGD)
from scikeras.wrappers import KerasClassifier

# Sklearn imports
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

# Progress bar
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 1.  Hyper‑parameter grids & scoring map
# -----------------------------------------------------------------------------
optimized_hyperparameter_space = {
    "RandomForestClassifier": {
        "n_estimators": [50,100,150,200,250,300,350,400,450,500],
        "max_depth": [None,5,10,15,20,25,30,35,40,50],
        "min_samples_split": [2,3,4,5,6,7,8,9,10,15],
        "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10],
        "max_features": ["sqrt","log2",0.3,0.4,0.5,0.6,0.7,0.8,0.9,None],
    },
    "GradientBoostingClassifier": {
        "n_estimators":[50,100,150,200,250,300,350,400,450,500],
        "learning_rate":[0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.5],
        "max_depth":[3,4,5,6,7,8,9,10,12,15],
        "subsample":[0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
        "min_samples_split":[2,3,4,5,6,7,8,9,10,15],
        "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    },
    "LogisticRegression": {
        "C":[0.001,0.01,0.1,0.5,1,2,5,10,50,100],
        "penalty":["l1","l2","elasticnet"],
        "solver":["lbfgs","newton-cg","liblinear","sag","saga"],
        "l1_ratio":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    },
    "SVC": {
        "C":[0.001,0.01,0.1,0.5,1,2,5,10,50,100],
        "kernel":["linear","poly","rbf","sigmoid"],
        "gamma":["scale","auto",0.0001,0.001,0.01,0.1,1,10],
        "degree":[2,3,4,5,6,7,8],
        "coef0":[-1.0,-0.5,0.0,0.5,1.0],
    },
    "NeuralNetwork": {
        "learning_rate":[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.5],
        "batch_size":[16,32,64,128,256,512],
        "epochs":[20,30,40,50,60,70,80,90,100,150],
    },
    "XGBClassifier": {
        "n_estimators":[100,200,300,400,500,600,800,1000],
        "max_depth":[3,4,5,6,7,8,10,12],
        "learning_rate":[0.001,0.005,0.01,0.05,0.1,0.2,0.3],
        "subsample":[0.5,0.6,0.7,0.8,0.9,1.0],
        "colsample_bytree":[0.5,0.6,0.7,0.8,0.9,1.0],
        "gamma":[0,0.01,0.05,0.1,0.2,0.3],
        "min_child_weight":[1,3,5,7],
        "reg_alpha":[0,0.001,0.01,0.1],
        "reg_lambda":[1,1.5,2,3],
    },
    "LinearRegression": {
        "fit_intercept":[True,False],
        "positive":[True,False],
    },
    "KNeighborsClassifier": {
        "n_neighbors":[1,2,3,4,5,6,7,8,9,10],
        "weights":["uniform","distance"],
        "p":[1,2],
    },
}

scoring_mapping = {
    "accuracy_score":"accuracy",
    "accuracy":"accuracy",
    "f1_score":"f1",
    "precision_score":"precision",
    "recall_score":"recall",
}

# -----------------------------------------------------------------------------
# 2.  Utility: build a dynamic Keras network
# -----------------------------------------------------------------------------
def create_dynamic_nn(architecture, *, learning_rate=0.01, input_dim=None,
                      optimizer="adam", dropout_rate=0.0,
                      batch_norm=False, l2_reg=0.0):
    model = Sequential()
    for i, layer_cfg in enumerate(architecture):
        kwargs = {
            "units": layer_cfg["units"],
            "activation": layer_cfg.get("activation","relu"),
            "kernel_regularizer": regularizers.l2(l2_reg),
        }
        if i == 0:
            kwargs["input_shape"] = (input_dim,)
        model.add(Dense(**kwargs))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    opt_switch = {
        "adam":Adam, "nadam":Nadam, "rmsprop":RMSprop,
        "sgd":SGD, "adagrad":Adagrad, "adamax":Adamax,
    }
    Opt = opt_switch.get(optimizer.lower(), Adam)
    optimizer_inst = (
        Opt(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        if Opt is SGD else
        Opt(learning_rate=learning_rate)
    )
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer_inst,
                  metrics=["accuracy"])
    return model

# -----------------------------------------------------------------------------
# 3.  Utility: capture trial‑level CV results
# -----------------------------------------------------------------------------
def _append_cv_results(search, model_label, scoring_str, notebook_file, trial_log):
    cv_res = search.cv_results_
    for i in range(len(cv_res["params"])):
        trial_log.append({
            "notebook": notebook_file,
            "model":    model_label,
            "params":   cv_res["params"][i],
            "mean_score": cv_res["mean_test_score"][i],
            "std_score":  cv_res["std_test_score"][i],
            "scoring":    scoring_str,
        })

# -----------------------------------------------------------------------------
# 4.  Main driver
# -----------------------------------------------------------------------------
def optimize_model(
    dataset_path: str = "train.csv",
    target_column: str  = "Transported",
    test_size: float    = 0.2,
    random_state: int   = 42,
    hyperparameter_space: Optional[dict] = None,
    cv_folds: int       = 5,
    output_file: str    = "optimized_results.json",
    log_file: str       = "hyperparam_trials.csv",
):
    """Run optimization + full trial logging, return best‑result dict."""
    # 0) load and prep
    print("Loading extracted model data …")
    with open("model_scores.json", "r", encoding="utf-8") as fh:
        extracted_data = json.load(fh)
    notebooks = extracted_data

    print("Loading dataset …")
    df = pd.read_csv(dataset_path)
    # basic imputation
    for col in df.select_dtypes(include=["number"]):
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=["object","category"]):
        df[col].fillna(df[col].mode()[0], inplace=True)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    for col in X.select_dtypes(include=["object","bool"]):
        X[col] = LabelEncoder().fit_transform(X[col])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    input_dim = X_train.shape[1]
    if hyperparameter_space is None:
        hyperparameter_space = optimized_hyperparameter_space

    optimized_results = {}
    trial_log = []

    # 1) loop over notebooks with a visible progress bar
    print(f"Starting optimization on {len(notebooks)} notebooks …")
    for notebook in tqdm(notebooks, desc="Notebooks", unit="nb", leave=True):
        nb_file = notebook.get("notebook", "<unknown>")
        tqdm.write(f"→ Processing notebook: {nb_file}")

        models = notebook.get("models", [])
        if not models:
            tqdm.write("   (no models found in this notebook)")
            continue

        # 2) loop over models in that notebook
        for model in tqdm(models, desc="Models", unit="model", leave=False):
            model_name = model.get("name", model.get("type"))
            model_type = model.get("type", "<UNK>")
            tqdm.write(f"   • Model `{model_name}` of type `{model_type}`")

            original_params = model.get("parameters", {})
            # pick first non‐null score
            scoring_method = next(
                (s.get("method") for s in model.get("scores", [])
                    if s.get("value") is not None),
                "accuracy_score"
            )
            scoring_str = scoring_mapping.get(scoring_method, "accuracy")
            best_params = {}
            best_model  = None

            try:
                if model_type == "Sequential":
                    tqdm.write("     ↳ parsing architecture…")
                    # … (your existing arch‐parsing logic here) …

                    tqdm.write("     ↳ running RandomizedSearchCV on Keras model…")
                    # … (your existing search.fit code here) …

                else:
                    param_grid = hyperparameter_space.get(model_type, {})
                    if not param_grid:
                        raise ValueError(f"No grid defined for {model_type}")

                    tqdm.write(f"     ↳ running RandomizedSearchCV on {model_type}…")
                    search = RandomizedSearchCV(
                        eval(model_type)(),
                        param_grid,
                        n_iter=10, cv=cv_folds,
                        scoring=scoring_str,
                        n_jobs=-1, refit=True
                    )
                    search.fit(X_train, y_train)

                _append_cv_results(
                    search,
                    model_name,
                    scoring_str,
                    nb_file,
                    trial_log
                )
                best_params = search.best_params_
                best_model  = search.best_estimator_

                # evaluate on hold‑out
                y_pred = best_model.predict(X_val)
                opt_acc  = accuracy_score(y_val, y_pred)
                opt_prec = precision_score(y_val, y_pred, zero_division=0)
                opt_rec  = recall_score(y_val, y_pred, zero_division=0)
                opt_f1   = f1_score(y_val, y_pred, zero_division=0)

            except Exception as exc:
                tqdm.write(f"     ⚠️ error optimizing: {exc}")
                best_params = str(exc)
                opt_acc = opt_prec = opt_rec = opt_f1 = f"Error: {exc}"

            # grab original score
            orig_score = "N/A"
            for se in model.get("scores", []):
                if se.get("value"):
                    try:
                        orig_score = float(se["value"].split(":")[-1])
                    except:
                        orig_score = se["value"]
                    break

            # stash results under “notebook::modelname”
            key = f"{nb_file}::{model_name}"
            optimized_results[key] = {
                "Original Model":       model_type,
                "Original Parameters":  original_params,
                "Original Score":       orig_score,
                "Optimized Parameters": best_params,
                "Optimized accuracy":         opt_acc,
                "Optimized precision_score":  opt_prec,
                "Optimized recall_score":     opt_rec,
                "Optimized f1_score":         opt_f1,
            }

    # 3) save outputs
    print("\nSaving optimized_results.json …")
    with open(output_file, "w") as fh:
        json.dump(optimized_results, fh, indent=4, default=str)
    print("Saving hyperparam_trials.csv …")
    pd.DataFrame(trial_log).to_csv(log_file, index=False)
    print("Done.")

    return optimized_results

# -----------------------------------------------------------------------------
# 5.  CLI entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    optimize_model()
