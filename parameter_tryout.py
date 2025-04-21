#!/usr/bin/env python
"""
Standalone script that:
  1) Loads extracted `model_scores.json`
  2) Imputes & encodes `train.csv`
  3) Rebuilds & scores original models
  4) Runs RandomizedSearchCV on each model
  5) Outputs:
     - trials_<notebook>.csv
     - best_optimized_results.csv
     - optimized_results.json
     - submission_<notebook>.csv  # <-- NEW
"""
import re
import os, json, warnings, ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb
import lightgbm as lgb


MODEL_REGISTRY = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "KNeighborsClassifier": KNeighborsClassifier,
    "XGBClassifier": xgb.XGBClassifier,
    "LGBMClassifier": lgb.LGBMClassifier,
    "LinearRegression": LinearRegression
}

# -----------------------------------------------------------------------------
# 1) KerasWrapper (unchanged)
# -----------------------------------------------------------------------------
class KerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, architecture, input_dim, learning_rate=0.01,
                 batch_size=32, epochs=30, optimizer="adam",
                 dropout_rate=0.0, batch_norm=False, l2_reg=0.0):
        self.architecture  = architecture
        self.input_dim     = input_dim
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.optimizer     = optimizer
        self.dropout_rate  = dropout_rate
        self.batch_norm    = batch_norm
        self.l2_reg        = l2_reg

    def _build_model(self):
        m = Sequential()
        for i, cfg in enumerate(self.architecture):
            kw = {
                "units": cfg["units"],
                "activation": cfg.get("activation","relu"),
                "kernel_regularizer": regularizers.l2(self.l2_reg),
            }
            if i == 0:
                kw["input_shape"] = (self.input_dim,)
            m.add(Dense(**kw))
            if self.batch_norm:
                m.add(BatchNormalization())
            if self.dropout_rate > 0:
                m.add(Dropout(self.dropout_rate))
        m.add(Dense(1, activation="sigmoid"))

        opt_map = {
            "adam": tf.keras.optimizers.Adam,
            "nadam": tf.keras.optimizers.Nadam,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "sgd": tf.keras.optimizers.SGD,
            "adagrad": tf.keras.optimizers.Adagrad,
            "adamax": tf.keras.optimizers.Adamax,
        }
        Opt = opt_map.get(self.optimizer.lower(), tf.keras.optimizers.Adam)
        opt_inst = (Opt(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
                    if Opt is tf.keras.optimizers.SGD
                    else Opt(learning_rate=self.learning_rate))
        m.compile(loss="binary_crossentropy", optimizer=opt_inst, metrics=["accuracy"])
        return m

    def fit(self, X, y):
        self.model_ = self._build_model()
        self.model_.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)],
            verbose=0
        )
        return self

    def predict(self, X):
        probs = self.model_.predict(X, verbose=0)
        return (probs.ravel() > 0.5).astype(int)


# -----------------------------------------------------------------------------
# 2. Hyper‑parameter grids & scoring map
# -----------------------------------------------------------------------------
optimized_hyperparameter_space = {
    "RandomForestClassifier": {
        "n_estimators":[50,100,150,200,250,300,350,400,450,500],
        "max_depth":[None,5,10,15,20,25,30,35,40,50],
        "min_samples_split":[2,3,4,5,6,7,8,9,10,15],
        "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
        "max_features":["sqrt","log2",0.3,0.4,0.5,0.6,0.7,0.8,0.9,None],
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
# 3) Helper to dump every CV trial
# -----------------------------------------------------------------------------
def _append_cv_results(search, notebook, model_label, scoring, trial_log):
    cv = search.cv_results_
    for i in range(len(cv["params"])):
        row = {"notebook":notebook, "model":model_label, "scoring":scoring}
        row.update(cv["params"][i])
        row["mean_score"] = cv["mean_test_score"][i]
        row["std_score"]  = cv["std_test_score"][i]
        trial_log.append(row)
# -----------------------------------------------------------------------------
# 4) Main optimizer + submission generator
# -----------------------------------------------------------------------------
def optimize_model(
    model_scores_json: str="model_scores.json",
    train_csv:         str="train.csv",
    test_csv:          str="test.csv",                      # NEW
    output_json:       str="optimized_results.json",
    best_csv:         str="best_optimized_results.csv",
    test_size:        float=0.2,
    random_state:      int=42,
    cv_folds:          int=5,
    hyper_space:     Optional[dict]=None
):
    # --- load extracted metadata ---
    with open(model_scores_json) as f:
        raw = json.load(f)
    notebooks = raw.get("notebooks") if isinstance(raw,dict) else raw

    # --- 1) LOAD & PREPROCESS TRAINING DATA ---
    df = pd.read_csv(train_csv)
    # remember imputation values
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","bool","category"]).columns.tolist()
    impute_means = df[num_cols].mean()
    impute_modes = {c: df[c].mode()[0] for c in cat_cols}

    # fill
    df[num_cols] = df[num_cols].fillna(impute_means)
    for c in cat_cols:
        df[c].fillna(impute_modes[c], inplace=True)

    # label‐encode cats and store encoders
    label_encoders = {}
    for c in cat_cols:
        le = LabelEncoder().fit(df[c])
        df[c] = le.transform(df[c])
        label_encoders[c] = le

    # split features/target
    X_full = df.drop(columns=["Transported"])
    y_full = df["Transported"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=test_size,
        random_state=random_state
    )
    input_dim = X_train.shape[1]
    hyper = hyper_space or optimized_hyperparameter_space

    optimized = {}
    best_rows = []
    arch_map   = {}   # to remember Sequential architectures per notebook

    # --- 2) LOOP OVER NOTEBOOKS & MODELS ---
    for nb in tqdm(notebooks, desc="Notebooks"):
        stem     = Path(nb["notebook"]).stem
        trial_log= []

        for m in nb.get("models", []):
            mtype   = m["type"]
            orig_kw = m["parameters"].get("kwargs", {})
            score_m = m.get("score") or "accuracy_score"
            scoring = scoring_mapping.get(score_m, "accuracy")

            # ── 2a) REBUILD & SCORE ORIGINAL ──
            try:
                if mtype == "Sequential":
                    raw = m["parameters"]["args"][0]
                    # find all Dense(...) argument strings
                    layer_defs = re.findall(r"Dense\(([^)]*)\)", raw)
                    arch = []
                    for layer_def in layer_defs:
                        # strip out the input_shape=... bit
                        clean = re.sub(r"input_shape\s*=\s*\([^)]*\),?", "", layer_def)
                        # extract activation if present
                        act_m = re.search(r"activation\s*=\s*['\"](.*?)['\"]", clean)
                        activation = act_m.group(1) if act_m else "relu"
                        # first positional argument is units
                        units = int(clean.split(",")[0].strip())
                        arch.append({"units": units, "activation": activation})
                        arch_map[stem] = arch
                    

                    orig_est = KerasWrapper(
                        architecture=arch, input_dim=input_dim,
                        optimizer=orig_kw.get("optimizer","adam")
                    ).fit(X_train, y_train)
                    y0 = orig_est.predict(X_val)
                else:
                    ModelClass = MODEL_REGISTRY.get(mtype)
                    if ModelClass is None:
                        raise ValueError(f"Unknown model type “{mtype}” in notebook {stem}")
                    orig_est = ModelClass(**orig_kw).fit(X_train, y_train)
                    y0 = orig_est.predict(X_val)


                orig_acc, orig_prec, orig_rec, orig_f1 = (
                    accuracy_score(y_val, y0),
                    precision_score(y_val, y0, zero_division=0),
                    recall_score(y_val, y0, zero_division=0),
                    f1_score(y_val, y0, zero_division=0),
                )

            except Exception as e:
                tqdm.write(f"⚠️  {stem}::{mtype} rebuild failed → {e}")
                continue

            # ── 2b) HYPERPARAMETER SEARCH ──
            grid = hyper.get(mtype, {})
            if mtype=="Sequential":
                estimator = KerasWrapper(architecture=arch, input_dim=input_dim)
                grid = hyper["NeuralNetwork"]
            elif grid:
                ModelClass = MODEL_REGISTRY.get(mtype)
                if ModelClass is None:
                    raise ValueError(f"No grid for unknown model “{mtype}” in notebook {stem}")
                estimator = ModelClass()
            else:
                best_params, best_est = {}, orig_est


            if grid:
                search = RandomizedSearchCV(
                    estimator, grid, n_iter=10, cv=cv_folds,
                    scoring=scoring, n_jobs=-1, refit=True, random_state=random_state
                )
                search.fit(X_train, y_train)
                best_params = search.best_params_
                best_est    = search.best_estimator_
                _append_cv_results(search, stem, mtype, scoring, trial_log)

            # ── 2c) EVALUATE OPTIMIZED ──
            y1 = best_est.predict(X_val)
            opt_acc, opt_prec, opt_rec, opt_f1 = (
                accuracy_score(y_val, y1),
                precision_score(y_val, y1, zero_division=0),
                recall_score(y_val, y1, zero_division=0),
                f1_score(y_val, y1, zero_division=0),
            )

            # store results
            key = f"{stem}::{mtype}"
            optimized[key] = {
                "Original Model":      mtype,
                "Original Parameters": orig_kw,
                "Original_accuracy":   orig_acc,
                "Original_precision":  orig_prec,
                "Original_recall":     orig_rec,
                "Original_f1":         orig_f1,
                "Optimized Parameters": best_params,
                "Optimized_accuracy":  opt_acc,
                "Optimized_precision": opt_prec,
                "Optimized_recall":    opt_rec,
                "Optimized_f1":        opt_f1,
            }
            best_rows.append({
                "notebook":     stem,
                "model":        mtype,
                **best_params,
                "val_accuracy":  opt_acc,
                "val_precision": opt_prec,
                "val_recall":    opt_rec,
                "val_f1":        opt_f1
            })

        # per‑notebook trial dump
        if trial_log:
            pd.DataFrame(trial_log).to_csv(f"trials_{stem}.csv", index=False)

    # --- 3) FINAL OUTPUTS: optimized JSON & best CSV ---
    with open(output_json,"w") as f:
        json.dump(optimized, f, indent=2, default=str)
    pd.DataFrame(best_rows).to_csv(best_csv, index=False)

    # --- 4) GENERATE per‑NOTEBOOK SUBMISSIONS ---
    test_df = pd.read_csv(test_csv)
    # preprocess test exactly as train
    test_df[num_cols] = test_df[num_cols].fillna(impute_means)
    for c in cat_cols:
        test_df[c].fillna(impute_modes[c], inplace=True)
        test_df[c] = label_encoders[c].transform(test_df[c])

    X_test = test_df.drop(columns=["PassengerId"], errors="ignore")
    ids    = test_df["PassengerId"]

    # pick best model per notebook by val_accuracy
    best_df = pd.DataFrame(best_rows)
    for stem in best_df["notebook"].unique():
        df_nb = best_df[best_df["notebook"]==stem]
        best = df_nb.loc[df_nb["val_accuracy"].idxmax()]
        mtype = best["model"]
        params = {k: best[k] for k in optimized_hyperparameter_space.get(mtype, {}) if k in best}

        # retrain on full data
        if mtype=="Sequential":
            arch = arch_map[stem]
            model = KerasWrapper(architecture=arch, input_dim=input_dim, **params)
        elif mtype=="XGBClassifier":
            model = xgb.XGBClassifier(**params)
        elif mtype=="LGBMClassifier":
            model = lgb.LGBMClassifier(**params)
        else:
            ModelClass = MODEL_REGISTRY.get(mtype)
            if ModelClass is None:
                raise ValueError(f"Cannot retrain unknown model “{mtype}” for {stem}")
            model = ModelClass(**params)

        model.fit(X_full, y_full)

        # predict & write submission
        preds = model.predict(X_test).astype(bool)
        sub   = pd.DataFrame({
            "PassengerId": ids,
            "Transported": preds
        })
        sub.to_csv(f"submission_{stem}.csv", index=False)

    print("✅ All done.")


if __name__=="__main__":
    optimize_model()