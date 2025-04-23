#!/usr/bin/env python -u
"""
Standalone script that:
  1) Loads extracted model_scores.json
  2) Imputes & encodes train.csv
  3) Rebuilds & scores original models
  4) Runs RandomizedSearchCV on each model (including Sequential nets via a small sklearn wrapper)
  5) Outputs:
     - trials_<notebook>.csv       # all CV trials per notebook
     - best_optimized_results.csv  # one row per model
     - optimized_results.json
     - submission_<notebook>.csv   # kaggle-style PassengerId,Transported
"""
import os
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
print("Script has launched", flush=True)

# suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


# -----------------------------------------------------------------------------
# 1) KerasWrapper: make Sequential behave like a sklearn classifier
# -----------------------------------------------------------------------------
class KerasWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, architecture, input_dim,
                 learning_rate=0.01, batch_size=32, epochs=30,
                 optimizer="adam", dropout_rate=0.0, batch_norm=False, l2_reg=0.0):
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
                "activation": cfg.get("activation", "relu"),
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
            "adam":    tf.keras.optimizers.Adam,
            "nadam":   tf.keras.optimizers.Nadam,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "sgd":     tf.keras.optimizers.SGD,
            "adagrad": tf.keras.optimizers.Adagrad,
            "adamax":  tf.keras.optimizers.Adamax,
        }
        Opt = opt_map.get(self.optimizer.lower(), tf.keras.optimizers.Adam)
        if Opt is tf.keras.optimizers.SGD:
            opt_inst = Opt(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        else:
            opt_inst = Opt(learning_rate=self.learning_rate)

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
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        probs = self.model_.predict(X, verbose=0)
        return (probs.ravel() > 0.5).astype(int)


# -----------------------------------------------------------------------------
# 2) Hyper-parameter spaces & scoring
# -----------------------------------------------------------------------------
optimized_hyperparameter_space = {
    "RandomForestClassifier": {
        "n_estimators": list(range(50,501,50)),
        "max_depth":    [None] + list(range(5,51,5)),
        "min_samples_split": [2,3,4,5,6,7,8,9,10,15],
        "min_samples_leaf":  list(range(1,11)),
        "max_features": ["sqrt","log2",0.3,0.4,0.5,0.6,0.7,0.8,0.9,None],
    },
    "GradientBoostingClassifier": {
        "n_estimators": list(range(50,501,50)),
        "learning_rate": [0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.5],
        "max_depth":     [3,4,5,6,7,8,9,10,12,15],
        "subsample":     [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
        "min_samples_split": [2,3,4,5,6,7,8,9,10,15],
        "min_samples_leaf":  list(range(1,11)),
    },
    "LogisticRegression": {
        "C": [0.001,0.01,0.1,0.5,1,2,5,10,50,100],
        "penalty":   ["l1","l2","elasticnet"],
        "solver":    ["lbfgs","newton-cg","liblinear","sag","saga"],
        "l1_ratio":  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    },
    "SVC": {
        "C":      [0.001,0.01,0.1,0.5,1,2,5,10,50,100],
        "kernel": ["linear","poly","rbf","sigmoid"],
        "gamma":  ["scale","auto",0.0001,0.001,0.01,0.1,1,10],
        "degree": [2,3,4,5,6,7,8],
        "coef0":  [-1.0,-0.5,0.0,0.5,1.0],
    },
    "NeuralNetwork": {
        "learning_rate": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.5],
        "batch_size":    [16,32,64,128,256,512],
        "epochs":        [20,30,40,50,60,70,80,90,100,150],
    },
    "XGBClassifier": {
        "n_estimators":    [100,200,300,400,500,600,800,1000],
        "max_depth":       [3,4,5,6,7,8,10,12],
        "learning_rate":   [0.001,0.005,0.01,0.05,0.1,0.2,0.3],
        "subsample":       [0.5,0.6,0.7,0.8,0.9,1.0],
        "colsample_bytree":[0.5,0.6,0.7,0.8,0.9,1.0],
        "gamma":           [0,0.01,0.05,0.1,0.2,0.3],
        "min_child_weight": [1,3,5,7],
        "reg_alpha":       [0,0.001,0.01,0.1],
        "reg_lambda":      [1,1.5,2,3],
    },
    "LinearRegression": {
        "fit_intercept": [True, False],
        "positive":      [True, False],
    },
    "KNeighborsClassifier": {
        "n_neighbors": list(range(1,11)),
        "weights":     ["uniform","distance"],
        "p":           [1,2],
    },
}

scoring_mapping = {
    "accuracy_score": "accuracy",
    "accuracy":       "accuracy",
    "f1_score":       "f1",
    "precision_score":"precision",
    "recall_score":   "recall",
}


# -----------------------------------------------------------------------------
# 3) Dump every CV trial
# -----------------------------------------------------------------------------
def _append_cv_results(search, notebook, model_label, scoring, trial_log):
    cv = search.cv_results_
    for i in range(len(cv["params"])):
        row = {"notebook": notebook, "model": model_label, "scoring": scoring}
        row.update(cv["params"][i])
        row["mean_score"] = cv["mean_test_score"][i]
        row["std_score"]  = cv["std_test_score"][i]
        trial_log.append(row)


# -----------------------------------------------------------------------------
# 4) Main optimizer
# -----------------------------------------------------------------------------
def optimize_model(
    model_scores_json: str = "model_scores.json",
    data_csv:           str = "train.csv",
    test_csv:           str = "test.csv",
    output_json:        str = "optimized_results.json",
    best_csv:           str = "best_optimized_results.csv",
    test_size:         float = 0.2,
    random_state:       int = 42,
    cv_folds:           int = 5,
    hyper_space:      Optional[dict] = None
):
    print("🚀 Starting optimization…", flush=True)

    # load metadata
    print(f"Loading metadata from {model_scores_json}", flush=True)
    with open(model_scores_json) as f:
        raw = json.load(f)
    notebooks = raw.get("notebooks") if isinstance(raw, dict) else raw

    # --- TRAIN data ---
    print(f"Reading training data from {data_csv}", flush=True)
    df = pd.read_csv(data_csv)
    # drop target + PassengerId
    df.drop(columns=["Transported","PassengerId"], inplace=True)
    # impute
    for c in df.select_dtypes(include=["number"]):
        df[c].fillna(df[c].mean(), inplace=True)
    for c in df.select_dtypes(include=["object","category"]):
        df[c].fillna(df[c].mode()[0], inplace=True)
    X = df.drop(columns=["Transported"], errors="ignore")
    y = pd.read_csv(data_csv)["Transported"]  # re-read target

    for c in X.select_dtypes(include=["object","bool"]):
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    # --- TEST data ---
    print(f"Reading test data from {test_csv}", flush=True)
    test_df = pd.read_csv(test_csv)
    test_ids = test_df["PassengerId"]
    test_X   = test_df.drop(columns=["PassengerId"])
    # impute with train stats
    for c in test_X.select_dtypes(include=["number"]):
        test_X[c].fillna(df[c].mean(), inplace=True)
    for c in test_X.select_dtypes(include=["object","category"]):
        test_X[c].fillna(df[c].mode()[0], inplace=True)
    # encode on union of categories
    for c in test_X.select_dtypes(include=["object","bool"]):
        all_vals = pd.concat([X[c].astype(str), test_X[c].astype(str)], ignore_index=True)
        le = LabelEncoder().fit(all_vals)
        test_X[c] = le.transform(test_X[c].astype(str))

    # ensure column order matches!
    test_X = test_X[X.columns]

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    input_dim = X_train.shape[1]
    hyper     = hyper_space or optimized_hyperparameter_space

    optimized = {}
    best_rows = []

    for nb in tqdm(notebooks, desc="Notebooks"):
        stem      = Path(nb["notebook"]).stem
        trial_log = []

        print(f"\nprocessing notebook: {stem}", flush=True)

        # initialize best for this notebook
        best_f1_nb = -1.0
        best_est_nb = None

        for m in nb.get("models", []):
            mtype   = m["type"]
            orig_kw = m["parameters"].get("kwargs", {})
            score_m = m.get("score") or "accuracy_score"
            scoring = scoring_mapping.get(score_m, "accuracy")

            # rebuild & score original
            try:
                if mtype == "Sequential":
                    raw_layers = m["parameters"]["args"][0].strip("[]")
                    arch = []
                    for piece in raw_layers.split("),"):
                        piece = piece.strip()
                        if "(" not in piece: continue
                        inside = piece.split("(",1)[1].rsplit(")",1)[0]
                        parts  = [p.strip() for p in inside.split(",")]
                        ut     = parts[0]
                        units  = input_dim if "train" in ut.lower() else int(ut)
                        act    = next((x.split("=")[1].strip().strip("'\"")
                                      for x in parts if x.startswith("activation")), "relu")
                        arch.append({"units": units, "activation": act})

                    orig_est = KerasWrapper(
                        architecture=arch,
                        input_dim=input_dim,
                        optimizer=orig_kw.get("optimizer","adam")
                    ).fit(X_train, y_train)
                    y0 = orig_est.predict(X_val)

                else:
                    if mtype == "XGBClassifier":
                        orig_est = xgb.XGBClassifier(**orig_kw).fit(X_train, y_train)
                    elif mtype == "LGBMClassifier":
                        orig_est = lgb.LGBMClassifier(**orig_kw).fit(X_train, y_train)
                    else:
                        orig_est = eval(mtype)(**orig_kw).fit(X_train, y_train)
                    y0 = orig_est.predict(X_val)

                orig_acc  = accuracy_score(y_val, y0)
                orig_prec = precision_score(y_val, y0, zero_division=0)
                orig_rec  = recall_score(y_val, y0, zero_division=0)
                orig_f1   = f1_score(y_val, y0, zero_division=0)

            except Exception as e:
                tqdm.write(f"⚠️ {stem}::{mtype} original rebuild failed → {e}")
                continue

            # hyperparameter search
            grid = hyper.get(mtype, {})
            if mtype == "Sequential":
                estimator = KerasWrapper(architecture=arch, input_dim=input_dim)
                grid = {
                    "learning_rate": hyper["NeuralNetwork"]["learning_rate"],
                    "batch_size":    hyper["NeuralNetwork"]["batch_size"],
                    "epochs":        hyper["NeuralNetwork"]["epochs"],
                }
            elif not grid:
                best_params, best_est = {}, orig_est
            else:
                estimator = xgb.XGBClassifier() if mtype=="XGBClassifier" else eval(mtype)()

            if grid:
                search = RandomizedSearchCV(
                    estimator, grid,
                    n_iter=10, cv=cv_folds,
                    scoring=scoring, n_jobs=-1, refit=True
                )
                search.fit(X_train, y_train)
                best_params = search.best_params_
                best_est    = search.best_estimator_
                _append_cv_results(search, stem, mtype, scoring, trial_log)

            # evaluate optimized
            y1     = best_est.predict(X_val)
            opt_acc= accuracy_score(y_val, y1)
            opt_prec = precision_score(y_val, y1, zero_division=0)
            opt_rec  = recall_score(y_val, y1, zero_division=0)
            opt_f1   = f1_score(y_val, y1, zero_division=0)

            key = f"{stem}::{mtype}"
            optimized[key] = {
                "Original Model":      mtype,
                "Original Parameters": orig_kw,
                "Original_accuracy":   orig_acc,
                "Original_precision":  orig_prec,
                "Original_recall":     orig_rec,
                "Original_f1":         orig_f1,
                "Optimized Parameters":best_params,
                "Optimized_accuracy":  opt_acc,
                "Optimized_precision": opt_prec,
                "Optimized_recall":    opt_rec,
                "Optimized_f1":        opt_f1,
                "optimized_based_on":   scoring

            }
            best_rows.append({
                "notebook":      stem,
                "model":         mtype,
                **best_params,
                "val_accuracy":  opt_acc,
                "val_precision": opt_prec,
                "val_recall":    opt_rec,
                "val_f1":        opt_f1
            })

            # update best‐per‐notebook
            if opt_f1 > best_f1_nb:
                best_f1_nb  = opt_f1
                best_est_nb = best_est

        # write out trials
        if trial_log:
            fname = f"trials_{stem}.csv"
            pd.DataFrame(trial_log).to_csv(fname, index=False)
            print(f"saved {fname}", flush=True)

        # write out Kaggle submission
        if best_est_nb is not None:
            preds      = best_est_nb.predict(test_X)
            submission = pd.DataFrame({
                "PassengerId": test_ids,
                "Transported": preds.astype(bool)
            })
            sub_name = f"submission_{stem}.csv"
            submission.to_csv(sub_name, index=False)
            print(f"saved {sub_name}", flush=True)

        print(f"finished processing notebook: {stem}", flush=True)

    # final outputs
    pd.DataFrame(best_rows).to_csv(best_csv, index=False)
    print(f"saved {best_csv}", flush=True)

    with open(output_json, "w") as f:
        json.dump(optimized, f, indent=2, default=str)
    print(f"saved {output_json}", flush=True)

    print("✅ All done.", flush=True)


if __name__ == "__main__":
    optimize_model()
