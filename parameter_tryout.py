#!/usr/bin/env python
import os
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    make_scorer
)
# preprocessing & model selection
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# base models
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# You can drop in your SPSA routine here if you still want it for XGBClassifier
# (omitted for brevity in this example).

CLASSIFIERS = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "KNeighborsClassifier": KNeighborsClassifier,
    "XGBClassifier": xgb.XGBClassifier,
    "LGBMClassifier": lgb.LGBMClassifier,
}

REGRESSORS = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "LinearRegression": LinearRegression,
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "XGBRegressor": xgb.XGBRegressor,
    "LGBMRegressor": lgb.LGBMRegressor,
}

# hyper‐parameter spaces for RandomizedSearchCV
HYPER_SPACE_CLASS = {
    "RandomForestClassifier": {
        "n_estimators": [100,200,300,400,500],
        "max_depth":    [None,5,10,20,30],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4],
        "max_features":["sqrt","log2",0.5],
    },
    "GradientBoostingClassifier": {
        "n_estimators":[100,200,300],
        "learning_rate":[0.01,0.05,0.1],
        "max_depth":[3,5,7],
        "subsample":[0.7,0.8,1.0],
    },
    "LogisticRegression": {
        "C":[0.01,0.1,1,10],
        "penalty":["l1","l2"],
        "solver":["saga"],
    },
    "SVC": {
        "C":[0.1,1,10],
        "kernel":["rbf","linear"],
        "gamma":["scale","auto"],
    },
    "KNeighborsClassifier": {
        "n_neighbors":[3,5,7,9],
        "weights":["uniform","distance"],
    },
}

HYPER_SPACE_REGR = {
    "RandomForestRegressor": {
        "n_estimators":[100,200,300],
        "max_depth":[None,5,10,20],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4],
        "max_features":["sqrt","log2",0.7],
    },
    "GradientBoostingRegressor": {
        "n_estimators":[100,200,300],
        "learning_rate":[0.01,0.05,0.1],
        "max_depth":[3,5,7],
        "subsample":[0.7,0.8,1.0],
    },
    "SVR": {
        "C":[0.1,1,10],
        "kernel":["rbf","linear"],
        "gamma":["scale","auto"],
    },
}

SCORING_CLASS = {
    "accuracy_score":"accuracy",
    "precision_score":"precision",
    "recall_score":"recall",
    "f1_score":"f1",
}

SCORING_REGR = {
    "mean_squared_error": "neg_mean_squared_error",
    "mean_absolute_error": "neg_mean_absolute_error",
    "r2_score":"r2",
}

def _append_cv_results(search, stem, mtype, scoring, trial_log):
    for i,p in enumerate(search.cv_results_["params"]):
        trial_log.append({
            "notebook": stem,
            "model": mtype,
            "scoring": scoring,
            **p,
            "mean_score": search.cv_results_["mean_test_score"][i],
            "std_score":  search.cv_results_["std_test_score"][i],
        })

def optimize_model(model_scores_json: str,
                   data_csv: str,
                   test_csv: str,
                   output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    raw = json.load(open(model_scores_json))
    notebooks = raw if isinstance(raw, list) else raw.get("notebooks", [])

    # --- load & preprocess train ---
    df = pd.read_csv(data_csv)
    y  = df["Transported"]
    df = df.drop(columns=["Transported","PassengerId"], errors="ignore")
    for c in df.select_dtypes(include="number"):
        df[c].fillna(df[c].mean(), inplace=True)
    for c in df.select_dtypes(include=["object","category"]):
        df[c].fillna(df[c].mode()[0], inplace=True)
    X = df.copy()
    for c in X.select_dtypes(include=["object","bool"]):
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    # --- load & preprocess test ---
    tdf      = pd.read_csv(test_csv)
    test_ids = tdf["PassengerId"]
    test_X   = tdf.drop(columns=["PassengerId"], errors="ignore")
    for c in test_X.select_dtypes(include="number"):
        test_X[c].fillna(df[c].mean(), inplace=True)
    for c in test_X.select_dtypes(include=["object","category"]):
        test_X[c].fillna(df[c].mode()[0], inplace=True)
    for c in test_X.select_dtypes(include=["object","bool"]):
        le = LabelEncoder().fit(list(X[c].astype(str)) + list(test_X[c].astype(str)))
        test_X[c] = le.transform(test_X[c].astype(str))
    test_X = test_X[X.columns]

    # split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    optimized = {}
    for nb in tqdm(notebooks, desc="Notebooks"):
        stem      = Path(nb["notebook"]).stem
        trial_log = []
        best_est  = None

        for m in nb.get("models", []):
            mtype   = m["type"]
            orig_kw = m["parameters"].get("kwargs", {})
            score_m = m.get("score") or next(iter(SCORING_CLASS), "accuracy_score")

            # --- original rebuild & metric ---
            if mtype in CLASSIFIERS:
                Est = CLASSIFIERS[mtype]
                orig_est = (Pipeline([("scaler",StandardScaler()),("est",Est(**orig_kw))])
                            if mtype=="LogisticRegression"
                            else Est(**orig_kw))\
                            .fit(X_train, y_train)
                y0 = orig_est.predict(X_val)
                orig_metrics = {
                    "accuracy": accuracy_score(y_val, y0),
                    "precision": precision_score(y_val, y0, zero_division=0),
                    "recall": recall_score(y_val, y0, zero_division=0),
                    "f1": f1_score(y_val, y0, zero_division=0),
                }
                scoring = SCORING_CLASS.get(score_m, "accuracy")

                # hyper­param search
                space = HYPER_SPACE_CLASS.get(mtype, {})
                if space:
                    base = CLASSIFIERS[mtype]()
                    if mtype=="LogisticRegression":
                        base = Pipeline([("scaler",StandardScaler()),("est",LogisticRegression(max_iter=1000))])
                    search = RandomizedSearchCV(
                        base, space, n_iter=20, cv=5,
                        scoring=scoring, n_jobs=-1, refit=True, error_score="raise"
                    )
                    search.fit(X_train, y_train)
                    best_est = search.best_estimator_
                    _append_cv_results(search, stem, mtype, scoring, trial_log)
                else:
                    best_est = orig_est

                y1 = best_est.predict(X_val)
                opt_metrics = {
                    "accuracy": accuracy_score(y_val, y1),
                    "precision": precision_score(y_val, y1, zero_division=0),
                    "recall": recall_score(y_val, y1, zero_division=0),
                    "f1": f1_score(y_val, y1, zero_division=0),
                }

                optimized[f"{stem}::{mtype}"] = {
                    "Original Model":       mtype,
                    "Original Parameters":  orig_kw,
                    **{f"Original_{k}":v for k,v in orig_metrics.items()},
                    "Optimized Parameters": getattr(best_est, "get_params", lambda **_: {})() if best_est else orig_kw,
                    **{f"Optimized_{k}":v for k,v in opt_metrics.items()},
                    "optimized_based_on":   scoring
                }

            elif mtype in REGRESSORS:
                Est = REGRESSORS[mtype]
                orig_est = Est(**orig_kw).fit(X_train, y_train)
                y0 = orig_est.predict(X_val)
                orig_mse = mean_squared_error(y_val, y0)
                orig_mae = mean_absolute_error(y_val, y0)
                orig_r2  = r2_score(y_val, y0)
                scoring = SCORING_REGR.get(score_m, "r2")

                # hyper­param search
                space = HYPER_SPACE_REGR.get(mtype, {})
                if space:
                    search = RandomizedSearchCV(
                        Est(), space, n_iter=20, cv=5,
                        scoring=scoring, n_jobs=-1, refit=True, error_score="raise"
                    )
                    search.fit(X_train, y_train)
                    best_est = search.best_estimator_
                    _append_cv_results(search, stem, mtype, scoring, trial_log)
                else:
                    best_est = orig_est

                y1 = best_est.predict(X_val)
                opt_mse = mean_squared_error(y_val, y1)
                opt_mae = mean_absolute_error(y_val, y1)
                opt_r2  = r2_score(y_val, y1)

                optimized[f"{stem}::{mtype}"] = {
                    "Original Model":       mtype,
                    "Original Parameters":  orig_kw,
                    "Original_mse":         orig_mse,
                    "Original_mae":         orig_mae,
                    "Original_r2":          orig_r2,
                    "Optimized Parameters": getattr(best_est, "get_params", lambda **_: {})(),
                    "Optimized_mse":        opt_mse,
                    "Optimized_mae":        opt_mae,
                    "Optimized_r2":         opt_r2,
                    "optimized_based_on":   scoring
                }

            else:
                tqdm.write(f"⚠️  {stem}::{mtype} not recognized; skipping.")
                continue

        # write trials
        pd.DataFrame(trial_log).to_csv(Path(output_dir)/f"trials_{stem}.csv", index=False)
        # write submission
        preds = best_est.predict(test_X)
        submission = pd.DataFrame({
            "PassengerId": test_ids,
            "Prediction":  preds
        })
        submission.to_csv(Path(output_dir)/f"submission_{stem}.csv", index=False)

    # write JSON
    with open(Path(output_dir)/"optimized_results.json", "w") as fp:
        json.dump(optimized, fp, indent=2, default=str)


if __name__ == "__main__":
    MODEL_SCORES = "model_scores.json"
    for suffix,outdir in [("", "original"), ("_transformed","transformed")]:
        optimize_model(
            model_scores_json=MODEL_SCORES,
            data_csv=f"train{suffix}.csv",
            test_csv =f"test{suffix}.csv",
            output_dir=outdir
        )
