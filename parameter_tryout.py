import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier

#Sample output:
#{'spaceship-titanic.ipynb': {'Best Model': 'RandomForestClassifier', 
# 'Best Score': np.float64(0.8022740226223046), 
# 'Best Parameters': {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}}}

model_types = {
    "spaceship-titanic.ipynb": ["RandomForestClassifier"],
}

df = pd.read_csv("train.csv")
# Fill numeric columns with mean
for col in df.select_dtypes(include=['number']):
    df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical columns with mode
for col in df.select_dtypes(include=['object', 'category']):
    df[col].fillna(df[col].mode()[0], inplace=True)
    df[col].fillna(df[col].mode()[0], inplace=True)

target_column = "Transported"  
X = df.drop(columns=[target_column])
y = df[target_column]

for col in X.select_dtypes(include=["object", "bool"]):
    X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grids = {
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "GradientBoostingClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10]
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "max_iter": [200, 500, 1000]
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }
}

def create_nn(learning_rate=0.01):
    model = Sequential([
        Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
        Dense(24, activation="relu"),
        Dense(12, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])
    return model

best_results = {}

for file, models in model_types.items():
    best_model = None
    best_score = -np.inf
    best_params = None

    for model_name in models:
        if model_name in param_grids:
            model = eval(model_name)()
            param_grid = param_grids[model_name]

            grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
            grid_search.fit(X_train, y_train)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = model_name
                best_params = grid_search.best_params_

        elif model_name == "Sequential":
            best_score_nn = -np.inf
            best_config = None

            for lr in [0.001, 0.01, 0.03]:
                nn_model = KerasClassifier(build_fn=create_nn, learning_rate=lr, epochs=20, batch_size=32, verbose=0)
                scores = cross_val_score(nn_model, X_train, y_train, cv=5, scoring="accuracy")

                avg_score = np.mean(scores)
                if avg_score > best_score_nn:
                    best_score_nn = avg_score
                    best_config = {"learning_rate": lr}

            if best_score_nn > best_score:
                best_score = best_score_nn
                best_model = "Sequential"
                best_params = best_config

    best_results[file] = {
        "Best Model": best_model,
        "Best Score": best_score,
        "Best Parameters": best_params
    }

print(best_results)
