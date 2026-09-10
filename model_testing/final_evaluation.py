import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "voice_features.csv"
)

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["sample_id", "label"])
y = df["label"].map({"HC": 0, "PwPD": 1})

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000))
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    },

    "Random Forest": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                random_state=42
            ))
        ]),
        "params": {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4]
        }
    },

    "SVM": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf"))
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.001, 0.01, 0.1]
        }
    }
}

print("\n========================================")
print("FINAL NESTED CROSS-VALIDATION")
print("========================================")

print(f"\nDataset size: {len(df)}")
print(f"Number of features: {X.shape[1]}")
print(f"HC samples: {(y == 0).sum()}")
print(f"PwPD samples: {(y == 1).sum()}")

all_results = {}

for model_name, config in models.items():

    print("\n========================================")
    print(model_name)
    print("========================================")

    outer_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "sensitivity": [],
        "specificity": []
    }

    selected_params = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        else:
            y_score = best_model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )
        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
        roc_auc = roc_auc_score(
            y_test,
            y_score
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        outer_metrics["accuracy"].append(accuracy)
        outer_metrics["precision"].append(precision)
        outer_metrics["recall"].append(recall)
        outer_metrics["f1"].append(f1)
        outer_metrics["roc_auc"].append(roc_auc)
        outer_metrics["sensitivity"].append(sensitivity)
        outer_metrics["specificity"].append(specificity)

        selected_params.append(grid.best_params_)

        print(f"\nFold {fold}")
        print("Best parameters:", grid.best_params_)
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1:          {f1:.4f}")
        print(f"ROC-AUC:     {roc_auc:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

    all_results[model_name] = outer_metrics

    print("\n----------------------------------------")
    print("FINAL NESTED CV RESULT")
    print("----------------------------------------")

    for metric, values in outer_metrics.items():
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        print(
            f"{metric.capitalize():12s}: "
            f"{mean:.4f} +/- {std:.4f}"
        )

    print("\nParameters selected in each fold:")
    for i, params in enumerate(selected_params, start=1):
        print(f"Fold {i}: {params}")


print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

comparison = []

for model_name, metrics in all_results.items():

    row = {
        "Model": model_name
    }

    for metric, values in metrics.items():
        row[metric] = np.mean(values)

    comparison.append(row)

comparison_df = pd.DataFrame(comparison)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

best_model = comparison_df.loc[
    comparison_df["f1"].idxmax(),
    "Model"
]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print(f"Best model based on mean F1: {best_model}")

print("\n===== EVALUATION COMPLETE =====")