"""
Strict Leakage-Free Nested Cross-Validation Benchmark for Candidate Models.
Evaluates:
- Logistic Regression (L1, L2)
- Linear SVM
- RBF SVM
- Random Forest
- Gradient Boosting
Inside 5 outer folds x 3 inner folds, with feature selection strictly inside folds.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


def run_nested_cv(csv_path: str = "data/processed/canonical_features.csv"):
    df = pd.read_csv(csv_path)
    meta_cols = ["sample_id", "subject_id", "label", "label_str", "duration_sec", "missing_count"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feat_cols]
    y = df["label"]

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    candidate_models = {
        "Logistic Regression (L2)": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(f_classif)),
                ("clf", LogisticRegression(penalty="l2", solver="liblinear", random_state=42))
            ]),
            {
                "selector__k": [8, 10, 12, 16],
                "clf__C": [0.05, 0.1, 0.5, 1.0, 2.0]
            }
        ),
        "Linear SVM": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(f_classif)),
                ("clf", SVC(kernel="linear", probability=True, random_state=42))
            ]),
            {
                "selector__k": [8, 10, 12, 16],
                "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0]
            }
        ),
        "RBF SVM": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(f_classif)),
                ("clf", SVC(kernel="rbf", probability=True, random_state=42))
            ]),
            {
                "selector__k": [8, 10, 12, 16],
                "clf__C": [0.1, 0.5, 1.0, 2.0],
                "clf__gamma": ["scale", 0.01, 0.05]
            }
        ),
        "Random Forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(f_classif)),
                ("clf", RandomForestClassifier(random_state=42))
            ]),
            {
                "selector__k": [8, 10, 12, 16],
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [2, 3, 4],
                "clf__min_samples_leaf": [2, 3]
            }
        ),
        "Gradient Boosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(f_classif)),
                ("clf", GradientBoostingClassifier(random_state=42))
            ]),
            {
                "selector__k": [8, 10, 12],
                "clf__n_estimators": [30, 40, 50],
                "clf__max_depth": [2, 3],
                "clf__learning_rate": [0.05, 0.08, 0.1]
            }
        )
    }

    print("\n" + "="*80)
    print("STRICT NESTED CROSS-VALIDATION BENCHMARK (5 Outer x 3 Inner Folds)")
    print("="*80)
    print(f"Total samples: {len(df)} (HC: {(y==0).sum()}, PD: {(y==1).sum()})")
    print(f"Candidate feature pool: {len(feat_cols)}")

    results = {}
    oof_predictions = {}

    for name, (pipe, grid) in candidate_models.items():
        print(f"\nEvaluating: {name} ...", end=" ", flush=True)
        metrics = {
            "accuracy": [],
            "bal_acc": [],
            "auc": [],
            "f1": [],
            "sensitivity": [],
            "specificity": []
        }
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))
        selected_params = []

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            search = GridSearchCV(
                pipe,
                grid,
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            preds = best_model.predict(X_test)
            probs = best_model.predict_proba(X_test)[:, 1]

            oof_preds[test_idx] = preds
            oof_probs[test_idx] = probs
            selected_params.append(search.best_params_)

            metrics["accuracy"].append(accuracy_score(y_test, preds))
            metrics["bal_acc"].append(balanced_accuracy_score(y_test, preds))
            metrics["auc"].append(roc_auc_score(y_test, probs))
            metrics["f1"].append(f1_score(y_test, preds, zero_division=0))
            metrics["sensitivity"].append(recall_score(y_test, preds, pos_label=1, zero_division=0))
            metrics["specificity"].append(recall_score(y_test, preds, pos_label=0, zero_division=0))

        # Overall Out-of-fold metrics
        oof_acc = accuracy_score(y, oof_preds)
        oof_bal_acc = balanced_accuracy_score(y, oof_preds)
        oof_auc = roc_auc_score(y, oof_probs)
        oof_f1 = f1_score(y, oof_preds, zero_division=0)
        oof_sens = recall_score(y, oof_preds, pos_label=1, zero_division=0)
        oof_spec = recall_score(y, oof_preds, pos_label=0, zero_division=0)
        cm = confusion_matrix(y, oof_preds).tolist()

        results[name] = {
            "mean_auc": float(np.mean(metrics["auc"])),
            "std_auc": float(np.std(metrics["auc"])),
            "mean_bal_acc": float(np.mean(metrics["bal_acc"])),
            "std_bal_acc": float(np.std(metrics["bal_acc"])),
            "mean_f1": float(np.mean(metrics["f1"])),
            "std_f1": float(np.std(metrics["f1"])),
            "mean_sensitivity": float(np.mean(metrics["sensitivity"])),
            "mean_specificity": float(np.mean(metrics["specificity"])),
            "oof_auc": float(oof_auc),
            "oof_bal_acc": float(oof_bal_acc),
            "oof_accuracy": float(oof_acc),
            "oof_f1": float(oof_f1),
            "oof_sensitivity": float(oof_sens),
            "oof_specificity": float(oof_spec),
            "confusion_matrix": cm,
            "selected_params": selected_params,
            "oof_probs": oof_probs.tolist(),
            "oof_preds": oof_preds.tolist()
        }
        oof_predictions[name] = {
            "probs": oof_probs.tolist(),
            "preds": oof_preds.tolist()
        }
        print("Done.")

    summary_df = pd.DataFrame([
        {
            "Model": name,
            "Mean_AUC": f"{data['mean_auc']:.3f} +/- {data['std_auc']:.3f}",
            "Mean_Bal_Acc": f"{data['mean_bal_acc']:.3f} +/- {data['std_bal_acc']:.3f}",
            "Sensitivity": f"{data['mean_sensitivity']:.3f}",
            "Specificity": f"{data['mean_specificity']:.3f}",
            "F1_Score": f"{data['mean_f1']:.3f}",
            "OOF_AUC": f"{data['oof_auc']:.3f}",
            "OOF_Bal_Acc": f"{data['oof_bal_acc']:.3f}"
        }
        for name, data in results.items()
    ]).sort_values(by="OOF_AUC", ascending=False)

    print("\n" + "="*80)
    print("NESTED CV SUMMARY COMPARISON")
    print("="*80)
    print(summary_df.to_string(index=False))

    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/nested_cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, summary_df


if __name__ == "__main__":
    run_nested_cv()
