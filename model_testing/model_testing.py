import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DATA_PATH = "xxxxxxxxxxx.csv"
TARGET_COLUMN = "status"


def load_data():
    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    if "name" in X.columns:
        X = X.drop(columns=["name"])
    X = X.select_dtypes(include=[np.number])

    return X, y


def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                class_weight="balanced"
            ))
        ])
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ])
    return models


def evaluate_models(X, y):
    models = build_models()
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    results = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=[
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"
            ],
            n_jobs=-1
        )
        results.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean()
        })
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )
    return results_df

def main():
    print("PARKINSON'S VOICE MODEL EXPERIMENT")
    X, y = load_data()

    print("\nFeatures used:", X.shape[1])
    print("Samples:", X.shape[0])

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nRunning model comparison")

    results = evaluate_models(X, y)
    print("\nMODEL RESULTS")
    print("=" * 80)
    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results.to_csv(
        "model_results.csv",
        index=False
    )
    print("\nResults saved to: model_results.csv")


if __name__ == "__main__":
    main()