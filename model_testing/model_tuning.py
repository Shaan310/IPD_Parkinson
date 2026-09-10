import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "voice_features.csv"
)


print("\n===== HYPERPARAMETER TUNING =====\n")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["sample_id", "label"])
y = df["label"].map({"HC": 0, "PwPD": 1})

print("Dataset:", X.shape)
print("HC:", (y == 0).sum())
print("PwPD:", (y == 1).sum())


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


models = {

    "Logistic Regression": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000))
        ]),
        {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    ),

    "Random Forest": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=42))
        ]),
        {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4]
        }
    ),

    "SVM": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                random_state=42
            ))
        ]),
        {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.001, 0.01, 0.1],
            "model__kernel": ["rbf"]
        }
    )
}


for name, (pipeline, parameters) in models.items():

    print(f"\n===== {name} =====")

    search = GridSearchCV(
        pipeline,
        parameters,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        return_train_score=False
    )

    search.fit(X, y)

    print("Best parameters:")
    print(search.best_params_)

    print(
        "Best cross-validation ROC-AUC:",
        round(search.best_score_, 4)
    )

    best_model = search.best_estimator_

    predictions = []
    probabilities = []
    actual = []

    for train_idx, test_idx in cv.split(X, y):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        best_model.fit(X_train, y_train)

        predictions.extend(best_model.predict(X_test))
        probabilities.extend(
            best_model.predict_proba(X_test)[:, 1]
        )
        actual.extend(y_test)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    actual = np.array(actual)

    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        actual,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        actual,
        predictions,
        zero_division=0
    )
    auc = roc_auc_score(
        actual,
        probabilities
    )

    cm = confusion_matrix(actual, predictions)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\nCross-validation predictions:")
    print("Accuracy   :", round(accuracy, 4))
    print("Precision  :", round(precision, 4))
    print("Recall     :", round(recall, 4))
    print("F1 Score   :", round(f1, 4))
    print("ROC-AUC    :", round(auc, 4))
    print("Sensitivity:", round(sensitivity, 4))
    print("Specificity:", round(specificity, 4))

    print("\nConfusion Matrix:")
    print(cm)


print("\n===== HYPERPARAMETER TUNING COMPLETE =====")