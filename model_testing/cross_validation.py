import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "voice_features.csv"
)


print("\n===== 5-FOLD CROSS-VALIDATION =====\n")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["sample_id", "label"])
y = df["label"].map({"HC": 0, "PwPD": 1})

print("Dataset:", X.shape)
print("HC:", (y == 0).sum())
print("PwPD:", (y == 1).sum())


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


for name, model in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    print(f"\n===== {name} =====")

    for metric in scoring:

        scores = results[f"test_{metric}"]

        print(
            f"{metric.upper():9}: "
            f"{scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )


print("\n===== CROSS-VALIDATION COMPLETE =====")