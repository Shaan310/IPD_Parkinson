import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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


print("\n===== FEATURE SELECTION + NESTED CROSS-VALIDATION =====\n")

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


feature_counts = [10, 20, 30, 50, 80, 100, "all"]

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


for name, model in models.items():

    print(f"\n===== {name} =====")

    outer_scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "sensitivity": [],
        "specificity": []
    }

    selected_features = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), 1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(score_func=f_classif)),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        parameter_grid = {
            "selector__k": feature_counts
        }

        search = GridSearchCV(
            pipeline,
            parameter_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )
        auc = roc_auc_score(
            y_test,
            probabilities
        )

        cm = confusion_matrix(y_test, predictions)

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0

        outer_scores["accuracy"].append(accuracy)
        outer_scores["precision"].append(precision)
        outer_scores["recall"].append(recall)
        outer_scores["f1"].append(f1)
        outer_scores["roc_auc"].append(auc)
        outer_scores["sensitivity"].append(sensitivity)
        outer_scores["specificity"].append(specificity)

        selected_features.append(
            search.best_params_["selector__k"]
        )

        print(
            f"Fold {fold}: "
            f"best features = {search.best_params_['selector__k']}, "
            f"ROC-AUC = {auc:.4f}"
        )

    print("\nSelected feature counts:")
    print(selected_features)

    print("\nFinal Results:")

    for metric, scores in outer_scores.items():

        scores = np.array(scores)

        print(
            f"{metric.upper():11}: "
            f"{scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )


print("\n===== FEATURE SELECTION COMPLETE =====")