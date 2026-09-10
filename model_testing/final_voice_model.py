import os
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "voice_features.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "model_testing",
    "final_voice_model.joblib"
)

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["sample_id", "label"])
y = df["label"].map({
    "HC": 0,
    "PwPD": 1
})

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        random_state=42
    ))
])

param_grid = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

print("\n========================================")
print("TRAINING FINAL VOICE MODEL")
print("========================================")

print(f"\nTraining samples: {len(X)}")
print(f"Features: {X.shape[1]}")

grid.fit(X, y)

final_model = grid.best_estimator_

print("\nBest parameters:")
print(grid.best_params_)

print(f"\nBest CV ROC-AUC: {grid.best_score_:.4f}")

joblib.dump(
    final_model,
    MODEL_FILE
)

print("\nFinal model saved to:")
print(MODEL_FILE)

print("\n===== FINAL VOICE MODEL READY =====")