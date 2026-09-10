import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_depth=None,
        random_state=42
    ))
])

model.fit(X, y)

rf = model.named_steps["model"]

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n========================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("========================================")

print("\nTop 20 features:\n")
print(
    importance.head(20).to_string(index=False)
)

output_file = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "feature_importance.csv"
)

importance.to_csv(
    output_file,
    index=False
)

top = importance.head(20).sort_values(
    "importance"
)

plt.figure(figsize=(10, 8))
plt.barh(
    top["feature"],
    top["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Random Forest Voice Features")
plt.tight_layout()

plot_file = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "feature_importance.png"
)

plt.savefig(plot_file, dpi=300)
plt.show()

print("\nSaved:")
print(output_file)
print(plot_file)

print("\n===== DONE =====")