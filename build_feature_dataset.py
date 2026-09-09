import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_testing")

sys.path.insert(0, MODEL_DIR)

from feature_extraction import extract_features


DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

PD_DIR = os.path.join(DATA_DIR, "PD")
HC_DIR = os.path.join(DATA_DIR, "HC")

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "voice_features.csv"
)


records = []


def process_folder(folder, label):
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            audio_path = os.path.join(root, file)

            print(f"Processing: {file}")

            try:
                features = extract_features(audio_path)

                row = features.iloc[0].to_dict()

                row["sample_id"] = os.path.splitext(file)[0]
                row["label"] = label

                records.append(row)

            except Exception as e:
                print(f"ERROR: {file}")
                print(e)


print("\n===== BUILDING FEATURE DATASET =====\n")

process_folder(PD_DIR, "PwPD")
process_folder(HC_DIR, "HC")


df = pd.DataFrame(records)

first_columns = ["sample_id", "label"]

feature_columns = [
    col for col in df.columns
    if col not in first_columns
]

df = df[first_columns + feature_columns]

df.to_csv(OUTPUT_FILE, index=False)


print("\n===== DATASET COMPLETE =====")

print(f"Total recordings: {len(df)}")
print(f"PD recordings: {(df['label'] == 'PwPD').sum()}")
print(f"HC recordings: {(df['label'] == 'HC').sum()}")
print(f"Total features: {len(feature_columns)}")

print("\nMissing values per feature:")

missing = df[feature_columns].isna().sum()

print(
    missing[missing > 0]
    .sort_values(ascending=False)
    .to_string()
)

print("\nTotal missing values:", int(missing.sum()))

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\n===== DONE =====")