import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_testing")

sys.path.insert(0, MODEL_DIR)

from preprocessing import preprocess_audio
from feature_extraction import extract_features


audio_path = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "PD",
    "PD_AH",
    os.listdir(
        os.path.join(BASE_DIR, "data", "raw", "PD", "PD_AH")
    )[0]
)

print("\n===== PIPELINE TEST =====")
print("Testing:", audio_path)

y, sr = preprocess_audio(audio_path)

print("Preprocessed sample rate:", sr)
print("Preprocessed samples:", len(y))
print("Preprocessed duration:", len(y) / sr)

features = extract_features(audio_path)

print("\nNumber of extracted features:", len(features.columns))

print("\nFeatures:")
print(features.to_string(index=False))

print("\n===== TEST COMPLETE =====")