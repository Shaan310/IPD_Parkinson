"""
Canonical Feature Extraction Module.
Extracts physiologically grounded, Nyquist-compliant acoustic features
for Parkinson's voice analysis from sustained vowel phonation (/a/).
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import librosa
from scipy.stats import entropy
import parselmouth

from src.preprocessing import preprocess_audio_signal, preprocess_audio, CANONICAL_SAMPLE_RATE

FEATURE_NAMES = [
    "f0_mean", "f0_std", "f0_cv",
    "jitter_local", "jitter_rap", "jitter_ppq5",
    "shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5",
    "hnr",
    "f1_mean", "f1_std", "f2_mean", "f2_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_rolloff_mean", "spectral_flatness_mean",
    "zcr_mean", "rms_mean", "rms_std",
    "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean", "mfcc_5_mean",
    "mfcc_6_mean", "mfcc_7_mean", "mfcc_8_mean", "mfcc_9_mean", "mfcc_10_mean",
    "mfcc_11_mean", "mfcc_12_mean", "mfcc_13_mean",
    "mfcc_1_std", "mfcc_2_std", "mfcc_3_std", "mfcc_4_std",
    "delta_mfcc_1_mean", "delta_mfcc_2_mean", "delta_mfcc_3_mean", "delta_mfcc_4_mean",
    "ppe"
]


def safe_stat(arr: np.ndarray, func) -> float:
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.nan
    return float(func(valid))


def extract_pitch_features(sound: parselmouth.Sound) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Extracts fundamental frequency (F0) using Praat's autocorrelation algorithm.
    Range 75 to 500 Hz (suitable for human phonation).
    """
    features = {
        "f0_mean": np.nan,
        "f0_std": np.nan,
        "f0_cv": np.nan,
    }
    try:
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0_raw = pitch.selected_array["frequency"]
        f0_voiced = f0_raw[f0_raw > 0]
        if len(f0_voiced) >= 5:
            m = float(np.mean(f0_voiced))
            s = float(np.std(f0_voiced))
            features["f0_mean"] = m
            features["f0_std"] = s
            features["f0_cv"] = (s / m) if m > 0 else np.nan
            return features, f0_voiced
    except Exception:
        pass
    return features, np.array([])


def extract_jitter_shimmer_hnr(sound: parselmouth.Sound) -> Dict[str, float]:
    """
    Extracts cycle-to-cycle frequency (jitter) and amplitude (shimmer) perturbation,
    plus Harmonics-to-Noise Ratio (HNR) via Praat PointProcess and Harmonicity.
    """
    feats = {
        "jitter_local": np.nan,
        "jitter_rap": np.nan,
        "jitter_ppq5": np.nan,
        "shimmer_local": np.nan,
        "shimmer_db": np.nan,
        "shimmer_apq3": np.nan,
        "shimmer_apq5": np.nan,
        "hnr": np.nan,
    }
    try:
        pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
        feats["jitter_local"] = float(parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        feats["jitter_rap"] = float(parselmouth.praat.call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
        feats["jitter_ppq5"] = float(parselmouth.praat.call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))

        feats["shimmer_local"] = float(parselmouth.praat.call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats["shimmer_db"] = float(parselmouth.praat.call([sound, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats["shimmer_apq3"] = float(parselmouth.praat.call([sound, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats["shimmer_apq5"] = float(parselmouth.praat.call([sound, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        pass

    try:
        harm = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        feats["hnr"] = float(parselmouth.praat.call(harm, "Get mean", 0, 0))
    except Exception:
        pass

    return feats


def extract_formants_8k(sound: parselmouth.Sound) -> Dict[str, float]:
    """
    Extracts vowel formants (F1, F2) using Burg algorithm respecting 8 kHz Nyquist limit (4000 Hz).
    Maximum formant ceiling set to 3800 Hz.
    """
    feats = {
        "f1_mean": np.nan,
        "f1_std": np.nan,
        "f2_mean": np.nan,
        "f2_std": np.nan,
    }
    try:
        formant = sound.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=3,
            maximum_formant=3800,
            window_length=0.025,
            pre_emphasis_from=50
        )
        f1_m = parselmouth.praat.call(formant, "Get mean", 1, 0, 0, "Hertz")
        f1_s = parselmouth.praat.call(formant, "Get standard deviation", 1, 0, 0, "Hertz")
        f2_m = parselmouth.praat.call(formant, "Get mean", 2, 0, 0, "Hertz")
        f2_s = parselmouth.praat.call(formant, "Get standard deviation", 2, 0, 0, "Hertz")

        if f1_m is not None and np.isfinite(f1_m):
            feats["f1_mean"] = float(f1_m)
        if f1_s is not None and np.isfinite(f1_s):
            feats["f1_std"] = float(f1_s)
        if f2_m is not None and np.isfinite(f2_m):
            feats["f2_mean"] = float(f2_m)
        if f2_s is not None and np.isfinite(f2_s):
            feats["f2_std"] = float(f2_s)
    except Exception:
        pass

    return feats


def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extracts frequency-domain spectral characteristics.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]

    return {
        "spectral_centroid_mean": safe_stat(centroid, np.mean),
        "spectral_centroid_std": safe_stat(centroid, np.std),
        "spectral_bandwidth_mean": safe_stat(bandwidth, np.mean),
        "spectral_rolloff_mean": safe_stat(rolloff, np.mean),
        "spectral_flatness_mean": safe_stat(flatness, np.mean),
        "zcr_mean": safe_stat(zcr, np.mean),
        "rms_mean": safe_stat(rms, np.mean),
        "rms_std": safe_stat(rms, np.std),
    }


def extract_compact_mfccs(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extracts compact MFCC representation: 13 mean coefficients + 4 variance + 4 delta means.
    """
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)

    # 13 MFCC means
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = safe_stat(mfcc[i], np.mean)

    # Top 4 MFCC standard deviations
    for i in range(4):
        feats[f"mfcc_{i+1}_std"] = safe_stat(mfcc[i], np.std)

    # Top 4 Delta MFCC means
    for i in range(4):
        feats[f"delta_mfcc_{i+1}_mean"] = safe_stat(delta[i], np.mean)

    return feats


def extract_ppe(f0_voiced: np.ndarray) -> Dict[str, float]:
    """
    Computes Pitch Period Entropy (PPE) from voiced F0 observations.
    Quantifies logarithmic pitch jitter/fluctuations.
    """
    if len(f0_voiced) < 10:
        return {"ppe": np.nan}
    try:
        med = float(np.median(f0_voiced))
        if med <= 0:
            return {"ppe": np.nan}
        semitones = 12 * np.log2(f0_voiced / med)
        hist, _ = np.histogram(semitones, bins=30, density=False)
        total = hist.sum()
        if total == 0:
            return {"ppe": np.nan}
        p = hist.astype(float) / total
        p = p[p > 0]
        return {"ppe": float(entropy(p))}
    except Exception:
        return {"ppe": np.nan}


def extract_canonical_features(y: np.ndarray, sr: int) -> pd.DataFrame:
    """
    Canonical feature extraction pipeline.
    Returns a single-row DataFrame containing all canonical acoustic features.
    """
    features = {}

    sound = parselmouth.Sound(y, sampling_frequency=sr)

    # 1. Pitch
    pitch_feats, f0_voiced = extract_pitch_features(sound)
    features.update(pitch_feats)

    # 2. Jitter, Shimmer, HNR
    features.update(extract_jitter_shimmer_hnr(sound))

    # 3. Formants
    features.update(extract_formants_8k(sound))

    # 4. Spectral & Dynamics
    features.update(extract_spectral_features(y, sr))

    # 5. Compact MFCCs
    features.update(extract_compact_mfccs(y, sr))

    # 6. PPE
    features.update(extract_ppe(f0_voiced))

    df = pd.DataFrame([features])
    return df


def extract_features_from_audio(audio_input) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full extraction from raw input (file path, bytes, or numpy array).
    Applies canonical preprocessing then canonical feature extraction.
    Returns (feature_df, metadata_dict).
    """
    if isinstance(audio_input, (str, bytes)):
        y, sr = preprocess_audio(audio_input)
    elif isinstance(audio_input, tuple):
        y, sr = preprocess_audio_signal(audio_input[0], audio_input[1])
    else:
        raise ValueError("Unsupported audio input type.")

    feature_df = extract_canonical_features(y, sr)

    n_missing = int(feature_df.isna().sum().sum())
    missing_cols = feature_df.columns[feature_df.isna().iloc[0]].tolist()

    meta = {
        "preprocessed_sr": sr,
        "preprocessed_samples": len(y),
        "duration_sec": round(len(y) / sr, 2),
        "total_features": feature_df.shape[1],
        "missing_features_count": n_missing,
        "missing_features": missing_cols,
    }

    return feature_df, meta
