from pyexpat import features

import numpy as np
import pandas as pd
import librosa
import parselmouth
from scipy.stats import entropy
import nolds
from pyrpde import rpde
import parselmouth.praat

def safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.std(values))


def safe_min(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.min(values))


def safe_max(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.max(values))


def extract_pitch_features(sound, y, sr):
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    f0 = f0[np.isfinite(f0)]

    features = {
        "f0_mean": safe_mean(f0),
        "f0_std": safe_std(f0),
        "f0_min": safe_min(f0),
        "f0_max": safe_max(f0),
    }

    if len(f0) > 0:
        features["f0_range"] = float(np.max(f0) - np.min(f0))
        features["f0_median"] = float(np.median(f0))
    else:
        features["f0_range"] = np.nan
        features["f0_median"] = np.nan

    return features


def extract_jitter_shimmer_hnr(sound):
    features = {
        "jitter_local": np.nan,
        "jitter_rap": np.nan,
        "jitter_ppq5": np.nan,
        "shimmer_local": np.nan,
        "shimmer_db": np.nan,
        "shimmer_apq3": np.nan,
        "shimmer_apq5": np.nan,
        "shimmer_apq11": np.nan,
        "hnr": np.nan
    }

    try:
        point_process = sound.to_point_process_cc(
            75,
            500
        )

        features["jitter_local"] = point_process.get_jitter_local(
            0,
            0,
            0.0001,
            0.02,
            1.3
        )

        features["jitter_rap"] = point_process.get_jitter_rap(
            0,
            0,
            0.0001,
            0.02,
            1.3
        )

        features["jitter_ppq5"] = point_process.get_jitter_ppq5(
            0,
            0,
            0.0001,
            0.02,
            1.3
        )

        features["shimmer_local"] = point_process.get_shimmer_local(
            sound,
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6
        )

        features["shimmer_db"] = point_process.get_shimmer_local_db(
            sound,
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6
        )

        features["shimmer_apq3"] = point_process.get_shimmer_apq3(
            sound,
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6
        )

        features["shimmer_apq5"] = point_process.get_shimmer_apq5(
            sound,
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6
        )

        features["shimmer_apq11"] = point_process.get_shimmer_apq11(
            sound,
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6
        )

    except Exception:
        pass

    try:
        harmonicity = sound.to_harmonicity_cc(
            time_step=0.01,
            minimum_pitch=75,
            silence_threshold=0.1,
            periods_per_window=1.0
        )

        features["hnr"] = harmonicity.get_mean(
            0,
            0
        )

    except Exception:
        pass

    return features


def extract_mfcc_features(y, sr):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = {}

    for i in range(13):
        features[f"mfcc_{i + 1}_mean"] = float(
            np.mean(mfcc[i])
        )

        features[f"mfcc_{i + 1}_std"] = float(
            np.std(mfcc[i])
        )

        features[f"delta_mfcc_{i + 1}_mean"] = float(
            np.mean(delta[i])
        )

        features[f"delta_mfcc_{i + 1}_std"] = float(
            np.std(delta[i])
        )

        features[f"delta2_mfcc_{i + 1}_mean"] = float(
            np.mean(delta2[i])
        )

        features[f"delta2_mfcc_{i + 1}_std"] = float(
            np.std(delta2[i])
        )

    return features


def extract_spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y
    )[0]

    flux = librosa.feature.rms(
        y=y
    )[0]

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    features = {
        "spectral_centroid_mean": safe_mean(centroid),
        "spectral_centroid_std": safe_std(centroid),

        "spectral_bandwidth_mean": safe_mean(bandwidth),
        "spectral_bandwidth_std": safe_std(bandwidth),

        "spectral_rolloff_mean": safe_mean(rolloff),
        "spectral_rolloff_std": safe_std(rolloff),

        "spectral_flatness_mean": safe_mean(flatness),
        "spectral_flatness_std": safe_std(flatness),

        "spectral_contrast_mean": float(
            np.mean(contrast)
        ),
        "spectral_contrast_std": float(
            np.std(contrast)
        )
    }

    return features


def extract_formant_features(sound):
    features = {}

    try:
        formant = sound.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=5500,
            window_length=0.025,
            pre_emphasis_from=50
        )

        times = np.linspace(
            sound.xmin,
            sound.xmax,
            100
        )

        for formant_number in range(1, 5):
            values = []

            for time in times:
                value = formant.get_value_at_time(
                    formant_number,
                    time
                )

                if value is not None and np.isfinite(value):
                    values.append(value)

            features[
                f"f{formant_number}_mean"
            ] = safe_mean(values)

            features[
                f"f{formant_number}_std"
            ] = safe_std(values)

    except Exception:
        for formant_number in range(1, 5):
            features[
                f"f{formant_number}_mean"
            ] = np.nan

            features[
                f"f{formant_number}_std"
            ] = np.nan

    return features


def extract_energy_features(y):
    rms = librosa.feature.rms(
        y=y
    )[0]

    return {
        "rms_mean": safe_mean(rms),
        "rms_std": safe_std(rms),
        "rms_min": safe_min(rms),
        "rms_max": safe_max(rms)
    }


def extract_zcr_features(y):
    zcr = librosa.feature.zero_crossing_rate(
        y
    )[0]

    return {
        "zcr_mean": safe_mean(zcr),
        "zcr_std": safe_std(zcr)
    }



def extract_ppe(y, sr):
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    f0 = f0[np.isfinite(f0)]

    if len(f0) < 10:
        return {"ppe": np.nan}

    semitones = 12 * np.log2(f0 / np.median(f0))

    hist, _ = np.histogram(
        semitones,
        bins=50,
        density=False
    )

    hist = hist.astype(float)

    if hist.sum() == 0:
        return {"ppe": np.nan}

    probabilities = hist / hist.sum()
    probabilities = probabilities[probabilities > 0]

    return {
        "ppe": float(entropy(probabilities))
    }


def extract_rpde(y):
    try:
        signal = np.asarray(
            y,
            dtype=np.float32
        )

        peak = np.max(np.abs(signal))

        if peak == 0:
            return {"rpde": np.nan}

        signal = signal / peak

        value, _ = rpde(
            signal,
            tau=30,
            dim=4,
            epsilon=0.01,
            tmax=1500
        )

        return {
            "rpde": float(value)
        }

    except Exception:
        return {
            "rpde": np.nan
        }


def extract_dfa(y):
    try:
        signal = np.asarray(
            y,
            dtype=float
        )

        if len(signal) < 100:
            return {"dfa": np.nan}

        value = nolds.dfa(signal)

        return {
            "dfa": float(value)
        }

    except Exception:
        return {
            "dfa": np.nan
        }


def extract_gne(sound):
    try:
        harmonicity_gne = sound.to_harmonicity_gne(
            minimum_frequency=500,
            maximum_frequency=4500,
            bandwidth=1000,
            step=80
        )

        values = harmonicity_gne.values.flatten()

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            return {"gne": np.nan}

        values = values[values > -200]

        if len(values) == 0:
            return {"gne": np.nan}

        return {
            "gne": float(np.mean(values))
        }

    except Exception:
        return {
            "gne": np.nan
        }


def extract_cpp(sound):
    try:
        power_cepstrogram = sound.to_power_cepstrogram(
            pitch_floor=60,
            time_step=0.002,
            maximum_frequency=5000,
            pre_emphasis_from=50
        )

        cpp = parselmouth.praat.call(
            power_cepstrogram,
            "Get CPPS",
            0.01,
            0.001,
            60,
            330,
            0.05,
            "Parabolic",
            0.001,
            0.0,
            "Straight",
            "Robust"
        )

        if cpp is None or not np.isfinite(cpp):
            return {"cpps": np.nan}

        return {
            "cpps": float(cpp)
        }

    except Exception as e:
        print("CPPS extraction error:", e)
        return {
            "cpps": np.nan
        }

def extract_features(audio_path):
    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    y, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    if len(y) == 0:
        raise ValueError(
            "The audio file contains no usable audio."
        )

    sound = parselmouth.Sound(y, sampling_frequency=sr)

    features = {}

    features.update(
        extract_pitch_features(
            sound,
            y,
            sr
        )
    )

    features.update(
        extract_jitter_shimmer_hnr(
            sound
        )
    )

    features.update(
        extract_mfcc_features(
            y,
            sr
        )
    )

    features.update(
        extract_spectral_features(
            y,
            sr
        )
    )

    features.update(
        extract_formant_features(
            sound
        )
    )

    features.update(
        extract_energy_features(
            y
        )
    )

    features.update(
        extract_zcr_features(
            y
        )
    )

    features.update(
        extract_ppe(y, sr)
    )

    features.update(
        extract_rpde(y)
    )

    features.update(
        extract_dfa(y)
    )

    features.update(
        extract_gne(sound)
    )

    features.update(
        extract_cpp(sound)
    )

        
    return pd.DataFrame(
        [features]
    )


if __name__ == "__main__":
    print(
        "Feature extraction module ready."
    )