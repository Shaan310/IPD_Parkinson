"""
Canonical Audio Preprocessing Module.
Ensures identical audio conditioning across training and inference.
Handles mono, multichannel (both (samples, channels) and (channels, samples)),
resampling to canonical 8 kHz, silence trimming, and peak normalization.
"""
from typing import Tuple, Union, Optional
import os
import io
import numpy as np
import librosa
import soundfile as sf

CANONICAL_SAMPLE_RATE = 8000


def convert_to_mono(y: np.ndarray) -> np.ndarray:
    """
    Robustly converts 1D or 2D audio arrays to 1D mono.
    Correctly identifies the channel axis for both:
      - (samples, channels) e.g. SoundFile format
      - (channels, samples) e.g. Librosa/PyTorch format
    """
    if y is None or len(y) == 0:
        raise ValueError("Audio array is empty.")

    y = np.squeeze(y)
    if y.ndim == 1:
        return y.astype(np.float32)

    if y.ndim == 2:
        # Determine channel axis: channels is almost always the smaller dimension (e.g. 2 channels vs 24000 samples)
        if y.shape[0] > y.shape[1]:
            # Shape is (samples, channels) -> average along axis 1
            y_mono = np.mean(y, axis=1)
        else:
            # Shape is (channels, samples) -> average along axis 0
            y_mono = np.mean(y, axis=0)
        return y_mono.astype(np.float32)

    # For 3D or higher, flatten to 1D
    y_mono = np.mean(y, axis=tuple(range(y.ndim - 1)))
    return y_mono.astype(np.float32)


def preprocess_audio_signal(
    y: np.ndarray,
    sr: int,
    target_sr: int = CANONICAL_SAMPLE_RATE,
    trim_top_db: float = 25.0
) -> Tuple[np.ndarray, int]:
    """
    Conditions an audio signal into the canonical format:
    1. Robust mono conversion
    2. Resampling to target_sr (8,000 Hz)
    3. Silence trimming
    4. Peak amplitude normalization to [-1.0, 1.0]
    """
    if y is None or len(y) == 0:
        raise ValueError("Audio array is empty.")

    # 1. Convert to mono
    y = convert_to_mono(y)

    # Convert to float32
    y = y.astype(np.float32)

    # 2. Resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 3. Trim leading and trailing silence
    if len(y) > 0:
        y_trimmed, _ = librosa.effects.trim(y, top_db=trim_top_db)
        if len(y_trimmed) >= int(0.5 * sr):
            y = y_trimmed

    # 4. Peak amplitude normalization to [-1.0, 1.0]
    max_amp = float(np.max(np.abs(y)))
    if max_amp > 1e-6:
        y = y / max_amp

    return y, sr


def preprocess_audio(
    file_or_bytes: Union[str, bytes, io.BytesIO],
    target_sr: int = CANONICAL_SAMPLE_RATE,
    trim_top_db: float = 25.0
) -> Tuple[np.ndarray, int]:
    """
    Loads and preprocesses audio from file path or byte stream.
    Tries soundfile first, with librosa fallback.
    """
    if isinstance(file_or_bytes, bytes):
        file_obj = io.BytesIO(file_or_bytes)
    elif isinstance(file_or_bytes, io.BytesIO):
        file_obj = file_or_bytes
    else:
        file_obj = file_or_bytes

    try:
        y, sr = sf.read(file_obj)
    except Exception:
        if isinstance(file_obj, io.BytesIO):
            file_obj.seek(0)
        y, sr = librosa.load(file_obj, sr=None, mono=False)

    return preprocess_audio_signal(y, sr, target_sr=target_sr, trim_top_db=trim_top_db)
