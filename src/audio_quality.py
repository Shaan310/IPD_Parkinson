"""
Inference Audio Quality Gate: Validates audio recordings prior to feature extraction.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import librosa


from src.preprocessing import convert_to_mono


@dataclass
class QualityGateResult:
    is_valid: bool
    reason: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


def evaluate_audio_quality(
    y: np.ndarray,
    sr: int,
    min_duration: float = 1.2,
    max_duration: float = 10.0,
    max_clipping_ratio: float = 0.02,
    min_rms_energy: float = 0.008,
) -> QualityGateResult:
    """
    Evaluates whether an audio signal meets technical screening standards.
    Checks duration, digital clipping, peak amplitude, and signal activity.
    Note: This is an Audio Quality & Signal Activity Check, not a clinical VAD.
    """
    metrics = {}

    if y is None or len(y) == 0:
        return QualityGateResult(
            is_valid=False,
            reason="Audio recording is empty or unreadable.",
            metrics={"duration": 0.0}
        )

    # Ensure mono for quality assessment
    try:
        y = convert_to_mono(y)
    except Exception as exc:
        return QualityGateResult(
            is_valid=False,
            reason=f"Failed to process audio channels: {exc}",
            metrics={"duration": 0.0}
        )

    if not np.all(np.isfinite(y)):
        return QualityGateResult(
            is_valid=False,
            reason="Audio signal contains invalid non-numeric values (NaN/Inf).",
            metrics={"duration": len(y) / sr if sr > 0 else 0.0}
        )

    duration = float(len(y) / sr)
    metrics["raw_duration_sec"] = round(duration, 3)

    if duration < min_duration:
        return QualityGateResult(
            is_valid=False,
            reason=(
                f"Audio duration ({duration:.2f} s) is too short. "
                f"A sustained vowel phonation of at least {min_duration:.1f} seconds is required."
            ),
            metrics=metrics
        )

    if duration > max_duration:
        return QualityGateResult(
            is_valid=False,
            reason=(
                f"Audio duration ({duration:.2f} s) exceeds maximum allowed length ({max_duration:.1f} s). "
                "Please provide a 3 to 5 second sustained vowel sample."
            ),
            metrics=metrics
        )

    # Peak amplitude and clipping check
    peak_amp = float(np.max(np.abs(y)))
    metrics["peak_amplitude"] = round(peak_amp, 4)

    if peak_amp < 1e-4:
        return QualityGateResult(
            is_valid=False,
            reason="Audio is completely silent (peak amplitude below audible threshold).",
            metrics=metrics
        )

    # Clipping estimation (fraction of samples near +/- 1.0)
    clip_thresh = 0.995 * (peak_amp if peak_amp > 1.0 else 1.0)
    clipping_ratio = float(np.mean(np.abs(y) >= clip_thresh))
    metrics["clipping_ratio"] = round(clipping_ratio, 5)

    if clipping_ratio > max_clipping_ratio:
        return QualityGateResult(
            is_valid=False,
            reason=(
                f"Severe audio clipping detected ({clipping_ratio * 100:.1f}% of samples saturated). "
                "Please lower microphone input gain or move slightly further from the microphone."
            ),
            metrics=metrics
        )

    # RMS Energy check
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    metrics["mean_rms_energy"] = round(mean_rms, 5)

    if mean_rms < min_rms_energy:
        return QualityGateResult(
            is_valid=False,
            reason=(
                f"Audio energy is too low (RMS = {mean_rms:.4f} < {min_rms_energy:.4f}). "
                "The recording contains ambient silence or is too quiet to extract vocal features reliably."
            ),
            metrics=metrics
        )

    # Check for active vocal content (non-silent fraction)
    non_silent_frames = np.sum(rms > (mean_rms * 0.2))
    non_silent_ratio = float(non_silent_frames / len(rms)) if len(rms) > 0 else 0.0
    metrics["voiced_ratio"] = round(non_silent_ratio, 3)

    if non_silent_ratio < 0.35:
        return QualityGateResult(
            is_valid=False,
            reason=(
                "Insufficient continuous voice activity detected in the recording. "
                "Please sustain a steady 'aaah' sound without long pauses."
            ),
            metrics=metrics
        )

    return QualityGateResult(
        is_valid=True,
        reason=None,
        metrics=metrics
    )
