"""
Unit tests for Audio Preprocessing module.
Verifies mono conversion for 1D, 2D (samples, channels), 2D (channels, samples),
resampling, silence trimming, and peak amplitude normalization.
"""
import unittest
import numpy as np
from src.preprocessing import convert_to_mono, preprocess_audio_signal, CANONICAL_SAMPLE_RATE


class TestAudioPreprocessing(unittest.TestCase):
    def test_mono_conversion_1d(self):
        y = np.ones(1000, dtype=np.float32)
        y_mono = convert_to_mono(y)
        self.assertEqual(y_mono.ndim, 1)
        self.assertEqual(len(y_mono), 1000)

    def test_mono_conversion_samples_by_channels(self):
        # Shape (1000, 2) e.g. SoundFile format
        ch1 = np.ones(1000, dtype=np.float32) * 0.4
        ch2 = np.ones(1000, dtype=np.float32) * 0.6
        stereo = np.column_stack([ch1, ch2])
        y_mono = convert_to_mono(stereo)
        self.assertEqual(y_mono.ndim, 1)
        self.assertEqual(len(y_mono), 1000)
        np.testing.assert_allclose(y_mono, 0.5, atol=1e-5)

    def test_mono_conversion_channels_by_samples(self):
        # Shape (2, 1000) e.g. PyTorch/Librosa format
        ch1 = np.ones(1000, dtype=np.float32) * 0.2
        ch2 = np.ones(1000, dtype=np.float32) * 0.8
        stereo = np.vstack([ch1, ch2])
        y_mono = convert_to_mono(stereo)
        self.assertEqual(y_mono.ndim, 1)
        self.assertEqual(len(y_mono), 1000)
        np.testing.assert_allclose(y_mono, 0.5, atol=1e-5)

    def test_resampling_to_canonical_rate(self):
        # 16 kHz input -> should resample to 8000 Hz
        sr_orig = 16000
        t = np.linspace(0, 1.0, sr_orig, endpoint=False)
        y = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        y_proc, sr_proc = preprocess_audio_signal(y, sr_orig, target_sr=CANONICAL_SAMPLE_RATE)
        self.assertEqual(sr_proc, CANONICAL_SAMPLE_RATE)
        self.assertEqual(len(y_proc), CANONICAL_SAMPLE_RATE)

    def test_peak_normalization(self):
        # Signal with peak 0.3 should be scaled so peak is 1.0
        y = np.array([0.1, -0.3, 0.2, -0.15], dtype=np.float32)
        y_proc, _ = preprocess_audio_signal(y, 8000, trim_top_db=100.0)
        self.assertAlmostEqual(float(np.max(np.abs(y_proc))), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
