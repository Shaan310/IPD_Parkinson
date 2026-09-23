"""
Unit tests for Audio Quality Gate.
Verifies rejection of short, silent, clipped, and unvoiced signals,
and acceptance of valid sustained vowels.
"""
import unittest
import numpy as np
from src.audio_quality import evaluate_audio_quality, QualityGateResult


class TestAudioQualityGate(unittest.TestCase):
    def setUp(self):
        self.sr = 8000

    def test_short_audio_rejection(self):
        # 0.8 seconds (under 1.2s threshold)
        y = np.sin(2 * np.pi * 150 * np.linspace(0, 0.8, int(0.8 * self.sr), endpoint=False)).astype(np.float32)
        res = evaluate_audio_quality(y, self.sr)
        self.assertFalse(res.is_valid)
        self.assertIn("short", res.reason.lower())

    def test_pure_silence_rejection(self):
        # 2.5 seconds of zeros
        y = np.zeros(int(2.5 * self.sr), dtype=np.float32)
        res = evaluate_audio_quality(y, self.sr)
        self.assertFalse(res.is_valid)
        self.assertTrue("silent" in res.reason.lower() or "silence" in res.reason.lower() or "energy" in res.reason.lower())

    def test_extreme_clipping_rejection(self):
        # 2.5 seconds clipped at max amplitude (> 10% clipped)
        t = np.linspace(0, 2.5, int(2.5 * self.sr), endpoint=False)
        y = np.clip(5.0 * np.sin(2 * np.pi * 200 * t), -0.999, 0.999).astype(np.float32)
        res = evaluate_audio_quality(y, self.sr)
        self.assertFalse(res.is_valid)
        self.assertIn("clipping", res.reason.lower())

    def test_valid_sustained_phonation_acceptance(self):
        # 3.0 seconds synthetic sustained harmonic vowel /a/ at 150 Hz
        t = np.linspace(0, 3.0, int(3.0 * self.sr), endpoint=False)
        # Fundamental + 3 harmonics
        y = (
            0.5 * np.sin(2 * np.pi * 150 * t) +
            0.3 * np.sin(2 * np.pi * 300 * t) +
            0.15 * np.sin(2 * np.pi * 450 * t)
        ).astype(np.float32)
        res = evaluate_audio_quality(y, self.sr)
        self.assertTrue(res.is_valid)
        self.assertIsNone(res.reason)
        self.assertGreater(res.metrics["raw_duration_sec"], 1.2)


if __name__ == "__main__":
    unittest.main()
