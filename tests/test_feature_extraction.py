"""
Unit tests for Feature Extraction module.
Verifies dimensionality, Nyquist compliance, determinism, and absence of NaNs.
"""
import unittest
import numpy as np
import pandas as pd
from src.feature_extraction import extract_canonical_features, FEATURE_NAMES


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.sr = 8000
        # 3.0s harmonic synthetic voice signal
        t = np.linspace(0, 3.0, int(3.0 * self.sr), endpoint=False)
        self.signal = (
            0.5 * np.sin(2 * np.pi * 140 * t) +
            0.25 * np.sin(2 * np.pi * 280 * t) +
            0.15 * np.sin(2 * np.pi * 700 * t) +
            0.1 * np.sin(2 * np.pi * 1200 * t)
        ).astype(np.float32)

    def test_feature_names_and_dimension(self):
        df = extract_canonical_features(self.signal, self.sr)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.shape[1], len(FEATURE_NAMES))
        self.assertListEqual(list(df.columns), FEATURE_NAMES)

    def test_no_nans_or_infs(self):
        df = extract_canonical_features(self.signal, self.sr)
        self.assertEqual(df.isna().sum().sum(), 0, "Feature extraction produced NaNs")
        self.assertEqual(np.isinf(df.values).sum(), 0, "Feature extraction produced Infs")

    def test_determinism(self):
        df1 = extract_canonical_features(self.signal, self.sr)
        df2 = extract_canonical_features(self.signal, self.sr)
        np.testing.assert_allclose(df1.values, df2.values, rtol=1e-5, atol=1e-5)

    def test_formants_nyquist_compliance(self):
        df = extract_canonical_features(self.signal, self.sr)
        # At 8kHz sr, Nyquist is 4000 Hz; formants should be capped below 3800 Hz
        self.assertLessEqual(df["f1_mean"].iloc[0], 3800)
        self.assertLessEqual(df["f2_mean"].iloc[0], 3800)


if __name__ == "__main__":
    unittest.main()
