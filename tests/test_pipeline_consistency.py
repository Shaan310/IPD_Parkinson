"""
Integration tests for CanonicalVoicePipeline consistency.
Verifies model loading, serialization consistency, end-to-end audio inference,
feature contribution fields, input compatibility check, and quality rejection.
"""
import unittest
import os
import glob
import numpy as np
from src.pipeline import CanonicalVoicePipeline, PredictionOutput


class TestPipelineConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join("model", "canonical_voice_pipeline.joblib")
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Model file not found at {cls.model_path}")
        cls.pipeline = CanonicalVoicePipeline.load(cls.model_path)

    def test_pipeline_loaded_attributes(self):
        self.assertIsNotNone(self.pipeline.sklearn_pipeline)
        self.assertIsNotNone(self.pipeline.feature_names_in_)
        self.assertEqual(len(self.pipeline.feature_names_in_), 45)
        self.assertEqual(len(self.pipeline.selected_feature_names_), 10)
        self.assertIsNotNone(self.pipeline.reference_feature_stats_)

    def test_inference_on_real_pd_recording(self):
        pd_files = glob.glob(os.path.join("data", "raw", "PD", "PD_AH", "*.wav"))
        if not pd_files:
            self.skipTest("No PD audio files found.")
        audio_file = pd_files[0]
        output = self.pipeline.predict_from_audio(audio_file)

        self.assertIsInstance(output, PredictionOutput)
        self.assertIn(output.status, ["CONFIDENT_PD", "CONFIDENT_HC", "INCONCLUSIVE"])
        self.assertGreaterEqual(output.pd_model_score, 0.0)
        self.assertLessEqual(output.pd_model_score, 1.0)
        self.assertGreaterEqual(output.confidence_score, 0.0)
        self.assertLessEqual(output.confidence_score, 1.0)
        self.assertGreater(len(output.top_feature_contributions), 0)
        self.assertIsInstance(output.explanation_text, str)
        self.assertGreater(len(output.explanation_text), 10)

        # Verify Streamlit feature contribution keys exist
        first_contrib = output.top_feature_contributions[0]
        self.assertIn("feature", first_contrib)
        self.assertIn("value", first_contrib)
        self.assertIn("z_score", first_contrib)
        self.assertIn("importance_weight", first_contrib)
        self.assertIn("direction", first_contrib)

    def test_inference_on_real_hc_recording(self):
        hc_files = glob.glob(os.path.join("data", "raw", "HC", "HC_AH", "*.wav"))
        if not hc_files:
            self.skipTest("No HC audio files found.")
        audio_file = hc_files[0]
        output = self.pipeline.predict_from_audio(audio_file)

        self.assertIsInstance(output, PredictionOutput)
        self.assertIn(output.status, ["CONFIDENT_PD", "CONFIDENT_HC", "INCONCLUSIVE"])
        self.assertGreaterEqual(output.pd_model_score, 0.0)
        self.assertLessEqual(output.pd_model_score, 1.0)

    def test_multichannel_audio_inference(self):
        # 3.0s stereo signal
        t = np.linspace(0, 3.0, 8000 * 3, endpoint=False)
        sig = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        stereo = np.column_stack([sig, sig * 0.9])
        output = self.pipeline.predict_from_audio(stereo, sr=8000)
        self.assertIn(output.status, ["CONFIDENT_PD", "CONFIDENT_HC", "INCONCLUSIVE"])

    def test_rejection_on_silent_audio(self):
        silence = np.zeros(8000 * 2, dtype=np.float32)
        output = self.pipeline.predict_from_audio(silence, sr=8000)
        self.assertEqual(output.status, "REJECTED")
        self.assertTrue(output.is_inconclusive)
        self.assertIsNotNone(output.rejection_reason)

    def test_input_compatibility_ood_check(self):
        # Non-sustained / synthetic signal compatibility test (frequency modulation)
        t = np.linspace(0, 3.0, 8000 * 3, endpoint=False)
        freq_swing = 100 + 350 * np.sin(2 * np.pi * 2.0 * t)
        ood_sig = (0.5 * np.sin(2 * np.pi * freq_swing * t)).astype(np.float32)
        output = self.pipeline.predict_from_audio(ood_sig, sr=8000)
        self.assertEqual(output.status, "INCONCLUSIVE")
        self.assertIn(output.category_label, ["Inconclusive", "Unsupported Recording Characteristics"])
        self.assertTrue(output.is_inconclusive)
        self.assertIn("falls outside the acoustic range", output.explanation_text)

    def test_indeterminate_threshold_logic(self):
        self.assertEqual(self.pipeline.low_threshold, 0.40)
        self.assertEqual(self.pipeline.high_threshold, 0.60)


if __name__ == "__main__":
    unittest.main()
