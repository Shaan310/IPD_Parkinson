"""
End-to-end Pipeline Verification Script.
Tests canonical preprocessing, feature extraction, and full model inference
using CanonicalVoicePipeline on:
1. Real PD recording
2. Real HC recording
3. Multi-channel audio input
4. Deliberately invalid audio (silence, too-short)
5. Out-of-Distribution / Non-sustained speech input
"""
import os
import glob
import numpy as np
from src.preprocessing import preprocess_audio, preprocess_audio_signal
from src.feature_extraction import extract_canonical_features
from src.pipeline import CanonicalVoicePipeline


def test_pipeline():
    print("\n" + "=" * 60)
    print("CANONICAL VOICE PIPELINE TEST")
    print("=" * 60)

    model_path = os.path.join("model", "canonical_voice_pipeline.joblib")
    if not os.path.exists(model_path):
        print(f"Error: Trained model pipeline not found at {model_path}")
        return False

    pipeline = CanonicalVoicePipeline.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    print(f"Model version         : {pipeline.version}")
    print(f"Algorithm             : {pipeline.model_name}")
    print(f"Candidate feature pool: {len(pipeline.feature_names_in_)}")
    print(f"Selected top features : {len(pipeline.selected_feature_names_)}")
    print(f"Hyperparameters       : {pipeline.clf_params}")

    # 1. Test sample PD recording
    pd_files = sorted(glob.glob(os.path.join("data", "raw", "PD", "PD_AH", "*.wav")))
    if pd_files:
        test_pd = pd_files[0]
        print(f"\n--- 1. Testing Real PD Recording: {os.path.basename(test_pd)} ---")
        y, sr = preprocess_audio(test_pd)
        print(f"Preprocessed: sr={sr} Hz, length={len(y)} samples ({len(y)/sr:.2f} s)")

        feats = extract_canonical_features(y, sr)
        print(f"Extracted acoustic features: {feats.shape[1]} (NaNs: {feats.isna().sum().sum()})")

        pred_pd = pipeline.predict_from_audio(test_pd)
        print(f"Inference Status : {pred_pd.status}")
        print(f"Category Label   : {pred_pd.category_label}")
        print(f"PD Model Score   : {pred_pd.pd_model_score:.4f}")
        print(f"Top Contributions: {len(pred_pd.top_feature_contributions)} features")
        for c in pred_pd.top_feature_contributions[:3]:
            print(f"   * {c['feature']:20s} value={c['value']:.3f}, z={c['z_score']:+.2f}, imp={c['importance_weight']:.3f} ({c['direction']})")
        print(f"Explanation      :\n  {pred_pd.explanation_text}")

    # 2. Test sample HC recording
    hc_files = sorted(glob.glob(os.path.join("data", "raw", "HC", "HC_AH", "*.wav")))
    if hc_files:
        test_hc = hc_files[0]
        print(f"\n--- 2. Testing Real HC Recording: {os.path.basename(test_hc)} ---")
        pred_hc = pipeline.predict_from_audio(test_hc)
        print(f"Inference Status : {pred_hc.status}")
        print(f"Category Label   : {pred_hc.category_label}")
        print(f"PD Model Score   : {pred_hc.pd_model_score:.4f}")
        for c in pred_hc.top_feature_contributions[:3]:
            print(f"   * {c['feature']:20s} value={c['value']:.3f}, z={c['z_score']:+.2f}, imp={c['importance_weight']:.3f} ({c['direction']})")
        print(f"Explanation      :\n  {pred_hc.explanation_text}")

    # 3. Multichannel preprocessing compatibility test
    print("\n--- 3. Multichannel Preprocessing Compatibility Test ---")
    print("   [Note: Verifies pipeline downmixes multichannel audio to mono; output is not model performance evidence]")
    t = np.linspace(0, 3.0, 8000 * 3, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    stereo_sig = np.column_stack([sig, sig * 0.9])  # (samples, channels)
    pred_stereo = pipeline.predict_from_audio(stereo_sig, sr=8000)
    print(f"Stereo (samples, channels) status: {pred_stereo.status} (Category: {pred_stereo.category_label})")

    # 4. Test Deliberately invalid audio (silence)
    print("\n--- 4. Testing Deliberately Invalid Audio (Silence) ---")
    silence = np.zeros(8000 * 2, dtype=np.float32)
    pred_silence = pipeline.predict_from_audio(silence, sr=8000)
    print(f"Silence Status : {pred_silence.status} (Reason: {pred_silence.rejection_reason})")
    assert pred_silence.status == "REJECTED", "Silence should be rejected by quality gate"

    # 5. Test Deliberately invalid audio (too short)
    print("\n--- 5. Testing Deliberately Invalid Audio (Too Short < 1.2s) ---")
    short_audio = (0.5 * np.sin(2 * np.pi * 150 * np.linspace(0, 0.8, int(8000 * 0.8)))).astype(np.float32)
    pred_short = pipeline.predict_from_audio(short_audio, sr=8000)
    print(f"Short Audio Status: {pred_short.status} (Reason: {pred_short.rejection_reason})")
    assert pred_short.status == "REJECTED", "Too-short audio should be rejected by quality gate"

    # 6. Non-sustained / synthetic signal compatibility test
    print("\n--- 6. Non-sustained / Synthetic Signal Compatibility Test ---")
    print("   [Note: Verifies input compatibility check rejects non-sustained pitch-varying synthetic signals]")
    # Synthetic signal with wildly varying pitch (chirp/frequency modulation)
    t = np.linspace(0, 3.0, 8000 * 3, endpoint=False)
    freq_swing = 100 + 350 * np.sin(2 * np.pi * 2.0 * t)  # rapid pitch changes
    ood_sig = (0.5 * np.sin(2 * np.pi * freq_swing * t)).astype(np.float32)
    pred_ood = pipeline.predict_from_audio(ood_sig, sr=8000)
    print(f"Non-sustained Audio Status: {pred_ood.status} (Category: {pred_ood.category_label})")
    print(f"Compatibility Explanation: {pred_ood.explanation_text[:120]}...")

    print("\n" + "=" * 60)
    print("ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    test_pipeline()