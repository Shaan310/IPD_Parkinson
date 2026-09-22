import io
import os
import sys
import time

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_testing")
sys.path.insert(0, MODEL_DIR)

from feature_extraction import extract_features  # noqa: E402

MODEL_FILE = os.path.join(MODEL_DIR, "final_voice_model.joblib")

# Nested 5-fold CV results (inner 3-fold), from final_evaluation.py, n=81
CV_RESULTS = pd.DataFrame(
    [
        {"Model": "Random Forest", "Accuracy": 0.728, "AUC": 0.759},
        {"Model": "Logistic Regression", "Accuracy": 0.717, "AUC": 0.772},
        {"Model": "SVM", "Accuracy": 0.579, "AUC": 0.738},
    ]
).set_index("Model")

DATASET_SUMMARY = {
    "Total recordings": 81,
    "PwPD": 40,
    "HC": 41,
    "Features extracted": 122,
    "Selected model": "Random Forest",
}


st.set_page_config(
    page_title="Parkinson's Voice Screening",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


def align_features(feature_row: pd.DataFrame, model) -> pd.DataFrame:
    """Reindex extracted features to the column order the model was trained on."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return feature_row
    missing = [c for c in expected if c not in feature_row.columns]
    for col in missing:
        feature_row[col] = np.nan
    return feature_row[expected]


def render_sidebar():
    with st.sidebar:
        st.header("Parkinson's Voice Screening")
        st.caption("Voice-based screening demonstration, not a diagnostic tool.")

        st.subheader("How it works")
        st.markdown(
            "1. Upload or record a short voice sample\n"
            "2. Acoustic features are extracted (pitch, jitter, shimmer, MFCCs, "
            "and related measures)\n"
            "3. A Random Forest model, trained on a small labeled dataset, "
            "estimates how closely the recording matches PwPD or HC patterns"
        )

        st.subheader("Dataset")
        st.markdown(
            f"- {DATASET_SUMMARY['Total recordings']} recordings "
            f"({DATASET_SUMMARY['PwPD']} PwPD, {DATASET_SUMMARY['HC']} HC)\n"
            f"- {DATASET_SUMMARY['Features extracted']} acoustic features per recording\n"
            f"- Nested cross-validation AUC around 0.76"
        )

        st.divider()
        st.caption(
            "This is a course project demonstration built on a small dataset. "
            "It has not been clinically validated and must not be used to make "
            "or delay a medical decision."
        )


def plot_waveform(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4C8BF5")
    ax.set_xlabel("Time (s)", color="#AAAAAA")
    ax.set_ylabel("Amplitude", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_gauge(pd_probability: float):
    color = "#E05A5A" if pd_probability >= 0.5 else "#4CAF7D"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pd_probability * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#888888"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "rgba(76, 175, 125, 0.25)"},
                    {"range": [50, 100], "color": "rgba(224, 90, 90, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
            title={"text": "PD-pattern probability"},
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#DDDDDD"},
    )
    st.plotly_chart(fig, width="stretch")


def get_audio_input():
    """Returns (bytes, filename_hint) from either the upload or record tab, or None."""
    upload_tab, record_tab = st.tabs(["Upload file", "Record live"])

    with upload_tab:
        uploaded = st.file_uploader("WAV file", type=["wav"])
        if uploaded is not None:
            st.audio(uploaded)
            return uploaded.getvalue(), uploaded.name

    with record_tab:
        st.caption("Record a sustained vowel or a few seconds of speech.")
        recorded = st.audio_input("Record")
        if recorded is not None:
            return recorded.getvalue(), "recording.wav"

    return None, None


def render_prediction_tab(model):
    st.subheader("Voice sample")
    st.caption(
        "Sustained vowel or short speech sample. "
        "Feature extraction takes roughly 15 to 20 seconds."
    )

    audio_bytes, source_name = get_audio_input()

    if audio_bytes is None:
        st.info("Upload or record a .wav sample to run a prediction.")
        return

    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        with st.expander("Waveform preview", expanded=True):
            plot_waveform(y, sr)
    except Exception:
        y, sr = None, None

    if not st.button("Run prediction", type="primary"):
        return

    if y is None:
        st.error("Could not decode this audio file.")
        return

    tmp_path = os.path.join(BASE_DIR, "_uploaded_tmp.wav")
    sf.write(tmp_path, y, sr)

    try:
        with st.status("Running pipeline...", expanded=True) as status:
            status.write("Loading audio")
            time.sleep(0.2)

            status.write("Extracting acoustic features (pitch, jitter, shimmer, MFCCs)")
            t0 = time.time()
            features = extract_features(tmp_path)
            elapsed = time.time() - t0

            status.write("Running model")
            features_aligned = align_features(features.copy(), model)
            proba = model.predict_proba(features_aligned)[0]
            pd_probability = float(proba[1])

            status.update(label="Done", state="complete", expanded=False)

        n_missing = int(features.isna().sum().sum())
        prediction = "PwPD signal pattern" if pd_probability >= 0.5 else "HC signal pattern"

        st.divider()

        col1, col2 = st.columns([1, 1])
        with col1:
            render_gauge(pd_probability)
        with col2:
            st.metric("Model output", prediction)
            st.metric("Extraction time", f"{elapsed:.1f} s")

        if n_missing > 0:
            st.warning(
                f"{n_missing} feature(s) could not be computed from this recording "
                "and were imputed with training-set medians. Results may be less reliable."
            )

        with st.expander("What this number means"):
            st.markdown(
                "This is the probability the model assigns to the recording's acoustic "
                "pattern resembling the PwPD group in the training data, not a clinical "
                "probability of having Parkinson's disease. The model was trained on 81 "
                "recordings and has not been clinically validated. It is a screening "
                "demonstration, not a diagnostic tool, and should not be used to make or "
                "delay a medical decision."
            )

        with st.expander("Extracted feature values"):
            st.dataframe(features.T.rename(columns={0: "value"}), width="stretch")

    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def render_results_tab(model):
    st.subheader("Dataset")
    cols = st.columns(len(DATASET_SUMMARY))
    for col, (label, value) in zip(cols, DATASET_SUMMARY.items()):
        col.metric(label, value)

    st.subheader("Model comparison, nested cross-validation")
    st.caption("5 outer folds, 3 inner folds, stratified. n = 81.")
    st.dataframe(
        CV_RESULTS.style.format({"Accuracy": "{:.1%}", "AUC": "{:.3f}"}),
        width="stretch",
    )
    st.bar_chart(CV_RESULTS)

    if model is not None and hasattr(model, "named_steps"):
        rf = model.named_steps.get("model")
        feature_names = getattr(model, "feature_names_in_", None)
        if rf is not None and hasattr(rf, "feature_importances_") and feature_names is not None:
            st.subheader("Feature importance, Random Forest")
            st.caption(
                "Impurity-based importance from the final model. Global, not specific "
                "to any single prediction."
            )
            importances = pd.Series(
                rf.feature_importances_, index=feature_names
            ).sort_values(ascending=False).head(15)
            st.bar_chart(importances)

    st.caption(
        "This dataset has 81 recordings and is not clinically validated. "
        "Figures here summarize model behavior on this dataset only."
    )


def main():
    render_sidebar()

    st.title("Parkinson's Disease Voice Screening")
    st.caption(
        "Voice-based screening demonstration. Not a diagnostic tool. "
        "Consult a medical professional for any health concern."
    )

    model = load_model()

    if model is None:
        st.error(
            f"Model file not found at {MODEL_FILE}. "
            "Train it with model_testing/final_voice_model.py and place the "
            ".joblib file in model_testing/, then reload this page."
        )

    tab1, tab2 = st.tabs(["Live prediction", "Model results"])
    with tab1:
        if model is not None:
            render_prediction_tab(model)
        else:
            st.info("Live prediction is unavailable until the model file is present.")
    with tab2:
        render_results_tab(model)


if __name__ == "__main__":
    main()