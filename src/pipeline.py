"""
Canonical End-to-End Parkinson's Voice Inference Pipeline.
Provides a unified, reproducible interface across training, validation, and Streamlit inference.

Includes:
1. Audio Quality & Signal Activity Gate (duration, clipping, RMS, voiced ratio)
2. Canonical Audio Preprocessing (mono conversion, 8 kHz resampling, trimming, peak normalization)
3. Nyquist-Compliant Feature Extraction (45 acoustic features)
4. Input Compatibility / Out-of-Distribution (OOD) Screening
5. Scikit-Learn Pipeline (Imputer + Scaler + SelectKBest + GradientBoostingClassifier)
6. Indeterminate Model-Score Band ([0.40, 0.60])
7. Transparent Feature Contribution Breakdown
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
import os
import io
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

from src.audio_quality import evaluate_audio_quality, QualityGateResult
from src.preprocessing import preprocess_audio_signal, preprocess_audio, CANONICAL_SAMPLE_RATE
from src.feature_extraction import extract_canonical_features


@dataclass
class PredictionOutput:
    status: str  # "CONFIDENT_PD", "CONFIDENT_HC", "INCONCLUSIVE", "REJECTED"
    category_label: str  # "PD-Associated Acoustic Pattern", "HC-Associated Acoustic Pattern", "Inconclusive Result", "Unsuitable Audio"
    confidence_score: float  # [0.0, 1.0]
    pd_model_score: float  # [0.0, 1.0]
    pd_probability: float  # alias for backward compatibility
    is_inconclusive: bool
    rejection_reason: Optional[str]
    quality_metrics: Dict[str, Any]
    feature_values: Dict[str, float]
    top_feature_contributions: List[Dict[str, Any]]
    explanation_text: str


class CanonicalVoicePipeline:
    """
    Unified Voice Screening Pipeline containing:
    1. Pre-inference Audio Quality & Signal Activity Gate
    2. Canonical Audio Preprocessing (8kHz mono normalization)
    3. Nyquist-Compliant Feature Extraction (45 acoustic features)
    4. Input Compatibility / Out-of-Distribution Check
    5. Trained Statistical ML Pipeline (Imputer + Scaler + SelectKBest + Estimator)
    6. Indeterminate Model-Score Decision Layer
    """

    def __init__(
        self,
        model_name: str = "GradientBoostingClassifier",
        k_features: int = 10,
        clf_params: Optional[Dict[str, Any]] = None,
        low_threshold: float = 0.40,
        high_threshold: float = 0.60,
        version: str = "2.1.0"
    ):
        self.model_name = model_name
        self.k_features = k_features
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.version = version

        # Default classifier hyperparameters (empirically validated via inner CV)
        default_clf_params = {
            "n_estimators": 40,
            "max_depth": 2,
            "learning_rate": 0.08,
            "random_state": 42
        }
        if clf_params:
            default_clf_params.update(clf_params)
        self.clf_params = default_clf_params

        # Sklearn ML pipeline
        self.sklearn_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k_features)),
            ("classifier", GradientBoostingClassifier(**self.clf_params))
        ])

        self.feature_names_in_: Optional[List[str]] = None
        self.selected_feature_names_: Optional[List[str]] = None
        self.training_summary_: Optional[Dict[str, Any]] = None
        self.reference_feature_stats_: Optional[Dict[str, Dict[str, float]]] = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], feature_names: Optional[List[str]] = None):
        """
        Fits the ML pipeline on feature matrix X and target y.
        Computes reference feature statistics for input compatibility checks.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_df = X
            X_arr = X.values
        else:
            self.feature_names_in_ = feature_names if feature_names is not None else [f"feat_{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X)
            X_df = pd.DataFrame(X_arr, columns=self.feature_names_in_)

        y_arr = np.asarray(y)

        # Fit sklearn pipeline
        self.sklearn_pipeline.fit(X_arr, y_arr)

        # Extract selected feature names
        selector = self.sklearn_pipeline.named_steps["selector"]
        support_mask = selector.get_support()
        self.selected_feature_names_ = [
            self.feature_names_in_[i] for i, supp in enumerate(support_mask) if supp
        ]

        # Compute reference cohort feature statistics for input compatibility verification
        stats = {}
        for col in self.feature_names_in_:
            col_vals = pd.to_numeric(X_df[col], errors="coerce").dropna()
            if len(col_vals) > 0:
                stats[col] = {
                    "mean": float(col_vals.mean()),
                    "std": float(col_vals.std()) if col_vals.std() > 1e-6 else 1.0,
                    "min": float(col_vals.min()),
                    "max": float(col_vals.max()),
                    "q25": float(col_vals.quantile(0.25)),
                    "q75": float(col_vals.quantile(0.75)),
                }
        self.reference_feature_stats_ = stats

        # Store training metadata
        clf = self.sklearn_pipeline.named_steps["classifier"]
        self.training_summary_ = {
            "version": self.version,
            "sample_rate": CANONICAL_SAMPLE_RATE,
            "n_samples": int(len(y_arr)),
            "n_features_in": int(len(self.feature_names_in_)),
            "k_selected": int(self.k_features),
            "selected_features": self.selected_feature_names_,
            "clf_params": self.clf_params,
            "feature_importances": dict(zip(self.selected_feature_names_, [float(v) for v in clf.feature_importances_]))
        }
        return self

    def _check_input_compatibility(self, features_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Input Compatibility / Out-of-Distribution (OOD) check.
        Calibrated strictly from the training cohort reference feature distributions.
        Verifies whether an extracted feature vector conforms to sustained /a/ phonation
        within the reference boundaries of the training cohort.
        """
        if self.reference_feature_stats_ is None:
            return True, None

        exact_msg = (
            "This recording falls outside the acoustic range represented by the model's training data. "
            "The system cannot provide a reliable experimental classification from this recording."
        )

        # 1. Pitch bounds check (modal sustained vowel phonation range in training cohort: 65 Hz to 380 Hz)
        f0_mean = features_df["f0_mean"].iloc[0] if "f0_mean" in features_df.columns else None
        if f0_mean is not None and np.isfinite(f0_mean):
            if f0_mean < 65.0 or f0_mean > 380.0:
                return False, exact_msg

        # 2. Pitch variation check (sustained vowel has low F0 CV; speech/conversation has F0 CV > 0.40)
        f0_cv = features_df["f0_cv"].iloc[0] if "f0_cv" in features_df.columns else None
        if f0_cv is not None and np.isfinite(f0_cv):
            if f0_cv > 0.40:
                return False, exact_msg

        # 3. Standardized distance across selected features
        z_scores = []
        extreme_z_count = 0
        for feat in (self.selected_feature_names_ or []):
            if feat in features_df.columns and feat in self.reference_feature_stats_:
                val = features_df[feat].iloc[0]
                if np.isfinite(val):
                    ref = self.reference_feature_stats_[feat]
                    std_val = ref["std"] if ref["std"] > 1e-6 else 1e-6
                    z = abs((val - ref["mean"]) / std_val)
                    z_scores.append(z)
                    if z > 4.5:
                        extreme_z_count += 1

        if len(z_scores) > 0:
            mean_z = float(np.mean(z_scores))
            max_z = float(np.max(z_scores))
            # Calibrated against 81-sample training distribution (99th pct mean |z| = 2.42, max |z| = 4.76)
            if mean_z > 3.20 or max_z > 6.0 or extreme_z_count >= 3:
                return False, exact_msg

        return True, None

    def predict_from_audio(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        sr: Optional[int] = None
    ) -> PredictionOutput:
        """
        Executes end-to-end inference directly from raw audio:
        Gate -> Preprocess -> Extract -> Input Compatibility Check -> Predict -> Indeterminate Band.
        """
        # Step 1: Decode raw audio
        try:
            if isinstance(audio_input, (str, bytes, io.BytesIO)):
                if isinstance(audio_input, bytes):
                    f_obj = io.BytesIO(audio_input)
                elif isinstance(audio_input, io.BytesIO):
                    f_obj = audio_input
                else:
                    f_obj = audio_input
                try:
                    y_raw, sample_rate = sf.read(f_obj)
                except Exception:
                    if isinstance(f_obj, io.BytesIO):
                        f_obj.seek(0)
                    y_raw, sample_rate = librosa.load(f_obj, sr=None, mono=False)
            elif isinstance(audio_input, np.ndarray):
                y_raw = audio_input
                sample_rate = sr if sr is not None else CANONICAL_SAMPLE_RATE
            else:
                raise ValueError("Unsupported audio input format.")
        except Exception as exc:
            return PredictionOutput(
                status="REJECTED",
                category_label="Audio Decode Failure",
                confidence_score=0.0,
                pd_model_score=0.5,
                pd_probability=0.5,
                is_inconclusive=True,
                rejection_reason=f"Failed to decode audio file: {exc}",
                quality_metrics={},
                feature_values={},
                top_feature_contributions=[],
                explanation_text="The uploaded file could not be decoded as valid audio."
            )

        # Audio Quality & Signal Activity Gate (evaluated on raw audio)
        quality_res: QualityGateResult = evaluate_audio_quality(y_raw, sample_rate)
        if not quality_res.is_valid:
            return PredictionOutput(
                status="REJECTED",
                category_label="Unsuitable Audio",
                confidence_score=0.0,
                pd_model_score=0.5,
                pd_probability=0.5,
                is_inconclusive=True,
                rejection_reason=quality_res.reason,
                quality_metrics=quality_res.metrics or {},
                feature_values={},
                top_feature_contributions=[],
                explanation_text=(
                    f"Audio rejected by Quality Gate: {quality_res.reason}\n"
                    "Please refer to the Recording Protocol and re-record a steady sustained vowel /a/."
                )
            )

        # Step 2: Canonical Preprocessing (8kHz, trimmed, normalized)
        y_proc, proc_sr = preprocess_audio_signal(y_raw, sample_rate)

        # Step 3: Canonical Feature Extraction
        try:
            features_df = extract_canonical_features(y_proc, proc_sr)
        except Exception as exc:
            return PredictionOutput(
                status="REJECTED",
                category_label="Feature Extraction Failure",
                confidence_score=0.0,
                pd_model_score=0.5,
                pd_probability=0.5,
                is_inconclusive=True,
                rejection_reason=f"Feature extraction failed: {exc}",
                quality_metrics=quality_res.metrics or {},
                feature_values={},
                top_feature_contributions=[],
                explanation_text="Acoustic feature calculation failed on this recording."
            )

        # Step 3: Check missing features
        n_missing = int(features_df.isna().sum().sum())
        if n_missing > (0.15 * features_df.shape[1]):
            return PredictionOutput(
                status="INCONCLUSIVE",
                category_label="Unreliable Feature Extraction",
                confidence_score=0.0,
                pd_model_score=0.5,
                pd_probability=0.5,
                is_inconclusive=True,
                rejection_reason=f"Too many acoustic features could not be reliably computed ({n_missing} missing).",
                quality_metrics=quality_res.metrics or {},
                feature_values=features_df.iloc[0].to_dict(),
                top_feature_contributions=[],
                explanation_text=(
                    "Unable to reliably analyze this recording. Essential acoustic parameters could not be stably computed. "
                    "Please record again following the protocol."
                )
            )

        # Step 4: Input Compatibility / OOD Check
        is_compat, ood_reason = self._check_input_compatibility(features_df)
        if not is_compat:
            exact_ood_msg = (
                "This recording falls outside the acoustic range represented by the model's training data. "
                "The system cannot provide a reliable experimental classification from this recording."
            )
            return PredictionOutput(
                status="INCONCLUSIVE",
                category_label="Inconclusive",
                confidence_score=0.0,
                pd_model_score=0.5,
                pd_probability=0.5,
                is_inconclusive=True,
                rejection_reason=ood_reason or exact_ood_msg,
                quality_metrics=quality_res.metrics or {},
                feature_values=features_df.iloc[0].to_dict(),
                top_feature_contributions=[],
                explanation_text=exact_ood_msg
            )

        # Step 5: Align features to training column order
        if self.feature_names_in_ is not None:
            for col in self.feature_names_in_:
                if col not in features_df.columns:
                    features_df[col] = np.nan
            features_aligned = features_df[self.feature_names_in_].values
        else:
            features_aligned = features_df.values

        # Step 6: Model Inference
        proba = self.sklearn_pipeline.predict_proba(features_aligned)[0]
        pd_score = float(proba[1])

        # Step 7: Feature Contributions and Indeterminate Decision Layer
        feature_dict = {k: float(v) for k, v in features_df.iloc[0].items()}
        contributions = self._compute_feature_contributions(features_aligned[0], feature_dict)

        if pd_score >= self.high_threshold:
            status = "CONFIDENT_PD"
            category_label = "PD-Associated Acoustic Pattern"
            confidence_score = pd_score
            is_inconclusive = False
            explanation = (
                f"PD-associated acoustic pattern identified (PD Model Score: {pd_score:.1%}). "
                "The acoustic parameters in this recording align more closely with the Parkinson's cohort in the training dataset. "
                "This is an exploratory screening result, NOT a medical diagnosis."
            )
        elif pd_score <= self.low_threshold:
            status = "CONFIDENT_HC"
            category_label = "HC-Associated Acoustic Pattern"
            confidence_score = 1.0 - pd_score
            is_inconclusive = False
            explanation = (
                f"HC-associated acoustic pattern identified (PD Model Score: {pd_score:.1%}). "
                "The acoustic parameters in this recording align more closely with the Healthy Control cohort in the training dataset. "
                "This is an exploratory screening result, NOT a medical clearance."
            )
        else:
            status = "INCONCLUSIVE"
            category_label = "Inconclusive Result"
            confidence_score = float(max(pd_score, 1.0 - pd_score))
            is_inconclusive = True
            explanation = (
                f"Model score ({pd_score:.1%}) falls within the Indeterminate Model-Score Band "
                f"[{self.low_threshold:.0%}, {self.high_threshold:.0%}]. The vocal characteristics do not provide sufficient "
                "evidence for a reliable model-supported classification. Please repeat the recording using the recommended protocol."
            )

        return PredictionOutput(
            status=status,
            category_label=category_label,
            confidence_score=round(confidence_score, 4),
            pd_model_score=round(pd_score, 4),
            pd_probability=round(pd_score, 4),
            is_inconclusive=is_inconclusive,
            rejection_reason=None,
            quality_metrics=quality_res.metrics or {},
            feature_values=feature_dict,
            top_feature_contributions=contributions,
            explanation_text=explanation
        )

    def _compute_feature_contributions(
        self,
        feature_row: np.ndarray,
        feature_dict: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Computes standardized feature deviations from the reference training cohort mean
        along with the classifier's tree-split importance weights.
        """
        if self.selected_feature_names_ is None:
            return []

        scaler = self.sklearn_pipeline.named_steps["scaler"]
        selector = self.sklearn_pipeline.named_steps["selector"]
        clf = self.sklearn_pipeline.named_steps["classifier"]

        # Scaled values for selected features
        imputer = self.sklearn_pipeline.named_steps["imputer"]
        X_imp = imputer.transform([feature_row])
        X_scaled = scaler.transform(X_imp)
        X_sel = selector.transform(X_scaled)[0]

        importances = clf.feature_importances_

        contributions = []
        for name, scaled_val, imp in zip(self.selected_feature_names_, X_sel, importances):
            raw_val = feature_dict.get(name, float("nan"))
            direction = "Elevated vs Reference Mean" if scaled_val > 0 else "Reduced vs Reference Mean"
            contributions.append({
                "feature": name,
                "value": round(float(raw_val), 4) if np.isfinite(raw_val) else 0.0,
                "raw_value": round(float(raw_val), 4) if np.isfinite(raw_val) else 0.0,
                "z_score": round(float(scaled_val), 2),
                "normalized_deviation": round(float(scaled_val), 2),
                "importance_weight": round(float(imp), 4),
                "relative_importance": f"{float(imp):.1%}",
                "direction": direction
            })

        contributions.sort(key=lambda x: x["importance_weight"], reverse=True)
        return contributions

    def save(self, filepath: str):
        """Serializes the canonical pipeline bundle with complete metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "version": self.version,
            "model_name": self.model_name,
            "sample_rate": CANONICAL_SAMPLE_RATE,
            "k_features": self.k_features,
            "clf_params": self.clf_params,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "feature_names_in": self.feature_names_in_,
            "selected_feature_names": self.selected_feature_names_,
            "training_summary": self.training_summary_,
            "reference_feature_stats": self.reference_feature_stats_,
            "sklearn_pipeline": self.sklearn_pipeline
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CanonicalVoicePipeline":
        """Loads a serialized canonical pipeline bundle."""
        bundle = joblib.load(filepath)
        instance = cls(
            model_name=bundle.get("model_name", "GradientBoostingClassifier"),
            k_features=bundle.get("k_features", 10),
            clf_params=bundle.get("clf_params"),
            low_threshold=bundle.get("low_threshold", 0.40),
            high_threshold=bundle.get("high_threshold", 0.60),
            version=bundle.get("version", "2.1.0")
        )
        instance.feature_names_in_ = bundle.get("feature_names_in")
        instance.selected_feature_names_ = bundle.get("selected_feature_names")
        instance.training_summary_ = bundle.get("training_summary")
        instance.reference_feature_stats_ = bundle.get("reference_feature_stats")
        instance.sklearn_pipeline = bundle.get("sklearn_pipeline")
        return instance
