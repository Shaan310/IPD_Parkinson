"""
Main Reproducible Training and Validation Script for Parkinson's Voice Pipeline.

Executes:
1. Dataset loading and subject-independence verification (81 recordings).
2. Canonical feature extraction check.
3. Leakage-free nested cross-validation across candidate models (5 outer x 3 inner folds).
4. Direct hyperparameter selection and final locked canonical pipeline fitting.
5. Inconclusive / indeterminate subset performance analysis.
6. Export of comprehensive evaluation metrics to data/evaluation/training_report.json.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix
)

from src.data_loader import get_dataset_files, audit_dataset_integrity
from src.feature_extraction import extract_features_from_audio
from src.evaluate_candidates import run_nested_cv
from src.pipeline import CanonicalVoicePipeline


def main():
    print("\n" + "="*80)
    print("PARKINSON'S DISEASE VOICE PIPELINE: REPRODUCIBLE TRAINING & AUDIT")
    print("="*80)

    # Step 1: Dataset Verification
    print("\n--- 1. DATASET INTEGRITY & LEAKAGE CHECK ---")
    df_files = get_dataset_files()
    audit = audit_dataset_integrity(df_files)
    print(f"Total recordings located: {audit['total_recordings']}")
    print(f"Healthy Controls (HC): {audit['hc_count']}")
    print(f"Parkinson's Disease (PwPD): {audit['pd_count']}")
    print(f"Unique subjects: {audit['total_unique_subjects']}")
    print(f"Subject overlap between classes: {audit['subject_overlap']}")
    print(f"Subject-independent validation verified: {audit['is_subject_independent']}")

    if audit['total_recordings'] != 81:
        print(f"WARNING: Expected 81 recordings, found {audit['total_recordings']}.")

    # Step 2: Canonical Feature Dataset Check / Generation
    features_csv = "data/processed/canonical_features.csv"
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(features_csv):
        print("\n--- 2. EXTRACTING CANONICAL FEATURES ---")
        records = []
        for idx, row in df_files.iterrows():
            feats, meta = extract_features_from_audio(row["file_path"])
            rec = {
                "sample_id": row["sample_id"],
                "subject_id": row["subject_id"],
                "label": row["label"],
                "label_str": row["label_str"],
                "duration_sec": meta["duration_sec"],
                "missing_count": meta["missing_features_count"],
            }
            rec.update(feats.iloc[0].to_dict())
            records.append(rec)
        df_feats = pd.DataFrame(records)
        df_feats.to_csv(features_csv, index=False)
        print(f"Extracted features saved to {features_csv}")
    else:
        print(f"\n--- 2. USING CANONICAL FEATURES: {features_csv} ---")
        df_feats = pd.read_csv(features_csv)

    meta_cols = ["sample_id", "subject_id", "label", "label_str", "duration_sec", "missing_count"]
    feat_cols = [c for c in df_feats.columns if c not in meta_cols]
    print(f"Feature pool: {len(feat_cols)} candidate acoustic features.")
    print(f"Missing values in dataset: {df_feats[feat_cols].isna().sum().sum()}")

    # Step 3: Run Strict Nested Cross-Validation Benchmark
    print("\n--- 3. CANDIDATE BENCHMARKING (STRICT NESTED CV) ---")
    cv_results, summary_df = run_nested_cv(features_csv)

    # Step 4: Fit Final Canonical Pipeline using Formal Grid Search Selection
    print("\n--- 4. SELECTING & TRAINING FINAL LOCKED CANONICAL PIPELINE ---")
    X = df_feats[feat_cols]
    y = df_feats["label"]

    base_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "selector__k": [8, 10, 12],
        "clf__n_estimators": [30, 40, 50],
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.05, 0.08, 0.1]
    }

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_pipe,
        param_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=-1
    )
    grid_search.fit(X.values, y.values)

    best_k = grid_search.best_params_["selector__k"]
    best_clf_params = {
        "n_estimators": grid_search.best_params_["clf__n_estimators"],
        "max_depth": grid_search.best_params_["clf__max_depth"],
        "learning_rate": grid_search.best_params_["clf__learning_rate"],
        "random_state": 42
    }
    print(f"Grid search selected hyperparameters (primary criterion: balanced_accuracy):")
    print(f"  k = {best_k}")
    print(f"  n_estimators = {best_clf_params['n_estimators']}")
    print(f"  max_depth = {best_clf_params['max_depth']}")
    print(f"  learning_rate = {best_clf_params['learning_rate']}")

    pipeline = CanonicalVoicePipeline(
        model_name="GradientBoostingClassifier",
        k_features=best_k,
        clf_params=best_clf_params,
        low_threshold=0.40,
        high_threshold=0.60,
        version="2.1.0"
    )
    pipeline.fit(X, y)

    print(f"\nSelected top {len(pipeline.selected_feature_names_)} features:")
    for f in pipeline.selected_feature_names_:
        imp = pipeline.training_summary_["feature_importances"].get(f, 0.0)
        print(f"  - {f:25s} (importance: {imp:.4f})")

    # Step 5: Save Canonical Pipeline (Single Source of Truth)
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    canonical_model_path = os.path.join(model_dir, "canonical_voice_pipeline.joblib")
    pipeline.save(canonical_model_path)
    print(f"\nCanonical pipeline saved to: {canonical_model_path}")

    # Step 6: Conclusive-Subset & Indeterminate Band Analysis
    print("\n--- 5. INDETERMINATE MODEL-SCORE BAND & CONCLUSIVE-SUBSET ANALYSIS ---")
    gb_oof_probs = np.array(cv_results["Gradient Boosting"]["oof_probs"])
    gb_oof_targets = y.values

    total_samples = len(gb_oof_targets)
    inconclusive_mask = (gb_oof_probs > 0.40) & (gb_oof_probs < 0.60)
    conclusive_mask = ~inconclusive_mask

    n_inconclusive = int(np.sum(inconclusive_mask))
    n_conclusive = int(np.sum(conclusive_mask))
    coverage = float(n_conclusive / total_samples)

    conclusive_preds = (gb_oof_probs[conclusive_mask] >= 0.50).astype(int)
    conclusive_targets = gb_oof_targets[conclusive_mask]
    correct_conclusive = int(np.sum(conclusive_preds == conclusive_targets))
    conclusive_acc = float(correct_conclusive / n_conclusive) if n_conclusive > 0 else 0.0
    conclusive_bal_acc = float(balanced_accuracy_score(conclusive_targets, conclusive_preds)) if n_conclusive > 0 else 0.0

    print(f"Total recordings evaluated       : {total_samples}")
    print(f"Conclusive predictions (p<=0.40 or p>=0.60): {n_conclusive} ({coverage:.1%} coverage)")
    print(f"Inconclusive predictions (0.40 < p < 0.60) : {n_inconclusive} ({n_inconclusive/total_samples:.1%})")
    print(f"Correct conclusive predictions   : {correct_conclusive} / {n_conclusive}")
    print(f"Conclusive-subset accuracy       : {conclusive_acc:.1%}")
    print(f"Conclusive-subset balanced acc   : {conclusive_bal_acc:.1%}")
    print(f"Overall Out-of-Fold Balanced Acc : {cv_results['Gradient Boosting']['oof_bal_acc']:.1%}")
    print(f"Overall Out-of-Fold ROC-AUC      : {cv_results['Gradient Boosting']['oof_auc']:.3f}")

    # Step 7: Export Comprehensive Training Report
    subset_analysis = {
        "total_samples": total_samples,
        "conclusive_samples": n_conclusive,
        "inconclusive_samples": n_inconclusive,
        "coverage": round(coverage, 4),
        "correct_conclusive": correct_conclusive,
        "conclusive_subset_accuracy": round(conclusive_acc, 4),
        "conclusive_subset_balanced_acc": round(conclusive_bal_acc, 4),
        "overall_oof_accuracy": round(cv_results['Gradient Boosting']['oof_accuracy'], 4),
        "overall_oof_balanced_acc": round(cv_results['Gradient Boosting']['oof_bal_acc'], 4),
        "overall_oof_auc": round(cv_results['Gradient Boosting']['oof_auc'], 4),
        "low_threshold": 0.40,
        "high_threshold": 0.60,
    }

    report = {
        "dataset": audit,
        "features": {
            "total_extracted": len(feat_cols),
            "retained_by_selector": pipeline.selected_feature_names_,
            "feature_importances": pipeline.training_summary_["feature_importances"]
        },
        "nested_cross_validation": cv_results,
        "indeterminate_band_analysis": subset_analysis,
        "selected_model": {
            "algorithm": "GradientBoostingClassifier",
            "hyperparameters": best_clf_params,
            "k_features": best_k,
            "preprocessing": "Canonical 8kHz Mono Standardized Normalization",
            "feature_selection": f"ANOVA F-score (k={best_k})",
            "model_selection_criterion": "Stratified 3-fold inner balanced_accuracy",
            "indeterminate_score_band": "[0.40, 0.60]"
        }
    }

    report_path = "data/evaluation/training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nComprehensive training report written to: {report_path}")

    print("\n" + "="*80)
    print("TRAINING & VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
