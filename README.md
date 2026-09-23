# Early Detection of Parkinson's Disease (Voice Screening Prototype)

An academic research prototype for voice-based acoustic screening of Parkinson's Disease (PwPD vs. Healthy Controls). Built on a verified cohort of **81 real audio recordings** under a standardized sustained vowel phonation `/a/` protocol.

> [!WARNING]
> **MEDICAL DISCLAIMER**: This software is an experimental, non-clinical research prototype developed solely for educational and investigative purposes. It is **NOT** a certified medical diagnostic device or diagnostic instrument. It cannot diagnose, confirm, rule out, or monitor Parkinson's Disease or any other neurological or speech-language pathology. All medical evaluations must be conducted by qualified healthcare professionals.

---

## 1. Project Overview & Scope

Parkinson's disease frequently affects the vocal system early in its clinical trajectory (hypokinetic dysarthria), presenting with reduced pitch range, subtle tremor/micro-instabilities in phonation, micro-fluctuations in amplitude, and breathiness.

This repository provides an end-to-end, scientifically grounded, reproducible audio screening pipeline:
1. **Standardized Recording Task**: Sustained open vowel phonation `/a/` ("ah") for 3–5 seconds.
2. **Pre-Inference Audio Quality Gate**: Automated validation of duration, digital clipping, silence, and vocal energy prior to analysis.
3. **Canonical Signal Conditioning**: Robust mono conversion for multi-channel audio, 8,000 Hz resampling, silence trimming, and peak amplitude normalization.
4. **Physiologically Grounded Feature Extraction**: 45 acoustic parameters spanning fundamental frequency stability (F0 mean/std/cv), vocal perturbation (jitter, shimmer, HNR), vocal tract resonances (F1, F2 Burg formants capped at 3,800 Hz), spectral dynamics (centroid, bandwidth, rolloff, flatness, ZCR, RMS), compact MFCCs (1–13 mean, 1–4 std, 1–4 delta), and Pitch Period Entropy (PPE). Zero missing values. Fast C++ execution (< 0.15s per recording).
5. **Input Compatibility / OOD Check**: Screens for pitch stability ($F_0\text{ CV} \le 0.40$) and standardized feature distances against the reference cohort to prevent false predictions on conversational speech, music, or noise.
6. **Regularized Machine Learning Model**: Gradient Boosting Classifier ($n=30$, $\text{max\_depth}=3$, $\text{learning\_rate}=0.08$) with univariate ANOVA F-score feature selection ($k=10$), rigorously evaluated using strict Nested Cross-Validation (5 outer $\times$ 3 inner folds).
7. **Indeterminate Model-Score Band**: Audio samples with model scores between $0.40$ and $0.60$ are classified as **Inconclusive Result** to prevent forced false classifications on ambiguous recordings.
8. **Interactive Streamlit Web Interface**: Complete screening app with sample audio testing, file upload, live microphone recording, waveform and spectrogram inspection, and transparent feature deviation metrics.

---

## 2. Dataset Reality & Integrity Audit

- **Designated Dataset**: Exactly **81 real audio recordings** (40 Parkinson's Disease `PD_AH` and 41 Healthy Controls `HC_AH`).
- **Subject Independence**: Audited filename subject prefixes confirmed 81 unique subjects (40 PD, 41 HC) with **zero cross-cohort overlap**. Every outer fold in cross-validation evaluates strictly unseen individuals.
- **Audio Format**:
  - Sample rate: **8,000 Hz** (Mono, 32-bit float PCM)
  - Phonation duration: 1.52 s to 7.21 s (mean: 3.32 s)
- **Nyquist Constraint**: At 8,000 Hz, the Nyquist frequency is 4,000 Hz. Formant extraction is strictly capped at 3,800 Hz, capturing F1 and F2 cleanly. Claims of higher formants (F3, F4, F5) or full-bandwidth spectral measures are mathematically invalid for 8 kHz speech and were removed.

---

## 3. Empirical Model Validation (Strict Nested CV)

To prevent data leakage and provide realistic generalization estimates on small-sample biomedical data ($n=81$), all models were evaluated using **5-Fold Outer $\times$ 3-Fold Inner Stratified Nested Cross-Validation**. Feature selection ($k \in [8, 10, 12]$) and hyperparameter tuning were performed strictly inside the inner folds.

| Model Candidate | Outer CV AUC | Out-of-Fold AUC | Mean Bal Acc | Out-of-Fold Bal Acc | Sensitivity | Specificity | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Selected)** | **0.675 ± 0.058** | **0.706** | **66.4% ± 9.5%** | **66.7%** | **67.5%** | **65.3%** | **0.671** |
| Random Forest ($d=3, n=50$) | 0.666 ± 0.059 | 0.684 | 59.0% ± 5.6% | 59.2% | 57.5% | 60.6% | 0.576 |
| Logistic Regression (L2) | 0.666 ± 0.111 | 0.671 | 60.1% ± 12.5%| 60.5% | 60.0% | 60.3% | 0.598 |
| Linear SVM ($C=0.1$) | 0.688 ± 0.106 | 0.660 | 62.5% ± 10.5%| 62.8% | 47.5% | 77.5% | 0.562 |
| RBF SVM | 0.580 ± 0.178 | 0.623 | 60.1% ± 11.8%| 60.1% | 27.5% | 92.8% | 0.333 |

### Indeterminate Model-Score Band Analysis:
- Total Recordings Evaluated: **81**
- Conclusive Predictions ($p \le 0.40$ or $p \ge 0.60$): **63 samples (77.8% coverage)**
- Inconclusive Predictions ($0.40 < p < 0.60$): **18 samples (22.2%)**
- Correct Conclusive Predictions: **43 / 63**
- **Conclusive-Subset Accuracy: 68.3%**
- **Overall Out-of-Fold Balanced Accuracy (all 81 recordings): 66.7%** (ROC-AUC: 0.706)

*Note: 68.3% represents performance solely on the conclusive subset. The overall generalization metric across all 81 recordings is 66.7% Balanced Accuracy / 0.706 ROC-AUC.*

### Top 10 Selected Discriminative Features & Importance:
1. `mfcc_1_std` (Importance: **0.4143**) — Low-frequency energy envelope stability.
2. `mfcc_2_std` (Importance: **0.2158**) — Spectral tilt stability.
3. `f0_mean` (Importance: **0.1160**) — Mean fundamental frequency.
4. `f1_std` (Importance: **0.0813**) — First formant stability.
5. `rms_std` (Importance: **0.0561**) — Vocal amplitude fluctuation.
6. `f2_std` (Importance: **0.0522**) — Second formant stability.
7. `mfcc_4_std` (Importance: **0.0233**) — Higher-order cepstral variation.
8. `mfcc_3_mean` (Importance: **0.0201**) — Mid-frequency spectral shape.
9. `mfcc_3_std` (Importance: **0.0124**) — Mid-band spectral stability.
10. `spectral_centroid_std` (Importance: **0.0083**) — Spectral brightness variation.

---

## 4. Repository Structure

```text
IPD_Parkinson/
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # End-to-end reproducible training & benchmark entry point
├── verify_dataset.py           # Dataset integrity & subject leakage audit script
├── test_pipeline.py            # Quick end-to-end inference verification script
├── recording_protocol.md       # Standardized clinical recording protocol guide
├── requirements.txt            # Python dependencies
├── src/
│   ├── __init__.py             # Package marker
│   ├── data_loader.py          # Dataset scanner, subject ID extractor, and leakage audit
│   ├── audio_quality.py        # Pre-inference Audio Quality Gate
│   ├── preprocessing.py        # Canonical 8 kHz mono resampling, trimming, normalization
│   ├── feature_extraction.py   # 45 Nyquist-compliant acoustic features (Praat + Librosa)
│   ├── evaluate_candidates.py  # Strict 5x3 nested CV benchmark
│   └── pipeline.py             # CanonicalVoicePipeline class bundle
├── data/
│   ├── raw/                    # 81 raw audio recordings (HC: 41, PD: 40)
│   ├── processed/              # Canonical extracted feature matrix (canonical_features.csv)
│   └── evaluation/             # Training report and nested CV benchmark (training_report.json)
├── model/
│   └── canonical_voice_pipeline.joblib  # Serialized single canonical model
├── archive/
│   └── legacy_model_testing/   # Archived legacy experiments (isolated from production)
└── tests/
    ├── test_audio_quality.py       # Quality gate unit tests
    ├── test_preprocessing.py       # Preprocessing & multichannel unit tests
    ├── test_feature_extraction.py  # Feature extraction unit tests
    └── test_pipeline_consistency.py# End-to-end integration & OOD tests
```

---

## 5. Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Dataset Integrity
```bash
python verify_dataset.py
```

### 3. Run Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 4. Train & Benchmark Pipeline
```bash
python train.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 6. Scientific Limitations

1. **Sample Size Constraints ($n=81$)**: While subject-independent cross-validation was strictly enforced, 81 recordings cannot capture the broad demographic and phonetic variability of the global population.
2. **Single Vowel Context**: Sustained phonation `/a/` isolates phonatory stability but cannot capture prosodic deficits, speech rate abnormalities, or dysfluency that appear in continuous reading or monologue tasks.
3. **Telephonic/Downsampled Bandwidth (8 kHz)**: Acoustic information above 4,000 Hz is absent, precluding high-frequency fricative or upper-formant analysis.
4. **Etiological Specificity**: Acoustic dysphonic markers (jitter, shimmer, spectral envelope instability) are not exclusive to Parkinson's disease. They can also appear in presbyphonia, laryngeal dystonia, essential tremor, and acute vocal strain.
