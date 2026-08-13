
# AI-Based Multimodal Parkinson's Risk Screening System

An AI-based research project for Parkinson's disease risk screening using **voice analysis and machine learning**, with **gait analysis from video planned as a future multimodal extension**.

## Project Status

**Currently in development**

The current implementation focuses on building and testing the **voice-based analysis pipeline**. The gait/video module will be added in a later phase.

## Project Goal

The goal is to develop a non-invasive AI-assisted screening system that analyzes characteristics of a person's voice to identify patterns associated with Parkinson's disease.

The project will eventually combine multiple modalities:

-  Voice analysis
-  Gait analysis from video
-  Machine learning
- Multimodal prediction fusion

The system is intended for **research and screening purposes**, not medical diagnosis.

## Proposed Architecture

### Voice Module

```text
Voice Recording
      ↓
Audio Preprocessing
      ↓
Feature Extraction
      ↓
Feature Selection
      ↓
Machine Learning Models
      ↓
Model Evaluation
      ↓
Hybrid Model
      ↓
Voice Risk Score
```

### Future Gait Module
```text
Walking Video
      ↓
Pose Estimation
      ↓
Gait Feature Extraction
      ↓
Machine Learning Model
      ↓
Gait Risk Score
```
### Future Multimodal Fusion
```text
Voice Risk Score ─────┐
                      ↓
                 Fusion Model
                      ↓
                Combined Risk
                      ↑
                      │
Gait Risk Score ──────┘
```
## Voice Features
```text
The voice pipeline is designed to extract a 
broad set of acoustic and signal-processing features, 
including:

Fundamental frequency (F0 / Pitch)
Jitter
Shimmer
Harmonics-to-Noise Ratio (HNR)
Noise-to-Harmonics Ratio (NHR)
Mel-Frequency Cepstral Coefficients (MFCCs)
Delta and Delta-Delta MFCCs
Spectral Centroid
Spectral Bandwidth
Spectral Rolloff
Spectral Contrast
Spectral Flatness
Formants (F1–F4)
Cepstral features including CPP/CPPS
Recurrence Period Density Entropy (RPDE)
Detrended Fluctuation Analysis (DFA)
Pitch Period Entropy (PPE)
Glottal-to-Noise Excitation Ratio (GNE)
RMS Energy
Zero-Crossing Rate (ZCR)

Feature selection will be performed after obtaining and 
evaluating the final dataset. Not all extracted features 
are expected to be used in the final model.
```
## Machine Learning Models
```text
The project will experimentally evaluate multiple machine 
learning algorithms:

Logistic Regression
Support Vector Machine (SVM)
Random Forest
XGBoost

The models will first be evaluated independently.
The best-performing and most complementary models will then 
be considered for the final hybrid model.
```

## Hybrid Model
```text
The proposed hybrid approach will combine predictions 
from multiple machine learning models rather than relying
on a single classifier

Voice Features
      ↓
 ┌────┴─────┐
 ↓          ↓
 SVM      XGBoost
 ↓          ↓
 └────┬─────┘
      ↓
 Prediction Fusion
      ↓
 Final Risk Score

The final model combination and fusion method will be 
determined experimentally using the selected dataset 
and validation results.
```

## Technologies
### Programming
```text
Python
```
### Audio Processing
```text
Librosa
Parselmouth / Praat
NumPy
SciPy
Pandas
```
### Machine Learning
```text
Scikit-learn
XGBoost
```
### Future Computer Vision
```text
OpenCV
MediaPipe
```
### Planned Interface
```text
Streamlit
```
### Version Control
```text
Git
GitHub
```

## Disclaimer
This project is an academic/research prototype and is not intended to diagnose Parkinson's disease or replace evaluation by qualified healthcare professionals.
Any risk score produced by the system should be interpreted as a machine-learning screening result rather than a medical diagnosis.

## Team

This project is being developed as part of an academic Innovative Product Design (IPD) project by
```text
Shaan Suvarna
Mohammad Jaffer Hussin
Tanay Gala
Yoshita Pol
```
