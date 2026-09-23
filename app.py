"""
Early Detection of Parkinson's Disease — Research Prototype & Acoustic Analysis Workspace.

A scientifically grounded, reproducible Streamlit application providing voice-based
screening analysis using sustained vowel /a/ phonations.

Academic & Medical-Engineering Interface:
- Standardized 8 kHz audio conditioning & pre-inference quality gating
- Nyquist-compliant 45-dimensional acoustic feature extraction
- Input compatibility / out-of-distribution pitch stability check
- Canonical Gradient Boosting decision model with [0.40, 0.60] indeterminate band
- Independent Two-Recording / Repeat Recording consistency evaluation
- Restrained academic styling (black, white, charcoal, restrained light blue accent)
"""
import io
import os
import glob
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import CanonicalVoicePipeline, PredictionOutput
from src.preprocessing import convert_to_mono

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model", "canonical_voice_pipeline.joblib")
REPORT_FILE = os.path.join(BASE_DIR, "data", "evaluation", "training_report.json")


# ==============================================================================
# ACADEMIC RESEARCH & MEDICAL ENGINEERING DESIGN SYSTEM CSS
# ==============================================================================
CUSTOM_CSS = """
<style>
/* Academic Research & Medical-Engineering Color Tokens */
:root {
    --bg-base: #FFFFFF;
    --card-bg: #FFFFFF;
    --border-subtle: #E2E8F0;
    --border-strong: #CBD5E1;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --accent-blue: #0284C7;
    --accent-blue-dark: #0369A1;
    --accent-blue-light: #F0F9FF;
    --accent-blue-border: #BAE6FD;
    --neutral-surface: #F8FAFC;
    --neutral-border: #94A3B8;
}

/* Base application typography and surfaces */
.stApp {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

.stApp p {
    color: #1E293B;
}

.stApp label {
    color: #0F172A !important;
    font-weight: 600;
}

/* Captions and muted text */
.stApp small,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
    font-size: 0.82rem !important;
}

/* Sidebar Academic Styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    padding-top: 1.25rem;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #0F172A;
}

[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #1E293B;
}

/* Sidebar Radio Navigation */
[data-testid="stSidebar"] div[data-testid="stRadio"] label span p,
[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
[data-testid="stSidebar"] div[data-testid="stRadio"] label span,
[data-testid="stSidebar"] div[data-testid="stRadio"] div {
    color: #0F172A !important;
    font-weight: 500 !important;
    font-size: 0.90rem !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] span p,
[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] span {
    color: #0284C7 !important;
    font-weight: 700 !important;
}

/* Form labels and select elements */
div[data-testid="stRadio"] label span p,
div[data-testid="stRadio"] label p,
div[data-testid="stSelectbox"] label p,
div[data-testid="stFileUploader"] label p {
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 0.90rem !important;
}

/* BaseWeb Select Input */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #0284C7 !important;
}

div[data-baseweb="select"] * {
    color: #0F172A !important;
}

div[data-baseweb="select"] svg {
    fill: #0F172A !important;
}

/* BaseWeb Popovers (Dropdown menu options portal attached to body) */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.10) !important;
}

div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    padding: 4px 0 !important;
}

div[data-baseweb="popover"] li[role="option"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    padding: 8px 14px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}

div[data-baseweb="popover"] li[role="option"] * {
    color: #0F172A !important;
}

div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
    background-color: #F0F9FF !important;
}

div[data-baseweb="popover"] li[role="option"]:hover *,
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] * {
    color: #0284C7 !important;
    font-weight: 600 !important;
}

/* Primary Action Buttons (Dark Charcoal/Navy background with CRISP WHITE text) */
button[kind="primary"],
button[data-testid="baseButton-primary"],
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stButton"] button[data-testid="baseButton-primary"] {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border: 1px solid #0F172A !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.15rem !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease-in-out;
}

button[kind="primary"] *,
button[data-testid="baseButton-primary"] *,
div[data-testid="stButton"] button[kind="primary"] *,
div[data-testid="stButton"] button[data-testid="baseButton-primary"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
div[data-testid="stButton"] button[kind="primary"]:hover,
div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover {
    background-color: #334155 !important;
    border-color: #334155 !important;
    color: #FFFFFF !important;
}

button[kind="primary"]:hover *,
button[data-testid="baseButton-primary"]:hover *,
div[data-testid="stButton"] button[kind="primary"]:hover *,
div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

button[kind="primary"]:active,
button[data-testid="baseButton-primary"]:active {
    background-color: #1E293B !important;
    border-color: #1E293B !important;
}

/* Secondary Action Buttons (White background with CRISP CHARCOAL text) */
button[kind="secondary"],
button[data-testid="baseButton-secondary"],
div[data-testid="stButton"] button[kind="secondary"],
div[data-testid="stButton"] button[data-testid="baseButton-secondary"],
div[data-testid="stDownloadButton"] button {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    transition: all 0.15s ease-in-out;
}

button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
div[data-testid="stButton"] button[kind="secondary"] *,
div[data-testid="stButton"] button[data-testid="baseButton-secondary"] *,
div[data-testid="stDownloadButton"] button * {
    color: #0F172A !important;
    fill: #0F172A !important;
}

button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
div[data-testid="stButton"] button[kind="secondary"]:hover,
div[data-testid="stButton"] button[data-testid="baseButton-secondary"]:hover,
div[data-testid="stDownloadButton"] button:hover {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    border-color: #94A3B8 !important;
}

button[kind="secondary"]:hover *,
button[data-testid="baseButton-secondary"]:hover *,
div[data-testid="stButton"] button[kind="secondary"]:hover *,
div[data-testid="stButton"] button[data-testid="baseButton-secondary"]:hover *,
div[data-testid="stDownloadButton"] button:hover * {
    color: #0F172A !important;
    fill: #0F172A !important;
}

/* File Uploader Component */
div[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
    background-color: #F8FAFC !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 20px !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #0284C7 !important;
    background-color: #F0F9FF !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] span,
div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small,
div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] div {
    color: #334155 !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button * {
    color: #0F172A !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:hover {
    background-color: #F1F5F9 !important;
    border-color: #94A3B8 !important;
}

div[data-testid="stFileUploaderFileData"] {
    background-color: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
}

div[data-testid="stFileUploaderFileData"] * {
    color: #0F172A !important;
}

/* Audio Input (Live Microphone) */
div[data-testid="stAudioInput"] {
    background-color: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}

div[data-testid="stAudioInput"] * {
    color: #0F172A !important;
}

/* Expander Component */
div[data-testid="stExpander"] {
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    background-color: #FFFFFF !important;
    margin-bottom: 14px !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] details summary {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    padding: 10px 14px !important;
    font-weight: 600 !important;
    border: none !important;
}

div[data-testid="stExpander"] details summary:hover {
    background-color: #F1F5F9 !important;
}

div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span {
    color: #0F172A !important;
    font-weight: 600 !important;
}

div[data-testid="stExpander"] details summary svg {
    fill: #0F172A !important;
    stroke: #0F172A !important;
}

div[data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid #E2E8F0 !important;
}

div[data-testid="stExpander"] details[open] > div:not(summary) {
    background-color: #FFFFFF !important;
    padding: 14px !important;
}

/* Native Metrics */
div[data-testid="stMetric"] {
    background-color: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    padding: 12px 14px !important;
}

div[data-testid="stMetricLabel"] * {
    color: #64748B !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

div[data-testid="stMetricValue"] * {
    color: #0F172A !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricDelta"] * {
    color: #334155 !important;
    font-size: 0.78rem !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 0.90rem !important;
    background-color: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
}

button[data-baseweb="tab"]:hover {
    color: #0F172A !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #0284C7 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #0284C7 !important;
}

button[data-baseweb="tab"] * {
    color: inherit !important;
}

/* Streamlit Alerts */
div[data-testid="stAlert"] {
    border-radius: 6px !important;
    border: 1px solid #CBD5E1 !important;
    padding: 12px 16px !important;
}

div[data-testid="stAlert"] * {
    color: #0F172A !important;
}

div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] p {
    color: #0F172A !important;
}

div[data-testid="stAlert"][data-test="stAlert-info"],
div[data-testid="stNotification-info"] {
    background-color: #F0F9FF !important;
    border-color: #BAE6FD !important;
    border-left: 4px solid #0284C7 !important;
}
div[data-testid="stAlert"][data-test="stAlert-info"] *,
div[data-testid="stNotification-info"] * {
    color: #0369A1 !important;
}

div[data-testid="stAlert"][data-test="stAlert-warning"],
div[data-testid="stNotification-warning"] {
    background-color: #FFFBEB !important;
    border-color: #FCD34D !important;
    border-left: 4px solid #D97706 !important;
}
div[data-testid="stAlert"][data-test="stAlert-warning"] *,
div[data-testid="stNotification-warning"] * {
    color: #78350F !important;
}

div[data-testid="stAlert"][data-test="stAlert-error"],
div[data-testid="stNotification-error"] {
    background-color: #FEF2F2 !important;
    border-color: #FCA5A5 !important;
    border-left: 4px solid #DC2626 !important;
}
div[data-testid="stAlert"][data-test="stAlert-error"] *,
div[data-testid="stNotification-error"] * {
    color: #7F1D1D !important;
}

div[data-testid="stAlert"][data-test="stAlert-success"],
div[data-testid="stNotification-success"] {
    background-color: #F0FDF4 !important;
    border-color: #86EFAC !important;
    border-left: 4px solid #16A34A !important;
}
div[data-testid="stAlert"][data-test="stAlert-success"] *,
div[data-testid="stNotification-success"] * {
    color: #14532D !important;
}

/* Dataframes & Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    background-color: #FFFFFF !important;
}

div[data-testid="stTable"] {
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

div[data-testid="stTable"] table {
    color: #0F172A !important;
    background-color: #FFFFFF !important;
}

div[data-testid="stTable"] th {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #CBD5E1 !important;
    padding: 8px 12px !important;
}

div[data-testid="stTable"] td {
    color: #1E293B !important;
    border-bottom: 1px solid #E2E8F0 !important;
    padding: 8px 12px !important;
}

/* Clean Academic Card Styles */
.research-card {
    background: #FFFFFF;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.header-banner {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-left: 4px solid #0284C7;
    border-radius: 8px;
    padding: 24px 28px;
    margin-bottom: 24px;
}

.spec-chip {
    display: inline-flex;
    align-items: center;
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    padding: 3px 9px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #334155 !important;
    margin-right: 6px;
    margin-bottom: 6px;
}

/* Restrained Semantic Status Treatments */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}

/* 1. HC-Associated / Healthy Control Pattern (Restrained Green) */
.badge-hc {
    background-color: #F0FDF4 !important;
    color: #166534 !important;
    border: 1px solid #86EFAC !important;
}

.badge-hc * {
    color: #166534 !important;
}

.dot-hc {
    color: #16A34A !important;
}

/* 2. PD-Associated Acoustic Pattern (Restrained Red) */
.badge-pd {
    background-color: #FEF2F2 !important;
    color: #991B1B !important;
    border: 1px solid #FCA5A5 !important;
}

.badge-pd * {
    color: #991B1B !important;
}

.dot-pd {
    color: #DC2626 !important;
}

/* 3. Inconclusive Pattern (Restrained Amber) */
.badge-inconclusive {
    background-color: #FFFBEB !important;
    color: #92400E !important;
    border: 1px solid #FCD34D !important;
}

.badge-inconclusive * {
    color: #92400E !important;
}

.dot-inconclusive {
    color: #D97706 !important;
}

/* Model Output Card (Restrained Semantic Card Accents) */
.model-output-card {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.model-output-card.card-hc {
    border-left: 4px solid #16A34A;
}

.model-output-card.card-pd {
    border-left: 4px solid #DC2626;
}

.model-output-card.card-inconclusive {
    border-left: 4px solid #D97706;
}

/* Audio Quality Gate Rejection Card */
.card-rejection {
    background-color: #FEF2F2 !important;
    border: 1px solid #FECACA !important;
    border-left: 4px solid #DC2626 !important;
}

/* Metric Display Boxes (Neutral Academic Styling) */
.metric-container {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: left;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748B !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 3px;
}

.metric-val {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A !important;
}

.metric-sub {
    font-size: 0.78rem;
    color: #334155 !important;
    margin-top: 2px;
}

/* Recording Guide Box */
.guide-card {
    background: #F8FAFC;
    border: 1px solid #BAE6FD;
    border-left: 4px solid #0284C7;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 20px;
}

.guide-title {
    font-size: 0.90rem;
    font-weight: 700;
    color: #0369A1 !important;
    margin-bottom: 6px;
}

.guide-list {
    margin: 0;
    padding-left: 18px;
    font-size: 0.85rem;
    color: #1E293B !important;
    line-height: 1.6;
}

/* Comparison Summary Card */
.comparison-card {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-left: 4px solid #0284C7;
    border-radius: 8px;
    padding: 18px 22px;
    margin-top: 16px;
    margin-bottom: 20px;
}

.disclaimer-box {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-left: 3px solid #0284C7;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.80rem;
    color: #334155 !important;
    line-height: 1.5;
}

.transparency-bar {
    font-size: 0.75rem;
    color: #64748B !important;
    margin-top: 12px;
    padding-top: 8px;
    border-top: 1px solid #E2E8F0;
}

/* Inline code formatting */
code {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 4px !important;
    padding: 2px 5px !important;
    font-size: 0.85em !important;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the trained canonical voice pipeline."""
    if not os.path.exists(MODEL_FILE):
        return None
    return CanonicalVoicePipeline.load(MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_training_report():
    """Load the comprehensive training report JSON."""
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as f:
            return json.load(f)
    return None


# ==============================================================================
# HELPER: USER-FRIENDLY QUALITY GATE MESSAGES
# ==============================================================================

def format_user_friendly_rejection(reason: Optional[str]) -> Tuple[str, str]:
    """
    Translates technical rejection reasons into clear, accessible messages
    without exposing unnecessary technical implementation details.
    """
    reason_str = str(reason or "").lower()

    if "too short" in reason_str:
        headline = "Recording rejected: audio is too short."
        guidance = "A sustained vowel phonation of at least 1.2 seconds is required."
    elif "silent" in reason_str or "energy" in reason_str or "amplitude" in reason_str:
        headline = "Recording rejected: insufficient usable speech signal."
        guidance = "The microphone did not detect a clear vocal sound. Check your microphone and phonate at normal volume."
    elif "clipping" in reason_str or "clip" in reason_str:
        headline = "Recording rejected: excessive clipping."
        guidance = "The recording was too loud and caused audio distortion. Please speak at a moderate volume slightly further from the microphone."
    elif "instability" in reason_str or "unsupported" in reason_str or "f0" in reason_str or "diverge" in reason_str:
        headline = "Recording rejected: audio does not meet the expected sustained-vowel recording profile."
        guidance = "Please steady your pitch on a single continuous vowel 'ah' sound rather than conversational speech or words."
    else:
        headline = "Recording rejected: audio does not meet quality requirements."
        guidance = "Please re-record a steady sustained 'ah' vowel in a quiet environment."

    return headline, guidance


def map_category_label(status: str) -> str:
    """
    Ensures model output is formatted strictly as exactly one of:
    - PD-Associated Acoustic Pattern
    - HC-Associated Acoustic Pattern
    - Inconclusive
    """
    if status == "CONFIDENT_PD":
        return "PD-Associated Acoustic Pattern"
    elif status == "CONFIDENT_HC":
        return "HC-Associated Acoustic Pattern"
    else:
        return "Inconclusive"


# ==============================================================================
# VISUALIZATION COMPONENTS
# ==============================================================================

def plot_acoustic_waveform_and_spectrogram(y: np.ndarray, sr: int):
    """
    Renders synchronized Waveform (Plotly) and Spectrogram (Matplotlib).
    Uses clean academic styling with restrained steel blue and charcoal.
    """
    y_mono = convert_to_mono(y)
    duration = len(y_mono) / sr
    time_axis = np.linspace(0, duration, len(y_mono))

    downsample_factor = max(1, len(y_mono) // 4000)
    t_down = time_axis[::downsample_factor]
    y_down = y_mono[::downsample_factor]

    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=t_down,
        y=y_down,
        mode="lines",
        line=dict(color="#0284C7", width=1.2),
        name="Acoustic Signal",
        hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>"
    ))
    fig_wave.update_layout(
        height=140,
        margin=dict(l=15, r=15, t=10, b=10),
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(color="#334155", size=10)),
            tickfont=dict(color="#334155", size=9),
            showgrid=True,
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5E1"
        ),
        yaxis=dict(
            title=dict(text="Amplitude", font=dict(color="#334155", size=10)),
            tickfont=dict(color="#334155", size=9),
            range=[-1.05, 1.05],
            showgrid=True,
            gridcolor="#E2E8F0"
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(size=10, color="#334155")
    )
    st.plotly_chart(fig_wave, use_container_width=True, config={"displayModeBar": False})

    fig_spec, ax = plt.subplots(figsize=(10, 2.3), dpi=100)
    fig_spec.patch.set_facecolor("#FFFFFF")
    ax.patch.set_facecolor("#FFFFFF")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_mono)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="linear", ax=ax, cmap="magma")
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)", color="#334155", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color="#334155", fontsize=9)
    ax.tick_params(colors="#334155", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")

    plt.tight_layout(pad=0.8)
    st.pyplot(fig_spec, use_container_width=True)
    plt.close(fig_spec)


def render_feature_importance_barchart(report: dict, top_k: int = 10):
    """Renders a clean horizontal bar chart of canonical feature importances with academic charcoal styling."""
    if not report or "features" not in report or "feature_importances" not in report["features"]:
        return

    importances = report["features"]["feature_importances"]
    s = pd.Series(importances).sort_values(ascending=True).tail(top_k)

    fig = go.Figure(go.Bar(
        x=s.values,
        y=s.index,
        orientation="h",
        marker=dict(color="#334155", line=dict(color="#0284C7", width=1)),
        hovertemplate="Feature: %{y}<br>Tree Split Weight: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=20, t=15, b=20),
        xaxis=dict(
            title=dict(text="Relative Tree-Split Weight", font=dict(color="#334155", size=10)),
            tickfont=dict(color="#334155", size=9),
            showgrid=True,
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            tickfont=dict(family="monospace", size=10, color="#0F172A")
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(size=10, color="#0F172A")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: 8px;">
                <span style="color: #0284C7; font-size: 0.9rem;">●</span> IPD Voice Screening
            </div>
            <div style="font-size: 0.78rem; font-weight: 500; color: #64748B; margin-top: 2px;">
                Acoustic Screening Research Prototype • v2.1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = [
            "Overview",
            "Voice Analysis",
            "Model Insights",
            "About / Protocol"
        ]

        if "current_page" not in st.session_state:
            st.session_state.current_page = "Overview"

        selected_page = st.radio(
            "Navigation",
            pages,
            index=pages.index(st.session_state.current_page),
            label_visibility="collapsed"
        )
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Quick Model Specifications
        st.markdown("""
        <div class="research-card" style="padding: 14px 16px; margin-bottom: 16px; background: #F8FAFC;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                System Specifications
            </div>
            <div style="font-size: 0.80rem; line-height: 1.6; color: #334155;">
                • <b>Dataset</b>: 81 real recordings (40 PD, 41 HC)<br>
                • <b>Algorithm</b>: Gradient Boosting (k=10)<br>
                • <b>Validation</b>: 5-Fold Nested CV<br>
                • <b>Generalization</b>: ROC-AUC 0.706 (OOF)<br>
                • <b>Band Thresholds</b>: [0.40, 0.60] Indeterminate<br>
                • <b>Bandwidth</b>: 8,000 Hz (Nyquist: 4 kHz)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Medical Disclaimer in Sidebar
        st.markdown("""
        <div class="disclaimer-box">
            <b style="color: #0F172A;">Research Prototype Only</b>: This system is an experimental academic screening tool, not an approved medical device.
            It provides statistical pattern analysis on sustained vowels, not a medical diagnosis.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# HELP COMPONENT: WHAT DOES THIS RESULT MEAN?
# ==============================================================================

def render_what_does_result_mean(category: str):
    """
    Renders the concise expandable help section explaining the model output.
    Inherits the appropriate semantic accent for the identified finding.
    """
    if category == "PD-Associated Acoustic Pattern":
        finding_color = "#991B1B"
        box_border = "border-left: 3px solid #DC2626;"
        box_bg = "background: #FEF2F2;"
    elif category == "HC-Associated Acoustic Pattern":
        finding_color = "#166534"
        box_border = "border-left: 3px solid #16A34A;"
        box_bg = "background: #F0FDF4;"
    else:
        finding_color = "#92400E"
        box_border = "border-left: 3px solid #D97706;"
        box_bg = "background: #FFFBEB;"

    with st.expander("What does this result mean?", expanded=True):
        st.markdown(f"""
        <div style="font-size: 0.88rem; color: #0F172A; line-height: 1.6;">
            <div style="margin-bottom: 8px;">
                <b>Identified Finding</b>: <span style="font-weight: 700; color: {finding_color};">{category}</span>
            </div>
            <div style="margin-bottom: 6px; font-weight: 600; color: #475569; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">
                Interpretation:
            </div>
            <div style="{box_bg} border: 1px solid #CBD5E1; {box_border} border-radius: 6px; padding: 12px 14px; margin-bottom: 12px; font-size: 0.85rem; color: #1E293B;">
        """, unsafe_allow_html=True)

        if category == "PD-Associated Acoustic Pattern":
            st.markdown(
                "The recording contains acoustic characteristics that are more similar to the "
                "Parkinson's-associated patterns learned during model development."
            )
        elif category == "HC-Associated Acoustic Pattern":
            st.markdown(
                "The recording contains acoustic characteristics that are more similar to the "
                "healthy-control patterns learned during model development."
            )
        else:
            st.markdown(
                "The model score falls in the indeterminate range, so the system cannot produce a "
                "conclusive experimental classification from this recording."
            )

        st.markdown("""
            </div>
            <div style="font-size: 0.80rem; color: #64748B; line-height: 1.5;">
                <b>Important</b>: These findings represent statistical model outputs from sustained vowel recordings, 
                <b>NOT</b> a clinical or medical diagnosis. This screening tool does not replace a neurological examination.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# COMPONENT: PROMINENT MODEL OUTPUT CARD (ACADEMIC / MEDICAL-ENGINEERING)
# ==============================================================================

def render_prominent_model_output(
    category: str,
    pd_score: float,
    is_inconclusive: bool,
    timestamp_str: str,
    recording_label: str = "Recording 1"
):
    """
    Renders the prominent MODEL OUTPUT section with restrained academic styling:
    restrained semantic card accent, color-coded dot & badge, secondary PD Model Score,
    and transparency metadata.
    """
    if category == "PD-Associated Acoustic Pattern":
        card_class = "card-pd"
        badge_class = "badge-pd"
        dot_color = "#DC2626"
    elif category == "HC-Associated Acoustic Pattern":
        card_class = "card-hc"
        badge_class = "badge-hc"
        dot_color = "#16A34A"
    else:
        card_class = "card-inconclusive"
        badge_class = "badge-inconclusive"
        dot_color = "#D97706"

    st.markdown("### MODEL OUTPUT")
    st.markdown(f"""
    <div class="model-output-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
            <div>
                <span class="status-badge {badge_class}">
                    <span style="color: {dot_color}; font-size: 0.85rem;">●</span> {category}
                </span>
            </div>
            <div style="font-size: 0.80rem; font-weight: 600; color: #64748B;">
                Scope: {recording_label}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 12px;">
            <div class="metric-container">
                <div class="metric-label">PD Model Score</div>
                <div class="metric-val" style="color: #0F172A;">
                    {pd_score:.1%}
                </div>
                <div class="metric-sub">Continuous empirical model score (not a clinical probability)</div>
            </div>
            <div class="metric-container">
                <div class="metric-label">Decision Boundary Band</div>
                <div class="metric-val" style="color: #334155; font-size: 1.2rem;">[40.0% – 60.0%]</div>
                <div class="metric-sub">Indeterminate guardrail for borderline phonations</div>
            </div>
        </div>
        <div class="transparency-bar">
            Model Version: <b>2.1.0</b> &nbsp;•&nbsp; 
            Analysis Timestamp: <b>{timestamp_str}</b> &nbsp;•&nbsp; 
            Recording Status: <b>{recording_label} Analyzed</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: OVERVIEW (ACADEMIC RESEARCH PRESENTATION)
# ==============================================================================

def render_overview_page():
    st.markdown("""
    <div class="header-banner">
        <div style="font-size: 1.65rem; font-weight: 700; color: #0F172A; margin-bottom: 6px; letter-spacing: -0.01em;">
            Early Detection of Parkinson's Disease
        </div>
        <div style="font-size: 0.95rem; color: #334155; margin-bottom: 18px; line-height: 1.5;">
            Acoustic Feature Extraction, Quality-Gated Preprocessing & Decision-Support Model for Sustained Phonation.
        </div>
        <div>
            <span class="spec-chip">81 Real Subject Audio Files</span>
            <span class="spec-chip">100% Subject-Independent CV</span>
            <span class="spec-chip">8,000 Hz Standardized Sampling</span>
            <span class="spec-chip">Indeterminate Score Guardrail [0.40, 0.60]</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Launch Voice Analysis Workspace", type="primary", use_container_width=True):
            st.session_state.current_page = "Voice Analysis"
            st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Key Validation Metrics Grid
    st.markdown("### Validation Highlights on the 81-Subject Cohort")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Out-of-Fold ROC-AUC</div>
            <div class="metric-val" style="color: #0284C7;">0.706</div>
            <div class="metric-sub">Mean Outer: 0.675 ± 0.058</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Overall Balanced Accuracy</div>
            <div class="metric-val">66.7%</div>
            <div class="metric-sub">Mean Outer: 66.4% ± 9.5%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Conclusive-Subset Acc</div>
            <div class="metric-val">68.3%</div>
            <div class="metric-sub">63/81 Samples (77.8% Coverage)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Indeterminate Rate</div>
            <div class="metric-val">22.2%</div>
            <div class="metric-sub">18/81 Samples in [0.40, 0.60]</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Architectural Pipeline Flow
    st.markdown("### End-to-End Scientific Architecture")
    st.markdown("""
    <div class="research-card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
            <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px;">
                <div style="font-weight: 700; font-size: 0.85rem; color: #0284C7; margin-bottom: 4px;">1. Audio Quality Gate</div>
                <div style="font-size: 0.80rem; color: #334155; line-height: 1.5;">
                    Screens peak clipping (&lt;2.0%), minimum duration (&ge;1.2s), audible energy, and voicing before inference.
                </div>
            </div>
            <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px;">
                <div style="font-weight: 700; font-size: 0.85rem; color: #0284C7; margin-bottom: 4px;">2. Acoustic Extraction</div>
                <div style="font-size: 0.80rem; color: #334155; line-height: 1.5;">
                    Extracts 45 Nyquist-compliant (0–4 kHz) acoustic biomarkers including Praat Jitter, Shimmer, HNR, Formants, and MFCCs.
                </div>
            </div>
            <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px;">
                <div style="font-weight: 700; font-size: 0.85rem; color: #0284C7; margin-bottom: 4px;">3. Input Compatibility</div>
                <div style="font-size: 0.80rem; color: #334155; line-height: 1.5;">
                    Screens pitch coefficient of variation (F0 CV &le; 0.40) and feature distributions to prevent false inferences on non-vowel audio.
                </div>
            </div>
            <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px;">
                <div style="font-weight: 700; font-size: 0.85rem; color: #0284C7; margin-bottom: 4px;">4. Decision & Guardrail</div>
                <div style="font-size: 0.80rem; color: #334155; line-height: 1.5;">
                    Trained Gradient Boosting model evaluates score. Scores inside [0.40, 0.60] are declared Inconclusive rather than forced into error.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# HELPER: AUDIO INPUT SELECTOR
# ==============================================================================

def render_audio_input_selector(key_prefix: str = "rec1") -> Tuple[Optional[bytes], str, str]:
    """
    Renders the 3-mode audio input selector (Benchmark, Upload WAV, Live Mic).
    Returns (audio_bytes, source_name, source_cohort).
    """
    input_mode = st.radio(
        f"Choose Voice Input Method ({key_prefix})",
        ["Select Pre-loaded Benchmark Sample", "Upload Audio File (.wav)", "Record Live Microphone"],
        horizontal=True,
        key=f"{key_prefix}_input_mode",
        label_visibility="collapsed"
    )

    audio_bytes = None
    source_name = ""
    source_cohort = "Unknown"

    if input_mode == "Select Pre-loaded Benchmark Sample":
        hc_files = sorted(glob.glob(os.path.join(BASE_DIR, "data", "raw", "HC", "HC_AH", "*.wav")))
        pd_files = sorted(glob.glob(os.path.join(BASE_DIR, "data", "raw", "PD", "PD_AH", "*.wav")))

        col_c, col_f = st.columns([1, 2])
        with col_c:
            cohort_pick = st.selectbox(
                "Cohort Group",
                ["Healthy Control (HC) — 41 Available", "Parkinson's Disease (PD) — 40 Available"],
                key=f"{key_prefix}_cohort_pick"
            )
        with col_f:
            file_list = hc_files if "Healthy" in cohort_pick else pd_files
            if file_list:
                file_map = {os.path.basename(f): f for f in file_list}
                chosen_name = st.selectbox("Select Recording File", list(file_map.keys()), key=f"{key_prefix}_file_pick")
                chosen_path = file_map[chosen_name]
                source_name = chosen_name
                source_cohort = "Healthy Control" if "Healthy" in cohort_pick else "Parkinson's Disease"
                with open(chosen_path, "rb") as f:
                    audio_bytes = f.read()

    elif input_mode == "Upload Audio File (.wav)":
        uploaded_file = st.file_uploader(
            "Upload audio file (.wav format)",
            type=["wav"],
            key=f"{key_prefix}_uploader",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            source_name = uploaded_file.name
            source_cohort = "Uploaded External File"

    else:
        st.markdown(
            "<div style='font-size: 0.85rem; font-weight: 600; color: #1E293B; margin-bottom: 6px;'>"
            "Record a sustained vowel 'ah' for 3 to 5 seconds:</div>",
            unsafe_allow_html=True
        )
        try:
            if hasattr(st, "audio_input"):
                recorded = st.audio_input("Record Phonation", key=f"{key_prefix}_audio_input", label_visibility="collapsed")
                if recorded is not None:
                    audio_bytes = recorded.getvalue()
                    source_name = f"Live_Microphone_{key_prefix}.wav"
                    source_cohort = "Live Recording"
            else:
                st.warning("Live microphone recording is not supported in this Streamlit version. Please use WAV file upload instead.")
        except Exception as exc:
            st.warning(f"Microphone input is unavailable ({exc}). Please use WAV file upload instead.")

    return audio_bytes, source_name, source_cohort


# ==============================================================================
# PAGE 2: VOICE ANALYSIS WORKSPACE (CORE WORKSPACE & REPEAT WORKFLOW)
# ==============================================================================

def render_voice_analysis_page(pipeline: CanonicalVoicePipeline):
    st.markdown("## Voice Analysis Workspace")
    st.caption("Acoustic screening analysis on sustained phonations using the canonical Gradient Boosting pipeline.")

    # Initialize Session State Variables
    if "rec1_result" not in st.session_state:
        st.session_state.rec1_result = None
    if "rec2_result" not in st.session_state:
        st.session_state.rec2_result = None
    if "is_repeating" not in st.session_state:
        st.session_state.is_repeating = False

    # Top Action: "Analyze Another Recording" (if any analysis has been run)
    if st.session_state.rec1_result is not None:
        col_hdr, col_reset = st.columns([3, 1])
        with col_reset:
            if st.button("Analyze Another Recording", key="top_reset_btn", type="secondary", use_container_width=True):
                st.session_state.rec1_result = None
                st.session_state.rec2_result = None
                st.session_state.is_repeating = False
                st.rerun()

    # --------------------------------------------------------------------------
    # CASE A: INITIAL RECORDING (Recording 1)
    # --------------------------------------------------------------------------
    if st.session_state.rec1_result is None:
        # Short Recording Instructions
        st.markdown("""
        <div class="guide-card">
            <div class="guide-title">Recording Instructions</div>
            <ul class="guide-list">
                <li><b>Record in a quiet environment</b>: Choose a quiet room without background chatter or echoes.</li>
                <li><b>Sustain the vowel "ah" steadily</b>: Hold the sound steadily for 3 to 5 seconds on a single comfortable breath.</li>
                <li><b>Avoid background noise</b>: Keep away from fans, air conditioners, and noise sources.</li>
                <li><b>Keep microphone reasonably close</b>: Position 10 to 15 cm from lips, angled slightly off-axis.</li>
                <li><b>Single sustained vowel only</b>: Record only one sustained vowel rather than conversational speech or words.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
            Step 1: Provide Sustained Vowel /a/ Recording
        </div>
        """, unsafe_allow_html=True)

        audio_bytes, source_name, source_cohort = render_audio_input_selector(key_prefix="rec1")

        if audio_bytes is None:
            st.info("Select a benchmark sample, upload a WAV file, or record using your microphone to begin.")
            return

        # Audio Decode & Preview
        try:
            y_raw, sr_raw = sf.read(io.BytesIO(audio_bytes))
            y_mono = convert_to_mono(y_raw)
            duration_sec = len(y_mono) / sr_raw
            peak_amp = float(np.max(np.abs(y_mono)))
            rms_val = float(np.sqrt(np.mean(y_mono ** 2)))
        except Exception as exc:
            st.error(f"Failed to decode audio file: {exc}")
            return

        with st.expander("Audio Playback & Signal Inspection", expanded=True):
            col_ply, col_meta = st.columns([1, 2])
            with col_ply:
                st.audio(audio_bytes, format="audio/wav")
                st.caption(f"Source: `{source_name}` ({source_cohort})")
            with col_meta:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Duration", f"{duration_sec:.2f} s")
                m2.metric("Sample Rate", f"{sr_raw} Hz")
                m3.metric("Peak Amp", f"{peak_amp:.3f}")
                m4.metric("RMS Power", f"{rms_val:.4f}")

            plot_acoustic_waveform_and_spectrogram(y_mono, sr_raw)

        # Trigger Analysis on Recording 1
        if st.button("Run Acoustic Analysis", key="btn_run_rec1", type="primary", use_container_width=True):
            with st.spinner("Executing Audio Quality Gate & Canonical Model Inference..."):
                t_start = time.time()
                output: PredictionOutput = pipeline.predict_from_audio(y_raw, sr=sr_raw)
                elapsed_sec = time.time() - t_start

            # Handle Quality Gate Rejections
            if output.status == "REJECTED":
                headline, guidance = format_user_friendly_rejection(output.rejection_reason)
                st.markdown(f"""
                <div class="research-card card-rejection">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #991B1B; margin-bottom: 6px;">
                        {headline}
                    </div>
                    <div style="font-size: 0.88rem; color: #7F1D1D; margin-bottom: 8px;">
                        {guidance}
                    </div>
                    <div style="font-size: 0.78rem; color: #991B1B;">
                        Please review the recording guide above and try again with a steady sustained vowel phonation.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                return

            # Store Valid Result
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            category = map_category_label(output.status)
            st.session_state.rec1_result = {
                "output": output,
                "elapsed": elapsed_sec,
                "source_name": source_name,
                "source_cohort": source_cohort,
                "timestamp": now_utc,
                "category": category,
                "score": output.pd_model_score,
                "is_inconclusive": output.is_inconclusive
            }
            st.rerun()

    # --------------------------------------------------------------------------
    # CASE B: FIRST RECORDING COMPLETED (and Not Repeating)
    # --------------------------------------------------------------------------
    elif st.session_state.rec1_result is not None and not st.session_state.is_repeating:
        rec1 = st.session_state.rec1_result
        output1: PredictionOutput = rec1["output"]

        # Prominent MODEL OUTPUT Section
        render_prominent_model_output(
            category=rec1["category"],
            pd_score=rec1["score"],
            is_inconclusive=rec1["is_inconclusive"],
            timestamp_str=rec1["timestamp"],
            recording_label="Recording 1"
        )

        # What does this result mean?
        render_what_does_result_mean(rec1["category"])

        # Top Acoustic Feature Deviations
        st.markdown("### Top Acoustic Feature Deviations")
        st.caption(
            "Feature deviations relative to training cohort reference mean. "
            "Relative Model Importance represents tree-split weight across training folds, not individual biological causation."
        )

        cols = st.columns(min(len(output1.top_feature_contributions), 5))
        for idx, feat in enumerate(output1.top_feature_contributions[:5]):
            col = cols[idx % len(cols)]
            with col:
                st.markdown(f"""
                <div class="metric-container" style="padding: 12px; margin-bottom: 8px;">
                    <div style="font-family: monospace; font-weight: 700; font-size: 0.82rem; color: #0F172A;">{feat['feature']}</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 3px 0;">{feat['value']:.2f}</div>
                    <div style="margin-top: 3px;">
                        <span style="background: #F1F5F9; color: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; font-family: monospace;">
                            {feat['z_score']:+.2f} SD
                        </span>
                    </div>
                    <div style="font-size: 0.74rem; color: #64748B; margin-top: 5px;">
                        Tree Weight: {feat['importance_weight']:.3f}<br>{feat['direction']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Offer Second Recording Workflow
        is_pd_or_inconclusive = rec1["category"] in ["PD-Associated Acoustic Pattern", "Inconclusive"]
        
        st.markdown("""
        <div class="research-card" style="border-left: 3px solid #0284C7; padding: 18px 20px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                Repeat Recording Workflow
            </div>
            <div style="font-size: 0.85rem; color: #334155; line-height: 1.5; margin-bottom: 12px;">
        """, unsafe_allow_html=True)

        if is_pd_or_inconclusive:
            st.markdown(
                f"Because the initial analysis produced a **{rec1['category']}**, "
                "providing a second independent recording is recommended to evaluate acoustic consistency across phonations."
            )
        else:
            st.markdown(
                "You may provide a second independent recording to verify consistency across separate phonations."
            )

        st.markdown("</div>", unsafe_allow_html=True)

        col_rep, col_new = st.columns([1, 1])
        with col_rep:
            if st.button("Record Another Sample (Provide Second Recording)", key="btn_start_repeat", type="primary", use_container_width=True):
                st.session_state.is_repeating = True
                st.rerun()

        with col_new:
            if st.button("Analyze Another Recording", key="btn_reset_after_rec1", type="secondary", use_container_width=True):
                st.session_state.rec1_result = None
                st.session_state.rec2_result = None
                st.session_state.is_repeating = False
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # CASE C: REPEAT RECORDING IN PROGRESS (Recording 2)
    # --------------------------------------------------------------------------
    elif st.session_state.is_repeating and st.session_state.rec2_result is None:
        rec1 = st.session_state.rec1_result

        # Compact Reference Banner for Recording 1
        rec1_border = "#DC2626" if rec1["category"] == "PD-Associated Acoustic Pattern" else ("#16A34A" if rec1["category"] == "HC-Associated Acoustic Pattern" else "#D97706")
        rec1_text_color = "#991B1B" if rec1["category"] == "PD-Associated Acoustic Pattern" else ("#166534" if rec1["category"] == "HC-Associated Acoustic Pattern" else "#92400E")
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid {rec1_border}; border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.85rem; color: #0F172A;">
                <b>Recording 1 Baseline</b>: <span style="font-weight: 700; color: {rec1_text_color};">{rec1['category']}</span> (PD Model Score: {rec1['score']:.1%})
            </div>
            <div style="font-size: 0.78rem; color: #64748B;">{rec1['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
            Step 2: Provide Second Sustained Vowel /a/ Recording (Independent Phonation)
        </div>
        """, unsafe_allow_html=True)

        audio_bytes2, source_name2, source_cohort2 = render_audio_input_selector(key_prefix="rec2")

        if audio_bytes2 is None:
            st.info("Select, upload, or record your second sample to proceed with the repeat analysis.")
            return

        try:
            y_raw2, sr_raw2 = sf.read(io.BytesIO(audio_bytes2))
            y_mono2 = convert_to_mono(y_raw2)
            duration_sec2 = len(y_mono2) / sr_raw2
            peak_amp2 = float(np.max(np.abs(y_mono2)))
            rms_val2 = float(np.sqrt(np.mean(y_mono2 ** 2)))
        except Exception as exc:
            st.error(f"Failed to decode second audio file: {exc}")
            return

        with st.expander("Second Sample Audio Playback & Signal Inspection", expanded=True):
            c_p, c_m = st.columns([1, 2])
            with c_p:
                st.audio(audio_bytes2, format="audio/wav")
                st.caption(f"Second Sample: `{source_name2}`")
            with c_m:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Duration", f"{duration_sec2:.2f} s")
                m2.metric("Sample Rate", f"{sr_raw2} Hz")
                m3.metric("Peak Amp", f"{peak_amp2:.3f}")
                m4.metric("RMS Power", f"{rms_val2:.4f}")

            plot_acoustic_waveform_and_spectrogram(y_mono2, sr_raw2)

        if st.button("Run Analysis on Second Recording", key="btn_run_rec2", type="primary", use_container_width=True):
            with st.spinner("Analyzing second recording independently through canonical pipeline..."):
                t_start = time.time()
                output2: PredictionOutput = pipeline.predict_from_audio(y_raw2, sr=sr_raw2)
                elapsed_sec2 = time.time() - t_start

            if output2.status == "REJECTED":
                headline, guidance = format_user_friendly_rejection(output2.rejection_reason)
                st.markdown(f"""
                <div class="research-card card-rejection">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #991B1B; margin-bottom: 6px;">
                        Second Sample Rejected: {headline}
                    </div>
                    <div style="font-size: 0.88rem; color: #7F1D1D; margin-bottom: 8px;">
                        {guidance}
                    </div>
                    <div style="font-size: 0.78rem; color: #991B1B;">
                        Please provide a valid second recording according to the protocol.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                return

            now_utc2 = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            category2 = map_category_label(output2.status)
            st.session_state.rec2_result = {
                "output": output2,
                "elapsed": elapsed_sec2,
                "source_name": source_name2,
                "source_cohort": source_cohort2,
                "timestamp": now_utc2,
                "category": category2,
                "score": output2.pd_model_score,
                "is_inconclusive": output2.is_inconclusive
            }
            st.rerun()

    # --------------------------------------------------------------------------
    # CASE D: BOTH RECORDINGS COMPLETED — INDEPENDENT & COMBINED FINDINGS
    # --------------------------------------------------------------------------
    elif st.session_state.rec1_result is not None and st.session_state.rec2_result is not None:
        rec1 = st.session_state.rec1_result
        rec2 = st.session_state.rec2_result

        st.markdown("### MODEL OUTPUT — TWO-RECORDING ANALYSIS")
        st.caption("Each recording was evaluated independently through the canonical pipeline. Scores are never averaged.")

        # Side-by-Side Independent Result Cards
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if rec1["category"] == "PD-Associated Acoustic Pattern":
                badge_cls1 = "badge-pd"
                dot_c1 = "#DC2626"
                card_border1 = "#DC2626"
            elif rec1["category"] == "HC-Associated Acoustic Pattern":
                badge_cls1 = "badge-hc"
                dot_c1 = "#16A34A"
                card_border1 = "#16A34A"
            else:
                badge_cls1 = "badge-inconclusive"
                dot_c1 = "#D97706"
                card_border1 = "#D97706"

            st.markdown(f"""
            <div class="research-card" style="border-top: 3px solid {card_border1};">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                    Recording 1 Result (Independent Phonation)
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="status-badge {badge_cls1}">
                        <span style="color: {dot_c1}; font-size: 0.85rem;">●</span> {rec1['category']}
                    </span>
                </div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A; margin-bottom: 4px;">
                    PD Model Score: {rec1['score']:.1%}
                </div>
                <div style="font-size: 0.75rem; color: #64748B; line-height: 1.5;">
                    Sample: <code>{rec1['source_name']}</code><br>
                    Analyzed: {rec1['timestamp']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            if rec2["category"] == "PD-Associated Acoustic Pattern":
                badge_cls2 = "badge-pd"
                dot_c2 = "#DC2626"
                card_border2 = "#DC2626"
            elif rec2["category"] == "HC-Associated Acoustic Pattern":
                badge_cls2 = "badge-hc"
                dot_c2 = "#16A34A"
                card_border2 = "#16A34A"
            else:
                badge_cls2 = "badge-inconclusive"
                dot_c2 = "#D97706"
                card_border2 = "#D97706"

            st.markdown(f"""
            <div class="research-card" style="border-top: 3px solid {card_border2};">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                    Recording 2 Result (Independent Phonation)
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="status-badge {badge_cls2}">
                        <span style="color: {dot_c2}; font-size: 0.85rem;">●</span> {rec2['category']}
                    </span>
                </div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A; margin-bottom: 4px;">
                    PD Model Score: {rec2['score']:.1%}
                </div>
                <div style="font-size: 0.75rem; color: #64748B; line-height: 1.5;">
                    Sample: <code>{rec2['source_name']}</code><br>
                    Analyzed: {rec2['timestamp']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Standardized Combined Comparison Finding
        cat1 = rec1["category"]
        cat2 = rec2["category"]

        if cat1 == "PD-Associated Acoustic Pattern" and cat2 == "PD-Associated Acoustic Pattern":
            comparison_text = "Both recordings produced a PD-associated acoustic pattern. This result is not a diagnosis. Consider discussing your concerns with a qualified healthcare professional."
            comp_title = "Consistent PD-Associated Pattern"
            comp_border = "#DC2626"
            comp_title_color = "#991B1B"
        elif cat1 == "HC-Associated Acoustic Pattern" and cat2 == "HC-Associated Acoustic Pattern":
            comparison_text = "Both recordings produced an HC-associated acoustic pattern. This does not rule out Parkinson's disease or replace a clinical evaluation."
            comp_title = "Consistent HC-Associated Pattern"
            comp_border = "#16A34A"
            comp_title_color = "#166534"
        elif cat1 == "Inconclusive" and cat2 == "Inconclusive":
            comparison_text = "Both recordings were inconclusive. Try recording again under the recommended recording conditions. If you remain concerned about your health, consider discussing your concerns with a qualified healthcare professional."
            comp_title = "Both Recordings Inconclusive"
            comp_border = "#D97706"
            comp_title_color = "#92400E"
        else:
            comparison_text = "The two recordings produced different results. The findings are not consistent across recordings. Consider repeating the recording under the recommended protocol if you wish to perform another analysis."
            comp_title = "Inconsistent Cross-Recording Findings"
            comp_border = "#D97706"
            comp_title_color = "#92400E"

        st.markdown(f"""
        <div class="comparison-card" style="border-left: 4px solid {comp_border};">
            <div style="font-size: 0.95rem; font-weight: 700; color: {comp_title_color}; margin-bottom: 6px;">
                Cross-Recording Finding: {comp_title}
            </div>
            <div style="font-size: 0.88rem; color: #1E293B; line-height: 1.6; font-weight: 500;">
                {comparison_text}
            </div>
            <div class="transparency-bar" style="margin-top: 10px;">
                Model Version: <b>2.1.0</b> &nbsp;•&nbsp; 
                Evaluation: <b>Two Independent Phonation Assessments</b> &nbsp;•&nbsp; 
                Non-Clinical Screening Output
            </div>
        </div>
        """, unsafe_allow_html=True)

        # What does this result mean?
        dominant_cat = cat1 if cat1 == cat2 else "Inconclusive"
        render_what_does_result_mean(dominant_cat)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        if st.button("Analyze Another Recording", key="btn_reset_full", type="primary", use_container_width=True):
            st.session_state.rec1_result = None
            st.session_state.rec2_result = None
            st.session_state.is_repeating = False
            st.rerun()


# ==============================================================================
# PAGE 3: MODEL INSIGHTS & VALIDATION
# ==============================================================================

def render_model_insights_page(report: dict):
    st.markdown("## Model Validation & Generalization Insights")
    st.caption("Transparent evaluation metrics on the 81-recording dataset using strict nested cross-validation.")

    # Headline Validation Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Mean Outer CV ROC-AUC</div>
            <div class="metric-val" style="color: #0284C7;">0.675 ± 0.058</div>
            <div class="metric-sub">5 Outer Folds × 3 Inner Folds</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Out-of-Fold ROC-AUC</div>
            <div class="metric-val">0.706</div>
            <div class="metric-sub">Aggregated Out-of-Fold Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Mean Balanced Accuracy</div>
            <div class="metric-val">66.4% ± 9.5%</div>
            <div class="metric-sub">Out-of-Fold Balanced Acc: 66.7%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Sensitivity / Specificity</div>
            <div class="metric-val" style="font-size: 1.25rem;">67.5% / 65.3%</div>
            <div class="metric-sub">Mean F1: 0.671 ± 0.055</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Conclusive-Subset Analysis Card
    st.markdown("### Conclusive-Subset Evaluation (Decision Guardrail)")
    st.markdown("""
    <div class="research-card">
        <div style="font-size: 0.90rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
            Conclusive-Subset Performance (Excluding Indeterminate Band [0.40, 0.60])
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 12px;">
            <div class="metric-container">
                <div class="metric-label">Conclusive-Subset Accuracy</div>
                <div class="metric-val">68.3%</div>
                <div class="metric-sub">43 of 63 conclusive samples</div>
            </div>
            <div class="metric-container">
                <div class="metric-label">Conclusive Coverage</div>
                <div class="metric-val">77.8%</div>
                <div class="metric-sub">63 of 81 total recordings</div>
            </div>
            <div class="metric-container">
                <div class="metric-label">Indeterminate Rate</div>
                <div class="metric-val">22.2%</div>
                <div class="metric-sub">18 recordings flagged for repeat</div>
            </div>
            <div class="metric-container">
                <div class="metric-label">Full Dataset OOF Balanced Acc</div>
                <div class="metric-val">66.7%</div>
                <div class="metric-sub">Across all 81 samples (no exclusions)</div>
            </div>
        </div>
        <div style="font-size: 0.80rem; color: #64748B; line-height: 1.5;">
            <b>Methodological Note</b>: The <b>68.3%</b> accuracy applies strictly to the conclusive subset (63 samples, 77.8% coverage).
            When all 81 samples are evaluated without an indeterminate band, the out-of-fold balanced accuracy is <b>66.7%</b> (ROC-AUC: <b>0.706</b>).
            Reserving ambiguous samples ([0.40, 0.60]) improves decision certainty on actionable screening cases.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Candidate Model Benchmark Comparison
    st.markdown("### Candidate Model Benchmark (5×3 Nested Cross-Validation)")
    st.caption("All models underwent identical inner-loop hyperparameter optimization and feature selection with zero test fold leakage.")

    if report and "nested_cross_validation" in report:
        nested_cv = report["nested_cross_validation"]
        rows = []
        for model_name, m in nested_cv.items():
            is_selected = " (Selected)" if "Gradient" in model_name else ""
            rows.append({
                "Algorithm": f"{model_name}{is_selected}",
                "Outer CV AUC": f"{m['mean_auc']:.3f} ± {m['std_auc']:.3f}",
                "OOF AUC": f"{m['oof_auc']:.3f}",
                "Balanced Accuracy": f"{m['mean_bal_acc']:.1%} ± {m['std_bal_acc']:.1%}",
                "Sensitivity": f"{m['mean_sensitivity']:.1%}",
                "Specificity": f"{m['mean_specificity']:.1%}",
                "F1 Score": f"{m['mean_f1']:.3f}"
            })
        df_bench = pd.DataFrame(rows).set_index("Algorithm")
        st.dataframe(df_bench, use_container_width=True)

    # Canonical Hyperparameters and Feature Selector
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### Selected Architecture & Hyperparameters")
    col_par, col_feat = st.columns([1, 1])

    with col_par:
        st.markdown("""
        <div class="research-card">
            <div style="font-weight: 700; font-size: 0.88rem; color: #0F172A; margin-bottom: 8px;">Gradient Boosting Pipeline Config</div>
            <div style="font-size: 0.82rem; line-height: 1.6; color: #334155;">
                • <b>Estimator</b>: <code>GradientBoostingClassifier</code><br>
                • <b>Number of Trees (n_estimators)</b>: <code>30</code> (restrained to prevent overfitting)<br>
                • <b>Max Tree Depth</b>: <code>3</code><br>
                • <b>Learning Rate</b>: <code>0.08</code><br>
                • <b>Feature Selector</b>: <code>SelectKBest(f_classif, k=10)</code> (fitted inside training folds)<br>
                • <b>Scaling & Imputation</b>: <code>StandardScaler</code> + <code>SimpleImputer(median)</code><br>
                • <b>Random State</b>: <code>42</code> (fully deterministic and reproducible)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_feat:
        st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;'>Global Top 10 Feature Importance</div>", unsafe_allow_html=True)
        render_feature_importance_barchart(report, top_k=10)

    # Dataset Independence Audit
    st.markdown("### Dataset Audit & Integrity Verification")
    st.markdown("""
    <div class="research-card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;">
            <div>
                <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">Total Real Audio Files</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">81 Recordings</div>
                <div style="font-size: 0.78rem; color: #334155;">40 Parkinson's Disease, 41 Healthy Controls</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">Subject Independence</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">100% Unique</div>
                <div style="font-size: 0.78rem; color: #334155;">81 distinct subjects, 0 cross-cohort duplicates</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">Data Completeness</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">0 NaNs / Missing</div>
                <div style="font-size: 0.78rem; color: #334155;">100% complete across all 45 acoustic features</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: ABOUT / PROTOCOL
# ==============================================================================

def render_protocol_page():
    st.markdown("## Recording Protocol & Research Methodology")
    st.caption("Standardized audio acquisition procedures, quality gating thresholds, and ethical research boundaries.")

    # Standardized Phonation Protocol
    st.markdown("""
    <div class="research-card">
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 10px;">
            1. Standardized Phonation Protocol
        </div>
        <div style="font-size: 0.85rem; color: #1E293B; line-height: 1.6;">
            To ensure reproducible acoustic biomarkers and prevent extraneous aerodynamic confounding:
            <ul style="margin-top: 6px;">
                <li><b>Phonation Task</b>: Sustained open vowel <b>/a/</b> ("ah", as in <i>father</i>).</li>
                <li><b>Target Duration</b>: <b>3 to 5 seconds</b> of steady vocalization on a single comfortable breath.</li>
                <li><b>Pitch & Loudness</b>: Natural modal pitch and conversational volume. Avoid pitch glides, singing, whispering, or vocal fry.</li>
                <li><b>Microphone Placement</b>: 10 to 15 cm from lips, oriented at approximately 45 degrees off-axis to avoid direct plosive airflow.</li>
                <li><b>Acoustic Environment</b>: Quiet room with minimal reverberation (&lt;40 dB ambient noise, no fans or AC).</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quality Gate Specifications Table
    st.markdown("""
    <div class="research-card">
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 10px;">
            2. Automated Audio Quality Gate Thresholds
        </div>
        <div style="font-size: 0.85rem; color: #1E293B; line-height: 1.6;">
            Every recording is automatically screened prior to acoustic feature extraction and model inference:
        </div>
        <div style="margin-top: 10px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; color: #1E293B;">
                <thead>
                    <tr style="border-bottom: 1px solid #CBD5E1; text-align: left; background: #F8FAFC;">
                        <th style="padding: 8px 10px; color: #0F172A;">Parameter</th>
                        <th style="padding: 8px 10px; color: #0F172A;">Threshold Requirement</th>
                        <th style="padding: 8px 10px; color: #0F172A;">Scientific Rationale</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 8px 10px; font-weight: 600;">Minimum Duration</td>
                        <td style="padding: 8px 10px;"><code>&ge; 1.2 seconds</code></td>
                        <td style="padding: 8px 10px;">Ensures sufficient vocal cycles for stable perturbation & HNR calculation.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 8px 10px; font-weight: 600;">Maximum Duration</td>
                        <td style="padding: 8px 10px;"><code>&le; 10.0 seconds</code></td>
                        <td style="padding: 8px 10px;">Avoids vocal fatigue and extraneous respiratory artifacts.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 8px 10px; font-weight: 600;">Clipping Limit</td>
                        <td style="padding: 8px 10px;"><code>&lt; 2.0% samples at full scale</code></td>
                        <td style="padding: 8px 10px;">Prevents spurious harmonic distortion and artificial spectral centroid elevation.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 8px 10px; font-weight: 600;">Audible Energy (RMS)</td>
                        <td style="padding: 8px 10px;"><code>RMS &ge; 0.008</code></td>
                        <td style="padding: 8px 10px;">Screens out ambient room noise, whispers, or silent recordings.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 8px 10px; font-weight: 600;">Pitch Stability (F0 CV)</td>
                        <td style="padding: 8px 10px;"><code>F0 CV &le; 0.40</code></td>
                        <td style="padding: 8px 10px;">Rejects non-sustained pitch glides, reading speech, or singing.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Scientific Limitations & Non-Clinical Disclaimer
    st.markdown("""
    <div class="research-card" style="border-left: 3px solid #0284C7;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
            3. Research Scope, Limitations & Non-Clinical Disclaimer
        </div>
        <div style="font-size: 0.84rem; color: #334155; line-height: 1.6;">
            <b>Academic Prototype Notice</b>: This software is an exploratory research prototype developed for academic investigation.
            It is <b>NOT</b> an approved medical device, diagnostic test, or clinical instrument.
            <br><br>
            <b>Key Limitations</b>:
            <ol style="margin-top: 4px; padding-left: 18px;">
                <li><b>Cohort Scale (n=81)</b>: Validated strictly across 81 real individuals. While subject-independent, it cannot represent global demographic or dialectal diversity.</li>
                <li><b>Telephonic Bandwidth (8 kHz)</b>: The training dataset was recorded at 8,000 Hz, limiting acoustic analysis to frequencies below 4,000 Hz.</li>
                <li><b>Etiological Non-Specificity</b>: Vocal perturbation (jitter, shimmer) and spectral irregularities are dysphonic signs that also manifest in non-Parkinsonian conditions (e.g. presbyphonia, muscle tension dysphonia, laryngitis).</li>
            </ol>
            <b>Clinical Guidance</b>: This system's predictions must never be used to initiate, withhold, or alter medical treatment.
            For any medical concerns, consult a qualified neurologist or physician.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Early Detection of Parkinson's Disease — Voice Screening Workspace",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_sidebar()

    pipeline = load_pipeline()
    report = load_training_report()

    if pipeline is None:
        st.error(
            f"Trained pipeline model not found at `{MODEL_FILE}`. "
            "Please run `python train.py` to train and serialize the canonical model."
        )
        return

    current = st.session_state.get("current_page", "Overview")

    if current == "Overview":
        render_overview_page()
    elif current == "Voice Analysis":
        render_voice_analysis_page(pipeline)
    elif current == "Model Insights":
        render_model_insights_page(report)
    elif current == "About / Protocol":
        render_protocol_page()
    else:
        render_overview_page()


if __name__ == "__main__":
    main()
