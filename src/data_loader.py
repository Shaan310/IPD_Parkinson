"""
Dataset loading and subject identity verification.
"""
import os
import re
import pandas as pd
from typing import Optional, Tuple, List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
FALLBACK_HC_DIR = os.environ.get("HC_AH_DIR", "")
FALLBACK_PD_DIR = os.environ.get("PD_AH_DIR", "")


def extract_subject_id(filename: str) -> str:
    """
    Extract subject identifier prefix from audio filename.
    Filenames follow patterns such as:
      AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav -> subject '064F'
      AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.wav -> subject '545616858'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) >= 2:
        # Check if second part has a hyphen (like AH_545616858-UUID)
        sub_part = parts[1].split("-")[0]
        return sub_part
    return base


def get_dataset_files(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Scan for HC and PD audio recordings. Returns DataFrame with:
    [sample_id, file_path, label_str, label, subject_id]
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    hc_dir = os.path.join(data_dir, "HC")
    pd_dir = os.path.join(data_dir, "PD")

    # If local data_dir is missing or empty, use fallback dirs
    def collect_wavs(folder: str, fallback_folder: str) -> List[str]:
        wavs = []
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(".wav"):
                        wavs.append(os.path.join(root, f))
        if not wavs and os.path.exists(fallback_folder):
            for root, _, files in os.walk(fallback_folder):
                for f in files:
                    if f.lower().endswith(".wav"):
                        wavs.append(os.path.join(root, f))
        return sorted(wavs)

    hc_files = collect_wavs(hc_dir, FALLBACK_HC_DIR)
    pd_files = collect_wavs(pd_dir, FALLBACK_PD_DIR)

    records = []
    for p in hc_files:
        filename = os.path.basename(p)
        sample_id = os.path.splitext(filename)[0]
        records.append({
            "sample_id": sample_id,
            "file_path": p,
            "label_str": "HC",
            "label": 0,
            "subject_id": extract_subject_id(filename)
        })

    for p in pd_files:
        filename = os.path.basename(p)
        sample_id = os.path.splitext(filename)[0]
        records.append({
            "sample_id": sample_id,
            "file_path": p,
            "label_str": "PwPD",
            "label": 1,
            "subject_id": extract_subject_id(filename)
        })

    df = pd.DataFrame(records)
    return df


def audit_dataset_integrity(df: pd.DataFrame) -> Dict[str, any]:
    """
    Audit dataset composition, subject leakage potential, and class balance.
    """
    total = len(df)
    n_hc = (df["label"] == 0).sum()
    n_pd = (df["label"] == 1).sum()

    hc_subjects = df[df["label"] == 0]["subject_id"]
    pd_subjects = df[df["label"] == 1]["subject_id"]

    unique_hc_subjects = hc_subjects.nunique()
    unique_pd_subjects = pd_subjects.nunique()
    total_unique_subjects = df["subject_id"].nunique()

    # Check for subject overlap between classes
    overlap = set(hc_subjects).intersection(set(pd_subjects))

    has_duplicate_subjects = (total != total_unique_subjects)

    audit = {
        "total_recordings": int(total),
        "hc_count": int(n_hc),
        "pd_count": int(n_pd),
        "hc_unique_subjects": int(unique_hc_subjects),
        "pd_unique_subjects": int(unique_pd_subjects),
        "total_unique_subjects": int(total_unique_subjects),
        "subject_overlap": list(overlap),
        "has_duplicate_subjects": bool(has_duplicate_subjects),
        "is_subject_independent": not has_duplicate_subjects and len(overlap) == 0
    }
    return audit


if __name__ == "__main__":
    df = get_dataset_files()
    print(f"Loaded {len(df)} recordings.")
    audit = audit_dataset_integrity(df)
    for k, v in audit.items():
        print(f"  {k}: {v}")
