"""
Dataset Verification Script.
Audits the raw Parkinson's Voice dataset in data/raw/:
- Checks file counts (40 PD, 41 HC = 81 total)
- Verifies subject IDs and cross-cohort independence (no subject overlap)
- Checks sample rate, channel count, and duration distribution
"""
import os
import soundfile as sf
import pandas as pd
from src.data_loader import get_dataset_files, audit_dataset_integrity


def verify_dataset():
    print("\n" + "=" * 60)
    print("PARKINSON'S VOICE DATASET AUDIT & INTEGRITY VERIFICATION")
    print("=" * 60)

    df_meta = get_dataset_files()
    audit = audit_dataset_integrity(df_meta)

    print(f"Total WAV files found : {audit['total_recordings']} (expected 81)")
    print(f"Healthy Controls (HC) : {audit['hc_count']} (expected 41)")
    print(f"Parkinson's (PwPD)    : {audit['pd_count']} (expected 40)")
    print(f"Unique Subjects       : {audit['total_unique_subjects']}")
    print(f"Subject Overlap       : {audit['subject_overlap']}")
    print(f"Subject Independence  : {audit['is_subject_independent']}")

    # Audio properties inspection
    sample_rates = set()
    channels = set()
    durations = []

    for path in df_meta["file_path"]:
        info = sf.info(path)
        sample_rates.add(info.samplerate)
        channels.add(info.channels)
        durations.append(info.duration)

    print(f"\nAudio Properties:")
    print(f"  Sample Rates        : {list(sample_rates)} Hz")
    print(f"  Channels            : {list(channels)} (1 = Mono)")
    print(f"  Duration Range      : {min(durations):.2f}s to {max(durations):.2f}s (mean {sum(durations)/len(durations):.2f}s)")

    # Result verification
    is_valid = (
        audit["total_recordings"] == 81 and
        audit["hc_count"] == 41 and
        audit["pd_count"] == 40 and
        len(audit["subject_overlap"]) == 0 and
        audit["is_subject_independent"]
    )

    print("\n" + "=" * 60)
    if is_valid:
        print("VERIFICATION RESULT: SUCCESS (Dataset is intact and subject-independent)")
    else:
        print("VERIFICATION RESULT: FAILED (Dataset mismatch or leakage detected)")
    print("=" * 60 + "\n")
    return is_valid


if __name__ == "__main__":
    import sys
    success = verify_dataset()
    sys.exit(0 if success else 1)