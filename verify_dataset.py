import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PD_DIR = os.path.join(BASE_DIR, "data", "raw", "PD")
HC_DIR = os.path.join(BASE_DIR, "data", "raw", "HC")
EXCEL_FILE = os.path.join(BASE_DIR, "data", "raw", "Demographics_age_sex.xlsx")


def get_wav_files(folder):
    wav_files = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(file)

    return wav_files


pd_files = get_wav_files(PD_DIR)
hc_files = get_wav_files(HC_DIR)

all_files = pd_files + hc_files

print("\n===== DATASET VERIFICATION =====\n")

print(f"PD WAV files : {len(pd_files)}")
print(f"HC WAV files : {len(hc_files)}")
print(f"Total WAVs   : {len(all_files)}")

df = pd.read_excel(EXCEL_FILE, sheet_name="Parselmouth")

excel_ids = set(df["Sample ID"].astype(str).str.strip())

wav_ids = set(os.path.splitext(file)[0] for file in all_files)

missing_wavs = excel_ids - wav_ids
extra_wavs = wav_ids - excel_ids

print(f"\nExcel samples : {len(excel_ids)}")
print(f"Missing WAVs  : {len(missing_wavs)}")
print(f"Extra WAVs    : {len(extra_wavs)}")

print("\nLabels in Excel:")
print(df["Label"].value_counts().to_string())

if missing_wavs:
    print("\nMissing WAV files:")
    for item in sorted(missing_wavs):
        print(item)

if extra_wavs:
    print("\nWAV files not found in Excel:")
    for item in sorted(extra_wavs):
        print(item)

print("\n===== RESULT =====")

if (
    len(pd_files) == 40
    and len(hc_files) == 41
    and len(all_files) == 81
    and len(missing_wavs) == 0
    and len(extra_wavs) == 0
):
    print("DATASET VERIFIED SUCCESSFULLY")
else:
    print("SOMETHING DOES NOT MATCH")
    