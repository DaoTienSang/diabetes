import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_raw


COL_MAP = {
    "SL.NO": "sl_no",
    "NAME": "name",
    "AGE": "age",
    "SEX": "sex",
    "BMI": "bmi",
    "SP": "sp",
    "BP": "bp",
    "HbA1c": "hba1c",
    "FPS": "fps",
    "PPS": "pps",
    "FAMILY H/O": "family_h",
    "ONSET AGE": "onset_age",
    "DIA LIFE": "dia_life",
    "SMOKING": "smoking",
    "PHY ACT": "phy_act",
    "MED USE": "med_use",
    "MED ADH": "med_adh",
    "NEP": "nep",
    "NEU": "neu",
    "RET": "ret",
    "CV": "cv",
    "PER VAS": "per_vas",
}

FEATURE_COLS = [
    "age",
    "sex",
    "bmi",
    "sp",
    "bp",
    "hba1c",
    "fps",
    "pps",
    "family_h",
    "onset_age",
    "dia_life",
    "smoking",
    "phy_act",
    "med_use",
    "med_adh",
]

TARGET_COLS = ["nep", "neu", "ret", "cv", "per_vas"]


def norm_cols(df):
    return df.rename(columns=COL_MAP)


def to_year(v):
    if pd.isna(v):
        return np.nan

    if isinstance(v, (int, float, np.integer, np.floating)):
        return v

    txt = str(v).strip()
    num = re.search(r"[-+]?\d*\.?\d+", txt)
    if not num:
        return np.nan

    val = float(num.group())
    low = txt.lower()
    if "month" in low or "months" in low or "mon" in low:
        return val / 12
    return val


def clean_df(df):
    rec = norm_cols(df.copy())

    if "dia_life" in rec.columns:
        rec["dia_life"] = rec["dia_life"].apply(to_year)

    cols = [c for c in FEATURE_COLS + TARGET_COLS if c in rec.columns]
    for col in cols:
        rec[col] = pd.to_numeric(rec[col], errors="coerce")

    id_cols = ["sl_no", "name"]
    data_cols = [c for c in rec.columns if c not in id_cols]
    rec = rec.drop_duplicates(subset=data_cols)
    return rec.reset_index(drop=True)


def save_clean(raw_path="data/raw/data.csv", out_path="data/processed/clean_data.csv"):
    df = load_raw(raw_path)
    rec0 = norm_cols(df.copy())

    if "dia_life" in rec0.columns:
        rec0["dia_life"] = rec0["dia_life"].apply(to_year)

    cols = [c for c in FEATURE_COLS + TARGET_COLS if c in rec0.columns]
    for col in cols:
        rec0[col] = pd.to_numeric(rec0[col], errors="coerce")

    id_cols = ["sl_no", "name"]
    data_cols = [c for c in rec0.columns if c not in id_cols]
    dup = rec0[rec0.duplicated(subset=data_cols, keep=False)]
    dup_fp = Path("reports/duplicate_rows.csv")
    dup_fp.parent.mkdir(parents=True, exist_ok=True)
    dup.to_csv(dup_fp, index=False)

    rec = rec0.drop_duplicates(subset=data_cols).reset_index(drop=True)
    fp = Path(out_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(fp, index=False)
    return rec


def main():
    raw_path = "data/raw/data.csv"
    out_path = "data/processed/clean_data.csv"
    df = load_raw(raw_path)
    rec0 = norm_cols(df.copy())
    if "dia_life" in rec0.columns:
        rec0["dia_life"] = rec0["dia_life"].apply(to_year)
    cols = [c for c in FEATURE_COLS + TARGET_COLS if c in rec0.columns]
    for col in cols:
        rec0[col] = pd.to_numeric(rec0[col], errors="coerce")
    id_cols = ["sl_no", "name"]
    data_cols = [c for c in rec0.columns if c not in id_cols]
    dup = int(rec0.duplicated(subset=data_cols).sum())
    rec = save_clean(raw_path, out_path)
    cnt = {c: int(rec[c].sum()) for c in TARGET_COLS if c in rec.columns}
    print("raw_shape", df.shape)
    print("dup_no_id", dup)
    print("clean_shape", rec.shape)
    print("target_counts", cnt)


if __name__ == "__main__":
    main()
