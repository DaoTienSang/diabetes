from pathlib import Path

import pandas as pd

from src.data.clean_data import FEATURE_COLS, TARGET_COLS, clean_df, norm_cols, save_clean, to_year
from src.data.load_data import load_raw


RAW_COLS = [
    "SL.NO",
    "NAME",
    "AGE",
    "SEX",
    "BMI",
    "SP",
    "BP",
    "HbA1c",
    "FPS",
    "PPS",
    "FAMILY H/O",
    "ONSET AGE",
    "DIA LIFE",
    "SMOKING",
    "PHY ACT",
    "MED USE",
    "MED ADH",
    "NEP",
    "NEU",
    "RET",
    "CV",
    "PER VAS",
]


def sample_df():
    rec = [
        [1, "A", 40, 1, 24.5, 120, 80, 7.1, 100, 130, 0, 35, "12 months", 0, 1, 1, 2, 0, 1, 0, 1, 0],
        [2, "B", 40, 1, 24.5, 120, 80, 7.1, 100, 130, 0, 35, "12 months", 0, 1, 1, 2, 0, 1, 0, 1, 0],
        [3, "C", 55, 0, 28.0, 130, 85, 8.2, 120, 150, 1, 45, "24 months", 1, 0, 2, 1, 1, 0, 1, 0, 1],
    ]
    return pd.DataFrame(rec, columns=RAW_COLS)


def test_load_raw_reads_existing_file(tmp_path):
    fp = tmp_path / "data.csv"
    sample_df().to_csv(fp, index=False)
    df = load_raw(fp)
    assert df.shape == (3, 22)


def test_norm_cols_renames_columns():
    df = norm_cols(sample_df())
    cols = [
        "sl_no",
        "name",
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
        "nep",
        "neu",
        "ret",
        "cv",
        "per_vas",
    ]
    assert list(df.columns) == cols


def test_to_year_values():
    assert to_year("12 months") == 1
    assert to_year("24 months") == 2
    assert to_year("6 months") == 0.5
    assert to_year("5") == 5
    assert to_year(3) == 3


def test_clean_df_has_feature_cols():
    df = clean_df(sample_df())
    for col in FEATURE_COLS:
        assert col in df.columns


def test_clean_df_has_target_cols():
    df = clean_df(sample_df())
    for col in TARGET_COLS:
        assert col in df.columns


def test_id_cols_not_in_features():
    assert "sl_no" not in FEATURE_COLS
    assert "name" not in FEATURE_COLS


def test_duplicate_drop_uses_data_cols():
    df = clean_df(sample_df())
    assert len(df) == 2
    assert list(df["name"]) == ["A", "C"]


def test_save_clean_creates_file(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "clean_data.csv"
    sample_df().to_csv(raw, index=False)
    save_clean(raw, out)
    assert Path(out).is_file()


def test_duplicate_report_created_if_duplicate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw.csv"
    out = tmp_path / "clean_data.csv"
    sample_df().to_csv(raw, index=False)
    save_clean(raw, out)
    assert Path("reports/duplicate_rows.csv").is_file()


def test_targets_are_binary_for_valid_data():
    df = clean_df(sample_df())
    for col in TARGET_COLS:
        vals = set(df[col].dropna().unique())
        assert vals <= {0, 1}
