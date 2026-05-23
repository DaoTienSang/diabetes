import duckdb
import pandas as pd

from src.db.analytics import (
    FEATURE_COLS,
    TARGET_COLS,
    get_training_counts,
    init_analytics,
    load_clean_to_duckdb,
    read_training_data,
)


def clean_csv(tmp_path):
    fp = tmp_path / "clean_data.csv"
    df = pd.DataFrame(
        [
            [1, "A", 40, 1, 24.5, 120, 80, 7.1, 100, 130, 0, 35, 1.0, 0, 1, 1, 2, 0, 1, 0, 1, 0],
            [2, "B", 55, 0, 28.0, 130, 85, 8.2, 120, 150, 1, 45, 2.0, 1, 0, 2, 1, 1, 0, 1, 0, 1],
        ],
        columns=[
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
        ],
    )
    df.to_csv(fp, index=False)
    return fp


def test_init_analytics_creates_table(tmp_path):
    db = tmp_path / "app.duckdb"
    init_analytics(db)
    con = duckdb.connect(str(db))
    try:
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'du_lieu_huan_luyen'"
        ).fetchall()
    finally:
        con.close()
    assert rows


def test_load_clean_to_duckdb_loads_rows(tmp_path):
    db = tmp_path / "app.duckdb"
    csv = clean_csv(tmp_path)
    rows = load_clean_to_duckdb(csv, db)
    assert rows == 2


def test_load_twice_does_not_duplicate(tmp_path):
    db = tmp_path / "app.duckdb"
    csv = clean_csv(tmp_path)
    load_clean_to_duckdb(csv, db)
    load_clean_to_duckdb(csv, db)
    df = read_training_data(db)
    assert len(df) == 2


def test_read_training_data_has_feature_and_target(tmp_path):
    db = tmp_path / "app.duckdb"
    csv = clean_csv(tmp_path)
    load_clean_to_duckdb(csv, db)
    df = read_training_data(db)
    for col in FEATURE_COLS + TARGET_COLS:
        assert col in df.columns


def test_get_training_counts_returns_dict(tmp_path):
    db = tmp_path / "app.duckdb"
    csv = clean_csv(tmp_path)
    load_clean_to_duckdb(csv, db)
    rec = get_training_counts(db)
    assert isinstance(rec, dict)
    assert rec["rows"] == 2
    assert rec["target_counts"]["nep"] == 1
