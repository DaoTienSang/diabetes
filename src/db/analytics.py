from pathlib import Path

import duckdb
import pandas as pd


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

TABLE_COLS = ["id"] + FEATURE_COLS + TARGET_COLS


def init_analytics(db_path="db/app.duckdb"):
    fp = Path(db_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(fp))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS du_lieu_huan_luyen (
                id BIGINT,
                age DOUBLE,
                sex INTEGER,
                bmi DOUBLE,
                sp DOUBLE,
                bp DOUBLE,
                hba1c DOUBLE,
                fps DOUBLE,
                pps DOUBLE,
                family_h INTEGER,
                onset_age INTEGER,
                dia_life DOUBLE,
                smoking INTEGER,
                phy_act INTEGER,
                med_use INTEGER,
                med_adh INTEGER,
                nep INTEGER,
                neu INTEGER,
                ret INTEGER,
                cv INTEGER,
                per_vas INTEGER
            )
            """
        )
    finally:
        con.close()


def load_clean_to_duckdb(csv_path="data/processed/clean_data.csv", db_path="db/app.duckdb"):
    fp = Path(csv_path)
    if not fp.exists():
        raise FileNotFoundError(f"Clean dataset not found: {fp}")

    init_analytics(db_path)
    df = pd.read_csv(fp)
    rec = pd.DataFrame()
    rec["id"] = df["sl_no"] if "sl_no" in df.columns else range(1, len(df) + 1)

    for col in FEATURE_COLS + TARGET_COLS:
        rec[col] = df[col]

    con = duckdb.connect(str(db_path))
    try:
        con.execute("DELETE FROM du_lieu_huan_luyen")
        con.register("rec", rec)
        con.execute(
            """
            INSERT INTO du_lieu_huan_luyen
            SELECT
                CAST(id AS BIGINT),
                CAST(age AS DOUBLE),
                CAST(sex AS INTEGER),
                CAST(bmi AS DOUBLE),
                CAST(sp AS DOUBLE),
                CAST(bp AS DOUBLE),
                CAST(hba1c AS DOUBLE),
                CAST(fps AS DOUBLE),
                CAST(pps AS DOUBLE),
                CAST(family_h AS INTEGER),
                CAST(onset_age AS INTEGER),
                CAST(dia_life AS DOUBLE),
                CAST(smoking AS INTEGER),
                CAST(phy_act AS INTEGER),
                CAST(med_use AS INTEGER),
                CAST(med_adh AS INTEGER),
                CAST(nep AS INTEGER),
                CAST(neu AS INTEGER),
                CAST(ret AS INTEGER),
                CAST(cv AS INTEGER),
                CAST(per_vas AS INTEGER)
            FROM rec
            """
        )
        cnt = con.execute("SELECT COUNT(*) FROM du_lieu_huan_luyen").fetchone()[0]
    finally:
        con.close()

    return int(cnt)


def read_training_data(db_path="db/app.duckdb"):
    init_analytics(db_path)
    con = duckdb.connect(str(db_path))
    try:
        return con.execute("SELECT * FROM du_lieu_huan_luyen ORDER BY id").fetchdf()
    finally:
        con.close()


def get_training_counts(db_path="db/app.duckdb"):
    init_analytics(db_path)
    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("SELECT COUNT(*) FROM du_lieu_huan_luyen").fetchone()[0]
        tg = {}
        for col in TARGET_COLS:
            val = con.execute(f"SELECT COALESCE(SUM({col}), 0) FROM du_lieu_huan_luyen").fetchone()[0]
            tg[col] = int(val)
    finally:
        con.close()

    return {"rows": int(rows), "target_counts": tg}
