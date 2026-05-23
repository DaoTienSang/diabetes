import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split


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


def get_cols():
    return FEATURE_COLS, TARGET_COLS


def make_xy(df):
    x = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()
    return x, y


def load_data(source="duckdb", db_path="db/app.duckdb", csv_path="data/processed/clean_data.csv"):
    if source == "csv":
        return pd.read_csv(csv_path)

    try:
        con = duckdb.connect(str(db_path))
        try:
            return con.execute("SELECT * FROM du_lieu_huan_luyen ORDER BY id").fetchdf()
        finally:
            con.close()
    except Exception:
        print("csv_fallback")
        return pd.read_csv(csv_path)


def load_xy(source="duckdb", db_path="db/app.duckdb", csv_path="data/processed/clean_data.csv"):
    df = load_data(source, db_path, csv_path)
    return make_xy(df)


def split_data(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
