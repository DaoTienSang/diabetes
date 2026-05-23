import duckdb
import pandas as pd

from src.features.build_features import get_cols, load_xy, make_xy, split_data


def clean_df():
    return pd.DataFrame(
        [
            [1, "A", 40, 1, 24.5, 120, 80, 7.1, 100, 130, 0, 35, 1.0, 0, 1, 1, 2, 0, 1, 0, 1, 0],
            [2, "B", 55, 0, 28.0, 130, 85, 8.2, 120, 150, 1, 45, 2.0, 1, 0, 2, 1, 1, 0, 1, 0, 1],
            [3, "C", 50, 1, 26.0, 125, 82, 7.5, 110, 140, 0, 42, 3.0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
            [4, "D", 60, 0, 30.0, 135, 88, 8.7, 130, 170, 1, 48, 4.0, 1, 0, 2, 2, 1, 1, 0, 0, 1],
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


def init_db(db, df):
    con = duckdb.connect(str(db))
    try:
        con.register("df", df)
        con.execute(
            """
            CREATE TABLE du_lieu_huan_luyen AS
            SELECT
                CAST(sl_no AS BIGINT) AS id,
                age, sex, bmi, sp, bp, hba1c, fps, pps,
                family_h, onset_age, dia_life, smoking, phy_act,
                med_use, med_adh, nep, neu, ret, cv, per_vas
            FROM df
            """
        )
    finally:
        con.close()


def test_get_cols_returns_15_features():
    ft, _ = get_cols()
    assert len(ft) == 15


def test_get_cols_returns_5_targets():
    _, tg = get_cols()
    assert len(tg) == 5


def test_load_xy_reads_duckdb_when_table_exists(tmp_path):
    db = tmp_path / "app.duckdb"
    init_db(db, clean_df())
    x, y = load_xy(db_path=db)
    assert x.shape == (4, 15)
    assert y.shape == (4, 5)


def test_load_xy_fallback_csv_when_db_missing(tmp_path, capsys):
    csv = tmp_path / "clean_data.csv"
    db = tmp_path / "missing.duckdb"
    clean_df().to_csv(csv, index=False)
    x, y = load_xy(db_path=db, csv_path=csv)
    out = capsys.readouterr().out
    assert "csv_fallback" in out
    assert x.shape == (4, 15)
    assert y.shape == (4, 5)


def test_x_does_not_contain_targets():
    x, _ = make_xy(clean_df())
    _, tg = get_cols()
    assert not set(tg) & set(x.columns)


def test_y_has_5_cols():
    _, y = make_xy(clean_df())
    assert y.shape[1] == 5


def test_split_data_row_counts():
    x, y = make_xy(clean_df())
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.25, random_state=42)
    assert len(x_train) == 3
    assert len(x_test) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1
