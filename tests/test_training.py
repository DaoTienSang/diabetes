import json

import duckdb
import joblib
import pandas as pd

from src.features.build_features import FEATURE_COLS
from src.features.build_features import TARGET_COLS
from src.train.train_models import make_models
from src.train.train_models import run_train


def train_csv(tmp_path):
    fp = tmp_path / "clean_data.csv"
    rows = []
    for i in range(60):
        rows.append(
            {
                "sl_no": i + 1,
                "name": f"p{i}",
                "age": 30 + i % 30,
                "sex": i % 2,
                "bmi": 22 + (i % 10) * 0.5,
                "sp": 115 + i % 20,
                "bp": 70 + i % 15,
                "hba1c": 6.5 + (i % 8) * 0.2,
                "fps": 90 + i % 40,
                "pps": 120 + i % 50,
                "family_h": i % 2,
                "onset_age": 25 + i % 25,
                "dia_life": 1 + i % 12,
                "smoking": i % 3,
                "phy_act": i % 2,
                "med_use": 1 + i % 2,
                "med_adh": 1 + i % 2,
                "nep": i % 2,
                "neu": (i // 2) % 2,
                "ret": (i // 3) % 2,
                "cv": (i // 4) % 2,
                "per_vas": (i // 5) % 2,
            }
        )
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def test_make_models_has_logistic_and_random_forest():
    rows = make_models()
    assert "logistic_regression" in rows
    assert "random_forest" in rows


def test_run_train_creates_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = train_csv(tmp_path)
    run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    assert (tmp_path / "models" / "best_model.pkl").is_file()
    assert (tmp_path / "models" / "meta.json").is_file()
    assert (tmp_path / "reports" / "metrics.csv").is_file()


def test_metrics_csv_has_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = train_csv(tmp_path)
    run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    df = pd.read_csv(tmp_path / "reports" / "metrics.csv")
    cols = {
        "exact_acc",
        "label_acc",
        "f1_micro",
        "f1_macro",
        "recall_micro",
        "recall_macro",
        "precision_micro",
        "precision_macro",
        "hamming_loss",
    }
    assert cols <= set(df.columns)


def test_best_model_predicts_one_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = train_csv(tmp_path)
    run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    mdl = joblib.load(tmp_path / "models" / "best_model.pkl")
    df = pd.read_csv(csv)
    pred = mdl.predict(df[FEATURE_COLS].head(1))
    assert pred.shape == (1, 5)


def test_meta_has_features_and_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = train_csv(tmp_path)
    run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    meta = json.loads((tmp_path / "models" / "meta.json").read_text(encoding="utf-8"))
    assert meta["features"] == FEATURE_COLS
    assert meta["targets"] == TARGET_COLS


def test_insert_model_version_when_table_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = train_csv(tmp_path)
    db = tmp_path / "app.duckdb"
    con = duckdb.connect(str(db))
    try:
        con.execute("CREATE TABLE model_version (model_version VARCHAR, model_name VARCHAR)")
    finally:
        con.close()
    run_train(source="csv", csv_path=csv, db_path=db)
    con = duckdb.connect(str(db))
    try:
        cnt = con.execute("SELECT COUNT(*) FROM model_version").fetchone()[0]
    finally:
        con.close()
    assert cnt == 1


def test_xgboost_optional_not_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = train_csv(tmp_path)
    rec = run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    assert rec["best_model"]
