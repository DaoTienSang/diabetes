import json

import pandas as pd

import src.train.mlflow_utils as mlflow_utils
from src.train.train_models import run_train


def train_csv(tmp_path):
    fp = tmp_path / "clean_data.csv"
    rows = []
    for i in range(50):
        rows.append(
            {
                "sl_no": i + 1,
                "name": f"p{i}",
                "age": 35 + i % 25,
                "sex": i % 2,
                "bmi": 22 + (i % 9) * 0.5,
                "sp": 115 + i % 15,
                "bp": 70 + i % 12,
                "hba1c": 6.5 + (i % 6) * 0.2,
                "fps": 90 + i % 35,
                "pps": 120 + i % 45,
                "family_h": i % 2,
                "onset_age": 25 + i % 20,
                "dia_life": 1 + i % 10,
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


def test_train_with_mlflow_creates_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    csv = train_csv(tmp_path)
    run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    assert (tmp_path / "models" / "best_model.pkl").is_file()
    assert (tmp_path / "models" / "meta.json").is_file()
    assert (tmp_path / "reports" / "metrics.csv").is_file()


def test_meta_has_mlflow_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    csv = train_csv(tmp_path)
    run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    meta = json.loads((tmp_path / "models" / "meta.json").read_text(encoding="utf-8"))
    assert "registered_model_name" in meta
    assert "registered_version" in meta
    assert "tracking_uri" in meta
    assert "metrics_path" in meta


def test_metrics_csv_has_all_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    csv = train_csv(tmp_path)
    run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    df = pd.read_csv(tmp_path / "reports" / "metrics.csv")
    vals = set(df["model_name"])
    assert "logistic_regression" in vals
    assert "random_forest" in vals


def test_mlflow_failure_does_not_break_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = train_csv(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("mlflow off")

    monkeypatch.setattr(mlflow_utils, "log_one_model_run", fail)
    import src.train.train_models as train_models

    monkeypatch.setattr(train_models, "log_one_model_run", fail)
    run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    assert (tmp_path / "models" / "best_model.pkl").is_file()
    assert (tmp_path / "reports" / "metrics.csv").is_file()


def test_best_model_insert_db_optional(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    csv = train_csv(tmp_path)
    rec = run_train(source="csv", csv_path=csv, db_path=tmp_path / "app.duckdb")
    assert rec["best_model"]
