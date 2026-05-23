import json

import joblib
import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.db.repo import get_counts


class DummyModel:
    def predict_proba(self, x):
        rows = len(x)
        vals = [0.8, 0.2, 0.5, 0.75, 0.1]
        return [np.tile([[1 - v, v]], (rows, 1)) for v in vals]


def payload():
    return {
        "patient_id": "bn_test",
        "name": "Test Patient",
        "age": 52,
        "sex": 1,
        "bmi": 26.5,
        "sp": 128,
        "bp": 82,
        "hba1c": 7.8,
        "fps": 115,
        "pps": 158,
        "family_h": 1,
        "onset_age": 44,
        "dia_life": 8.0,
        "smoking": 0,
        "phy_act": 1,
        "med_use": 1,
        "med_adh": 2,
    }


def make_app(tmp_path):
    model = tmp_path / "best_model.pkl"
    meta = tmp_path / "meta.json"
    db = tmp_path / "app.duckdb"
    joblib.dump(DummyModel(), model)
    meta.write_text(
        json.dumps(
            {
                "features": [
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
                ],
                "targets": ["nep", "neu", "ret", "cv", "per_vas"],
                "best_model": "dummy",
                "model_version": "test_v1",
            }
        ),
        encoding="utf-8",
    )
    app.state.db_path = str(db)
    app.state.model_path = str(model)
    app.state.meta_path = str(meta)
    return TestClient(app), db


def test_health_200(tmp_path):
    client, _ = make_app(tmp_path)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_200_and_schema(tmp_path):
    client, _ = make_app(tmp_path)
    res = client.post("/predict", json=payload())
    data = res.json()
    assert res.status_code == 200
    assert data["prediction_id"]
    assert len(data["risk_scores"]) == 5
    assert len(data["risk_labels"]) == 5
    assert len(data["risk_levels"]) == 5


def test_missing_field_422(tmp_path):
    client, _ = make_app(tmp_path)
    rec = payload()
    rec.pop("age")
    res = client.post("/predict", json=rec)
    assert res.status_code == 422


def test_metrics_200(tmp_path):
    client, _ = make_app(tmp_path)
    res = client.get("/metrics")
    assert res.status_code == 200


def test_predict_saves_db(tmp_path):
    client, db = make_app(tmp_path)
    res = client.post("/predict", json=payload())
    assert res.status_code == 200
    rec = get_counts(db)
    assert rec["benh_nhan"] == 1
    assert rec["ho_so_lam_sang"] == 1
    assert rec["ket_qua_du_doan"] == 1
    assert rec["chi_tiet_risk_score"] == 5
    assert rec["request_log"] == 1
