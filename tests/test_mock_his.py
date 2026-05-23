import numpy as np
import pandas as pd
import pytest
import requests

from src.mock_his import stream
from src.mock_his.stream import FEATURES
from src.mock_his.stream import TARGETS
from src.mock_his.stream import load_rows
from src.mock_his.stream import row_to_payload
from src.mock_his.stream import run_stream
from src.mock_his.stream import safe_val
from src.mock_his.stream import send_one
from src.mock_his.stream import short_level
from src.mock_his.stream import verify_db_log


def csv_file(tmp_path, rows=3, include_name=True):
    data = []
    for i in range(rows):
        rec = {
            "age": 30 + i,
            "sex": i % 2,
            "bmi": 25.0 + i,
            "sp": 120 + i,
            "bp": 80 + i,
            "hba1c": 7.0 + i,
            "fps": 100 + i,
            "pps": 130 + i,
            "family_h": i % 2,
            "onset_age": 25 + i,
            "dia_life": 1.0 + i,
            "smoking": i % 3,
            "phy_act": i % 2,
            "med_use": 1,
            "med_adh": 2,
            "nep": 0,
            "neu": 1,
            "ret": 0,
            "cv": 1,
            "per_vas": 0,
        }
        if include_name:
            rec["name"] = f"P{i}"
        data.append(rec)
    fp = tmp_path / "clean_data.csv"
    pd.DataFrame(data).to_csv(fp, index=False)
    return fp


def sample_row(include_name=True):
    rec = {
        "age": 32,
        "sex": 0,
        "bmi": 36.1,
        "sp": 130,
        "bp": 80,
        "hba1c": 10.7,
        "fps": 102,
        "pps": 119,
        "family_h": 0,
        "onset_age": 31,
        "dia_life": 1,
        "smoking": 0,
        "phy_act": 1,
        "med_use": 1,
        "med_adh": 2,
        "nep": 0,
        "neu": 1,
        "ret": 0,
        "cv": 0,
        "per_vas": 0,
    }
    if include_name:
        rec["name"] = "PL"
    return pd.Series(rec)


def test_load_rows_limit(tmp_path):
    fp = csv_file(tmp_path, rows=3)
    df = load_rows(fp, limit=2)
    assert len(df) == 2


def test_row_to_payload_has_features():
    payload = row_to_payload(sample_row(), 1)
    for col in FEATURES:
        assert col in payload
    assert payload["patient_id"] == "BN_1"


def test_row_to_payload_no_targets():
    payload = row_to_payload(sample_row(), 1)
    for col in TARGETS:
        assert col not in payload


def test_row_to_payload_name_optional():
    payload = row_to_payload(sample_row(False), 1)
    assert payload["name"] == ""
    payload = row_to_payload(sample_row(True), 1)
    assert payload["name"] == "PL"


def test_row_to_payload_missing_feature():
    rec = sample_row()
    rec = rec.drop("age")
    with pytest.raises(ValueError):
        row_to_payload(rec, 1)


def test_safe_val_nan():
    assert safe_val(np.nan) is None


def test_send_one_success(monkeypatch):
    class Res:
        status_code = 200

        def json(self):
            return {
                "warning_level": "high",
                "risk_scores": {
                    "nep": 0.8,
                    "neu": 0.2,
                    "ret": 0.1,
                    "cv": 0.3,
                    "per_vas": 0.0,
                },
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Res())
    rec = send_one("http://localhost:8000/predict", {"age": 1})
    assert rec["ok"] is True


def test_send_one_api_error(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("off")

    monkeypatch.setattr(requests, "post", fail)
    rec = send_one("http://localhost:8000/predict", {"age": 1})
    assert rec["ok"] is False


def test_short_level():
    assert short_level({"warning_level": "medium"}) == "medium"
    assert short_level(None) == "unknown"


def test_run_stream(tmp_path, monkeypatch):
    fp = csv_file(tmp_path, rows=3)
    monkeypatch.setattr(stream, "send_one", lambda url, payload: {"ok": True, "status_code": 200, "data": {"warning_level": "low"}, "error": None})
    rec = run_stream(fp, limit=2, delay=0)
    assert len(rec) == 2


def test_run_stream_continue_on_error(tmp_path, monkeypatch):
    fp = csv_file(tmp_path, rows=2)
    calls = [
        {"ok": False, "status_code": None, "data": None, "error": "api_not_available"},
        {"ok": True, "status_code": 200, "data": {"warning_level": "high"}, "error": None},
    ]

    def fake(url, payload):
        return calls.pop(0)

    monkeypatch.setattr(stream, "send_one", fake)
    rec = run_stream(fp, limit=2, delay=0)
    assert len(rec) == 2


def test_verify_db_log_no_db(tmp_path):
    rec = verify_db_log(tmp_path / "missing.duckdb")
    assert rec["ok"] is False


def test_payload_matches_api_schema():
    payload = row_to_payload(sample_row(), 1)
    assert set(payload) == {
        "patient_id",
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
    }


def test_no_target_leakage():
    payload = row_to_payload(sample_row(), 1)
    assert not set(TARGETS) & set(payload)
