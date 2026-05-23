import inspect
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.config.settings")

import django

django.setup()

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import Client

import dashboard.app.services as services
from src.db.repo import insert_alert
from src.db.repo import insert_clinical_record
from src.db.repo import insert_model_version
from src.db.repo import insert_patient
from src.db.repo import insert_prediction
from src.db.repo import insert_request_log
from src.db.repo import insert_risk_scores
from src.db.schema import init_db


def setup_auth():
    call_command("migrate", verbosity=0, interactive=False)
    User.objects.filter(username="doctor").delete()
    User.objects.create_user(username="doctor", password="doctor123")


def seed_db(db):
    init_db(db)
    ma_bn = insert_patient(db, {"ma_benh_nhan": "bn_demo", "ten": "Nguyen Van A", "tuoi": 55, "gioi_tinh": 1})
    ma_hs = insert_clinical_record(
        db,
        ma_bn,
        {
            "ma_ho_so": "hs_demo",
            "bmi": 26.5,
            "sp": 128,
            "bp": 82,
            "hba1c": 7.8,
            "fps": 115,
            "pps": 158,
            "family_history": 1,
            "onset_age": 44,
            "dia_life": 8.0,
            "smoking": 0,
            "phy_act": 1,
            "med_use": 1,
            "med_adh": 2,
        },
    )
    ma_model = insert_model_version(
        db,
        {
            "ma_model": "mdl_demo",
            "model_name": "random_forest",
            "version": "v1",
            "stage": "production",
            "f1_micro": 0.7,
            "f1_macro": 0.6,
            "recall_macro": 0.5,
            "hamming_loss": 0.2,
        },
    )
    ma_kq = insert_prediction(db, ma_bn, ma_hs, ma_model, "high")
    insert_risk_scores(
        db,
        ma_kq,
        {"nep": 0.8, "neu": 0.6, "ret": 0.5, "cv": 0.7, "per_vas": 0.2},
        {"nep": 1, "neu": 1, "ret": 1, "cv": 1, "per_vas": 0},
        {"nep": "high", "neu": "medium", "ret": "medium", "cv": "high", "per_vas": "low"},
    )
    insert_alert(db, ma_bn, ma_kq, "high", "Cảnh báo nguy cơ cao")
    insert_request_log(db, ma_kq, "/predict", 200, 12.5)
    return ma_bn, ma_kq


def login_client(tmp_path):
    setup_auth()
    db = tmp_path / "app.duckdb"
    ma_bn, ma_kq = seed_db(db)
    os.environ["APP_DB_PATH"] = str(db)
    client = Client()
    assert client.login(username="doctor", password="doctor123")
    return client, ma_bn, ma_kq


def test_dashboard_status_after_login(tmp_path):
    client, _, _ = login_client(tmp_path)
    assert client.get("/").status_code == 200


def test_dashboard_content_after_login(tmp_path):
    client, _, _ = login_client(tmp_path)
    res = client.get("/")
    assert "Dashboard dự đoán biến chứng tiểu đường".encode() in res.content
    assert "Tổng số bệnh nhân".encode() in res.content
    assert "Tổng số dự đoán".encode() in res.content


def test_patients_status_after_login(tmp_path):
    client, _, _ = login_client(tmp_path)
    assert client.get("/patients/").status_code == 200


def test_patients_content_after_login(tmp_path):
    client, ma_bn, _ = login_client(tmp_path)
    assert ma_bn.encode() in client.get("/patients/").content


def test_patient_detail_status_after_login(tmp_path):
    client, ma_bn, _ = login_client(tmp_path)
    assert client.get(f"/patients/{ma_bn}/").status_code == 200


def test_patient_detail_scores_after_login(tmp_path):
    client, ma_bn, _ = login_client(tmp_path)
    res = client.get(f"/patients/{ma_bn}/")
    for text in [
        "biến chứng thận",
        "biến chứng thần kinh",
        "biến chứng võng mạc",
        "biến chứng tim mạch",
        "biến chứng mạch máu ngoại biên",
    ]:
        assert text.encode() in res.content


def test_alerts_status_after_login(tmp_path):
    client, _, _ = login_client(tmp_path)
    assert client.get("/alerts/").status_code == 200


def test_alerts_content_after_login(tmp_path):
    client, _, _ = login_client(tmp_path)
    assert "Cảnh báo nguy cơ cao".encode() in client.get("/alerts/").content


def test_prediction_detail_status_after_login(tmp_path):
    client, _, ma_kq = login_client(tmp_path)
    assert client.get(f"/predictions/{ma_kq}/").status_code == 200


def test_filter_patients_query_after_login(tmp_path):
    client, ma_bn, _ = login_client(tmp_path)
    res = client.get(f"/patients/?q={ma_bn}")
    assert ma_bn.encode() in res.content


def test_filter_alert_level_after_login(tmp_path):
    client, _, _ = login_client(tmp_path)
    res = client.get("/alerts/?level=high")
    assert "Cảnh báo nguy cơ cao".encode() in res.content


def test_empty_db_no_crash_after_login(tmp_path):
    setup_auth()
    db = tmp_path / "empty.duckdb"
    init_db(db)
    os.environ["APP_DB_PATH"] = str(db)
    client = Client()
    assert client.login(username="doctor", password="doctor123")
    assert client.get("/").status_code == 200
    patients = client.get("/patients/")
    alerts = client.get("/alerts/")
    assert patients.status_code == 200
    assert alerts.status_code == 200
    assert "Chưa có dữ liệu bệnh nhân".encode() in patients.content
    assert "Chưa có cảnh báo".encode() in alerts.content


def test_no_mlops_ui_rebuild(tmp_path):
    client, _, _ = login_client(tmp_path)
    res = client.get("/")
    assert b"MLflow" in res.content
    assert b"Prometheus" in res.content
    assert b"Grafana" in res.content
    assert client.get("/mlflow-ui/").status_code == 404
    assert client.get("/grafana-ui/").status_code == 404
    assert client.get("/prometheus-ui/").status_code == 404


def test_uses_duckdb_repo():
    src = inspect.getsource(services)
    assert "from src.db import repo" in src
