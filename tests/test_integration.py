import json
import os
import shutil
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


pytestmark = pytest.mark.integration

FEATURES = [
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

TARGETS = ["nep", "neu", "ret", "cv", "per_vas"]

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


def make_raw_sample(path):
    rows = []
    for i in range(40):
        base = [
            i + 1,
            f"Patient {i + 1}",
            30 + i % 35,
            i % 2,
            20 + i % 12,
            110 + i % 35,
            70 + i % 18,
            6.5 + (i % 6) * 0.4,
            90 + i % 70,
            130 + i % 90,
            i % 2,
            25 + i % 25,
            "12 months" if i % 8 == 0 else ("24 months" if i % 8 == 1 else 1 + i % 12),
            i % 3,
            i % 2,
            i % 2,
            i % 3,
            i % 2,
            (i + 1) % 2,
            (i // 2) % 2,
            (i // 3) % 2,
            1 if i % 9 == 0 else 0,
        ]
        rows.append(base)
    dup = rows[0].copy()
    dup[0] = 1001
    dup[1] = "Duplicate A"
    rows.append(dup)
    dup = rows[5].copy()
    dup[0] = 1002
    dup[1] = "Duplicate B"
    rows.append(dup)
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RAW_COLS).to_csv(fp, index=False)
    return fp


def make_clean_sample(path):
    rows = []
    for i in range(40):
        rows.append(
            {
                "age": 30 + i % 35,
                "sex": i % 2,
                "bmi": 20 + i % 12,
                "sp": 110 + i % 35,
                "bp": 70 + i % 18,
                "hba1c": 6.5 + (i % 6) * 0.4,
                "fps": 90 + i % 70,
                "pps": 130 + i % 90,
                "family_h": i % 2,
                "onset_age": 25 + i % 25,
                "dia_life": 1 + i % 12,
                "smoking": i % 3,
                "phy_act": i % 2,
                "med_use": i % 2,
                "med_adh": i % 3,
                "nep": i % 2,
                "neu": (i + 1) % 2,
                "ret": (i // 2) % 2,
                "cv": (i // 3) % 2,
                "per_vas": 1 if i % 9 == 0 else 0,
            }
        )
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def patch_project_paths(monkeypatch, tmp_path):
    paths = {
        "raw": tmp_path / "data" / "raw" / "data.csv",
        "clean": tmp_path / "data" / "processed" / "clean_data.csv",
        "current": tmp_path / "data" / "processed" / "current_data.csv",
        "db": tmp_path / "db" / "app.duckdb",
        "model": tmp_path / "models" / "best_model.pkl",
        "meta": tmp_path / "models" / "meta.json",
        "reports": tmp_path / "reports",
        "mlruns": tmp_path / "mlruns",
    }
    for key in ["raw", "clean", "current", "db", "model", "meta"]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    paths["mlruns"].mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("APP_DB_PATH", str(paths["db"]))
    monkeypatch.setenv("MODEL_PATH", str(paths["model"]))
    monkeypatch.setenv("META_PATH", str(paths["meta"]))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(paths["mlruns"]))
    return paths


def make_prediction_payload():
    return {
        "patient_id": "BN_TEST_001",
        "name": "Test Patient",
        "age": 45,
        "sex": 1,
        "bmi": 26.5,
        "sp": 130,
        "bp": 80,
        "hba1c": 8.1,
        "fps": 145,
        "pps": 220,
        "family_h": 1,
        "onset_age": 38,
        "dia_life": 7,
        "smoking": 0,
        "phy_act": 1,
        "med_use": 1,
        "med_adh": 2,
    }


def make_dummy_model(path, meta_path):
    rows = []
    targets = []
    for i in range(60):
        rows.append(
            [
                30 + i % 35,
                i % 2,
                20 + i % 12,
                110 + i % 35,
                70 + i % 18,
                6.5 + (i % 6) * 0.4,
                90 + i % 70,
                130 + i % 90,
                i % 2,
                25 + i % 25,
                1 + i % 12,
                i % 3,
                i % 2,
                i % 2,
                i % 3,
            ]
        )
        targets.append([i % 2, (i + 1) % 2, (i // 2) % 2, (i // 3) % 2, 1 if i % 9 == 0 else 0])
    x = pd.DataFrame(rows, columns=FEATURES)
    y = pd.DataFrame(targets, columns=TARGETS)
    mdl = MultiOutputClassifier(RandomForestClassifier(n_estimators=8, random_state=42))
    mdl.fit(x, y)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl, path)
    meta = {
        "features": FEATURES,
        "targets": TARGETS,
        "best_model": "dummy_model",
        "model_version": "test_v1",
        "registered_model_name": "diabetes_complication_model",
        "registered_version": None,
        "tracking_uri": "test",
        "metrics_path": "reports/metrics.csv",
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path, meta_path


def seed_business_db(db_path):
    from src.db import repo
    from src.db.schema import init_db

    init_db(db_path)
    ma_bn = repo.insert_patient(
        db_path,
        {"ma_benh_nhan": "BN_INT_001", "ten": "Nguyen Van A", "tuoi": 55, "gioi_tinh": 1},
    )
    ma_hs = repo.insert_clinical_record(
        db_path,
        ma_bn,
        {
            "ma_ho_so": "HS_INT_001",
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
    ma_model = repo.insert_model_version(
        db_path,
        {
            "ma_model": "MDL_INT_001",
            "model_name": "random_forest",
            "version": "v1",
            "stage": "production",
            "f1_micro": 0.7,
            "f1_macro": 0.6,
            "recall_macro": 0.5,
            "hamming_loss": 0.2,
        },
    )
    ma_kq = repo.insert_prediction(db_path, ma_bn, ma_hs, ma_model, "high")
    repo.insert_risk_scores(
        db_path,
        ma_kq,
        {"nep": 0.8, "neu": 0.6, "ret": 0.5, "cv": 0.7, "per_vas": 0.2},
        {"nep": 1, "neu": 1, "ret": 1, "cv": 1, "per_vas": 0},
        {"nep": "high", "neu": "medium", "ret": "medium", "cv": "high", "per_vas": "low"},
    )
    repo.insert_alert(db_path, ma_bn, ma_kq, "high", "Canh bao nguy co cao")
    repo.insert_request_log(db_path, ma_kq, "/predict", 200, 12.5)
    repo.insert_monitoring_log(db_path, ma_model, 0.3, 0.5, 0.6, "no")
    return ma_bn, ma_kq


def test_domain_entities_integration():
    from src.domain.entities import AlertEngine
    from src.domain.entities import BacSi
    from src.domain.entities import BenhNhan
    from src.domain.entities import HoSoLamSang
    from src.domain.entities import LogisticRegressionModel
    from src.domain.entities import MoHinhDuDoan
    from src.domain.entities import MonitoringService
    from src.domain.entities import NguoiDung
    from src.domain.entities import QuanTriVien
    from src.domain.entities import RandomForestModel
    from src.domain.entities import RetrainingPipeline
    from src.domain.entities import XGBoostModel

    assert issubclass(BacSi, NguoiDung)
    assert issubclass(QuanTriVien, NguoiDung)
    assert issubclass(LogisticRegressionModel, MoHinhDuDoan)
    assert issubclass(RandomForestModel, MoHinhDuDoan)
    assert issubclass(XGBoostModel, MoHinhDuDoan)
    assert BenhNhan("BN1", "A", 50, 1).lay_thong_tin_co_ban()["ma_benh_nhan"] == "BN1"
    assert HoSoLamSang("HS1", 25, 120, 80, 7, 100, 150, 1, 40, 5, 0, 1, 1, 2).kiem_tra_hop_le()
    assert AlertEngine(0.5).threshold == 0.5
    assert MonitoringService("svc").service_name == "svc"
    assert RetrainingPipeline("pipe").pipeline_id == "pipe"


def test_data_cleaning_to_clean_csv(tmp_path, monkeypatch):
    from src.data.clean_data import FEATURE_COLS, TARGET_COLS, save_clean

    monkeypatch.chdir(tmp_path)
    raw = make_raw_sample(tmp_path / "raw.csv")
    out = tmp_path / "clean_data.csv"
    df = save_clean(raw, out)
    assert out.is_file()
    assert set(FEATURE_COLS) <= set(df.columns)
    assert set(TARGET_COLS) <= set(df.columns)
    assert "sl_no" not in FEATURE_COLS
    assert "name" not in FEATURE_COLS
    assert len(df) < 42
    assert 1.0 in set(df["dia_life"])


def test_clean_data_load_to_duckdb(tmp_path):
    from src.db.analytics import load_clean_to_duckdb, read_training_data

    clean = make_clean_sample(tmp_path / "clean_data.csv")
    db = tmp_path / "app.duckdb"
    cnt = load_clean_to_duckdb(clean, db)
    df = read_training_data(db)
    assert cnt == len(df)
    assert set(FEATURES + TARGETS) <= set(df.columns)
    load_clean_to_duckdb(clean, db)
    df2 = read_training_data(db)
    assert len(df2) == len(df)


def test_feature_builder_reads_from_duckdb(tmp_path):
    from src.db.analytics import load_clean_to_duckdb
    from src.features.build_features import load_xy

    clean = make_clean_sample(tmp_path / "clean_data.csv")
    db = tmp_path / "app.duckdb"
    load_clean_to_duckdb(clean, db)
    x, y = load_xy(source="duckdb", db_path=db, csv_path=clean)
    assert x.shape[1] == 15
    assert y.shape[1] == 5
    assert not set(TARGETS) & set(x.columns)
    assert list(y.columns) == TARGETS


def test_train_pipeline_creates_model_outputs(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import src.train.train_models as train_models

    clean = make_clean_sample(tmp_path / "clean_data.csv")
    db = tmp_path / "app.duckdb"
    from src.db.analytics import load_clean_to_duckdb

    load_clean_to_duckdb(clean, db)

    def small_models():
        return {
            "logistic_regression": Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("model", MultiOutputClassifier(RandomForestClassifier(n_estimators=6, random_state=1)))]
            ),
            "random_forest": Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("model", MultiOutputClassifier(RandomForestClassifier(n_estimators=8, random_state=2)))]
            ),
        }

    monkeypatch.setattr(train_models, "make_models", small_models)
    monkeypatch.setattr(train_models, "log_one_model_run", lambda *args, **kwargs: {"logged": False, "registered_version": None})

    model_path = tmp_path / "models" / "best_model.pkl"
    meta_path = tmp_path / "models" / "meta.json"
    metrics_path = tmp_path / "reports" / "metrics.csv"
    train_models.run_train(
        source="duckdb",
        db_path=str(db),
        csv_path=str(clean),
        output_model_path=str(model_path),
        output_meta_path=str(meta_path),
        output_metrics_path=str(metrics_path),
        register_production=False,
    )
    assert model_path.is_file()
    assert meta_path.is_file()
    assert metrics_path.is_file()
    met = pd.read_csv(metrics_path)
    for col in ["f1_micro", "f1_macro", "recall_macro", "hamming_loss"]:
        assert col in met.columns
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ["features", "targets", "best_model", "model_version"]:
        assert key in meta
    mdl = joblib.load(model_path)
    x = pd.read_csv(clean)[FEATURES].head(1)
    assert mdl.predict(x).shape == (1, 5)


def test_database_schema_and_repo_integration(tmp_path):
    from src.db import repo
    from src.db.schema import init_db, list_tables

    db = tmp_path / "app.duckdb"
    init_db(db)
    expected = {
        "benh_nhan",
        "ho_so_lam_sang",
        "model_version",
        "ket_qua_du_doan",
        "chi_tiet_risk_score",
        "canh_bao",
        "request_log",
        "monitoring_log",
        "du_lieu_huan_luyen",
    }
    assert expected <= set(list_tables(db))
    ma_bn, ma_kq = seed_business_db(db)
    counts = repo.get_counts(db)
    assert counts["benh_nhan"] >= 1
    assert counts["ket_qua_du_doan"] >= 1
    assert counts["canh_bao"] >= 1
    assert len(repo.get_prediction_scores(db, ma_kq)) == 5
    assert repo.get_patient(db, ma_bn)["ma_benh_nhan"] == ma_bn


def test_alert_engine_integration():
    from src.predict.alert import make_alert, make_labels, make_levels

    risk_scores = {"nep": 0.8, "neu": 0.2, "ret": 0.5, "cv": 0.1, "per_vas": 0.0}
    risk_levels = make_levels(risk_scores)
    risk_labels = make_labels(risk_scores)
    alert = make_alert(risk_scores, risk_levels)
    assert risk_levels["nep"] == "high"
    assert risk_levels["ret"] == "medium"
    assert set(risk_labels) == set(TARGETS)
    assert alert["level"] == "high"
    assert alert["message"]
    assert "biến chứng thận" in alert["message"]


def test_fastapi_health_predict_metrics_integration(tmp_path, monkeypatch):
    from src.api.main import app
    from src.db import repo

    paths = patch_project_paths(monkeypatch, tmp_path)
    make_dummy_model(paths["model"], paths["meta"])
    app.state.db_path = str(paths["db"])
    app.state.model_path = str(paths["model"])
    app.state.meta_path = str(paths["meta"])
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    res = client.post("/predict", json=make_prediction_payload())
    assert res.status_code == 200
    data = res.json()
    for key in [
        "patient_id",
        "prediction_id",
        "risk_scores",
        "risk_labels",
        "risk_levels",
        "warning_level",
        "warning_message",
        "model_version",
        "timestamp",
    ]:
        assert key in data
    assert set(data["risk_scores"]) == set(TARGETS)
    assert set(data["risk_labels"]) == set(TARGETS)
    assert set(data["risk_levels"]) == set(TARGETS)
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert b"predict_count" in metrics.content or b"request_count" in metrics.content
    counts = repo.get_counts(paths["db"])
    assert counts["benh_nhan"] >= 1
    assert counts["ket_qua_du_doan"] >= 1
    assert len(repo.get_prediction_scores(paths["db"], data["prediction_id"])) == 5
    assert counts["request_log"] >= 1


def test_mock_his_payload_integration(tmp_path, monkeypatch):
    from src.mock_his import stream

    clean = make_clean_sample(tmp_path / "clean_data.csv")
    df = pd.read_csv(clean)
    payload = stream.row_to_payload(df.iloc[0], 1)
    assert set(FEATURES) <= set(payload)
    assert payload["patient_id"] == "BN_1"
    assert not set(TARGETS) & set(payload)
    monkeypatch.setattr(
        stream,
        "send_one",
        lambda url, payload, timeout=10: {
            "ok": True,
            "status_code": 200,
            "data": {"warning_level": "low"},
            "error": None,
        },
    )
    rows = stream.run_stream(data_path=clean, limit=2, delay=0)
    assert len(rows) == 2


def test_django_dashboard_integration_after_login(tmp_path, monkeypatch):
    pytest.importorskip("django")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.config.settings")
    import django

    django.setup()
    from django.contrib.auth.models import User
    from django.core.management import call_command
    from django.test import Client

    db = tmp_path / "app.duckdb"
    ma_bn, ma_kq = seed_business_db(db)
    monkeypatch.setenv("APP_DB_PATH", str(db))
    call_command("migrate", verbosity=0, interactive=False)
    User.objects.filter(username="doctor_int").delete()
    User.objects.create_user(username="doctor_int", password="doctor123")
    client = Client()
    assert client.get("/").status_code == 302
    assert client.get("/login/").status_code == 200
    assert client.login(username="doctor_int", password="doctor123")
    assert client.get("/").status_code == 200
    assert client.get("/patients/").status_code == 200
    assert client.get("/alerts/").status_code == 200
    detail = client.get(f"/patients/{ma_bn}/")
    assert detail.status_code == 200
    assert client.get(f"/predictions/{ma_kq}/").status_code == 200
    names = [
        "biến chứng thận",
        "biến chứng thần kinh",
        "biến chứng võng mạc",
        "biến chứng tim mạch",
        "biến chứng mạch máu ngoại biên",
    ]
    content = detail.content.decode("utf-8", errors="ignore")
    assert all(name in content for name in names) or "biáº¿n chá»©ng tháº­n" in content
    assert client.get("/register/").status_code == 404


def test_evidently_monitoring_integration(tmp_path):
    from src.monitoring.drift import run_drift_monitoring

    clean = make_clean_sample(tmp_path / "clean_data.csv")
    cur = tmp_path / "current_data.csv"
    out = tmp_path / "reports" / "drift_report.html"
    summary_path = tmp_path / "reports" / "monitoring_summary.json"
    summary = run_drift_monitoring(
        ref_path=clean,
        cur_path=cur,
        db_path=tmp_path / "missing.duckdb",
        out_path=out,
        summary_path=summary_path,
        prefer_db=False,
    )
    assert summary_path.is_file()
    assert "retrain_flag" in summary
    assert "drift_share" in summary
    assert "drifted_features" in summary
    skip = Path("reports/drift_report_skip.txt")
    assert out.exists() or skip.exists() or Path(summary["report_path"]).exists()


def test_prefect_retrain_no_retrain_integration(tmp_path, monkeypatch):
    import src.retrain.flow as flow

    monkeypatch.chdir(tmp_path)
    flow.save_json({"retrain_flag": "no", "reason": "stable"}, "reports/monitoring_summary.json")
    monkeypatch.setattr(flow, "run_training_pipeline", lambda: (_ for _ in ()).throw(AssertionError("train should not run")))
    rec = flow.run_retrain_if_needed()
    assert rec["final_status"] == "no_retrain"
    assert not Path("models/candidate_model.pkl").exists()


def test_prefect_retrain_rules_fail_no_overwrite_integration(tmp_path, monkeypatch):
    import src.retrain.flow as flow

    monkeypatch.chdir(tmp_path)
    flow.save_json({"retrain_flag": "yes", "reason": "drift_high"}, "reports/monitoring_summary.json")
    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("reports/metrics.csv").write_text(
        "model_name,f1_micro,f1_macro,recall_macro,hamming_loss\nrandom_forest,0.6,0.5,0.4,0.2\n",
        encoding="utf-8",
    )
    Path("models/best_model.pkl").write_text("old_model", encoding="utf-8")

    def fake_train():
        Path("models/candidate_model.pkl").write_text("candidate", encoding="utf-8")
        Path("reports/candidate_metrics.csv").write_text(
            "model_name,f1_micro,f1_macro,recall_macro,hamming_loss\ncandidate,0.3,0.2,0.1,0.5\n",
            encoding="utf-8",
        )
        return {
            "ok": True,
            "model_path": "models/candidate_model.pkl",
            "metrics_path": "reports/candidate_metrics.csv",
            "meta_path": "models/candidate_meta.json",
            "metrics": {"model": "candidate", "f1_micro": 0.3, "recall_macro": 0.1},
            "model_name": "candidate",
            "error": None,
        }

    monkeypatch.setattr(flow, "run_training_pipeline", fake_train)
    rec = flow.run_retrain_if_needed()
    assert rec["final_status"] == "rejected"
    assert Path("models/best_model.pkl").read_text(encoding="utf-8") == "old_model"


def test_docker_compose_files_integration():
    root = Path(__file__).resolve().parents[1]
    assert (root / "Dockerfile").is_file()
    assert (root / "docker-compose.yml").is_file()
    assert (root / "docker" / "prometheus.yml").is_file()
    assert (root / "docker" / "grafana" / "provisioning" / "datasources" / "datasource.yml").is_file()
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    prom = (root / "docker" / "prometheus.yml").read_text(encoding="utf-8")
    for val in ["fastapi", "django", "mlflow", "prometheus", "grafana"]:
        assert val in compose
    for port in ["8000:8000", "8001:8001", "5000:5000", "9090:9090", "3000:3000"]:
        assert port in compose
    assert "fastapi:8000" in prom
    assert "/metrics" in prom
    if shutil.which("docker"):
        res = subprocess.run(["docker", "compose", "config"], cwd=root, capture_output=True, text=True, timeout=60)
        if res.returncode != 0 and "daemon" in (res.stderr + res.stdout).lower():
            pytest.skip("docker daemon not running")
        assert res.returncode == 0


def test_github_actions_files_integration():
    root = Path(__file__).resolve().parents[1]
    ci = root / ".github" / "workflows" / "ci.yml"
    assert ci.is_file()
    text = ci.read_text(encoding="utf-8")
    for val in ["push", "pull_request", "main", "pytest", "docker compose config", "docker build"]:
        assert val in text
    assert 'python-version: "3.11"' in text or "python-version: 3.11" in text
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/DaoTienSang/diabetes.git" in readme


def test_end_to_end_core_without_servers(tmp_path, monkeypatch):
    from src.api.main import app
    from src.data.clean_data import save_clean
    from src.db import repo
    from src.db.analytics import load_clean_to_duckdb
    from src.db.schema import init_db
    from src.features.build_features import load_xy
    from src.mock_his.stream import row_to_payload
    from src.monitoring.drift import run_drift_monitoring

    paths = patch_project_paths(monkeypatch, tmp_path)
    raw = make_raw_sample(paths["raw"])
    save_clean(raw, paths["clean"])
    load_clean_to_duckdb(paths["clean"], paths["db"])
    x, y = load_xy(source="duckdb", db_path=paths["db"], csv_path=paths["clean"])
    assert x.shape[1] == 15
    assert y.shape[1] == 5
    make_dummy_model(paths["model"], paths["meta"])
    init_db(paths["db"])
    app.state.db_path = str(paths["db"])
    app.state.model_path = str(paths["model"])
    app.state.meta_path = str(paths["meta"])
    client = TestClient(app)
    res = client.post("/predict", json=make_prediction_payload())
    assert res.status_code == 200
    data = res.json()
    assert len(data["risk_scores"]) == 5
    counts = repo.get_counts(paths["db"])
    assert counts["ket_qua_du_doan"] >= 1
    assert len(repo.get_prediction_scores(paths["db"], data["prediction_id"])) == 5
    payload = row_to_payload(pd.read_csv(paths["clean"]).iloc[0], 1)
    assert not set(TARGETS) & set(payload)
    summary = run_drift_monitoring(
        ref_path=paths["clean"],
        cur_path=paths["current"],
        db_path=paths["db"],
        out_path=paths["reports"] / "drift_report.html",
        summary_path=paths["reports"] / "monitoring_summary.json",
        prefer_db=False,
    )
    assert "retrain_flag" in summary
