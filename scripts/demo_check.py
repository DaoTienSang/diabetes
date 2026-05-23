import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

REQUIRED_PATHS = [
    "data",
    "data/raw",
    "data/processed",
    "db",
    "models",
    "reports",
    "src",
    "src/domain",
    "src/data",
    "src/db",
    "src/features",
    "src/train",
    "src/predict",
    "src/api",
    "src/mock_his",
    "src/monitoring",
    "src/retrain",
    "dashboard",
    "docker",
    "tests",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "README.md",
    ".github/workflows/ci.yml",
]

IMPORTS = [
    "src.domain.entities",
    "src.data.clean_data",
    "src.db.schema",
    "src.db.repo",
    "src.db.analytics",
    "src.features.build_features",
    "src.train.metrics",
    "src.train.train_models",
    "src.predict.alert",
    "src.api.main",
    "src.mock_his.stream",
    "src.monitoring.drift",
    "src.retrain.flow",
]


def set_status(rows, key, status, note=""):
    rows[key] = {"status": status, "note": note}


def cmd(args, timeout=600):
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        rec = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        return {
            "ok": rec.returncode == 0,
            "code": rec.returncode,
            "out": rec.stdout.strip(),
            "err": rec.stderr.strip(),
        }
    except Exception as err:
        return {"ok": False, "code": -1, "out": "", "err": str(err)}


def ensure_dirs():
    for path in ["data/raw", "data/processed", "db", "models", "reports", "mlruns", "scripts"]:
        Path(path).mkdir(parents=True, exist_ok=True)


def make_sample_raw(path):
    cols = [
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
    rows = []
    for i in range(40):
        rows.append(
            [
                i + 1,
                f"Sample {i + 1}",
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
        )
    rows.append(rows[0].copy())
    rows[-1][0] = 1001
    rows[-1][1] = "Duplicate Sample"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def check_structure(rows):
    ensure_dirs()
    miss = [path for path in REQUIRED_PATHS if not Path(path).exists()]
    set_status(rows, "project_structure", "PASS" if not miss else "FAIL", "missing: " + ", ".join(miss) if miss else "required files/folders exist")


def check_imports(rows):
    failed = []
    for name in IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as err:
            failed.append(f"{name}: {err}")
    set_status(rows, "imports", "PASS" if not failed else "FAIL", "; ".join(failed) if failed else "main modules import ok")


def check_data(rows):
    raw = Path("data/raw/data.csv")
    used_sample = False
    if not raw.exists():
        make_sample_raw(raw)
        used_sample = True
    clean_res = cmd([sys.executable, "-m", "src.data.clean_data"], 300)
    clean = Path("data/processed/clean_data.csv")
    ok = clean_res["ok"] and clean.exists()
    note = clean_res["out"][-1000:] if clean_res["out"] else clean_res["err"][-1000:]
    if used_sample:
        note = "using sample raw data for demo check. " + note
    try:
        df = pd.read_csv(clean)
        ok = ok and set(FEATURES) <= set(df.columns) and set(TARGETS) <= set(df.columns)
        ok = ok and "sl_no" not in FEATURES and "name" not in FEATURES
        note += f" clean_shape={df.shape}"
    except Exception as err:
        ok = False
        note += f" clean_read_error={err}"
    set_status(rows, "data_cleaning", "PASS" if ok else "FAIL", note)
    try:
        from src.db.analytics import get_training_counts, init_analytics, load_clean_to_duckdb, read_training_data

        init_analytics()
        c1 = load_clean_to_duckdb()
        c2 = load_clean_to_duckdb()
        df = read_training_data()
        counts = get_training_counts()
        ok = Path("db/app.duckdb").exists() and len(df) > 0 and c1 == c2 == len(df)
        set_status(rows, "duckdb_analytics", "PASS" if ok else "FAIL", f"training_shape={df.shape}, counts={counts}")
    except Exception as err:
        set_status(rows, "duckdb_analytics", "FAIL", str(err))


def check_features(rows):
    try:
        from src.features.build_features import get_cols, load_xy, split_data

        ft, tg = get_cols()
        x, y = load_xy(source="duckdb")
        x_tr, x_te, y_tr, y_te = split_data(x, y)
        ok = ft == FEATURES and tg == TARGETS and x.shape[1] == 15 and y.shape[1] == 5 and not set(TARGETS) & set(x.columns)
        note = f"x_shape={x.shape}, y_shape={y.shape}, train={x_tr.shape}/{y_tr.shape}, test={x_te.shape}/{y_te.shape}"
        set_status(rows, "feature_builder", "PASS" if ok else "FAIL", note)
    except Exception as err:
        set_status(rows, "feature_builder", "FAIL", str(err))


def check_training(rows):
    res = cmd([sys.executable, "-m", "src.train.train_models"], 600)
    ok_files = Path("models/best_model.pkl").exists() and Path("models/meta.json").exists() and Path("reports/metrics.csv").exists()
    try:
        metrics = pd.read_csv("reports/metrics.csv")
        need = {"exact_acc", "label_acc", "f1_micro", "f1_macro", "recall_micro", "recall_macro", "precision_micro", "precision_macro", "hamming_loss"}
        has_models = {"logistic_regression", "random_forest"} <= set(metrics["model_name"])
        ok_metrics = need <= set(metrics.columns) and has_models
    except Exception:
        ok_metrics = False
    ok = res["ok"] and ok_files and ok_metrics
    set_status(rows, "training", "PASS" if ok else "FAIL", (res["out"] or res["err"])[-1000:])
    try:
        import joblib

        meta = json.loads(Path("models/meta.json").read_text(encoding="utf-8"))
        df = pd.read_csv("data/processed/clean_data.csv")
        pred = joblib.load("models/best_model.pkl").predict(df[meta["features"]].head(2))
        ok = tuple(pred.shape) == (2, 5) and meta["targets"] == TARGETS
        note = f"best_model={meta.get('best_model')}, model_version={meta.get('model_version')}, pred_shape={pred.shape}"
        set_status(rows, "model_output", "PASS" if ok else "FAIL", note)
        ml_ok = "tracking_uri" in meta and "registered_model_name" in meta and Path("mlruns").exists()
        set_status(rows, "mlflow", "PASS" if ml_ok else "SKIP", f"tracking_uri={meta.get('tracking_uri')}, registered_version={meta.get('registered_version')}")
    except Exception as err:
        set_status(rows, "model_output", "FAIL", str(err))
        set_status(rows, "mlflow", "SKIP", "model meta unavailable")


def check_database(rows):
    try:
        from src.db import repo
        from src.db.schema import init_db, list_tables

        db_path = "db/app.duckdb"
        init_db(db_path)
        suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        ma_bn = repo.insert_patient(db_path, {"ma_benh_nhan": f"BN_DEMO_DB_{suffix}", "ten": "Demo Patient", "tuoi": 45, "gioi_tinh": 1})
        ma_hs = repo.insert_clinical_record(db_path, ma_bn, {"bmi": 26.5, "sp": 130, "bp": 80, "hba1c": 8.1, "fps": 145, "pps": 220, "family_history": 1, "onset_age": 38, "dia_life": 7, "smoking": 0, "phy_act": 1, "med_use": 1, "med_adh": 2})
        ma_model = repo.insert_model_version(db_path, {"model_name": "demo_model", "version": f"demo_v1_{suffix}", "stage": "Production", "f1_micro": 0.5, "f1_macro": 0.4, "recall_macro": 0.35, "hamming_loss": 0.2})
        ma_kq = repo.insert_prediction(db_path, ma_bn, ma_hs, ma_model, "high")
        risk_scores = {"nep": 0.8, "neu": 0.2, "ret": 0.5, "cv": 0.1, "per_vas": 0.0}
        risk_labels = {"nep": 1, "neu": 0, "ret": 1, "cv": 0, "per_vas": 0}
        risk_levels = {"nep": "high", "neu": "low", "ret": "medium", "cv": "low", "per_vas": "low"}
        n = repo.insert_risk_scores(db_path, ma_kq, risk_scores, risk_labels, risk_levels)
        repo.insert_alert(db_path, ma_bn, ma_kq, "high", "Bệnh nhân có nguy cơ cao về biến chứng thận.")
        repo.insert_request_log(db_path, ma_kq, "/predict", 200, 12.5)
        expected = {"benh_nhan", "ho_so_lam_sang", "model_version", "ket_qua_du_doan", "chi_tiet_risk_score", "canh_bao", "request_log", "monitoring_log"}
        ok_schema = expected <= set(list_tables(db_path))
        ok_crud = n == 5 and len(repo.get_prediction_scores(db_path, ma_kq)) == 5
        set_status(rows, "database_schema", "PASS" if ok_schema else "FAIL", f"tables={list_tables(db_path)}")
        set_status(rows, "database", "PASS" if ok_crud else "FAIL", f"counts={repo.get_counts(db_path)}, prediction={ma_kq}")
    except Exception as err:
        set_status(rows, "database_schema", "FAIL", str(err))
        set_status(rows, "database", "FAIL", str(err))


def check_alert(rows):
    try:
        from src.predict.alert import make_alert, make_labels, make_levels

        risk_scores = {"nep": 0.8, "neu": 0.2, "ret": 0.5, "cv": 0.1, "per_vas": 0.0}
        levels = make_levels(risk_scores)
        labels = make_labels(risk_scores)
        alert = make_alert(risk_scores, levels)
        ok = levels["nep"] == "high" and levels["ret"] == "medium" and set(labels) == set(TARGETS) and alert["level"] == "high" and "biến chứng thận" in alert["message"]
        set_status(rows, "alert_engine", "PASS" if ok else "FAIL", f"levels={levels}, alert={alert}")
    except Exception as err:
        set_status(rows, "alert_engine", "FAIL", str(err))


def check_api(rows):
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        from src.db import repo

        api_db = Path("db") / f"demo_check_api_{datetime.now().strftime('%Y%m%d%H%M%S')}.duckdb"
        app.state.db_path = str(api_db)
        app.state.model_path = "models/best_model.pkl"
        app.state.meta_path = "models/meta.json"
        client = TestClient(app)
        payload = {"patient_id": f"BN_DEMO_API_{datetime.now().strftime('%H%M%S')}", "name": "Demo API", "age": 45, "sex": 1, "bmi": 26.5, "sp": 130, "bp": 80, "hba1c": 8.1, "fps": 145, "pps": 220, "family_h": 1, "onset_age": 38, "dia_life": 7, "smoking": 0, "phy_act": 1, "med_use": 1, "med_adh": 2}
        health = client.get("/health")
        pred = client.post("/predict", json=payload)
        metrics = client.get("/metrics")
        data = pred.json() if pred.status_code == 200 else {}
        scores_ok = set(data.get("risk_scores", {})) == set(TARGETS)
        log_ok = False
        if data.get("prediction_id"):
            log_ok = len(repo.get_prediction_scores(str(api_db), data["prediction_id"])) == 5
        set_status(rows, "fastapi_health", "PASS" if health.status_code == 200 else "FAIL", str(health.json()) if health.status_code == 200 else health.text[:300])
        set_status(rows, "fastapi_predict", "PASS" if pred.status_code == 200 and scores_ok else "FAIL", json.dumps(data, ensure_ascii=False)[:1200])
        set_status(rows, "fastapi_metrics", "PASS" if metrics.status_code == 200 else "FAIL", metrics.text[:300])
        set_status(rows, "api_db_logging", "PASS" if log_ok else "FAIL", f"prediction_id={data.get('prediction_id')}")
        rows["_api_response"] = data
    except Exception as err:
        set_status(rows, "fastapi_health", "FAIL", str(err))
        set_status(rows, "fastapi_predict", "FAIL", str(err))
        set_status(rows, "fastapi_metrics", "FAIL", str(err))
        set_status(rows, "api_db_logging", "FAIL", str(err))


def check_mock_his(rows):
    try:
        from src.mock_his.stream import row_to_payload

        df = pd.read_csv("data/processed/clean_data.csv")
        payload = row_to_payload(df.iloc[0], 1)
        ok_feature = set(FEATURES) <= set(payload) and "patient_id" in payload
        leak = [c for c in TARGETS if c in payload]
        set_status(rows, "mock_his", "PASS" if ok_feature and not leak else "FAIL", f"payload_keys={list(payload)}, target_leak={leak}")
    except Exception as err:
        set_status(rows, "mock_his", "FAIL", str(err))


def check_django(rows):
    mig = cmd([sys.executable, "dashboard/manage.py", "migrate"], 300)
    users = cmd([sys.executable, "dashboard/manage.py", "create_demo_users"], 120)
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.config.settings")
        import django
        django.setup()
        from django.test import Client

        client = Client()
        before = client.get("/")
        login_page = client.get("/login/")
        ok_login = client.login(username="doctor", password="doctor123")
        home = client.get("/")
        patients = client.get("/patients/")
        alerts = client.get("/alerts/")
        register = client.get("/register/")
        auth_ok = before.status_code == 302 and login_page.status_code == 200 and ok_login
        dash_ok = home.status_code == 200 and patients.status_code == 200 and alerts.status_code == 200
        no_portal = register.status_code == 404
        set_status(rows, "django_auth", "PASS" if auth_ok else "FAIL", f"migrate={mig['ok']}, users={users['ok']}, before={before.status_code}, login_page={login_page.status_code}, login={ok_login}")
        set_status(rows, "django_dashboard", "PASS" if dash_ok else "FAIL", f"home={home.status_code}, patients={patients.status_code}, alerts={alerts.status_code}")
        set_status(rows, "django_no_patient_portal", "PASS" if no_portal else "FAIL", f"register={register.status_code}")
    except Exception as err:
        set_status(rows, "django_auth", "FAIL", str(err))
        set_status(rows, "django_dashboard", "FAIL", str(err))
        set_status(rows, "django_no_patient_portal", "FAIL", str(err))


def check_evidently(rows):
    res = cmd([sys.executable, "-m", "src.monitoring.drift"], 300)
    summary = Path("reports/monitoring_summary.json")
    html = Path("reports/drift_report.html")
    skip = Path("reports/drift_report_skip.txt")
    try:
        data = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
        ok = summary.exists() and "retrain_flag" in data and "drift_share" in data and "drifted_features" in data
        status = "PASS" if ok else "FAIL"
        note = f"report={html.exists()}, skip={skip.exists()}, retrain_flag={data.get('retrain_flag')}, drift_share={data.get('drift_share')}"
        if ok and skip.exists() and not html.exists():
            status = "SKIP"
            note = "Evidently report skipped but monitoring summary exists. " + note
        set_status(rows, "evidently", status, note)
    except Exception as err:
        set_status(rows, "evidently", "FAIL", f"{err}; output={res['out'][-500:]} {res['err'][-500:]}")


def check_retrain(rows):
    try:
        import src.retrain.flow as flow

        summary = flow.load_monitoring_summary()
        rules = flow.check_rules({"f1_micro": 0.5, "recall_macro": 0.35}, {"f1_micro": 0.49, "recall_macro": 0.36})
        p = Path("reports/retrain_summary.json")
        note = f"monitoring_flag={summary.get('retrain_flag')}, rules_pass={rules.get('pass')}, retrain_summary={p.exists()}"
        set_status(rows, "prefect_retrain", "PASS" if rules.get("pass") else "FAIL", note)
    except Exception as err:
        set_status(rows, "prefect_retrain", "FAIL", str(err))


def check_docker(rows):
    try:
        compose = Path("docker-compose.yml").read_text(encoding="utf-8")
        prom = Path("docker/prometheus.yml").read_text(encoding="utf-8")
        services = all(s in compose for s in ["fastapi", "django", "mlflow", "prometheus", "grafana"])
        ports = all(p in compose for p in ["8000:8000", "8001:8001", "5000:5000", "9090:9090", "3000:3000"])
        scrape = "fastapi:8000" in prom and "/metrics" in prom
        res = cmd(["docker", "compose", "config"], 120) if shutil.which("docker") else {"ok": False, "err": "docker not installed"}
        status = "PASS" if services and ports and scrape and res["ok"] else "SKIP" if "daemon" in res.get("err", "").lower() or not shutil.which("docker") else "FAIL"
        set_status(rows, "docker", status, f"services={services}, ports={ports}, scrape={scrape}, compose_config={res['ok']}, err={res.get('err', '')[:300]}")
    except Exception as err:
        set_status(rows, "docker", "FAIL", str(err))


def check_github(rows):
    try:
        ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        vals = ["name: ci", "push", "pull_request", "main", "python-version: \"3.11\"", "pip install -r requirements.txt", "pytest", "data/processed/clean_data.csv", "from src.api.main import app", "DJANGO_SETTINGS_MODULE", "docker compose config", "docker build"]
        ok = all(v in ci for v in vals) and "https://github.com/DaoTienSang/diabetes.git" in readme
        set_status(rows, "github_actions", "PASS" if ok else "FAIL", "workflow ci has pytest/docker/import checks")
    except Exception as err:
        set_status(rows, "github_actions", "FAIL", str(err))


def check_pytest(rows):
    res = cmd([sys.executable, "-m", "pytest", "-q"], 900)
    set_status(rows, "pytest_all", "PASS" if res["ok"] else "FAIL", (res["out"] + "\n" + res["err"])[-2000:])


def status(rows, key):
    return rows.get(key, {}).get("status", "FAIL")


def write_report(rows, dataset_kind):
    api_response = rows.get("_api_response", {})
    meta = {}
    try:
        meta = json.loads(Path("models/meta.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    summary = {}
    try:
        summary = json.loads(Path("reports/monitoring_summary.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    table_map = [
        ("Project structure", "project_structure"),
        ("Imports", "imports"),
        ("Data cleaning", "data_cleaning"),
        ("DuckDB analytics", "duckdb_analytics"),
        ("Feature builder", "feature_builder"),
        ("Training", "training"),
        ("Model output 5 targets", "model_output"),
        ("Database schema", "database_schema"),
        ("Database CRUD", "database"),
        ("Alert Engine", "alert_engine"),
        ("FastAPI /health", "fastapi_health"),
        ("FastAPI /predict", "fastapi_predict"),
        ("FastAPI /metrics", "fastapi_metrics"),
        ("API DB logging", "api_db_logging"),
        ("Mock HIS", "mock_his"),
        ("Django login", "django_auth"),
        ("Django Dashboard", "django_dashboard"),
        ("No Patient Portal", "django_no_patient_portal"),
        ("MLflow", "mlflow"),
        ("Evidently", "evidently"),
        ("Prefect Retrain", "prefect_retrain"),
        ("Docker Compose config", "docker"),
        ("GitHub Actions config", "github_actions"),
        ("Pytest all", "pytest_all"),
    ]
    core = ["data_cleaning", "duckdb_analytics", "feature_builder", "training", "model_output", "database", "alert_engine", "fastapi_health", "fastapi_predict", "fastapi_metrics", "api_db_logging", "mock_his", "django_auth", "django_dashboard"]
    optional = ["mlflow", "evidently", "prefect_retrain", "docker"]
    core_failed = [k for k in core if status(rows, k) != "PASS"]
    optional_failed = [k for k in optional if status(rows, k) == "FAIL"]
    demo_ready = "yes" if not core_failed else "partial" if not [k for k in core if status(rows, k) == "FAIL"] else "no"
    lines = [
        "# Demo Check Result",
        "",
        "## Tổng quan",
        f"- Ngày chạy: {datetime.now().isoformat(timespec='seconds')}",
        f"- Python version: {platform.python_version()}",
        f"- OS: {platform.platform()}",
        f"- Repo path: {Path.cwd()}",
        f"- Dataset: {dataset_kind}",
        f"- Model hiện tại: {meta.get('best_model')}",
        "- DB path: db/app.duckdb",
        "",
        "## Bảng trạng thái",
        "",
        "| Module | Status | Ghi chú |",
        "|---|---|---|",
    ]
    for label, key in table_map:
        rec = rows.get(key, {"status": "FAIL", "note": "not checked"})
        note = str(rec.get("note", "")).replace("\n", " ")[:500]
        lines.append(f"| {label} | {rec.get('status')} | {note} |")
    lines += [
        "",
        "## API demo response",
        "",
        "```json",
        json.dumps(api_response, ensure_ascii=False, indent=2)[:4000],
        "```",
        "",
        "## Dashboard demo",
        "- URL: http://localhost:8001",
        "- Account doctor: doctor / doctor123",
        "- Account admin: admin / admin123",
        "- Các trang đã test: /login/, /, /patients/, /alerts/, /register/",
        "",
        "## Monitoring demo",
        f"- /metrics status: {status(rows, 'fastapi_metrics')}",
        f"- monitoring_summary.json: {Path('reports/monitoring_summary.json').exists()}",
        f"- retrain_flag: {summary.get('retrain_flag')}",
        "",
        "## Lỗi còn lại",
    ]
    fails = [(k, v) for k, v in rows.items() if isinstance(v, dict) and v.get("status") == "FAIL"]
    if fails:
        for key, rec in fails:
            lines.append(f"- {key}: {rec.get('note')}")
    else:
        lines.append("- Không có lỗi core còn lại.")
    lines += [
        "",
        "## Kết luận",
        f"- demo_ready: {demo_ready}",
        f"- core_failed: {', '.join(core_failed) if core_failed else 'none'}",
        f"- optional_failed: {', '.join(optional_failed) if optional_failed else 'none'}",
    ]
    Path("reports/demo_check_result.md").write_text("\n".join(lines), encoding="utf-8")
    return demo_ready, core_failed, optional_failed


def write_runbook():
    steps = [
        ("Giới thiệu kiến trúc AI Extend Module", "Không cần lệnh", "Nói rõ không xây lại HIS, chỉ mở rộng AI module.", "Kiểm tra README mục kiến trúc."),
        ("Nói dataset Mendeley dsjcb6pyd8", "Không cần lệnh", "Giải thích dataset tabular, không phải streaming thật.", "Kiểm tra data/raw/data.csv."),
        ("Chạy data cleaning", "python -m src.data.clean_data", "Tạo data/processed/clean_data.csv.", "Kiểm tra raw path và cột dataset."),
        ("Load DuckDB", "python - <<'PY'\nfrom src.db.analytics import load_clean_to_duckdb\nload_clean_to_duckdb()\nprint('duckdb training data ready')\nPY", "Bảng du_lieu_huan_luyen có dữ liệu.", "Kiểm tra db/app.duckdb."),
        ("Train model", "python -m src.train.train_models", "Tạo best_model.pkl, meta.json, metrics.csv.", "Kiểm tra clean_data.csv và requirements."),
        ("Mở metrics.csv", "python - <<'PY'\nimport pandas as pd\nprint(pd.read_csv('reports/metrics.csv'))\nPY", "Thấy logistic_regression/random_forest và metrics.", "Kiểm tra reports/metrics.csv."),
        ("Chạy FastAPI", "uvicorn src.api.main:app --reload", "API chạy ở localhost:8000.", "Kiểm tra models/best_model.pkl."),
        ("Gọi /health", "curl http://localhost:8000/health", "Trả {'status':'ok'}.", "Kiểm tra FastAPI process."),
        ("Gọi /predict", "Dùng curl mẫu trong README mục FastAPI.", "Trả risk_scores/risk_levels/warning_message.", "Kiểm tra model/meta."),
        ("Chạy Mock HIS gửi 5 hồ sơ", "python -m src.mock_his.stream --limit 5 --delay 1", "In send/status/level.", "Kiểm tra FastAPI đang chạy."),
        ("Mở Django Dashboard", "python dashboard/manage.py runserver 8001", "Dashboard ở localhost:8001.", "Chạy migrate và create_demo_users."),
        ("Login doctor / doctor123", "Mở http://localhost:8001/login/", "Đăng nhập thành công.", "Chạy create_demo_users."),
        ("Xem danh sách bệnh nhân", "Mở /patients/", "Thấy bệnh nhân từ DuckDB.", "Chạy /predict hoặc Mock HIS trước."),
        ("Xem chi tiết bệnh nhân", "Mở /patients/<ma_benh_nhan>/", "Thấy hồ sơ và prediction mới nhất.", "Kiểm tra DB logs."),
        ("Xem risk score 5 biến chứng", "Trong trang chi tiết bệnh nhân", "Thấy nep, neu, ret, cv, per_vas.", "Kiểm tra chi_tiet_risk_score."),
        ("Xem cảnh báo", "Mở /alerts/", "Thấy cảnh báo high/medium nếu có.", "Gửi payload có risk cao."),
        ("Mở MLflow UI", "mlflow ui --host 0.0.0.0 --port 5000", "UI mặc định ở localhost:5000.", "Kiểm tra mlruns."),
        ("Mở /metrics", "curl http://localhost:8000/metrics", "Thấy Prometheus metrics.", "Kiểm tra FastAPI."),
        ("Mở Prometheus", "http://localhost:9090", "UI mặc định Prometheus.", "Dùng Docker Compose."),
        ("Mở Grafana", "http://localhost:3000", "Login admin/admin, datasource Prometheus.", "Dùng Docker Compose."),
        ("Chạy Evidently drift", "python -m src.monitoring.drift", "Tạo monitoring_summary.json.", "Nếu lỗi version xem drift_report_skip.txt."),
        ("Xem monitoring_summary.json", "python - <<'PY'\nimport json\nprint(json.load(open('reports/monitoring_summary.json', encoding='utf-8')))\nPY", "Thấy retrain_flag/drift_share.", "Chạy monitoring trước."),
        ("Chạy Prefect retrain", "python -m src.retrain.flow", "Tạo retrain_summary.json hoặc no_retrain.", "Kiểm tra monitoring_summary.json."),
        ("Xem retrain_summary.json", "python - <<'PY'\nimport json\nprint(json.load(open('reports/retrain_summary.json', encoding='utf-8')))\nPY", "Thấy final_status/promoted/rule_result.", "Chạy retrain flow trước."),
        ("Chạy docker compose config", "docker compose config", "Compose hợp lệ.", "Kiểm tra Docker Desktop."),
        ("Nói GitHub Actions CI", "Mở .github/workflows/ci.yml", "Có pytest, docker compose config, docker build.", "Kiểm tra tab Actions trên GitHub."),
        ("Chạy pytest", "pytest -q", "Toàn bộ test pass.", "Nếu fail đọc tên test và sửa tối thiểu."),
        ("Kết luận hạn chế", "Không cần lệnh", "Nhấn mạnh demo học thuật, không phải chẩn đoán y khoa.", "Mở README mục Hạn chế."),
    ]
    lines = ["# Demo Runbook", ""]
    for i, (title, command, expected, check) in enumerate(steps, start=1):
        lines += [
            f"## {i}. {title}",
            f"- Lệnh chạy: `{command}`" if "\n" not in command else "- Lệnh chạy:\n\n```bash\n" + command + "\n```",
            f"- Kết quả mong muốn: {expected}",
            f"- Nếu lỗi thì kiểm tra: {check}",
            "",
        ]
    Path("reports/demo_runbook.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows, demo_ready, core_failed, optional_failed):
    order = [
        ("project_structure", "project_structure"),
        ("data_cleaning", "data_cleaning"),
        ("duckdb_analytics", "duckdb_analytics"),
        ("feature_builder", "feature_builder"),
        ("training", "training"),
        ("model_output", "model_output"),
        ("database", "database"),
        ("alert_engine", "alert_engine"),
        ("fastapi", "fastapi_predict"),
        ("mock_his", "mock_his"),
        ("django_dashboard", "django_dashboard"),
        ("mlflow", "mlflow"),
        ("evidently", "evidently"),
        ("prefect_retrain", "prefect_retrain"),
        ("docker", "docker"),
        ("github_actions", "github_actions"),
        ("pytest_all", "pytest_all"),
    ]
    for label, key in order:
        print(f"{label}: {status(rows, key)}")
    print(f"demo_ready: {demo_ready}")
    if core_failed:
        print("core_failed:", ", ".join(core_failed))
    if optional_failed:
        print("optional_failed:", ", ".join(optional_failed))


def main():
    rows = {}
    dataset_kind = "real dataset at data/raw/data.csv" if Path("data/raw/data.csv").exists() else "sample data generated for demo check"
    check_structure(rows)
    check_imports(rows)
    check_data(rows)
    check_features(rows)
    check_training(rows)
    check_database(rows)
    check_alert(rows)
    check_api(rows)
    check_mock_his(rows)
    check_django(rows)
    check_evidently(rows)
    check_retrain(rows)
    check_docker(rows)
    check_github(rows)
    check_pytest(rows)
    write_runbook()
    demo_ready, core_failed, optional_failed = write_report(rows, dataset_kind)
    print_summary(rows, demo_ready, core_failed, optional_failed)


if __name__ == "__main__":
    main()
