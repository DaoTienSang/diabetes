from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def readme():
    return README.read_text(encoding="utf-8")


def test_readme_exists():
    assert README.is_file()


def test_readme_has_project_title():
    assert "Hệ thống dự đoán biến chứng tiểu đường theo thời gian thực" in readme()


def test_readme_has_ai_extend_module():
    assert "AI Extend Module" in readme()


def test_readme_has_dataset():
    assert "Mendeley dsjcb6pyd8" in readme()


def test_readme_has_features():
    text = readme()
    for val in ["age", "sex", "bmi", "hba1c", "fps", "pps", "dia_life", "med_adh"]:
        assert val in text


def test_readme_has_targets():
    text = readme()
    for val in ["nep", "neu", "ret", "cv", "per_vas"]:
        assert val in text


def test_readme_has_no_lstm_reason():
    text = readme()
    assert "không dùng LSTM" in text or "Không dùng LSTM" in text


def test_readme_has_mock_his():
    assert "Mock HIS" in readme()


def test_readme_has_fastapi_commands():
    text = readme()
    for val in ["uvicorn src.api.main:app --reload", "/health", "/predict", "/metrics"]:
        assert val in text


def test_readme_has_dashboard_commands():
    text = readme()
    for val in ["python dashboard/manage.py runserver 8001", "create_demo_users", "doctor", "admin"]:
        assert val in text


def test_readme_has_mlflow_prometheus_grafana():
    text = readme()
    for val in ["MLflow", "Prometheus", "Grafana"]:
        assert val in text


def test_readme_has_evidently_prefect():
    text = readme()
    assert "python -m src.monitoring.drift" in text
    assert "python -m src.retrain.flow" in text


def test_readme_has_docker():
    assert "docker compose up --build" in readme()


def test_readme_has_pytest():
    assert "pytest" in readme()


def test_readme_has_repo_url():
    assert "https://github.com/DaoTienSang/diabetes.git" in readme()


def test_readme_has_limitations():
    text = readme()
    for val in ["duplicate", "PER VAS", "demo học thuật", "không phải chẩn đoán y khoa"]:
        assert val in text


def test_readme_no_patient_portal():
    text = readme()
    assert "không có tài khoản bệnh nhân" in text or "không làm Patient Portal" in text


def test_readme_has_du_lieu_huan_luyen():
    assert "du_lieu_huan_luyen" in readme()


def test_readme_has_domain_entities():
    text = readme()
    assert "domain entities" in text or "Class Diagram" in text
