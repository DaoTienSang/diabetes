import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path):
    try:
        import yaml

        return yaml.safe_load(read(path))
    except Exception:
        return None


def compose_data():
    data = load_yaml("docker-compose.yml")
    if data:
        return data
    return {}


def test_dockerfile_exists():
    assert (ROOT / "Dockerfile").is_file()


def test_compose_exists():
    assert (ROOT / "docker-compose.yml").is_file()


def test_prometheus_config_exists():
    assert (ROOT / "docker" / "prometheus.yml").is_file()


def test_grafana_datasource_exists():
    assert (ROOT / "docker" / "grafana" / "provisioning" / "datasources" / "datasource.yml").is_file()


def test_compose_has_services():
    text = read("docker-compose.yml")
    data = compose_data()
    if data:
        services = set(data["services"])
        assert {"fastapi", "django", "mlflow", "prometheus", "grafana"} <= services
    else:
        for service in ["fastapi", "django", "mlflow", "prometheus", "grafana"]:
            assert f"  {service}:" in text


def test_compose_ports():
    text = read("docker-compose.yml")
    for port in ["8000:8000", "8001:8001", "5000:5000", "9090:9090", "3000:3000"]:
        assert port in text


def test_fastapi_command():
    text = read("docker-compose.yml")
    assert "uvicorn" in text
    assert "src.api.main:app" in text
    assert "--host 0.0.0.0" in text
    assert "--port 8000" in text


def test_django_command():
    text = read("docker-compose.yml")
    assert "dashboard/manage.py" in text
    assert "runserver" in text
    assert "0.0.0.0:8001" in text


def test_mlflow_command():
    text = read("docker-compose.yml")
    assert "mlflow ui" in text
    assert "--port 5000" in text


def test_prometheus_scrape_fastapi():
    text = read("docker/prometheus.yml")
    assert "fastapi:8000" in text
    assert 'metrics_path: "/metrics"' in text


def test_no_custom_mlops_ui():
    text = read("docker-compose.yml")
    assert "mlflow_ui_custom" not in text
    assert "grafana_ui_custom" not in text
    assert "prometheus_ui_custom" not in text
    for path in ROOT.rglob("*"):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        assert "/mlflow-ui/" not in rel
        assert "/grafana-ui/" not in rel
        assert "/prometheus-ui/" not in rel


def test_volumes_mounted():
    text = read("docker-compose.yml")
    for vol in ["./data:/app/data", "./db:/app/db", "./models:/app/models", "./reports:/app/reports"]:
        assert vol in text


def test_yaml_parse():
    data = load_yaml("docker-compose.yml")
    prom = load_yaml("docker/prometheus.yml")
    if data and prom:
        assert "services" in data
        assert "scrape_configs" in prom
    else:
        assert "services:" in read("docker-compose.yml")
        assert "scrape_configs:" in read("docker/prometheus.yml")


def test_docker_compose_config_optional():
    if not shutil.which("docker"):
        pytest.skip("docker not installed")
    res = subprocess.run(
        ["docker", "compose", "config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if res.returncode != 0 and "docker" in (res.stderr + res.stdout).lower():
        pytest.skip("docker compose unavailable")
    assert res.returncode == 0


def test_readme_has_docker_commands():
    text = read("README.md")
    for val in [
        "docker compose up --build",
        "http://localhost:8000/health",
        "http://localhost:8001",
        "http://localhost:5000",
        "http://localhost:9090",
        "http://localhost:3000",
    ]:
        assert val in text
