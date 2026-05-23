from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"


def read_ci():
    return CI_PATH.read_text(encoding="utf-8")


def test_ci_file_exists():
    assert CI_PATH.is_file()


def test_ci_has_main_triggers():
    text = read_ci()
    assert "push:" in text
    assert "pull_request:" in text
    assert "main" in text


def test_ci_uses_python_311():
    text = read_ci()
    assert 'python-version: "3.11"' in text or "python-version: 3.11" in text


def test_ci_runs_pytest():
    assert "pytest" in read_ci()


def test_ci_checks_docker_compose():
    assert "docker compose config" in read_ci()


def test_ci_builds_docker_image():
    assert "docker build" in read_ci()


def test_ci_no_deploy_secret():
    text = read_ci()
    assert "secrets.DOCKERHUB_TOKEN" not in text
    assert "docker push" not in text
    assert "deploy production" not in text.lower()


def test_ci_creates_sample_data():
    text = read_ci()
    assert "data/processed/clean_data.csv" in text
    assert "sample data ready" in text


def test_ci_imports_fastapi():
    assert "from src.api.main import app" in read_ci()


def test_ci_imports_django():
    assert "DJANGO_SETTINGS_MODULE" in read_ci()
