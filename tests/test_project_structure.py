import importlib
from pathlib import Path


def test_main_dirs_exist():
    dirs = [
        "data/raw",
        "data/processed",
        "db",
        "models",
        "reports",
        "src",
        "src/domain",
        "dashboard",
        "tests",
        "docker",
    ]

    for path in dirs:
        assert Path(path).is_dir()


def test_base_files_exist():
    files = [
        "requirements.txt",
        "README.md",
    ]

    for path in files:
        assert Path(path).is_file()


def test_import_src():
    mod = importlib.import_module("src")
    assert mod is not None
