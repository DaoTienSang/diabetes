import os

import numpy as np
from sklearn.dummy import DummyClassifier

from src.train.mlflow_utils import end_run_safe
from src.train.mlflow_utils import get_tracking_uri_safe
from src.train.mlflow_utils import log_artifact_safe
from src.train.mlflow_utils import log_metrics_safe
from src.train.mlflow_utils import log_model_safe
from src.train.mlflow_utils import log_one_model_run
from src.train.mlflow_utils import log_params_safe
from src.train.mlflow_utils import setup_mlflow
from src.train.mlflow_utils import start_run_safe


def use_tmp_mlruns(tmp_path):
    os.environ["MLFLOW_TRACKING_URI"] = str(tmp_path / "mlruns")


def fitted_model():
    x = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 0, 1])
    mdl = DummyClassifier(strategy="most_frequent")
    mdl.fit(x, y)
    return mdl


def test_setup_mlflow_no_crash(tmp_path):
    use_tmp_mlruns(tmp_path)
    assert setup_mlflow("test_diabetes") in {True, False}


def test_start_end_run_safe(tmp_path):
    use_tmp_mlruns(tmp_path)
    setup_mlflow("test_diabetes")
    start_run_safe("test_run")
    end_run_safe()


def test_log_params_safe(tmp_path):
    use_tmp_mlruns(tmp_path)
    setup_mlflow("test_diabetes")
    start_run_safe("params")
    log_params_safe({"model_name": "random_forest", "n_estimators": 300})
    end_run_safe()


def test_log_metrics_safe(tmp_path):
    use_tmp_mlruns(tmp_path)
    setup_mlflow("test_diabetes")
    start_run_safe("metrics")
    log_metrics_safe({"f1_micro": 0.5, "label_acc": 0.8})
    end_run_safe()


def test_log_artifact_safe_missing_file(tmp_path):
    use_tmp_mlruns(tmp_path)
    setup_mlflow("test_diabetes")
    start_run_safe("artifact")
    log_artifact_safe(tmp_path / "missing.csv")
    end_run_safe()


def test_log_model_safe_dummy_model(tmp_path):
    use_tmp_mlruns(tmp_path)
    setup_mlflow("test_diabetes")
    start_run_safe("model")
    log_model_safe(fitted_model())
    end_run_safe()


def test_log_one_model_run(tmp_path):
    use_tmp_mlruns(tmp_path)
    rec = log_one_model_run(
        "dummy",
        fitted_model(),
        {"model_name": "dummy"},
        {"f1_micro": 0.5},
        register=False,
    )
    assert "logged" in rec


def test_get_tracking_uri_safe(tmp_path):
    use_tmp_mlruns(tmp_path)
    assert isinstance(get_tracking_uri_safe(), str)
