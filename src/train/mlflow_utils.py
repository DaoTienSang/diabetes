import os
from pathlib import Path


def mlflow_mod():
    import mlflow

    return mlflow


def setup_mlflow(experiment_name="diabetes_complication_prediction"):
    try:
        mlflow = mlflow_mod()
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        return True
    except Exception:
        print("mlflow_setup_skip")
        return False


def start_run_safe(run_name):
    try:
        mlflow = mlflow_mod()
        if mlflow.active_run():
            mlflow.end_run()
        return mlflow.start_run(run_name=run_name)
    except Exception:
        print("mlflow_run_skip")
        return None


def end_run_safe():
    try:
        mlflow = mlflow_mod()
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        return None


def log_params_safe(params):
    try:
        mlflow = mlflow_mod()
        rec = {}
        for key, val in params.items():
            if isinstance(val, (str, int, float, bool)) or val is None:
                rec[key] = val
            else:
                rec[key] = str(val)
        mlflow.log_params(rec)
    except Exception:
        print("mlflow_params_skip")


def log_metrics_safe(metrics):
    try:
        mlflow = mlflow_mod()
        rec = {}
        for key, val in metrics.items():
            try:
                rec[key] = float(val)
            except Exception:
                pass
        mlflow.log_metrics(rec)
    except Exception:
        print("mlflow_metrics_skip")


def log_artifact_safe(path):
    try:
        fp = Path(path)
        if fp.exists():
            mlflow = mlflow_mod()
            mlflow.log_artifact(str(fp))
    except Exception:
        print("mlflow_artifact_skip")


def log_model_safe(model, artifact_path="model"):
    try:
        mlflow = mlflow_mod()
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    except Exception:
        print("mlflow_model_skip")


def register_model_safe(model_uri, name="diabetes_complication_model"):
    try:
        mlflow = mlflow_mod()
        rec = mlflow.register_model(model_uri=model_uri, name=name)
        return rec.version
    except Exception:
        print("mlflow_registry_skip")
        return None


def log_one_model_run(name, model, params, metrics, artifacts=None, register=False):
    ok = setup_mlflow()
    run = start_run_safe(name) if ok else None
    if run is None:
        return {"logged": False, "registered_version": None}
    registered_version = None
    try:
        log_params_safe(params)
        log_metrics_safe(metrics)
        for path in artifacts or []:
            log_artifact_safe(path)
        log_model_safe(model)
        if register:
            run_id = run.info.run_id
            registered_version = register_model_safe(f"runs:/{run_id}/model")
        return {"logged": True, "registered_version": registered_version}
    except Exception:
        return {"logged": False, "registered_version": None}
    finally:
        end_run_safe()


def get_tracking_uri_safe():
    try:
        mlflow = mlflow_mod()
        return str(mlflow.get_tracking_uri())
    except Exception:
        return "mlflow_unavailable"
