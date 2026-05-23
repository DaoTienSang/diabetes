import os

from src.db import repo


def get_db_path():
    return os.environ.get("APP_DB_PATH", "db/app.duckdb")


def safe_counts(db_path):
    try:
        rec = repo.get_counts(db_path)
        alerts = repo.get_alerts(db_path, 1000)
        return {
            "patients": rec.get("benh_nhan", 0),
            "predictions": rec.get("ket_qua_du_doan", 0),
            "alerts": rec.get("canh_bao", 0),
            "high_alerts": len([r for r in alerts if r.get("muc_canh_bao") == "high"]),
            "medium_alerts": len([r for r in alerts if r.get("muc_canh_bao") == "medium"]),
        }
    except Exception:
        return {
            "patients": 0,
            "predictions": 0,
            "alerts": 0,
            "high_alerts": 0,
            "medium_alerts": 0,
        }


def safe_latest_model(db_path):
    try:
        return repo.get_latest_model_version(db_path)
    except Exception:
        return None


def safe_recent_predictions(db_path, limit=20):
    try:
        return repo.get_recent_predictions(db_path, limit)
    except Exception:
        return []


def safe_patients(db_path, limit=100):
    try:
        return repo.list_patients(db_path, limit)
    except Exception:
        return []


def safe_patient_detail(db_path, ma_benh_nhan):
    try:
        patient = repo.get_patient(db_path, ma_benh_nhan)
        clinical = repo.get_latest_clinical_record(db_path, ma_benh_nhan)
        prediction = repo.get_latest_prediction(db_path, ma_benh_nhan)
        scores = repo.get_prediction_scores(db_path, prediction["ma_ket_qua"]) if prediction else []
        history = repo.get_patient_predictions(db_path, ma_benh_nhan)
        return {
            "patient": patient,
            "clinical": clinical,
            "prediction": prediction,
            "scores": scores,
            "history": history,
        }
    except Exception:
        return {
            "patient": None,
            "clinical": None,
            "prediction": None,
            "scores": [],
            "history": [],
        }


def safe_alerts(db_path, limit=100):
    try:
        return repo.get_alerts(db_path, limit)
    except Exception:
        return []


def safe_prediction_detail(db_path, ma_ket_qua):
    try:
        prediction = repo.get_prediction(db_path, ma_ket_qua)
        scores = repo.get_prediction_scores(db_path, ma_ket_qua) if prediction else []
        return {"prediction": prediction, "scores": scores}
    except Exception:
        return {"prediction": None, "scores": []}
