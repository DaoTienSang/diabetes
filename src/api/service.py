import json
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.db.repo import get_patient
from src.db.repo import insert_alert
from src.db.repo import insert_clinical_record
from src.db.repo import insert_model_version
from src.db.repo import insert_patient
from src.db.repo import insert_prediction
from src.db.repo import insert_request_log
from src.db.repo import insert_risk_scores
from src.db.repo import new_id
from src.db.schema import get_conn
from src.db.schema import init_db
from src.predict.alert import make_alert
from src.predict.alert import make_labels
from src.predict.alert import make_levels


@lru_cache(maxsize=8)
def load_meta(path="models/meta.json"):
    fp = Path(path)
    return json.loads(fp.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_model(path="models/best_model.pkl"):
    return joblib.load(path)


def data_dict(payload):
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def make_input_df(payload, ft):
    rec = data_dict(payload)
    return pd.DataFrame([{c: rec[c] for c in ft}])


def get_prob(mdl, x, tg):
    if hasattr(mdl, "predict_proba"):
        prob = mdl.predict_proba(x)
        if isinstance(prob, list):
            rec = {}
            for i, target in enumerate(tg):
                arr = prob[i]
                rec[target] = float(arr[0][1]) if arr.shape[1] > 1 else float(arr[0][0])
            return rec
        return {target: float(prob[0][i]) for i, target in enumerate(tg)}

    pred = mdl.predict(x)
    return {target: float(pred[0][i]) for i, target in enumerate(tg)}


def find_model_version(db_path, model_name, version):
    con = get_conn(db_path)
    try:
        rec = con.execute(
            """
            SELECT ma_model
            FROM model_version
            WHERE model_name = ? AND version = ?
            LIMIT 1
            """,
            [model_name, version],
        ).fetchone()
    finally:
        con.close()
    return rec[0] if rec else None


def get_or_create_model_version(db_path, meta):
    name = meta.get("best_model", "model")
    ver = meta.get("model_version", "unknown")
    ma = find_model_version(db_path, name, ver)
    if ma:
        return ma
    return insert_model_version(
        db_path,
        {
            "model_name": name,
            "version": ver,
            "stage": "production",
            "f1_micro": None,
            "f1_macro": None,
            "recall_macro": None,
            "hamming_loss": None,
        },
    )


def predict_one(
    payload,
    db_path="db/app.duckdb",
    model_path="models/best_model.pkl",
    meta_path="models/meta.json",
):
    st = time.perf_counter()
    meta = load_meta(meta_path)
    mdl = load_model(model_path)
    ft = meta["features"]
    tg = meta["targets"]
    x = make_input_df(payload, ft)
    risk_scores = get_prob(mdl, x, tg)
    risk_labels = make_labels(risk_scores)
    risk_levels = make_levels(risk_scores)
    alert = make_alert(risk_scores, risk_levels)
    rec = data_dict(payload)

    init_db(db_path)
    ma_bn = rec.get("patient_id") or new_id("bn")
    if not get_patient(db_path, ma_bn):
        insert_patient(
            db_path,
            {
                "ma_benh_nhan": ma_bn,
                "ten": rec.get("name"),
                "tuoi": rec["age"],
                "gioi_tinh": rec["sex"],
            },
        )

    ma_hs = insert_clinical_record(
        db_path,
        ma_bn,
        {
            "bmi": rec["bmi"],
            "sp": rec["sp"],
            "bp": rec["bp"],
            "hba1c": rec["hba1c"],
            "fps": rec["fps"],
            "pps": rec["pps"],
            "family_history": rec["family_h"],
            "onset_age": rec["onset_age"],
            "dia_life": rec["dia_life"],
            "smoking": rec["smoking"],
            "phy_act": rec["phy_act"],
            "med_use": rec["med_use"],
            "med_adh": rec["med_adh"],
        },
    )
    ma_model = get_or_create_model_version(db_path, meta)
    ma_kq = insert_prediction(db_path, ma_bn, ma_hs, ma_model, alert["level"])
    insert_risk_scores(db_path, ma_kq, risk_scores, risk_labels, risk_levels)

    if alert["level"] in {"medium", "high"}:
        insert_alert(db_path, ma_bn, ma_kq, alert["level"], alert["message"])

    latency = (time.perf_counter() - st) * 1000
    insert_request_log(db_path, ma_kq, "/predict", 200, latency)
    ts = datetime.now().isoformat()

    return {
        "patient_id": ma_bn,
        "prediction_id": ma_kq,
        "risk_scores": risk_scores,
        "risk_labels": risk_labels,
        "risk_levels": risk_levels,
        "warning_level": alert["level"],
        "warning_message": alert["message"],
        "model_version": meta.get("model_version", "unknown"),
        "timestamp": ts,
    }
