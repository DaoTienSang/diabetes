from datetime import datetime
from uuid import uuid4

from src.db.schema import get_conn
from src.db.schema import init_db


TARGETS = ["nep", "neu", "ret", "cv", "per_vas"]


def new_id(prefix):
    return f"{prefix}_{uuid4().hex[:12]}"


def val(rec, key, default=None):
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def rows(db_path, sql, params=None):
    init_db(db_path)
    con = get_conn(db_path)
    try:
        df = con.execute(sql, params or []).fetchdf()
    finally:
        con.close()
    return df.to_dict("records")


def one(db_path, sql, params=None):
    rec = rows(db_path, sql, params)
    return rec[0] if rec else None


def insert_patient(db_path, patient):
    init_db(db_path)
    ma = val(patient, "ma_benh_nhan") or new_id("bn")
    rec = (
        ma,
        val(patient, "ten"),
        val(patient, "tuoi"),
        val(patient, "gioi_tinh"),
        datetime.now(),
    )
    con = get_conn(db_path)
    try:
        con.execute("INSERT INTO benh_nhan VALUES (?, ?, ?, ?, ?)", rec)
    finally:
        con.close()
    return ma


def insert_clinical_record(db_path, ma_benh_nhan, rec):
    init_db(db_path)
    ma = val(rec, "ma_ho_so") or new_id("hs")
    data = (
        ma,
        ma_benh_nhan,
        val(rec, "bmi"),
        val(rec, "sp"),
        val(rec, "bp"),
        val(rec, "hba1c"),
        val(rec, "fps"),
        val(rec, "pps"),
        val(rec, "family_history", val(rec, "family_h")),
        val(rec, "onset_age"),
        val(rec, "dia_life"),
        val(rec, "smoking"),
        val(rec, "phy_act"),
        val(rec, "med_use"),
        val(rec, "med_adh"),
        datetime.now(),
    )
    con = get_conn(db_path)
    try:
        con.execute("INSERT INTO ho_so_lam_sang VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
    finally:
        con.close()
    return ma


def insert_model_version(db_path, model):
    init_db(db_path)
    ma = val(model, "ma_model") or new_id("mdl")
    data = (
        ma,
        val(model, "model_name"),
        val(model, "version", val(model, "model_version")),
        val(model, "stage", "staging"),
        val(model, "f1_micro"),
        val(model, "f1_macro"),
        val(model, "recall_macro"),
        val(model, "hamming_loss"),
        datetime.now(),
    )
    con = get_conn(db_path)
    try:
        con.execute("INSERT INTO model_version VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
    finally:
        con.close()
    return ma


def insert_prediction(db_path, ma_benh_nhan, ma_ho_so, ma_model, muc_nguy_co_tong):
    init_db(db_path)
    ma = new_id("kq")
    con = get_conn(db_path)
    try:
        con.execute(
            "INSERT INTO ket_qua_du_doan VALUES (?, ?, ?, ?, ?, ?)",
            (ma, ma_benh_nhan, ma_ho_so, ma_model, muc_nguy_co_tong, datetime.now()),
        )
    finally:
        con.close()
    return ma


def get_item(v, key, idx):
    if isinstance(v, dict):
        return v[key]
    return v[idx]


def insert_risk_scores(db_path, ma_ket_qua, risk_scores, risk_labels, risk_levels):
    init_db(db_path)
    recs = []
    for i, tg in enumerate(TARGETS):
        recs.append(
            (
                new_id("ct"),
                ma_ket_qua,
                tg,
                float(get_item(risk_scores, tg, i)),
                int(get_item(risk_labels, tg, i)),
                str(get_item(risk_levels, tg, i)),
            )
        )
    con = get_conn(db_path)
    try:
        con.executemany("INSERT INTO chi_tiet_risk_score VALUES (?, ?, ?, ?, ?, ?)", recs)
    finally:
        con.close()
    return len(recs)


def insert_alert(db_path, ma_benh_nhan, ma_ket_qua, muc_canh_bao, noi_dung, trang_thai="new"):
    init_db(db_path)
    ma = new_id("cb")
    con = get_conn(db_path)
    try:
        con.execute(
            "INSERT INTO canh_bao VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ma, ma_benh_nhan, ma_ket_qua, muc_canh_bao, noi_dung, trang_thai, datetime.now()),
        )
    finally:
        con.close()
    return ma


def insert_request_log(db_path, ma_ket_qua, endpoint, status_code, latency_ms):
    init_db(db_path)
    ma = new_id("log")
    con = get_conn(db_path)
    try:
        con.execute(
            "INSERT INTO request_log VALUES (?, ?, ?, ?, ?, ?)",
            (ma, ma_ket_qua, endpoint, status_code, latency_ms, datetime.now()),
        )
    finally:
        con.close()
    return ma


def insert_monitoring_log(db_path, ma_model, drift_score, recall_score, f1_score, retrain_flag):
    init_db(db_path)
    ma = new_id("mon")
    con = get_conn(db_path)
    try:
        con.execute(
            "INSERT INTO monitoring_log VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ma, ma_model, drift_score, recall_score, f1_score, retrain_flag, datetime.now()),
        )
    finally:
        con.close()
    return ma


def get_counts(db_path):
    init_db(db_path)
    tables = [
        "benh_nhan",
        "ho_so_lam_sang",
        "model_version",
        "ket_qua_du_doan",
        "chi_tiet_risk_score",
        "canh_bao",
        "request_log",
        "monitoring_log",
        "du_lieu_huan_luyen",
    ]
    con = get_conn(db_path)
    try:
        rec = {t: int(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tables}
    finally:
        con.close()
    return rec


def get_high_risk_alerts(db_path, limit=20):
    return rows(
        db_path,
        "SELECT * FROM canh_bao WHERE muc_canh_bao = 'high' ORDER BY thoi_gian_tao DESC LIMIT ?",
        [limit],
    )


def get_patient_predictions(db_path, ma_benh_nhan):
    return rows(
        db_path,
        "SELECT * FROM ket_qua_du_doan WHERE ma_benh_nhan = ? ORDER BY thoi_gian_du_doan DESC",
        [ma_benh_nhan],
    )


def list_patients(db_path, limit=100):
    return rows(db_path, "SELECT * FROM benh_nhan ORDER BY ngay_tao DESC LIMIT ?", [limit])


def get_patient(db_path, ma_benh_nhan):
    return one(db_path, "SELECT * FROM benh_nhan WHERE ma_benh_nhan = ?", [ma_benh_nhan])


def get_latest_clinical_record(db_path, ma_benh_nhan):
    return one(
        db_path,
        "SELECT * FROM ho_so_lam_sang WHERE ma_benh_nhan = ? ORDER BY thoi_gian_tao DESC LIMIT 1",
        [ma_benh_nhan],
    )


def get_latest_prediction(db_path, ma_benh_nhan):
    return one(
        db_path,
        "SELECT * FROM ket_qua_du_doan WHERE ma_benh_nhan = ? ORDER BY thoi_gian_du_doan DESC LIMIT 1",
        [ma_benh_nhan],
    )


def get_prediction_scores(db_path, ma_ket_qua):
    return rows(
        db_path,
        "SELECT * FROM chi_tiet_risk_score WHERE ma_ket_qua = ? ORDER BY ten_bien_chung",
        [ma_ket_qua],
    )


def get_alerts(db_path, limit=100):
    return rows(db_path, "SELECT * FROM canh_bao ORDER BY thoi_gian_tao DESC LIMIT ?", [limit])


def get_latest_model_version(db_path):
    rec = one(
        db_path,
        """
        SELECT *
        FROM model_version
        WHERE lower(stage) = 'production'
        ORDER BY ngay_train DESC
        LIMIT 1
        """,
    )
    if rec:
        return rec
    return one(db_path, "SELECT * FROM model_version ORDER BY ngay_train DESC LIMIT 1")


def get_recent_predictions(db_path, limit=20):
    return rows(
        db_path,
        "SELECT * FROM ket_qua_du_doan ORDER BY thoi_gian_du_doan DESC LIMIT ?",
        [limit],
    )


def get_prediction(db_path, ma_ket_qua):
    return one(db_path, "SELECT * FROM ket_qua_du_doan WHERE ma_ket_qua = ?", [ma_ket_qua])
