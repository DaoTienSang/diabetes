from src.db.repo import get_alerts
from src.db.repo import get_counts
from src.db.repo import get_high_risk_alerts
from src.db.repo import get_latest_clinical_record
from src.db.repo import get_latest_prediction
from src.db.repo import get_patient
from src.db.repo import get_patient_predictions
from src.db.repo import get_prediction_scores
from src.db.repo import insert_alert
from src.db.repo import insert_clinical_record
from src.db.repo import insert_model_version
from src.db.repo import insert_monitoring_log
from src.db.repo import insert_patient
from src.db.repo import insert_prediction
from src.db.repo import insert_request_log
from src.db.repo import insert_risk_scores
from src.db.repo import list_patients
from src.db.schema import init_db
from src.db.schema import list_tables
from src.db.schema import table_columns


REQ_TABLES = {
    "benh_nhan",
    "ho_so_lam_sang",
    "model_version",
    "ket_qua_du_doan",
    "chi_tiet_risk_score",
    "canh_bao",
    "request_log",
    "monitoring_log",
    "du_lieu_huan_luyen",
}


def db_path(tmp_path):
    return tmp_path / "app.duckdb"


def patient():
    return {"ma_benh_nhan": "bn1", "ten": "A", "tuoi": 50, "gioi_tinh": 1}


def clinical():
    return {
        "ma_ho_so": "hs1",
        "bmi": 25.0,
        "sp": 120,
        "bp": 80,
        "hba1c": 7.1,
        "fps": 100,
        "pps": 130,
        "family_history": 0,
        "onset_age": 40,
        "dia_life": 2.0,
        "smoking": 0,
        "phy_act": 1,
        "med_use": 1,
        "med_adh": 2,
    }


def model():
    return {
        "ma_model": "mdl1",
        "model_name": "random_forest",
        "version": "v1",
        "stage": "production",
        "f1_micro": 0.7,
        "f1_macro": 0.6,
        "recall_macro": 0.5,
        "hamming_loss": 0.2,
    }


def seed(db):
    ma_bn = insert_patient(db, patient())
    ma_hs = insert_clinical_record(db, ma_bn, clinical())
    ma_mdl = insert_model_version(db, model())
    ma_kq = insert_prediction(db, ma_bn, ma_hs, ma_mdl, "high")
    insert_risk_scores(
        db,
        ma_kq,
        {"nep": 0.1, "neu": 0.2, "ret": 0.3, "cv": 0.4, "per_vas": 0.5},
        {"nep": 0, "neu": 0, "ret": 0, "cv": 1, "per_vas": 1},
        {"nep": "low", "neu": "low", "ret": "medium", "cv": "high", "per_vas": "high"},
    )
    insert_alert(db, ma_bn, ma_kq, "high", "risk")
    insert_request_log(db, ma_kq, "/predict", 200, 10.5)
    insert_monitoring_log(db, ma_mdl, 0.1, 0.7, 0.8, "no")
    return ma_bn, ma_hs, ma_mdl, ma_kq


def test_init_db_creates_9_tables(tmp_path):
    db = db_path(tmp_path)
    init_db(db)
    assert REQ_TABLES <= set(list_tables(db))


def test_list_tables_returns_all_tables(tmp_path):
    db = db_path(tmp_path)
    assert REQ_TABLES <= set(list_tables(db))


def test_table_columns_have_pk_and_fk_columns(tmp_path):
    db = db_path(tmp_path)
    cols = table_columns(db, "benh_nhan")
    rec = {c["name"]: c for c in cols}
    assert rec["ma_benh_nhan"]["pk"]
    hs_cols = {c["name"] for c in table_columns(db, "ho_so_lam_sang")}
    kq_cols = {c["name"] for c in table_columns(db, "ket_qua_du_doan")}
    assert "ma_benh_nhan" in hs_cols
    assert {"ma_benh_nhan", "ma_ho_so", "ma_model"} <= kq_cols


def test_insert_flow_and_counts(tmp_path):
    db = db_path(tmp_path)
    seed(db)
    rec = get_counts(db)
    assert rec["benh_nhan"] == 1
    assert rec["ho_so_lam_sang"] == 1
    assert rec["model_version"] == 1
    assert rec["ket_qua_du_doan"] == 1
    assert rec["chi_tiet_risk_score"] == 5
    assert rec["canh_bao"] == 1
    assert rec["request_log"] == 1
    assert rec["monitoring_log"] == 1


def test_get_high_risk_alerts(tmp_path):
    db = db_path(tmp_path)
    seed(db)
    rec = get_high_risk_alerts(db)
    assert len(rec) == 1
    assert rec[0]["muc_canh_bao"] == "high"


def test_get_patient_predictions(tmp_path):
    db = db_path(tmp_path)
    ma_bn, _, _, _ = seed(db)
    rec = get_patient_predictions(db, ma_bn)
    assert len(rec) == 1


def test_list_patients_and_get_patient(tmp_path):
    db = db_path(tmp_path)
    ma_bn, _, _, _ = seed(db)
    assert len(list_patients(db)) == 1
    assert get_patient(db, ma_bn)["ma_benh_nhan"] == ma_bn


def test_latest_clinical_record_and_prediction(tmp_path):
    db = db_path(tmp_path)
    ma_bn, ma_hs, _, ma_kq = seed(db)
    assert get_latest_clinical_record(db, ma_bn)["ma_ho_so"] == ma_hs
    assert get_latest_prediction(db, ma_bn)["ma_ket_qua"] == ma_kq


def test_prediction_scores_and_alerts(tmp_path):
    db = db_path(tmp_path)
    _, _, _, ma_kq = seed(db)
    assert len(get_prediction_scores(db, ma_kq)) == 5
    assert len(get_alerts(db)) == 1


def test_init_db_twice_no_error(tmp_path):
    db = db_path(tmp_path)
    init_db(db)
    init_db(db)
    assert REQ_TABLES <= set(list_tables(db))
