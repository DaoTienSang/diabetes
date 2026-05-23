from pathlib import Path

import duckdb


TABLES = [
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


def get_conn(db_path):
    fp = Path(db_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(fp))


def init_db(db_path="db/app.duckdb"):
    con = get_conn(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS benh_nhan (
                ma_benh_nhan VARCHAR PRIMARY KEY,
                ten VARCHAR,
                tuoi INTEGER,
                gioi_tinh INTEGER,
                ngay_tao TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ho_so_lam_sang (
                ma_ho_so VARCHAR PRIMARY KEY,
                ma_benh_nhan VARCHAR,
                bmi DOUBLE,
                sp DOUBLE,
                bp DOUBLE,
                hba1c DOUBLE,
                fps DOUBLE,
                pps DOUBLE,
                family_history INTEGER,
                onset_age INTEGER,
                dia_life DOUBLE,
                smoking INTEGER,
                phy_act INTEGER,
                med_use INTEGER,
                med_adh INTEGER,
                thoi_gian_tao TIMESTAMP,
                FOREIGN KEY (ma_benh_nhan) REFERENCES benh_nhan(ma_benh_nhan)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS model_version (
                ma_model VARCHAR PRIMARY KEY,
                model_name VARCHAR,
                version VARCHAR,
                stage VARCHAR,
                f1_micro DOUBLE,
                f1_macro DOUBLE,
                recall_macro DOUBLE,
                hamming_loss DOUBLE,
                ngay_train TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ket_qua_du_doan (
                ma_ket_qua VARCHAR PRIMARY KEY,
                ma_benh_nhan VARCHAR,
                ma_ho_so VARCHAR,
                ma_model VARCHAR,
                muc_nguy_co_tong VARCHAR,
                thoi_gian_du_doan TIMESTAMP,
                FOREIGN KEY (ma_benh_nhan) REFERENCES benh_nhan(ma_benh_nhan),
                FOREIGN KEY (ma_ho_so) REFERENCES ho_so_lam_sang(ma_ho_so),
                FOREIGN KEY (ma_model) REFERENCES model_version(ma_model)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chi_tiet_risk_score (
                ma_chi_tiet VARCHAR PRIMARY KEY,
                ma_ket_qua VARCHAR,
                ten_bien_chung VARCHAR,
                risk_score DOUBLE,
                risk_label INTEGER,
                risk_level VARCHAR,
                FOREIGN KEY (ma_ket_qua) REFERENCES ket_qua_du_doan(ma_ket_qua)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS canh_bao (
                ma_canh_bao VARCHAR PRIMARY KEY,
                ma_benh_nhan VARCHAR,
                ma_ket_qua VARCHAR,
                muc_canh_bao VARCHAR,
                noi_dung VARCHAR,
                trang_thai VARCHAR,
                thoi_gian_tao TIMESTAMP,
                FOREIGN KEY (ma_benh_nhan) REFERENCES benh_nhan(ma_benh_nhan),
                FOREIGN KEY (ma_ket_qua) REFERENCES ket_qua_du_doan(ma_ket_qua)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS request_log (
                ma_log VARCHAR PRIMARY KEY,
                ma_ket_qua VARCHAR,
                endpoint VARCHAR,
                status_code INTEGER,
                latency_ms DOUBLE,
                thoi_gian TIMESTAMP,
                FOREIGN KEY (ma_ket_qua) REFERENCES ket_qua_du_doan(ma_ket_qua)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_log (
                ma_monitor VARCHAR PRIMARY KEY,
                ma_model VARCHAR,
                drift_score DOUBLE,
                recall_score DOUBLE,
                f1_score DOUBLE,
                retrain_flag VARCHAR,
                thoi_gian TIMESTAMP,
                FOREIGN KEY (ma_model) REFERENCES model_version(ma_model)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS du_lieu_huan_luyen (
                id BIGINT,
                age DOUBLE,
                sex INTEGER,
                bmi DOUBLE,
                sp DOUBLE,
                bp DOUBLE,
                hba1c DOUBLE,
                fps DOUBLE,
                pps DOUBLE,
                family_h INTEGER,
                onset_age INTEGER,
                dia_life DOUBLE,
                smoking INTEGER,
                phy_act INTEGER,
                med_use INTEGER,
                med_adh INTEGER,
                nep INTEGER,
                neu INTEGER,
                ret INTEGER,
                cv INTEGER,
                per_vas INTEGER
            )
            """
        )
    finally:
        con.close()


def list_tables(db_path="db/app.duckdb"):
    init_db(db_path)
    con = get_conn(db_path)
    try:
        rows = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def table_columns(db_path, table):
    init_db(db_path)
    con = get_conn(db_path)
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    finally:
        con.close()
    return [
        {
            "name": r[1],
            "type": r[2],
            "notnull": bool(r[3]),
            "default": r[4],
            "pk": bool(r[5]),
        }
        for r in rows
    ]
