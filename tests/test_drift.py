import json

import pandas as pd

from src.monitoring.drift import FEATURES
from src.monitoring.drift import TARGETS
from src.monitoring.drift import calc_basic_drift
from src.monitoring.drift import calc_performance_if_labels
from src.monitoring.drift import clean_for_evidently
from src.monitoring.drift import insert_monitoring_log
from src.monitoring.drift import load_current_from_duckdb
from src.monitoring.drift import load_ref
from src.monitoring.drift import main
from src.monitoring.drift import make_current_if_missing
from src.monitoring.drift import make_evidently_report
from src.monitoring.drift import make_summary
from src.monitoring.drift import run_drift_monitoring
from src.monitoring.drift import save_summary


def ref_df(n=20):
    rows = []
    for i in range(n):
        rec = {
            "age": 40 + i % 20,
            "sex": i % 2,
            "bmi": 24 + i * 0.1,
            "sp": 120 + i % 10,
            "bp": 80 + i % 5,
            "hba1c": 7 + i * 0.05,
            "fps": 100 + i,
            "pps": 130 + i,
            "family_h": i % 2,
            "onset_age": 35 + i % 10,
            "dia_life": 1 + i % 5,
            "smoking": i % 3,
            "phy_act": i % 2,
            "med_use": 1 + i % 2,
            "med_adh": 1 + i % 2,
            "nep": i % 2,
            "neu": (i // 2) % 2,
            "ret": (i // 3) % 2,
            "cv": (i // 4) % 2,
            "per_vas": (i // 5) % 2,
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def write_ref(tmp_path):
    fp = tmp_path / "clean_data.csv"
    ref_df().to_csv(fp, index=False)
    return fp


def test_load_ref(tmp_path):
    fp = write_ref(tmp_path)
    df = load_ref(fp)
    assert set(FEATURES) <= set(df.columns)


def test_make_current_if_missing(tmp_path):
    ref = ref_df()
    fp = tmp_path / "current_data.csv"
    cur = make_current_if_missing(ref, fp)
    assert fp.is_file()
    assert len(cur) > 0


def test_clean_for_evidently():
    ref = ref_df()
    cur = ref_df()
    cur.loc[0, "hba1c"] = None
    ref_x, cur_x = clean_for_evidently(ref, cur)
    assert list(ref_x.columns) == FEATURES
    assert list(cur_x.columns) == FEATURES
    assert not cur_x.isna().any().any()


def test_calc_basic_drift():
    ref_x = ref_df()[FEATURES]
    cur_x = ref_x.copy()
    cur_x["hba1c"] = cur_x["hba1c"] + 5
    rec = calc_basic_drift(ref_x, cur_x)
    assert 0 <= rec["drift_share"] <= 1
    assert isinstance(rec["drifted_features"], list)


def test_calc_performance_unknown():
    rec = calc_performance_if_labels(ref_df()[FEATURES])
    assert rec["performance_status"] == "unknown"


def test_calc_performance_available():
    cur = ref_df()
    for tg in TARGETS:
        cur[f"pred_{tg}"] = cur[tg]
    rec = calc_performance_if_labels(cur)
    assert rec["performance_status"] == "available"
    assert rec["recall_macro"] is not None


def test_make_summary_retrain_yes_by_drift():
    ref_x = ref_df()[FEATURES]
    cur_x = ref_x.copy()
    for col in FEATURES:
        cur_x[col] = cur_x[col] + 10
    report = {"ok": True, "path": "report.html", "error": None}
    rec = make_summary(ref_x, cur_x, cur_x, report, drift_threshold=0.1)
    assert rec["retrain_flag"] == "yes"


def test_make_summary_retrain_no():
    ref_x = ref_df()[FEATURES]
    cur_x = ref_x.copy()
    report = {"ok": True, "path": "report.html", "error": None}
    rec = make_summary(ref_x, cur_x, cur_x, report)
    assert rec["retrain_flag"] == "no"


def test_save_summary(tmp_path):
    fp = tmp_path / "summary.json"
    save_summary({"retrain_flag": "no"}, fp)
    data = json.loads(fp.read_text(encoding="utf-8"))
    assert data["retrain_flag"] == "no"


def test_make_evidently_report_no_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref_x = ref_df()[FEATURES]
    cur_x = ref_x.copy()
    rec = make_evidently_report(ref_x, cur_x, tmp_path / "drift_report.html")
    assert rec["ok"] is True or rec["ok"] is False
    assert (tmp_path / "drift_report.html").exists() or (tmp_path / "reports" / "drift_report_skip.txt").exists()


def test_run_drift_monitoring(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref = write_ref(tmp_path)
    summary = tmp_path / "monitoring_summary.json"
    rec = run_drift_monitoring(
        ref_path=ref,
        cur_path=tmp_path / "current_data.csv",
        db_path=tmp_path / "app.duckdb",
        out_path=tmp_path / "drift_report.html",
        summary_path=summary,
        prefer_db=False,
    )
    assert summary.is_file()
    assert "retrain_flag" in rec


def test_load_current_from_duckdb_no_db(tmp_path):
    assert load_current_from_duckdb(tmp_path / "missing.duckdb") is None


def test_insert_monitoring_log_no_crash(tmp_path):
    insert_monitoring_log({"drift_share": 0.1, "retrain_flag": "no"}, tmp_path / "missing.duckdb")


def test_main_import():
    assert main is not None
