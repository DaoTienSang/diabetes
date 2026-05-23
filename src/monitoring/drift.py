import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


FEATURES = [
    "age",
    "sex",
    "bmi",
    "sp",
    "bp",
    "hba1c",
    "fps",
    "pps",
    "family_h",
    "onset_age",
    "dia_life",
    "smoking",
    "phy_act",
    "med_use",
    "med_adh",
]

TARGETS = ["nep", "neu", "ret", "cv", "per_vas"]

DEFAULT_REF = "data/processed/clean_data.csv"
DEFAULT_CUR = "data/processed/current_data.csv"
DEFAULT_REPORT = "reports/drift_report.html"
DEFAULT_SUMMARY = "reports/monitoring_summary.json"
DEFAULT_SKIP = "reports/drift_report_skip.txt"


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_cols(df):
    rec = df.copy()
    cols = {}
    for col in rec.columns:
        key = str(col).strip()
        low = key.lower().replace(" ", "_")
        if low == "med_adherence":
            cols[col] = "med_adh"
        elif low == "family_history":
            cols[col] = "family_h"
        elif low in {"per_vas", "per_vas_"} or key.lower() == "per vas":
            cols[col] = "per_vas"
        else:
            cols[col] = low
    return rec.rename(columns=cols)


def select_cols(df):
    cols = FEATURES + [c for c in TARGETS if c in df.columns]
    return df[cols].copy()


def check_features(df):
    miss = [c for c in FEATURES if c not in df.columns]
    if miss:
        raise ValueError(f"Missing features: {miss}")


def load_ref(path=DEFAULT_REF):
    df = normalize_cols(pd.read_csv(path))
    check_features(df)
    return select_cols(df)


def make_current_if_missing(ref, path=DEFAULT_CUR, n=100):
    fp = Path(path)
    if fp.exists():
        return normalize_cols(pd.read_csv(fp))
    rec = ref.sample(n=min(n, len(ref)), random_state=42).copy()
    if "hba1c" in rec.columns:
        rec["hba1c"] = rec["hba1c"] + 0.2
    if "bmi" in rec.columns:
        rec["bmi"] = rec["bmi"] + 0.1
    if "fps" in rec.columns:
        rec["fps"] = rec["fps"] + 5
    if "pps" in rec.columns:
        rec["pps"] = rec["pps"] + 5
    ensure_dir(fp)
    rec.to_csv(fp, index=False)
    return rec


def load_cur(path=DEFAULT_CUR, ref=None):
    base = ref if ref is not None else load_ref()
    fp = Path(path)
    df = normalize_cols(pd.read_csv(fp)) if fp.exists() else make_current_if_missing(base, path)
    check_features(df)
    return select_cols(df)


def load_current_from_duckdb(db_path="db/app.duckdb"):
    try:
        fp = Path(db_path)
        if not fp.exists():
            return None
        import duckdb

        con = duckdb.connect(str(fp))
        try:
            ok = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'ho_so_lam_sang'"
            ).fetchone()[0]
            if not ok:
                return None
            df = con.execute(
                """
                SELECT
                    bn.tuoi AS age,
                    bn.gioi_tinh AS sex,
                    hs.bmi,
                    hs.sp,
                    hs.bp,
                    hs.hba1c,
                    hs.fps,
                    hs.pps,
                    hs.family_history AS family_h,
                    hs.onset_age,
                    hs.dia_life,
                    hs.smoking,
                    hs.phy_act,
                    hs.med_use,
                    hs.med_adh
                FROM ho_so_lam_sang hs
                LEFT JOIN benh_nhan bn ON bn.ma_benh_nhan = hs.ma_benh_nhan
                ORDER BY hs.thoi_gian_tao DESC
                LIMIT 500
                """
            ).fetchdf()
        finally:
            con.close()
        if df.empty:
            return None
        df = normalize_cols(df)
        check_features(df)
        return df[FEATURES].copy()
    except Exception:
        return None


def get_current_data(ref, cur_path=DEFAULT_CUR, db_path="db/app.duckdb", prefer_db=True):
    if prefer_db:
        df = load_current_from_duckdb(db_path)
        if df is not None:
            return df
    return load_cur(cur_path, ref)


def clean_for_evidently(ref, cur):
    ref_x = normalize_cols(ref.copy())
    cur_x = normalize_cols(cur.copy())
    check_features(ref_x)
    check_features(cur_x)
    ref_x = ref_x[FEATURES].apply(pd.to_numeric, errors="coerce")
    cur_x = cur_x[FEATURES].apply(pd.to_numeric, errors="coerce")
    med = ref_x.median(numeric_only=True)
    ref_x = ref_x.fillna(med).fillna(0)
    cur_x = cur_x.fillna(med).fillna(0)
    return ref_x, cur_x


def write_skip(err, path=DEFAULT_SKIP):
    ensure_dir(path)
    Path(path).write_text(str(err), encoding="utf-8")
    return {"ok": False, "path": path, "error": str(err)}


def make_evidently_report(ref_x, cur_x, out_path=DEFAULT_REPORT):
    ensure_dir(out_path)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        val = report.run(reference_data=ref_x, current_data=cur_x)
        val.save_html(out_path)
        return {"ok": True, "path": out_path, "error": None}
    except Exception as err_a:
        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_x, current_data=cur_x)
            report.save_html(out_path)
            return {"ok": True, "path": out_path, "error": None}
        except Exception as err_b:
            return write_skip(f"{err_a}; {err_b}")


def calc_basic_drift(ref_x, cur_x, threshold=0.1):
    stats = {}
    drifted = []
    for col in FEATURES:
        mean_ref = float(ref_x[col].mean())
        mean_cur = float(cur_x[col].mean())
        std_ref = float(ref_x[col].std())
        diff = abs(mean_cur - mean_ref)
        score = diff / std_ref if std_ref > 0 else 0.0
        if score >= threshold:
            drifted.append(col)
        stats[col] = {
            "mean_ref": mean_ref,
            "mean_cur": mean_cur,
            "std_ref": std_ref,
            "score": float(score),
        }
    total = len(FEATURES)
    return {
        "drift_score": len(drifted),
        "drift_share": len(drifted) / total if total else 0.0,
        "drifted_features": drifted,
        "feature_stats": stats,
    }


def calc_performance_if_labels(cur, pred_path=None):
    pred_cols = [f"pred_{c}" for c in TARGETS]
    if not set(TARGETS) <= set(cur.columns) or not set(pred_cols) <= set(cur.columns):
        return {
            "performance_status": "unknown",
            "recall_macro": None,
            "f1_micro": None,
            "f1_macro": None,
        }
    yt = cur[TARGETS].astype(int)
    yp = cur[pred_cols].astype(int)
    return {
        "performance_status": "available",
        "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def make_summary(ref_x, cur_x, cur_full, report_result, drift_threshold=0.25, recall_threshold=0.3):
    drift = calc_basic_drift(ref_x, cur_x)
    perf = calc_performance_if_labels(cur_full)
    retrain = "no"
    reason = "stable"
    if drift["drift_share"] >= drift_threshold:
        retrain = "yes"
        reason = "drift_high"
    if perf["recall_macro"] is not None and perf["recall_macro"] < recall_threshold:
        retrain = "yes"
        reason = "recall_low"
    if not report_result.get("ok") and reason == "stable":
        reason = "report_skip"
    return {
        "report_ok": bool(report_result.get("ok")),
        "report_path": report_result.get("path"),
        "drift_score": drift["drift_score"],
        "drift_share": drift["drift_share"],
        "drifted_features": drift["drifted_features"],
        "performance_status": perf["performance_status"],
        "recall_macro": perf["recall_macro"],
        "f1_micro": perf["f1_micro"],
        "f1_macro": perf["f1_macro"],
        "drift_threshold": drift_threshold,
        "recall_threshold": recall_threshold,
        "retrain_flag": retrain,
        "reason": reason,
        "created_at": datetime.now().isoformat(),
    }


def save_summary(summary, path=DEFAULT_SUMMARY):
    ensure_dir(path)
    Path(path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def insert_monitoring_log(summary, db_path="db/app.duckdb"):
    try:
        from src.db import repo

        model = repo.get_latest_model_version(db_path)
        ma_model = model.get("ma_model") if model else "unknown_model"
        repo.insert_monitoring_log(
            db_path,
            ma_model,
            summary.get("drift_share", 0),
            summary.get("recall_macro") if summary.get("recall_macro") is not None else -1,
            summary.get("f1_micro") if summary.get("f1_micro") is not None else -1,
            summary.get("retrain_flag", "no"),
        )
    except Exception:
        print("monitoring_db_skip")


def run_drift_monitoring(
    ref_path=DEFAULT_REF,
    cur_path=DEFAULT_CUR,
    db_path="db/app.duckdb",
    out_path=DEFAULT_REPORT,
    summary_path=DEFAULT_SUMMARY,
    prefer_db=True,
):
    print("drift_start")
    ref = load_ref(ref_path)
    cur = get_current_data(ref, cur_path, db_path, prefer_db)
    ref_x, cur_x = clean_for_evidently(ref, cur)
    report = make_evidently_report(ref_x, cur_x, out_path)
    summary = make_summary(ref_x, cur_x, cur, report)
    save_summary(summary, summary_path)
    insert_monitoring_log(summary, db_path)
    print("report_path", summary["report_path"])
    print("drift_share", summary["drift_share"])
    print("retrain_flag", summary["retrain_flag"])
    print("drift_done")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--cur", default=DEFAULT_CUR)
    parser.add_argument("--db", default="db/app.duckdb")
    parser.add_argument("--out", default=DEFAULT_REPORT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--no-db-current", action="store_true")
    args = parser.parse_args()
    run_drift_monitoring(
        ref_path=args.ref,
        cur_path=args.cur,
        db_path=args.db,
        out_path=args.out,
        summary_path=args.summary,
        prefer_db=not args.no_db_current,
    )


if __name__ == "__main__":
    main()
