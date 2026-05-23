import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_MONITORING_SUMMARY = "reports/monitoring_summary.json"
DEFAULT_OLD_METRICS = "reports/metrics.csv"
DEFAULT_PROD_MODEL = "models/best_model.pkl"
DEFAULT_PROD_META = "models/meta.json"
DEFAULT_CANDIDATE_MODEL = "models/candidate_model.pkl"
DEFAULT_CANDIDATE_META = "models/candidate_meta.json"
DEFAULT_CANDIDATE_METRICS = "reports/candidate_metrics.csv"
DEFAULT_RETRAIN_SUMMARY = "reports/retrain_summary.json"

TARGETS = ["nep", "neu", "ret", "cv", "per_vas"]


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    try:
        fp = Path(path)
        if not fp.exists():
            return None
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(data, path):
    ensure_dir(path)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_monitoring_summary(path=DEFAULT_MONITORING_SUMMARY):
    rec = load_json(path) or {}
    rec.setdefault("retrain_flag", "no")
    rec.setdefault("reason", "monitoring_summary_missing" if not Path(path).exists() else "unknown")
    return rec


def should_retrain(summary):
    if not summary:
        return False
    flag = summary.get("retrain_flag")
    return flag is True or str(flag).lower() == "yes"


def metric_float(rec, key):
    try:
        val = rec.get(key)
        return None if pd.isna(val) else float(val)
    except Exception:
        return None


def load_prod_meta(path=DEFAULT_PROD_META):
    return load_json(path)


def load_old_metrics(path=DEFAULT_OLD_METRICS):
    fp = Path(path)
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
        if df.empty:
            return None
        meta = load_prod_meta()
        best = meta.get("best_model") if meta else None
        if best and "model_name" in df.columns and best in set(df["model_name"]):
            rec = df[df["model_name"] == best].iloc[0].to_dict()
        else:
            sort_cols = [c for c in ["f1_micro", "recall_macro"] if c in df.columns]
            rec = df.sort_values(sort_cols, ascending=False).iloc[0].to_dict() if sort_cols else df.iloc[0].to_dict()
        return {
            "model": rec.get("model_name"),
            "f1_micro": metric_float(rec, "f1_micro"),
            "f1_macro": metric_float(rec, "f1_macro"),
            "recall_macro": metric_float(rec, "recall_macro"),
            "hamming_loss": metric_float(rec, "hamming_loss"),
        }
    except Exception:
        return None


def copy_file(src, dst):
    try:
        if not Path(src).exists():
            return False
        ensure_dir(dst)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def backup_production_model(prod_path=DEFAULT_PROD_MODEL):
    fp = Path(prod_path)
    if not fp.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    dst = fp.parent / f"backup_best_model_{ts}.pkl"
    return str(dst) if copy_file(fp, dst) else None


def load_candidate_metrics(path=DEFAULT_CANDIDATE_METRICS):
    fp = Path(path)
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
        if df.empty:
            return None
        df = df.sort_values(["f1_micro", "recall_macro"], ascending=False)
        rec = df.iloc[0].to_dict()
        return {
            "model": rec.get("model_name"),
            "f1_micro": metric_float(rec, "f1_micro"),
            "f1_macro": metric_float(rec, "f1_macro"),
            "recall_macro": metric_float(rec, "recall_macro"),
            "hamming_loss": metric_float(rec, "hamming_loss"),
        }
    except Exception:
        return None


def run_training_pipeline():
    try:
        from src.train.train_models import run_train

        rec = run_train(
            output_model_path=DEFAULT_CANDIDATE_MODEL,
            output_meta_path=DEFAULT_CANDIDATE_META,
            output_metrics_path=DEFAULT_CANDIDATE_METRICS,
            register_production=False,
        )
        met = load_candidate_metrics(DEFAULT_CANDIDATE_METRICS)
        return {
            "ok": True,
            "model_path": DEFAULT_CANDIDATE_MODEL,
            "metrics_path": DEFAULT_CANDIDATE_METRICS,
            "meta_path": DEFAULT_CANDIDATE_META,
            "metrics": met,
            "model_name": rec.get("best_model"),
            "error": None,
        }
    except Exception as err:
        return {
            "ok": False,
            "model_path": DEFAULT_CANDIDATE_MODEL,
            "metrics_path": DEFAULT_CANDIDATE_METRICS,
            "meta_path": DEFAULT_CANDIDATE_META,
            "metrics": None,
            "model_name": None,
            "error": str(err),
        }


def num(rec, key):
    if not rec:
        return None
    val = rec.get(key)
    return None if val is None else float(val)


def check_rules(old_metrics, new_metrics, min_recall_macro=0.30, min_f1_micro=0.30, max_f1_drop=0.02):
    old_f1 = num(old_metrics, "f1_micro")
    new_f1 = num(new_metrics, "f1_micro")
    old_recall = num(old_metrics, "recall_macro")
    new_recall = num(new_metrics, "recall_macro")
    rec = {
        "pass": False,
        "reason": "candidate_metrics_missing",
        "old_f1_micro": old_f1,
        "new_f1_micro": new_f1,
        "old_recall_macro": old_recall,
        "new_recall_macro": new_recall,
        "min_recall_macro": min_recall_macro,
        "min_f1_micro": min_f1_micro,
        "max_f1_drop": max_f1_drop,
    }
    if new_metrics is None or new_f1 is None or new_recall is None:
        return rec
    if old_metrics is None:
        if new_recall < min_recall_macro:
            rec["reason"] = "first_model_fail_low_recall"
            return rec
        if new_f1 < min_f1_micro:
            rec["reason"] = "first_model_fail_low_f1"
            return rec
        rec["pass"] = True
        rec["reason"] = "first_model_pass"
        return rec
    if old_f1 is None or old_recall is None:
        rec["reason"] = "missing_metric"
        return rec
    if new_f1 < old_f1 - max_f1_drop:
        rec["reason"] = "f1_drop_too_high"
        return rec
    if new_recall < min_recall_macro:
        rec["reason"] = "recall_below_threshold"
        return rec
    if new_recall < old_recall - 0.05:
        rec["reason"] = "recall_drop_too_high"
        return rec
    rec["pass"] = True
    rec["reason"] = "rules_pass"
    return rec


def save_candidate(model_path=DEFAULT_CANDIDATE_MODEL):
    return Path(model_path).exists()


def promote_candidate(
    candidate_path=DEFAULT_CANDIDATE_MODEL,
    prod_path=DEFAULT_PROD_MODEL,
    candidate_meta=DEFAULT_CANDIDATE_META,
    prod_meta=DEFAULT_PROD_META,
    candidate_metrics=DEFAULT_CANDIDATE_METRICS,
    prod_metrics=DEFAULT_OLD_METRICS,
):
    if not Path(candidate_path).exists():
        return {"promoted": False, "backup_path": None, "reason": "candidate_missing"}
    backup = backup_production_model(prod_path)
    ok = copy_file(candidate_path, prod_path)
    if Path(candidate_meta).exists():
        ok = copy_file(candidate_meta, prod_meta) and ok
    if Path(candidate_metrics).exists():
        ok = copy_file(candidate_metrics, prod_metrics) and ok
    return {
        "promoted": bool(ok),
        "backup_path": backup,
        "reason": "promoted" if ok else "copy_failed",
    }


def register_candidate_to_db(metrics, stage="Production", db_path="db/app.duckdb"):
    try:
        from src.db.repo import insert_model_version

        ma = insert_model_version(
            db_path,
            {
                "model_name": metrics.get("model", "candidate_model") if metrics else "candidate_model",
                "version": "v_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                "stage": stage,
                "f1_micro": num(metrics, "f1_micro"),
                "f1_macro": num(metrics, "f1_macro"),
                "recall_macro": num(metrics, "recall_macro"),
                "hamming_loss": num(metrics, "hamming_loss"),
            },
        )
        return {"ok": True, "ma_model": ma, "error": None}
    except Exception as err:
        print("db_register_skip")
        return {"ok": False, "ma_model": None, "error": str(err)}


def register_candidate_to_mlflow(metrics, model_path=DEFAULT_CANDIDATE_MODEL):
    try:
        import joblib
        from src.train.mlflow_utils import log_one_model_run

        if not Path(model_path).exists():
            return {"ok": False, "registered_version": None}
        mdl = joblib.load(model_path)
        met = {k: v for k, v in (metrics or {}).items() if k != "model" and v is not None}
        rec = log_one_model_run(
            "retrain_candidate",
            mdl,
            {"model_name": metrics.get("model", "candidate_model") if metrics else "candidate_model"},
            met,
            [DEFAULT_CANDIDATE_METRICS, DEFAULT_CANDIDATE_META],
            register=True,
        )
        return {"ok": bool(rec.get("logged")), "registered_version": rec.get("registered_version")}
    except Exception:
        print("mlflow_retrain_skip")
        return {"ok": False, "registered_version": None}


def make_retrain_summary(
    monitoring_summary,
    old_metrics,
    new_metrics,
    rule_result,
    train_result,
    promote_result=None,
    db_result=None,
    mlflow_result=None,
):
    promoted = bool((promote_result or {}).get("promoted"))
    if not should_retrain(monitoring_summary):
        status = "no_retrain"
    elif not train_result.get("ok"):
        status = "train_failed"
    elif rule_result and not rule_result.get("pass"):
        status = "rejected"
    elif rule_result and rule_result.get("pass") and promoted:
        status = "promoted"
    else:
        status = "promote_failed"
    return {
        "created_at": datetime.now().isoformat(),
        "monitoring_retrain_flag": (monitoring_summary or {}).get("retrain_flag"),
        "monitoring_reason": (monitoring_summary or {}).get("reason"),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "rule_result": rule_result,
        "train_ok": bool(train_result.get("ok")),
        "candidate_model_path": train_result.get("model_path"),
        "promoted": promoted,
        "promote_reason": (promote_result or {}).get("reason"),
        "db_registered": bool((db_result or {}).get("ok")),
        "mlflow_registered": bool((mlflow_result or {}).get("registered_version")),
        "final_status": status,
    }


def save_retrain_summary(summary, path=DEFAULT_RETRAIN_SUMMARY):
    return save_json(summary, path)


def restore_backup(backup, prod_path=DEFAULT_PROD_MODEL):
    if backup:
        copy_file(backup, prod_path)


def run_retrain_if_needed(monitoring_path=DEFAULT_MONITORING_SUMMARY, force=False):
    monitoring = load_monitoring_summary(monitoring_path)
    if force:
        monitoring["retrain_flag"] = "yes"
        monitoring["reason"] = "force"
    if not should_retrain(monitoring):
        summary = make_retrain_summary(monitoring, None, None, None, {"ok": False, "model_path": None})
        save_retrain_summary(summary)
        print("no_retrain")
        return summary

    print("retrain_start")
    old = load_old_metrics()
    backup = backup_production_model()
    train = run_training_pipeline()
    if not train.get("ok"):
        restore_backup(backup)
        summary = make_retrain_summary(monitoring, old, None, None, train)
        save_retrain_summary(summary)
        return summary
    new = load_candidate_metrics()
    rule = check_rules(old, new)
    if not rule["pass"]:
        restore_backup(backup)
        summary = make_retrain_summary(monitoring, old, new, rule, train)
        save_retrain_summary(summary)
        print("promote_no")
        return summary
    promote = promote_candidate()
    db_res = register_candidate_to_db(new) if promote.get("promoted") else {"ok": False}
    ml_res = register_candidate_to_mlflow(new) if promote.get("promoted") else {"ok": False}
    summary = make_retrain_summary(monitoring, old, new, rule, train, promote, db_res, ml_res)
    save_retrain_summary(summary)
    print("promote_yes" if promote.get("promoted") else "promote_no")
    return summary


try:
    from prefect import flow, task

    @task
    def load_monitoring_task(path=DEFAULT_MONITORING_SUMMARY):
        return load_monitoring_summary(path)

    @task
    def train_candidate_task():
        return run_training_pipeline()

    @task
    def check_rules_task(old_metrics, new_metrics):
        return check_rules(old_metrics, new_metrics)

    @task
    def promote_task():
        return promote_candidate()

    @task
    def save_summary_task(summary):
        return save_retrain_summary(summary)

    @flow(name="diabetes-retrain-flow")
    def retrain_flow(monitoring_path=DEFAULT_MONITORING_SUMMARY, force=False):
        return run_retrain_if_needed(monitoring_path, force)

except Exception:
    def load_monitoring_task(path=DEFAULT_MONITORING_SUMMARY):
        return load_monitoring_summary(path)

    def train_candidate_task():
        return run_training_pipeline()

    def check_rules_task(old_metrics, new_metrics):
        return check_rules(old_metrics, new_metrics)

    def promote_task():
        return promote_candidate()

    def save_summary_task(summary):
        return save_retrain_summary(summary)

    def retrain_flow(monitoring_path=DEFAULT_MONITORING_SUMMARY, force=False):
        return run_retrain_if_needed(monitoring_path, force)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitoring", default=DEFAULT_MONITORING_SUMMARY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = retrain_flow(args.monitoring, args.force)
    print("final_status", summary.get("final_status"))
    print("retrain_done")


if __name__ == "__main__":
    main()
