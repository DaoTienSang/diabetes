import json
from datetime import datetime
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import get_cols
from src.features.build_features import load_xy
from src.features.build_features import split_data
from src.train.metrics import calc_metrics
from src.train.mlflow_utils import get_tracking_uri_safe
from src.train.mlflow_utils import log_one_model_run


def make_models():
    rows = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputClassifier(
                        LogisticRegression(max_iter=1000, class_weight="balanced")
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    MultiOutputClassifier(
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=42,
                            class_weight="balanced",
                            min_samples_leaf=2,
                        )
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        rows["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    MultiOutputClassifier(
                        XGBClassifier(random_state=42, eval_metric="logloss")
                    ),
                ),
            ]
        )
    except Exception:
        print("xgboost_skip")

    return rows


def train_one(name, mdl, x_tr, y_tr, x_te, y_te):
    mdl.fit(x_tr, y_tr)
    pred = mdl.predict(x_te)
    met = calc_metrics(y_te, pred)
    rec = {"model_name": name}
    rec.update(met)
    return rec, mdl


def choose_best(rows):
    rec = sorted(rows, key=lambda r: (r["f1_micro"], r["recall_macro"]), reverse=True)
    return rec[0]["model_name"]


def save_model(mdl, path):
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl, fp)


def save_meta(ft, tg, best_name, path="models/meta.json", metrics_path="reports/metrics.csv"):
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    ver = datetime.now().strftime("%Y%m%d%H%M%S")
    rec = {
        "features": ft,
        "targets": tg,
        "best_model": best_name,
        "model_version": ver,
        "registered_model_name": "diabetes_complication_model",
        "registered_version": None,
        "tracking_uri": get_tracking_uri_safe(),
        "metrics_path": metrics_path,
    }
    fp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def save_model_version(db_path, best_name, model_version, rows=None):
    try:
        from src.db.schema import init_db

        con = duckdb.connect(str(db_path))
        try:
            ok = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'model_version'"
            ).fetchone()[0]
        finally:
            con.close()

        if not ok:
            init_db(db_path)

        con = duckdb.connect(str(db_path))
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info('model_version')").fetchall()]
            data = {
                "model_version": model_version,
                "version": model_version,
                "model_name": best_name,
                "best_model": best_name,
                "status": "trained",
                "stage": "Production",
            }
            if "ma_model" in cols:
                rec = next((r for r in rows or [] if r["model_name"] == best_name), {})
                data.update(
                    {
                        "ma_model": f"mdl_{model_version}",
                        "f1_micro": rec.get("f1_micro"),
                        "f1_macro": rec.get("f1_macro"),
                        "recall_macro": rec.get("recall_macro"),
                        "hamming_loss": rec.get("hamming_loss"),
                        "ngay_train": datetime.now(),
                    }
                )
            use = [c for c in cols if c in data]
            if not use:
                return
            marks = ", ".join(["?"] * len(use))
            names = ", ".join(use)
            vals = [data[c] for c in use]
            con.execute(f"INSERT INTO model_version ({names}) VALUES ({marks})", vals)
        finally:
            con.close()
    except Exception:
        print("db_model_version_skip")


def model_params(name, ft, tg, source, xgboost_available):
    rec = {
        "model_name": name,
        "target_count": len(tg),
        "feature_count": len(ft),
        "data_source": source,
        "imputer": "median",
        "scaler": "no",
        "class_weight": None,
        "xgboost_available": xgboost_available,
    }
    if name == "logistic_regression":
        rec["scaler"] = "yes"
        rec["class_weight"] = "balanced"
        rec["max_iter"] = 1000
    if name == "random_forest":
        rec["class_weight"] = "balanced"
        rec["n_estimators"] = 300
    return rec


def save_meta_file(meta, path="models/meta.json"):
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def run_train(
    source="duckdb",
    db_path="db/app.duckdb",
    csv_path="data/processed/clean_data.csv",
    output_model_path="models/best_model.pkl",
    output_meta_path="models/meta.json",
    output_metrics_path="reports/metrics.csv",
    register_production=True,
):
    x, y = load_xy(source=source, db_path=db_path, csv_path=csv_path)
    x_tr, x_te, y_tr, y_te = split_data(x, y)
    ft, tg = get_cols()
    rows = []
    trained = {}

    models = make_models()
    xgboost_available = "xgboost" in models

    for name, mdl in models.items():
        try:
            rec, fit = train_one(name, mdl, x_tr, y_tr, x_te, y_te)
            rows.append(rec)
            trained[name] = fit
            print("model_name", name)
            print("f1_micro", rec["f1_micro"])
            print("f1_macro", rec["f1_macro"])
            print("recall_macro", rec["recall_macro"])
        except Exception:
            if name == "xgboost":
                print("xgboost_skip")

    best_name = choose_best(rows)
    meta = save_meta(ft, tg, best_name, output_meta_path, output_metrics_path)
    save_model(trained[best_name], output_model_path)

    fp = Path(output_metrics_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(fp, index=False)

    artifacts = [
        output_metrics_path,
        output_meta_path,
        "reports/target_counts.png",
        "reports/corr.png",
    ]
    for rec in rows:
        name = rec["model_name"]
        metrics = {k: v for k, v in rec.items() if k != "model_name"}
        try:
            info = log_one_model_run(
                name,
                trained[name],
                model_params(name, ft, tg, source, xgboost_available),
                metrics,
                artifacts,
                register=name == best_name and register_production,
            )
            if name == best_name:
                meta["registered_version"] = info.get("registered_version")
        except Exception:
            print("mlflow_skip")

    meta["tracking_uri"] = get_tracking_uri_safe()
    save_meta_file(meta, output_meta_path)
    if register_production:
        save_model_version(db_path, best_name, meta["model_version"], rows)

    print("best_model", best_name)
    print("model_version", meta["model_version"])
    return {"rows": rows, "best_model": best_name, "model_version": meta["model_version"]}


def main():
    run_train()


if __name__ == "__main__":
    main()
