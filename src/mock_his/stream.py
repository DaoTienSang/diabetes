import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


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

DEFAULT_URL = "http://localhost:8000/predict"


def load_rows(path="data/processed/clean_data.csv", limit=10):
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Clean dataset not found: {fp}")
    df = pd.read_csv(fp)
    if limit is None or limit <= 0:
        return df
    return df.head(limit).copy()


def safe_val(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def row_to_payload(row, idx):
    rec = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    payload = {
        "patient_id": f"BN_{idx}",
        "name": "" if "name" not in rec or pd.isna(rec.get("name")) else str(rec.get("name")),
    }
    for col in FEATURES:
        if col not in rec:
            raise ValueError(f"Missing feature: {col}")
        payload[col] = safe_val(rec[col])
    return payload


def send_one(url, payload, timeout=10):
    try:
        res = requests.post(url, json=payload, timeout=timeout)
        try:
            data = res.json()
        except ValueError:
            return {
                "ok": False,
                "status_code": res.status_code,
                "data": None,
                "error": "invalid_json",
            }
        return {
            "ok": 200 <= res.status_code < 300,
            "status_code": res.status_code,
            "data": data,
            "error": None if 200 <= res.status_code < 300 else "api_error",
        }
    except requests.exceptions.RequestException as err:
        return {
            "ok": False,
            "status_code": getattr(getattr(err, "response", None), "status_code", None),
            "data": None,
            "error": "api_not_available",
        }


def short_level(data):
    if not data:
        return "unknown"
    return data.get("warning_level", "unknown")


def run_stream(data_path="data/processed/clean_data.csv", url=DEFAULT_URL, limit=10, delay=1.0):
    df = load_rows(data_path, limit)
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print("send", i)
        try:
            payload = row_to_payload(row, i)
            rec = send_one(url, payload)
        except Exception as err:
            rec = {"ok": False, "status_code": None, "data": None, "error": str(err)}
        rows.append(rec)
        if rec["ok"]:
            print("status", rec["status_code"])
            print("level", short_level(rec["data"]))
        else:
            print("status", "fail")
            print("error", rec["error"])
        if delay > 0:
            time.sleep(delay)
    return rows


def verify_db_log(db_path="db/app.duckdb"):
    try:
        if not Path(db_path).exists():
            return {"ok": False, "reason": "db_not_ready"}
        from src.db.repo import get_counts

        return {"ok": True, "counts": get_counts(db_path)}
    except Exception:
        return {"ok": False, "reason": "db_not_ready"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/clean_data.csv")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--delay", type=float, default=1)
    args = parser.parse_args()
    run_stream(args.data, args.url, args.limit, args.delay)
    print("stream_done")


if __name__ == "__main__":
    main()
