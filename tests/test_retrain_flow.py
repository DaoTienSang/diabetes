import json
from pathlib import Path

import src.retrain.flow as flow
from src.retrain.flow import check_rules
from src.retrain.flow import load_monitoring_summary
from src.retrain.flow import promote_candidate
from src.retrain.flow import register_candidate_to_db
from src.retrain.flow import register_candidate_to_mlflow
from src.retrain.flow import retrain_flow
from src.retrain.flow import run_retrain_if_needed
from src.retrain.flow import save_json
from src.retrain.flow import save_retrain_summary
from src.retrain.flow import should_retrain


def write_monitoring(path, flag="yes"):
    save_json({"retrain_flag": flag, "reason": "test"}, path)


def write_metrics(path, model="random_forest", f1=0.5, recall=0.35):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "model_name,f1_micro,f1_macro,recall_macro,hamming_loss\n"
        f"{model},{f1},0.4,{recall},0.2\n",
        encoding="utf-8",
    )


def test_should_retrain_yes():
    assert should_retrain({"retrain_flag": "yes"}) is True


def test_should_retrain_no():
    assert should_retrain({"retrain_flag": "no"}) is False


def test_load_monitoring_missing(tmp_path):
    rec = load_monitoring_summary(tmp_path / "missing.json")
    assert rec["retrain_flag"] == "no"


def test_check_rules_first_model_pass():
    rec = check_rules(None, {"f1_micro": 0.5, "recall_macro": 0.4})
    assert rec["pass"] is True


def test_check_rules_first_model_fail_low_recall():
    rec = check_rules(None, {"f1_micro": 0.5, "recall_macro": 0.1})
    assert rec["pass"] is False


def test_check_rules_f1_drop_too_high():
    old = {"f1_micro": 0.6, "recall_macro": 0.4}
    new = {"f1_micro": 0.5, "recall_macro": 0.4}
    rec = check_rules(old, new, max_f1_drop=0.02)
    assert rec["pass"] is False


def test_check_rules_recall_below_threshold():
    old = {"f1_micro": 0.5, "recall_macro": 0.4}
    new = {"f1_micro": 0.5, "recall_macro": 0.1}
    rec = check_rules(old, new)
    assert rec["pass"] is False


def test_check_rules_pass():
    old = {"f1_micro": 0.5, "recall_macro": 0.35}
    new = {"f1_micro": 0.49, "recall_macro": 0.36}
    rec = check_rules(old, new)
    assert rec["pass"] is True


def test_save_retrain_summary(tmp_path):
    fp = tmp_path / "retrain_summary.json"
    save_retrain_summary({"final_status": "no_retrain"}, fp)
    rec = json.loads(fp.read_text(encoding="utf-8"))
    assert rec["final_status"] == "no_retrain"


def test_promote_candidate(tmp_path):
    cand = tmp_path / "candidate_model.pkl"
    prod = tmp_path / "best_model.pkl"
    cand.write_text("candidate", encoding="utf-8")
    prod.write_text("old", encoding="utf-8")
    rec = promote_candidate(cand, prod)
    assert rec["promoted"] is True
    assert prod.read_text(encoding="utf-8") == "candidate"
    assert rec["backup_path"]


def test_promote_candidate_missing(tmp_path):
    rec = promote_candidate(tmp_path / "missing.pkl", tmp_path / "best_model.pkl")
    assert rec["promoted"] is False


def test_no_retrain_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_monitoring("reports/monitoring_summary.json", "no")

    def fail():
        raise AssertionError("train should not run")

    monkeypatch.setattr(flow, "run_training_pipeline", fail)
    rec = run_retrain_if_needed()
    assert rec["final_status"] == "no_retrain"


def test_retrain_train_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_monitoring("reports/monitoring_summary.json", "yes")
    monkeypatch.setattr(flow, "run_training_pipeline", lambda: {"ok": False, "model_path": None, "error": "x"})
    rec = run_retrain_if_needed()
    assert rec["final_status"] == "train_failed"


def test_retrain_rules_fail_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_monitoring("reports/monitoring_summary.json", "yes")
    write_metrics("reports/metrics.csv", f1=0.6, recall=0.4)
    Path("models").mkdir()
    Path("models/best_model.pkl").write_text("old", encoding="utf-8")

    def fake_train():
        Path("models/candidate_model.pkl").write_text("candidate", encoding="utf-8")
        write_metrics("reports/candidate_metrics.csv", f1=0.4, recall=0.2)
        return {
            "ok": True,
            "model_path": "models/candidate_model.pkl",
            "metrics_path": "reports/candidate_metrics.csv",
            "meta_path": "models/candidate_meta.json",
            "metrics": {"model": "candidate", "f1_micro": 0.4, "recall_macro": 0.2},
            "model_name": "candidate",
            "error": None,
        }

    monkeypatch.setattr(flow, "run_training_pipeline", fake_train)
    rec = run_retrain_if_needed()
    assert rec["final_status"] == "rejected"
    assert Path("models/best_model.pkl").read_text(encoding="utf-8") == "old"


def test_retrain_rules_pass_promote(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_monitoring("reports/monitoring_summary.json", "yes")
    write_metrics("reports/metrics.csv", f1=0.5, recall=0.35)
    Path("models").mkdir()
    Path("models/best_model.pkl").write_text("old", encoding="utf-8")

    def fake_train():
        Path("models/candidate_model.pkl").write_text("candidate", encoding="utf-8")
        write_metrics("reports/candidate_metrics.csv", f1=0.5, recall=0.36)
        return {
            "ok": True,
            "model_path": "models/candidate_model.pkl",
            "metrics_path": "reports/candidate_metrics.csv",
            "meta_path": "models/candidate_meta.json",
            "metrics": {"model": "candidate", "f1_micro": 0.5, "recall_macro": 0.36},
            "model_name": "candidate",
            "error": None,
        }

    monkeypatch.setattr(flow, "run_training_pipeline", fake_train)
    monkeypatch.setattr(flow, "register_candidate_to_db", lambda metrics: {"ok": True, "ma_model": "m", "error": None})
    monkeypatch.setattr(flow, "register_candidate_to_mlflow", lambda metrics: {"ok": True, "registered_version": "1"})
    rec = run_retrain_if_needed()
    assert rec["final_status"] == "promoted"
    assert Path("models/best_model.pkl").read_text(encoding="utf-8") == "candidate"


def test_register_candidate_to_db_no_crash(tmp_path):
    rec = register_candidate_to_db({"model": "candidate", "f1_micro": 0.5, "recall_macro": 0.4}, db_path=tmp_path / "app.duckdb")
    assert "ok" in rec


def test_register_candidate_to_mlflow_no_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = register_candidate_to_mlflow({"model": "candidate"})
    assert "ok" in rec


def test_retrain_flow_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_monitoring("reports/monitoring_summary.json", "no")
    rec = retrain_flow()
    assert rec["final_status"] == "no_retrain"


def test_force_retrain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_monitoring("reports/monitoring_summary.json", "no")
    monkeypatch.setattr(flow, "run_training_pipeline", lambda: {"ok": False, "model_path": None, "error": "x"})
    rec = run_retrain_if_needed(force=True)
    assert rec["monitoring_retrain_flag"] == "yes"


def test_summary_contains_required_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_monitoring("reports/monitoring_summary.json", "no")
    rec = run_retrain_if_needed()
    for key in ["created_at", "monitoring_retrain_flag", "old_metrics", "new_metrics", "rule_result", "final_status"]:
        assert key in rec
