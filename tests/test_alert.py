from src.predict.alert import get_level
from src.predict.alert import get_targets_by_level
from src.predict.alert import make_alert
from src.predict.alert import make_labels
from src.predict.alert import make_levels
from src.predict.alert import overall_level
from src.predict.alert import print_alert


def test_get_level_low():
    assert get_level(0.2) == "low"


def test_get_level_medium():
    assert get_level(0.5) == "medium"


def test_get_level_high():
    assert get_level(0.8) == "high"


def test_get_level_invalid():
    assert get_level(None) == "low"
    assert get_level("abc") == "low"


def test_make_levels_keys():
    rec = make_levels({"nep": 0.2, "neu": 0.8})
    assert set(rec) == {"nep", "neu", "ret", "cv", "per_vas"}


def test_make_labels_keys():
    rec = make_labels({"nep": 0.2, "neu": 0.8})
    assert set(rec) == {"nep", "neu", "ret", "cv", "per_vas"}


def test_make_labels_threshold():
    rec = make_labels({"nep": 0.49, "neu": 0.5}, threshold=0.5)
    assert rec["nep"] == 0
    assert rec["neu"] == 1


def test_overall_level_high():
    rec = {"nep": "low", "neu": "high", "ret": "low", "cv": "low", "per_vas": "low"}
    assert overall_level(rec) == "high"


def test_overall_level_medium():
    rec = {"nep": "low", "neu": "medium", "ret": "low", "cv": "low", "per_vas": "low"}
    assert overall_level(rec) == "medium"


def test_overall_level_low():
    rec = {"nep": "low", "neu": "low", "ret": "low", "cv": "low", "per_vas": "low"}
    assert overall_level(rec) == "low"


def test_get_targets_by_level():
    rec = {"nep": "high", "neu": "medium", "ret": "low", "cv": "high", "per_vas": "low"}
    assert get_targets_by_level(rec, "high") == ["nep", "cv"]


def test_make_alert_high():
    rec = make_alert({"nep": 0.8, "neu": 0.2, "ret": 0.3, "cv": 0.75, "per_vas": 0.1})
    assert rec["level"] == "high"
    assert "biến chứng thận" in rec["message"]
    assert rec["high_targets"]


def test_make_alert_medium():
    rec = make_alert({"nep": 0.2, "neu": 0.5, "ret": 0.3, "cv": 0.1, "per_vas": 0.0})
    assert rec["level"] == "medium"
    assert rec["medium_targets"]


def test_make_alert_low():
    rec = make_alert({"nep": 0.1, "neu": 0.2, "ret": 0.1, "cv": 0.3, "per_vas": 0.0})
    assert rec["level"] == "low"
    assert rec["high_targets"] == []
    assert rec["medium_targets"] == []


def test_make_alert_output_schema():
    rec = make_alert({"nep": 0.1})
    assert set(rec) == {"level", "message", "high_targets", "medium_targets", "risk_levels"}


def test_print_alert():
    rec = make_alert({"nep": 0.1})
    print_alert(rec)
