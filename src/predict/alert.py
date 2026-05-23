TARGETS = ["nep", "neu", "ret", "cv", "per_vas"]

TARGET_NAMES = {
    "nep": "biến chứng thận",
    "neu": "biến chứng thần kinh",
    "ret": "biến chứng võng mạc",
    "cv": "biến chứng tim mạch",
    "per_vas": "biến chứng mạch máu ngoại biên",
}


def score_float(score):
    try:
        if score is None:
            return 0.0
        return float(score)
    except Exception:
        return 0.0


def get_level(score):
    val = score_float(score)
    if val < 0.4:
        return "low"
    if val < 0.7:
        return "medium"
    return "high"


def make_levels(risk_scores):
    return {tg: get_level(risk_scores.get(tg, 0.0)) for tg in TARGETS}


def make_labels(risk_scores, threshold=0.5):
    return {tg: int(score_float(risk_scores.get(tg, 0.0)) >= threshold) for tg in TARGETS}


def overall_level(risk_levels):
    vals = [risk_levels.get(tg, "low") for tg in TARGETS]
    if "high" in vals:
        return "high"
    if "medium" in vals:
        return "medium"
    return "low"


def get_targets_by_level(risk_levels, level):
    return [tg for tg in TARGETS if risk_levels.get(tg, "low") == level]


def target_text(targets):
    return ", ".join(TARGET_NAMES[tg] for tg in targets)


def make_message(level, high_targets, medium_targets):
    if level == "high":
        names = target_text(high_targets)
        return f"Bệnh nhân có nguy cơ cao về {names}. Cần ưu tiên theo dõi và kiểm tra lâm sàng."
    if level == "medium":
        names = target_text(medium_targets)
        return f"Bệnh nhân có nguy cơ trung bình về {names}. Cần tiếp tục theo dõi các chỉ số lâm sàng."
    return "Nguy cơ biến chứng hiện tại ở mức thấp. Tiếp tục theo dõi định kỳ."


def make_alert(risk_scores, risk_levels=None):
    levels = risk_levels if risk_levels is not None else make_levels(risk_scores)
    levels = {tg: levels.get(tg, "low") for tg in TARGETS}
    level = overall_level(levels)
    high_targets = get_targets_by_level(levels, "high")
    medium_targets = get_targets_by_level(levels, "medium")
    return {
        "level": level,
        "message": make_message(level, high_targets, medium_targets),
        "high_targets": high_targets,
        "medium_targets": medium_targets,
        "risk_levels": levels,
    }


def print_alert(alert):
    print("level:", alert["level"])
    print("message:", alert["message"])
    print("high_targets:", alert["high_targets"])
    print("medium_targets:", alert["medium_targets"])
