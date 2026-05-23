from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from dashboard.app.services import get_db_path
from dashboard.app.services import safe_alerts
from dashboard.app.services import safe_counts
from dashboard.app.services import safe_latest_model
from dashboard.app.services import safe_patient_detail
from dashboard.app.services import safe_patients
from dashboard.app.services import safe_prediction_detail
from dashboard.app.services import safe_recent_predictions


TARGET_NAMES = {
    "nep": "biến chứng thận",
    "neu": "biến chứng thần kinh",
    "ret": "biến chứng võng mạc",
    "cv": "biến chứng tim mạch",
    "per_vas": "biến chứng mạch máu ngoại biên",
}


@login_required(login_url="/login/")
def dashboard_view(request):
    db_path = get_db_path()
    return render(
        request,
        "app/dashboard.html",
        {
            "counts": safe_counts(db_path),
            "latest_model": safe_latest_model(db_path),
            "recent_predictions": safe_recent_predictions(db_path),
            "mlflow_url": "http://localhost:5000",
            "prometheus_url": "http://localhost:9090",
            "grafana_url": "http://localhost:3000",
        },
    )


@login_required(login_url="/login/")
def patients_view(request):
    db_path = get_db_path()
    q = request.GET.get("q", "").strip()
    risk = request.GET.get("risk", "").strip()
    patients = safe_patients(db_path)
    if q:
        low = q.lower()
        patients = [
            p for p in patients
            if low in str(p.get("ma_benh_nhan", "")).lower() or low in str(p.get("ten", "")).lower()
        ]
    return render(request, "app/patients.html", {"patients": patients, "q": q, "risk": risk})


@login_required(login_url="/login/")
def patient_detail_view(request, ma_benh_nhan):
    db_path = get_db_path()
    rec = safe_patient_detail(db_path, ma_benh_nhan)
    msg = None if rec["patient"] else "Không tìm thấy bệnh nhân"
    rec["message"] = msg
    rec["target_names"] = TARGET_NAMES
    return render(request, "app/patient_detail.html", rec)


@login_required(login_url="/login/")
def alerts_view(request):
    db_path = get_db_path()
    level = request.GET.get("level", "").strip()
    alerts = safe_alerts(db_path)
    if level:
        alerts = [r for r in alerts if r.get("muc_canh_bao") == level]
    return render(request, "app/alerts.html", {"alerts": alerts, "level": level})


@login_required(login_url="/login/")
def prediction_detail_view(request, ma_ket_qua):
    db_path = get_db_path()
    rec = safe_prediction_detail(db_path, ma_ket_qua)
    rec["message"] = None if rec["prediction"] else "Không tìm thấy dự đoán"
    rec["target_names"] = TARGET_NAMES
    return render(request, "app/prediction_detail.html", rec)
