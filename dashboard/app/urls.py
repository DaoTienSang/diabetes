from django.urls import path

from dashboard.app.auth_views import login_view
from dashboard.app.auth_views import logout_view
from dashboard.app.views import alerts_view
from dashboard.app.views import dashboard_view
from dashboard.app.views import patient_detail_view
from dashboard.app.views import patients_view
from dashboard.app.views import prediction_detail_view


urlpatterns = [
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("", dashboard_view, name="dashboard"),
    path("patients/", patients_view, name="patients"),
    path("patients/<str:ma_benh_nhan>/", patient_detail_view, name="patient_detail"),
    path("alerts/", alerts_view, name="alerts"),
    path("predictions/<str:ma_ket_qua>/", prediction_detail_view, name="prediction_detail"),
]
