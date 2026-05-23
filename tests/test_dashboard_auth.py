import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.config.settings")

import django

django.setup()

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import Client


def setup_auth():
    call_command("migrate", verbosity=0, interactive=False)
    User.objects.filter(username__in=["doctor", "admin"]).delete()


def test_login_page_status():
    setup_auth()
    res = Client().get("/login/")
    assert res.status_code == 200
    assert "Đăng nhập Dashboard".encode() in res.content


def test_protected_dashboard_redirect():
    setup_auth()
    res = Client().get("/")
    assert res.status_code == 302
    assert res["Location"].startswith("/login/")


def test_protected_patients_redirect():
    setup_auth()
    assert Client().get("/patients/").status_code == 302


def test_protected_alerts_redirect():
    setup_auth()
    assert Client().get("/alerts/").status_code == 302


def test_login_success():
    setup_auth()
    User.objects.create_user(username="doctor", password="doctor123")
    client = Client()
    res = client.post("/login/", {"username": "doctor", "password": "doctor123"})
    assert res.status_code == 302
    assert res["Location"] == "/"
    assert client.get("/").status_code == 200


def test_login_fail():
    setup_auth()
    User.objects.create_user(username="doctor", password="doctor123")
    res = Client().post("/login/", {"username": "doctor", "password": "bad"})
    assert res.status_code == 200
    assert "không đúng".encode() in res.content


def test_logout():
    setup_auth()
    User.objects.create_user(username="doctor", password="doctor123")
    client = Client()
    assert client.login(username="doctor", password="doctor123")
    assert client.get("/logout/").status_code == 302
    assert client.get("/").status_code == 302


def test_no_register_route():
    setup_auth()
    assert Client().get("/register/").status_code == 404


def test_login_template_no_register_text():
    setup_auth()
    text = Client().get("/login/").content.decode("utf-8").lower()
    assert "đăng ký" not in text
    assert "register" not in text


def test_create_demo_users_command():
    setup_auth()
    call_command("create_demo_users")
    call_command("create_demo_users")
    assert User.objects.filter(username="doctor").count() == 1
    assert User.objects.filter(username="admin").count() == 1
    assert User.objects.get(username="admin").is_staff is True
