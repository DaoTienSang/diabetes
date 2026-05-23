from django.contrib.auth import authenticate
from django.contrib.auth import login
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.shortcuts import render


def login_view(request):
    if request.user.is_authenticated:
        return redirect("/")
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("/")
        return render(request, "app/login.html", {"error": "Tên đăng nhập hoặc mật khẩu không đúng"})
    return render(request, "app/login.html")


def logout_view(request):
    logout(request)
    return redirect("/login/")
