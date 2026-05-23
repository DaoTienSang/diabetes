from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        doctor, _ = User.objects.get_or_create(username="doctor")
        doctor.set_password("doctor123")
        doctor.is_staff = False
        doctor.is_superuser = False
        doctor.save()

        admin, _ = User.objects.get_or_create(username="admin")
        admin.set_password("admin123")
        admin.is_staff = True
        admin.is_superuser = True
        admin.save()

        print("demo_users_ready")
