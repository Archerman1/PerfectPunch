from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("training/", views.training, name="training"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("dashboard/", views.dashboard, name="dashboard"),
]