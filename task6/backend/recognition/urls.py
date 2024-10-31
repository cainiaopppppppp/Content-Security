from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import VisitorView, ValidateVisitorAPIView, VisitHistoryView

router = DefaultRouter()

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/visitors/', VisitorView.as_view(), name='add-visitor'),
    path('api/validate-visitor/', ValidateVisitorAPIView.as_view(), name='validate-visitor'),
    path('api/history/', VisitHistoryView.as_view(), name='visit-history'),
]
