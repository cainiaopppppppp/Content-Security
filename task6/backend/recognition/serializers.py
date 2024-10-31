from rest_framework import serializers
from .models import Visitor, VisitHistory

class VisitorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Visitor
        fields = ['student_id', 'name', 'face_image']

class VisitHistorySerializer(serializers.ModelSerializer):
    visitor_name = serializers.CharField(source='visitor.name', read_only=True)
    student_id = serializers.CharField(source='visitor.student_id', read_only=True)

    class Meta:
        model = VisitHistory
        fields = ['student_id', 'visitor_name', 'timestamp']
