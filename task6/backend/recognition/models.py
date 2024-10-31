from django.db import models


class Visitor(models.Model):
    student_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=255)
    face_image = models.ImageField(upload_to='visitor_images/', unique=True)
    face_features = models.BinaryField(null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.student_id})"


class VisitHistory(models.Model):
    visitor = models.ForeignKey(Visitor, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Visit by {self.visitor.name} on {self.timestamp}"
