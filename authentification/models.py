from django.db import models
from django.contrib.auth.models import AbstractUser

class Utilisateur(AbstractUser):
    photo = models.ImageField()
    
class FilesUpload(models.Model):
    userid = models.IntegerField()
    file = models.FileField()

class MetricasHeatmap(models.Model): 
    model = models.CharField(max_length=100, null=False) 
    accuracy = models.FloatField() 
    precision = models.FloatField()
    recall = models.FloatField()
    f1 = models.FloatField()
    roc_auc = models.FloatField()
    time = models.FloatField()
    
    
