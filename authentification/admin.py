from django.contrib import admin
from authentification.models import Utilisateur, FilesUpload, MetricasHeatmap

# Afficher toutes les colonnes
class colonnesTableUtilisateur(admin.ModelAdmin):
    list_display = [field.name for field in Utilisateur._meta.fields]

# J'attache mon model Utilisateur à la page d'administration
# pour pouvoir Create-Read-Update-Delete dessus
admin.site.register(Utilisateur, colonnesTableUtilisateur)


# Afficher toutes les colonnes
class colonnesTableFilesUpload(admin.ModelAdmin):
    list_display = [field.name for field in FilesUpload._meta.fields]

admin.site.register(FilesUpload, colonnesTableFilesUpload)


# Afficher toutes les colonnes
class colonnesTableMetricasHeatmap(admin.ModelAdmin):
    list_display = [field.name for field in MetricasHeatmap._meta.fields]

admin.site.register(MetricasHeatmap, colonnesTableMetricasHeatmap)
