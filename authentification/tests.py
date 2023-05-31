from django.test import TestCase
from rest_framework.authtoken.models import Token
from authentification.models import Utilisateur
from django.urls import reverse
from rest_framework.test import APIClient

# Create your tests here.

class MonTest(TestCase):
    def test_somme(self):
        resultat = 2 + 2
        self.assertEqual(resultat, 4)


class APITestCase(TestCase):
    def setUp(self):
        # self.base_url = 'http://localhost:8000'
        self.client = APIClient()
        self.username = 'projetSpam'
        self.password = 'projetSpam'
        self.user = Utilisateur.objects.create_user(username=self.username, password=self.password)
        self.token = Token.objects.create(user=self.user)

    def test_get_with_token(self):
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token}')

        # Effectuer la requête GET vers la vue protégée
        # reverse -> prend en paramètre le nom de la vue (ou le chemin de l'URL) et retourne l'URL 
        url = reverse('FilesUpload-list')
        response = self.client.get(url)

        # Vérifier le code de réponse et les données reçues
        self.assertEqual(response.status_code, 200)
        # Effectuer d'autres vérifications nécessaires sur les données reçues
        
    def test_get_with_token_2(self):
        # Configurer l'en-tête Authorization avec le jeton
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token}')

        # Effectuer la requête GET à l'URL spécifique
        url = '/api/FilesUpload/4'
        response = self.client.get(url)

        # Vérifier le code de réponse et les données reçues
        self.assertIn(response.status_code, [200, 301])
        # Effectuer d'autres vérifications nécessaires sur les données reçues
