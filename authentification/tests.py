from django.test import TestCase
from rest_framework.authtoken.models import Token
from authentification.models import Utilisateur
from django.urls import reverse
from rest_framework.test import APIClient
import requests
from urllib.parse import urljoin
import authentification.conf

# Create your tests here.

class APITestCase(TestCase):
    def setUp(self):
        # self.base_url = 'http://localhost:8000'
        self.client = APIClient()
        self.username = 'userTest1'
        self.password = 'userTest1'
        self.user = Utilisateur.objects.create_user(username=self.username, password=self.password)
        self.token = Token.objects.create(user=self.user)

    def test_get_with_token(self):
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token}')

        # Effectuer la requête GET vers la vue protégée
        # reverse -> prend en paramètre le nom de la vue (ou le chemin de l'URL) et retourne l'URL 
        url = reverse('FilesUpload-list')
        response = self.client.get(url)

        # Vérifier le code de réponse 
        self.assertEqual(response.status_code, 200)
        
        
        
    def test_get_with_token_2(self):
        # Configurer l'en-tête Authorization avec le jeton
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token}')

        # Effectuer la requête GET à l'URL spécifique
        url = '/api/FilesUpload/4'
        response = self.client.get(url)
        # print(response)

        # Vérifier le code de réponse et les données reçues
        self.assertIn(response.status_code, [200, 301])        

    def test_get_with_token_3(self):
        # Vérification que l'utilisateur peut accéder à un fichier qu'il a lui-même uploadé
        
        # URL pour obtenir le jeton
        url_token = 'http://localhost:8000/api-token-auth/'

        # Informations d'identification de l'utilisateur
        data_token = {
            'username': authentification.conf.username,
            'password': authentification.conf.password
        }

        # Effectuer la requête POST pour obtenir le jeton
        response_token = requests.post(url_token, data=data_token)
        token = response_token.json().get('token')

        # URL de la vue protégée
        url = 'http://localhost:8000/api/FilesUpload/24'

        headers = {'Authorization': f'Token {token}'}

        response = requests.get(url, headers=headers)
        self.assertEqual(response.json()['file'], 'http://localhost:8000/media/news_sms.txt')
        # print(response.json())      
    
    
    def test_get_with_token_4(self):
        # Ce test est réussi si l'API refuse l'accès 
        # L'utilisateur xxxx essaie d'accéder à un fichier qu'il n'a pas uploadé.
        
        # URL pour obtenir le token
        url_token = 'http://localhost:8000/api-token-auth/'

        # Informations d'identification de l'utilisateur
        data_token = {
            'username': authentification.conf.username,
            'password': authentification.conf.password
        }

        # Effectuer la requête POST pour obtenir le token
        response_token = requests.post(url_token, data=data_token)
        token = response_token.json().get('token')

        # URL de la vue protégée
        url = 'http://localhost:8000/api/FilesUpload/4'

        headers = {'Authorization': f'Token {token}'}

        response = requests.get(url, headers=headers)
        self.assertNotEqual(response.status_code, 200)
        print('- Accès à un fichier qui ne vous appartient pas')
        print(response.json())     
        
        
        
        
    