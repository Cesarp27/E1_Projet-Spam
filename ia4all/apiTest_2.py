import requests

# response = requests.get("http://localhost:8000/api/FilesUpload/24")
# print(response.json())  


class APITestCase():
    def setUp(self):
        # Configuration initiale pour les tests
        self.base_url = 'http://localhost:8000'
        self.username = 'projetSpam'
        self.password = 'projetSpam'
        self.token = None

    def test_get_with_token(self):
        # Obtenir le jeton d'authentification
        response = self.client.post('/api-token-auth/', {'username': self.username, 'password': self.password})
        # self.assertEqual(response.status_code, 200)
        # self.token = response.json().get('token')

        # # Configurer l'en-tête Authorization avec le jeton
        # headers = {
        #     'Authorization': f'Token {self.token}'
        # }

        # # Effectuer la requête GET vers la vue protégée
        # url = f'{self.base_url}/api/FilesUpload/'
        # response = requests.get(url, headers=headers)

        # # Vérifier le code de réponse et les données reçues
        # self.assertEqual(response.status_code, 200)
        # # Effectuer d'autres vérifications nécessaires sur les données reçues
        
        
        return 














# URL pour obtenir le jeton
url_token = 'http://localhost:8000/api-token-auth/'

# Informations d'identification de l'utilisateur
data_token = {
    'username': 'projetSpam',
    'password': 'projetSpam'
}

# Effectuer la requête POST pour obtenir le jeton
response_token = requests.post(url_token, data=data_token)
token = response_token.json().get('token')

# print(token)

# URL de la vue protégée
url = 'http://localhost:8000/api/FilesUpload/24'

# Token d'authentification généré à partir du terminal
# --> python manage.py drf_create_token root
# token = '#############################'

headers = {
    'Authorization': f'Token {token}'
}

response = requests.get(url, headers=headers)
print(response.json())