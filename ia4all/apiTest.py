import requests

# response = requests.get("http://localhost:8000/api/FilesUpload/24")
# print(response.json())  


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