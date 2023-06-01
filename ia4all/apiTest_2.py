from rest_framework.authtoken.models import Token
from django.test.client import Client

username = 'projetSpam'
password = 'projetSpam'

# Iniciar sesión con las credenciales
client = Client()
logged_in = client.login(username=username, password=password)

# Verificar si el inicio de sesión fue exitoso
if logged_in:
    # Obtener o crear el token de autenticación para el usuario
    user = User.objects.get(username=username)
    token, created = Token.objects.get_or_create(user=user)
    print("Token:", token.key)
else:
    print("Inicio de sesión fallido")