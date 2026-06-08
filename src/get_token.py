import requests
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
GRANT_TOKEN = "1000.b05cf37de92e4b98b528541888cd7ad0.48bedf55fe1b1157445e5a905350edd6"
DATACENTER_URL = "https://accounts.zoho.eu"

url = f"{DATACENTER_URL}/oauth/v2/token"

payload = {
    "grant_type": "authorization_code",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "code": GRANT_TOKEN
}

response = requests.post(url, data=payload)
print(response.json())

# Archivo usado para crear el TOKEN de acceso a Zoho