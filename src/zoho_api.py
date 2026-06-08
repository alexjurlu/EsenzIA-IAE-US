import os
import requests
from dotenv import load_dotenv

load_dotenv()

ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
DATACENTER_URL = "https://accounts.zoho.eu"
MAIL_API_URL = "https://mail.zoho.eu/api/accounts"

def obtener_access_token():
    url = f"{DATACENTER_URL}/oauth/v2/token"
    payload = {
        "refresh_token": ZOHO_REFRESH_TOKEN,
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "grant_type": "refresh_token"
    }
    response = requests.post(url, data=payload)
    response.raise_for_status()
    return response.json().get("access_token")

def obtener_correos():
    token = obtener_access_token()
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    acc_response = requests.get(MAIL_API_URL, headers=headers)
    acc_response.raise_for_status()
    accounts = acc_response.json().get("data")
    
    if not accounts:
        return []
        
    account_id = accounts[0].get("accountId")
    
    mensajes_url = f"{MAIL_API_URL}/{account_id}/messages/view"
    params = {"limit": 10}
    
    msg_response = requests.get(mensajes_url, headers=headers, params=params)
    msg_response.raise_for_status()
    return msg_response.json().get("data")

if __name__ == "__main__":
    correos = obtener_correos()
    for c in correos:
        print(f"ID: {c.get('messageId')}")
        print(f"Asunto: {c.get('subject')}")
        print(f"Remitente: {c.get('sender')}")
        print(f"Resumen: {c.get('summary')}")
        print("-" * 40)