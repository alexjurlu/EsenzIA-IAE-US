import os
import requests
import pandas as pd
import re
import html
from dotenv import load_dotenv

load_dotenv()

ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
DATACENTER_URL = "https://accounts.zoho.eu"
MAIL_API_URL = "https://mail.zoho.eu/api/accounts"

def obtener_access_token():
    """
    Conexión con la API de Zoho. Utiliza un sistema de renovación automática
    para que el pipeline funcione solo en segundo plano, sin requerir intervención manual.
    """
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

def limpiar_texto(texto):
    if not texto: return ""
    texto = html.unescape(texto)
    texto = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s\.,!?#]', '', str(texto))
    return re.sub(r'\s+', ' ', texto).strip()

def extraer_pedido(texto):
    """
    He desarrollado este extractor basado en expresiones regulares para normalizar los IDs.
    Diferencio entre el formato de Shopify (#) y el de Bigblue (ESENS) para asegurar
    que la trazabilidad de los pedidos sea consistente en todo el sistema.
    """
    if not texto: return None
    
    match_hash = re.search(r'#\s*(\d{5,7})', str(texto))
    if match_hash:
        return match_hash.group(1)
    
    match_bigblue = re.search(r'ESENS\s*(\d{5,7})', str(texto))
    if match_bigblue:
        pedido = match_bigblue.group(1)
        return pedido[-6:] if len(pedido) == 7 else pedido
    
    return None

def procesar_bandeja(ruta_salida="data/correos_recientes.csv"):
    token = obtener_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    acc_response = requests.get(MAIL_API_URL, headers=headers)
    acc_response.raise_for_status()
    accounts = acc_response.json().get("data")
    
    if not accounts:
        return
        
    account_id = accounts[0].get("accountId")
    mensajes_url = f"{MAIL_API_URL}/{account_id}/messages/view"
    
    datos = []
    start = 1
    limit = 100
    seguir = True
    
    while seguir and len(datos) < 2000:
        response = requests.get(mensajes_url, headers=headers, params={"limit": limit, "start": start})
        response.raise_for_status()
        data = response.json().get("data", [])
        
        if not data:
            seguir = False
            break
            
        for c in data:
            asunto = limpiar_texto(c.get('subject'))
            resumen = limpiar_texto(c.get('summary'))
            id_pedido = extraer_pedido(asunto) or extraer_pedido(resumen)
            
            datos.append({
                "ID_Mensaje": str(c.get('messageId')),
                "Remitente": c.get('sender'),
                "Asunto": asunto,
                "Resumen": resumen,
                "ID_Pedido": id_pedido
            })
        
        start += limit
        
    df = pd.DataFrame(datos)
    df = df[~df['Remitente'].str.contains('notion', case=False, na=False)]
    df.to_csv(ruta_salida, index=False)
    print(f"Descargados {len(df)} correos de Zoho.")

if __name__ == "__main__":
    procesar_bandeja()