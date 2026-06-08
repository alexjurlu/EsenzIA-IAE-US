import os
from dotenv import load_dotenv

load_dotenv()

ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

def validate_credentials():
    credentials = {
        "Zoho Client ID": ZOHO_CLIENT_ID,
        "Zoho Client Secret": ZOHO_CLIENT_SECRET,
        "Notion API Key": NOTION_API_KEY,
        "Notion Database ID": NOTION_DATABASE_ID
    }
    
    missing = [name for name, val in credentials.items() if not val]
    
    if missing:
        raise ValueError(f"Faltan credenciales en .env: {', '.join(missing)}")
    
    print("Credenciales cargadas correctamente.")
    return True

if __name__ == "__main__":
    validate_credentials()