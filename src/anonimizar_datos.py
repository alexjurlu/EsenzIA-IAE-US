import pandas as pd
import os
import hashlib

def anonymize_id(id_val):
    if pd.isna(id_val): return id_val
    return "ID_" + hashlib.md5(str(id_val).encode()).hexdigest()[:8]

def anonimizar_datos():
    archivos = [
        "data/dataset_nlp.csv",
        "data/tabla_incidencias.csv",
        "data/correos_recientes.csv",
        "data/dataset_enriquecido.csv",
        "data/dataset_maestro.csv",
        "data/dataset_maestro_limpio.csv"
    ]
    
    for archivo in archivos:
        if os.path.exists(archivo):
            print(f"Anonimizando {archivo}...")
            df = pd.read_csv(archivo)
            
            # Anonimizar IDs de pedido
            cols_id = ['ID_Pedido', 'id_pedido', 'Pedido']
            for col in cols_id:
                if col in df.columns:
                    df[col] = df[col].apply(anonymize_id)
            
            # Ocultar textos sensibles
            cols_texto = ['Texto_Incidencia', 'Texto_Completo_Mensaje', 'Asunto', 'Resumen', 'Resumen_IA', 'Remitente']
            for col in cols_texto:
                if col in df.columns:
                    df[col] = "[OCULTO POR PRIVACIDAD]"
            
            df.to_csv(archivo, index=False)
    
    print("Anonimización completada.")

if __name__ == "__main__":
    anonimizar_datos()
