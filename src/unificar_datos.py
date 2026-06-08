import pandas as pd
import os

def procesar_csvs():
    archivos = {
        "PAQUETES DUPLICADOS BIGBLUE 2bb788364f84806388edcae76dc25bdf.csv": {
            "categoria": "PAQUETE DUPLICADO",
            "col_texto": "Texto",
            "col_id": "NUMERO PEDIDO"
        },
        "INCIDENCIA PÉRDIDA PEDIDOS 2a3788364f8480bfb49ff6b3776b9b30.csv": {
            "categoria": "PEDIDO PERDIDO",
            "col_texto": "Texto",
            "col_id": "NÚMERO DE PEDIDO"
        },
        "INCIDENCIAS 19b788364f84804ba43bfc058366a595.csv": {
            "categoria": "INCIDENCIA GENERAL",
            "col_texto": "MOTIVO",
            "col_id": "Nº PEDIDO"
        },
        "INCIDENCIAS TARAS PROVEEDORES 2a3788364f84808ebdb3d6173787f9cc.csv": {
            "categoria": "TARA PROVEEDOR",
            "col_texto": "TIPO DE TARA",
            "col_id": "INCIDENCIA"
        }
    }
    dataframes_limpios = []
    for archivo, config in archivos.items():
        ruta = f"data/{archivo}"
        if not os.path.exists(ruta):
            continue
        df = pd.read_csv(ruta)
        df_limpio = pd.DataFrame()
        df_limpio['ID_Pedido'] = df.get(config['col_id'])
        df_limpio['Texto_Incidencia'] = df.get(config['col_texto'])
        df_limpio['Categoria_Origen'] = config['categoria']
        df_limpio = df_limpio.dropna(subset=['Texto_Incidencia'])
        dataframes_limpios.append(df_limpio)
    
    df_final = pd.concat(dataframes_limpios, ignore_index=True)
    df_final.to_csv("data/dataset_maestro.csv", index=False)
    print(f"Dataset maestro generado: {len(df_final)} registros.")

if __name__ == "__main__":
    procesar_csvs()