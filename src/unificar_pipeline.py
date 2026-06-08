import pandas as pd
import os

def crear_tabla_incidencias(
    ruta_enriquecido="data/dataset_enriquecido.csv",
    ruta_correos="data/correos_recientes.csv",
    ruta_salida="data/tabla_incidencias.csv"
):
    """
    Este script consolida el pipeline. He decidido agrupar los correos por ID de pedido
    para transformar mensajes aislados en 'Hilos de Incidencia', permitiendo un triaje
    mucho más eficiente basado en el historial completo de cada pedido.
    """
    if not os.path.exists(ruta_enriquecido):
        return None
    
    df_enriquecido = pd.read_csv(ruta_enriquecido)
    df_correos = None
    if os.path.exists(ruta_correos):
        df_correos = pd.read_csv(ruta_correos)
    
    df_final = procesar_enriquecidos(df_enriquecido, df_correos)
    
    if df_final.empty:
        return None
    
    df_final.to_csv(ruta_salida, index=False)
    print(f"Tabla de incidencias generada con {len(df_final)} casos.")
    return df_final

def procesar_enriquecidos(df_enriquecido, df_correos=None):
    df = df_enriquecido[df_enriquecido['ID_Pedido'].notna()].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    df['ID_Pedido_Clean'] = df['ID_Pedido'].astype(str).str.replace('#', '').str.strip()
    
    df_agrupado = df.groupby('ID_Pedido_Clean', as_index=False).agg({
        'Texto_Incidencia': lambda x: ' | '.join(x.astype(str).dropna()),
        'Categoria_Origen': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        'Sentimiento': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        'Emocion': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        'Prioridad': lambda x: x.iloc[0]
    }).rename(columns={
        'ID_Pedido_Clean': 'ID_Pedido',
        'Texto_Incidencia': 'Texto_Completo_Mensaje',
        'Categoria_Origen': 'Clasificacion_Incidencia',
        'Sentimiento': 'Sentimiento_Predominante',
        'Emocion': 'Emocion_Predominante'
    })
    
    num_emails_map = {}
    if df_correos is not None:
        df_correos_limpio = df_correos[df_correos['ID_Pedido'].notna()].copy()
        df_correos_limpio['ID_Clean'] = df_correos_limpio['ID_Pedido'].astype(str).str.replace('#', '').str.strip()
        num_emails_map = df_correos_limpio.groupby('ID_Clean').size().to_dict()
    
    df_agrupado['Num_Emails_Conversacion'] = df_agrupado['ID_Pedido'].map(lambda x: num_emails_map.get(x, 1))
    df_agrupado['Estado'] = 'Abierto'
    
    columnas_orden = ['ID_Pedido', 'Num_Emails_Conversacion', 'Texto_Completo_Mensaje', 'Clasificacion_Incidencia', 'Estado', 'Sentimiento_Predominante', 'Emocion_Predominante', 'Prioridad']
    df_agrupado = df_agrupado[columnas_orden]
    
    prioridad_orden = {'ALTA (Urgente)': 0, 'MEDIA': 1, 'BAJA': 2}
    df_agrupado['Prioridad_Sort'] = df_agrupado['Prioridad'].map(prioridad_orden).fillna(3)
    df_agrupado = df_agrupado.sort_values(['Prioridad_Sort', 'ID_Pedido']).drop('Prioridad_Sort', axis=1)
    
    return df_agrupado

if __name__ == "__main__":
    crear_tabla_incidencias()
