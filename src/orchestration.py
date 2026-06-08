from dagster import asset, Definitions
import os

"""
He implementado esta capa de orquestación con Dagster para transformar un conjunto de scripts
aislados en un pipeline de datos profesional y resiliente. 
Cada función representa un 'Software Defined Asset', lo que permite trazar el linaje de los datos
desde la API de Zoho hasta el Dashboard final.
Debido a que ha sido lo más complicado para mi durante el curso, me he asesorado con IA generativa.
"""

@asset
def dataset_maestro_notion():
    """Consolida los datos históricos de entrenamiento desde Notion."""
    from unificar_datos import procesar_csvs
    procesar_csvs()
    return "data/dataset_maestro.csv"

@asset(deps=[dataset_maestro_notion])
def modelo_clasificacion_entrenado():
    """Procesa el lenguaje natural y entrena la comparativa de 4 modelos (NB, SVM, KNN, LSTM) usando Word2Vec."""
    from tokenizar import ejecutar_nlp_entrenamiento
    from entrenamiento_avanzado import entrenar_modelos
    ejecutar_nlp_entrenamiento()
    entrenar_modelos()
    return "modelos/comparativa_modelos.pkl"

@asset
def correos_zoho_recientes():
    """Extrae de forma incremental los mensajes desde la API de Zoho Mail."""
    from extraer_zoho import procesar_bandeja
    procesar_bandeja()
    return "data/correos_recientes.csv"

@asset(deps=[correos_zoho_recientes, modelo_clasificacion_entrenado])
def dataset_enriquecido_ia():
    """Aplica el modelo de clasificación y el análisis de sentimiento RoBERTa."""
    from analisis_sentimiento import enriquecer_correos
    enriquecer_correos()
    return "data/dataset_enriquecido.csv"

@asset(deps=[dataset_enriquecido_ia])
def tabla_final_dashboard():
    """Genera la tabla final unificada y optimizada para el consumo en Streamlit."""
    from unificar_pipeline import crear_tabla_incidencias
    crear_tabla_incidencias()
    return "data/tabla_incidencias.csv"

defs = Definitions(
    assets=[
        dataset_maestro_notion, 
        modelo_clasificacion_entrenado, 
        correos_zoho_recientes, 
        dataset_enriquecido_ia, 
        tabla_final_dashboard
    ]
)
