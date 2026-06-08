# AI Customer Triage: Sistema Inteligente de Clasificación y Análisis Estadístico

## Descripción del Proyecto
Este proyecto implementa una solución avanzada de Inteligencia Artificial para automatizar el triage de atención al cliente. El sistema ingesta correos reales desde **Zoho Mail**, los clasifica comparando **4 modelos (NB, SVM, KNN, LSTM)** entrenados exclusivamente sobre embeddings de **Word2Vec** y analiza su carga emocional utilizando **Transformers (RoBERTa)**.

El objetivo es transformar una bandeja de entrada saturada en un panel de control operativo y estadístico que permite priorizar los casos urgentes y detectar puntos de fricción en el servicio.

## Arquitectura del Sistema
El proyecto está diseñado siguiendo estándares profesionales de ingeniería de datos:
*   **Gestor de Entorno**: `uv` para una gestión de dependencias rápida y determinista.
*   **Orquestación**: `Dagster` para la gestión del pipeline como Activos de Datos (Assets).
*   **IA de Producción**: Clasificación temática y análisis de sentimiento multimodelo.
*   **Visualización**: Dashboard interactivo en `Streamlit` con análisis estadístico automático.

## Instalación y Configuración
1.  Asegurar tener instalado `uv`.
2.  Clonar el repositorio y sincronizar el entorno:
    ```bash
    uv sync
    ```
3.  Configurar el archivo `.env` con las credenciales de la API de Zoho (ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN).

## Guía de Ejecución

### Opción A: Ejecución Profesional (Dagster)
Esta es la forma recomendada para ejecutar el pipeline completo con visibilidad total:
```bash
uv run dagster dev -f src/orchestration.py
```
Accede a `localhost:3000`, ve a **Asset Graph** y pulsa **Materialize All**. Esto ejecutará de forma orquestada:
1.  Descarga de correos (Zoho API).
2.  Entrenamiento y actualización del modelo IA.
3.  Enriquecimiento de datos (Categoría + Sentimiento + Prioridad).
4.  Generación de la tabla final de incidencias.

### Opción B: Visualización de Resultados (Streamlit)
Para ver el análisis estadístico y gestionar las incidencias:
```bash
uv run streamlit run app/main.py
```

## Estructura de Archivos
*   `app/`: Interfaz de usuario y motor de visualización estadística.
*   `src/`: Módulos de extracción, NLP, entrenamiento y orquestación.
*   `data/`: Almacenamiento de datasets (Notion y Zoho).
*   `modelos/`: Almacén de modelos serializados (Pickle/Joblib).
*   `Memoria.Rmd`: Informe técnico y análisis estadístico reproducible para RStudio.

## Resultados Clave y Comparativa de Modelos
- **Arquitectura de IA**: He evolucionado el sistema de un único modelo a una comparativa de **4 modelos** competitivos (Naive Bayes, SVM, KNN, LSTM) utilizando **Word2Vec** como motor de vectorización.
- **Métricas de Éxito**: El sistema ahora evalúa tanto **Accuracy** como **Recall**, asegurando que categorías minoritarias como "Pedidos Duplicados" sean detectadas con precisión mediante **estratificación**.
- **Entrenamiento**: 325 registros históricos procesados, seleccionando automáticamente la mejor combinación de vectorización y algoritmo para la clasificación.
- **Triaje Avanzado**: Integración de análisis de sentimiento RoBERTa para detectar casos de urgencia crítica basados en la carga emocional del cliente.