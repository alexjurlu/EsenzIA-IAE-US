# Caso Práctico: Automatización de Triaje con IA

Este documento resume los datos necesarios para la redacción de la aplicación práctica del TFG, centrada en el ahorro de tiempo y la eficiencia operativa lograda con el sistema de IA.

## 1. Parámetros del Sistema (Realidad Operativa)

Basado en el volumen de datos procesados (1999 correos descargados de Zoho Mail):

*   **Tasa de llegada ($\lambda$ total):**
    *   **Volumen:** 1999 correos.
    *   **Estimación Temporal:** Tomando como referencia un periodo de 3 meses (aprox. 90 días), la tasa de llegada es de **$\lambda \approx 22$ correos/día**.
    *   **Frecuencia:** Aproximadamente 1 correo por hora en una jornada de 24h, o ~3 correos/hora en jornada laboral.
*   **Tiempo medio de servicio ($1/\mu$):**
    *   **Situación Original (Manual):** Se estima en **8 minutos** por correo (lectura completa, identificación del pedido, apertura de Notion, clasificación manual y respuesta inicial).
    *   **Situación con IA (Actual):** El sistema automatiza la lectura y clasificación en Notion. Esto supone un **ahorro directo de 2 minutos por correo**.
    *   **Nuevo tiempo de servicio:** **6 minutos** por correo (solo resolución y respuesta).

## 2. Resultados de la Clasificación (Prioridades y Categorías)

El sistema ha sido entrenado con 325 registros históricos y ha clasificado los 1999 mensajes con la siguiente distribución:

### Distribución de Prioridades ($\lambda_i$)
Basado en el análisis de sentimiento (RoBERTa) y la urgencia detectada:

| Prioridad | Porcentaje | Frecuencia Est. ($\lambda_i$) |
| :--- | :--- | :--- |
| **ALTA (Urgente)** | 36.85% | ~8 correos/día |
| **MEDIA** | 60.09% | ~13 correos/día |
| **BAJA** | 3.06% | ~1 correo/día |

### Distribución por Categorías
| Categoría | Porcentaje |
| :--- | :--- |
| **INCIDENCIA GENERAL** | 82.86% |
| **PEDIDO PERDIDO** | 9.15% |
| **TARA PROVEEDOR** | 7.98% |

> **Nota sobre $\mu$:** Aunque el tiempo de resolución técnica del problema no ha variado, la capacidad de respuesta inmediata para la categoría "ALTA" mejora drásticamente al estar pre-clasificada.

## 3. Resumen del Modelo NLP (Introducción Técnica)

*   **Modelo de Clasificación:** **Naive Bayes Multinomial**. Elegido por su alta eficiencia en clasificación de texto con vocabularios específicos (500 características optimizadas).
*   **Modelo de Priorización:** **Transformer RoBERTa** (vía `pysentimiento`). Utilizado para detectar el sentimiento predominante (NEG/NEU/POS) y asignar la prioridad de forma inteligente.
*   **Métricas de Éxito:**
    *   **Accuracy (Exactitud):** 87.69%.
    *   **F1-Score:** 0.86.
*   **Orquestación:** Implementado con **Dagster**, asegurando un pipeline reproducible y robusto.

## 4. Conclusión de Ahorro y Eficiencia

Realizando una proyección anual basada en los datos extraídos:

1.  **Ahorro Diario:** 22 correos * 2 min = **44 minutos/día**.
2.  **Ahorro Mensual:** ~22 horas de trabajo (equivalente a **3 jornadas laborales completas** al mes).
3.  **Impacto Cualitativo:** El trabajador elimina la carga mental de clasificar manualmente 2000 mensajes, reduciendo el error humano y asegurando que las incidencias urgentes (37% del total) se identifiquen al instante sin necesidad de leer toda la bandeja de entrada.
