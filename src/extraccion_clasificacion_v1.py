import pandas as pd
import nltk
import random
from nltk.classify import NaiveBayesClassifier
from nltk.classify import accuracy

"""
def extraer_caracteristicas(texto):
    Convierte una cadena de texto procesada en el formato de diccionario que exige NLTK.
    Ej: "paquet perd" -> {'paquet': True, 'perd': True}
    palabras = set(texto.split())
    features = {}
    for palabra in palabras:
        features[palabra] = True
    return features

    """

# ==============================================================================
# ITERACIÓN 1: MODELO BASE (BASELINE) NLTK
# ------------------------------------------------------------------------------
# Ejecución inicial con Bag of Words booleano: 64.62% exactitud.
# Análisis de features detectó ruido numérico (ej. '17', '20').
# Tras aplicar filtro .isdigit() en tokenización, la exactitud subió a 66.15%.
# Siguiente acción: Implementar extracción de Bigramas para capturar contexto.
# ==============================================================================

from nltk import bigrams

def extraer_caracteristicas(texto):
    palabras = texto.split()
    features = {}
    
    for palabra in palabras:
        features[palabra] = True
        
    # Bigramas (Pares de palabras consecutivas)
    pares = list(bigrams(palabras))
    for par in pares:
        feature_name = f"{par[0]}_{par[1]}"
        features[feature_name] = True
        
    return features

# ==============================================================================
# ITERACIÓN 2: EXPANSIÓN CON BIGRAMAS (OVERFITTING)
# ------------------------------------------------------------------------------
# Al incorporar bigramas, la exactitud cayó drásticamente al 47.69%.
# Conclusión: La explosión del espacio de características (maldición de la 
# dimensionalidad) sobre un corpus pequeño (260 ejemplos) provoca sobreajuste.
# Siguiente Acción: Abandonar BoW booleano y NLTK puro
# ==============================================================================

def entrenar_modelo_nltk(ruta_entrada):
    print("Cargando datos limpios...")
    df = pd.read_csv(ruta_entrada)
    df = df.dropna(subset=['Texto_Procesado', 'Categoria_Origen'])
    
    #Crear la lista de tuplas (diccionario_features, etiqueta) que NLTK necesita
    documentos = [
        (extraer_caracteristicas(row['Texto_Procesado']), row['Categoria_Origen'])
        for index, row in df.iterrows()
    ]
    
    random.seed(42)
    random.shuffle(documentos)
    
    #Dividir en Entrenamiento (80%) y Prueba (20%)
    corte = int(len(documentos) * 0.8)
    train_set = documentos[:corte]
    test_set = documentos[corte:]
    
    print(f"Entrenando con {len(train_set)} ejemplos...")
    #Entrenar el clasificador Naive Bayes puro de NLTK
    clasificador = NaiveBayesClassifier.train(train_set)
    
    #Evaluar la eficacia (Accuracy)
    exactitud = accuracy(clasificador, test_set)
    print(f"\nExactitud del modelo NLTK: {exactitud:.2%}")
    
    #Mostrar las características más informativas
    print("\nLas 10 palabras más determinantes para clasificar la incidencia:")
    clasificador.show_most_informative_features(10)
    
    return clasificador

if __name__ == "__main__":
    modelo = entrenar_modelo_nltk("data/dataset_nlp.csv")