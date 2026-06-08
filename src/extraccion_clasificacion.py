import pandas as pd
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify import accuracy
from nltk.probability import FreqDist
import random
import pickle
import os

def obtener_palabras_frecuentes(df, num_palabras=500):
    todas_las_palabras = []
    for texto in df['Texto_Procesado']:
        if isinstance(texto, str):
            todas_las_palabras.extend(texto.split())
    return list(FreqDist(todas_las_palabras).keys())[:num_palabras]

def extraer_caracteristicas(texto, vocab):
    palabras_texto = set(str(texto).split())
    return {p: (p in palabras_texto) for p in vocab}

def entrenar_modelo():
    ruta_in = "data/dataset_nlp.csv"
    if not os.path.exists(ruta_in):
        return
    df = pd.read_csv(ruta_in).dropna(subset=['Texto_Procesado', 'Categoria_Origen'])
    vocab = obtener_palabras_frecuentes(df)
    documentos = [
        (extraer_caracteristicas(row['Texto_Procesado'], vocab), row['Categoria_Origen'])
        for _, row in df.iterrows()
    ]
    random.seed(42)
    random.shuffle(documentos)
    corte = int(len(documentos) * 0.8)
    train_set, test_set = documentos[:corte], documentos[corte:]
    
    modelo = NaiveBayesClassifier.train(train_set)
    
    # Cálculo de métricas para visualización
    acc = accuracy(modelo, test_set)
    
    # Para la matriz de confusión necesitamos las etiquetas reales y las predichas
    y_true = [label for (features, label) in test_set]
    y_pred = [modelo.classify(features) for (features, label) in test_set]
    labels = sorted(list(set(y_true)))
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Características más informativas
    # nltk no da una forma fácil de sacar la lista completa, así que extraemos las top 20
    informative_features = []
    for (fname, fval) in modelo.most_informative_features(20):
        # Para cada característica, sacamos los ratios de probabilidad
        cpdist = modelo._feature_probdist
        # Simplificamos para la tabla: nombre de la palabra y su impacto
        informative_features.append({"Característica": fname, "Valor": fval})

    print(f"Modelo entrenado. Accuracy: {acc:.2%}")
    
    os.makedirs('modelos', exist_ok=True)
    with open('modelos/clasificador_nltk.pkl', 'wb') as f:
        pickle.dump({'modelo': modelo, 'vocabulario': vocab}, f)
    
    # Guardamos las métricas por separado para que el dashboard las lea rápido
    with open('modelos/metricas_modelo.pkl', 'wb') as f:
        pickle.dump({
            'accuracy': acc,
            'confusion_matrix': cm,
            'labels': labels,
            'informative_features': informative_features
        }, f)

if __name__ == "__main__":
    entrenar_modelo()
