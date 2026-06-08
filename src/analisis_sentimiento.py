import os
import pandas as pd
import pickle
import nltk
from pysentimiento import create_analyzer
from tokenizar import procesar_texto_nlp

nltk.download('punkt', quiet=True)

def enriquecer_correos():
    """
    Este es el núcleo de mi sistema de triaje. Aquí combino dos modelos de IA:
    1. Un modelo Naive Bayes entrenado por mí para la categorización temática.
    2. Un modelo Transformer (RoBERTa) pre-entrenado para capturar matices emocionales y sentimientos.
    Esta combinación permite una priorización mucho más robusta que un análisis de palabras clave simple.
    """
    ruta_in = "data/correos_recientes.csv"
    ruta_mod = "modelos/clasificador_nltk.pkl"
    ruta_out = "data/dataset_enriquecido.csv"

    if not os.path.exists(ruta_in):
        return

    df = pd.read_csv(ruta_in)
    with open(ruta_mod, 'rb') as f:
        data = pickle.load(f)
    
    model = data['modelo']
    vocab = data['vocabulario']

    df['Texto_Incidencia'] = df['Asunto'].fillna('') + " " + df['Resumen'].fillna('')
    df['Texto_Procesado'] = df['Texto_Incidencia'].apply(procesar_texto_nlp)

    def predecir(txt):
        """
        Utilizo la extracción de características basada en el vocabulario (visto en tutoría con Andrés)
        que generé durante la fase de test para asegurar la consistencia.
        """
        feats = {p: (p in set(str(txt).split())) for p in vocab}
        return model.classify(feats)
    
    df['Categoria_Origen'] = df['Texto_Procesado'].apply(predecir)

    analyzer_s = create_analyzer(task="sentiment", lang="es")
    analyzer_e = create_analyzer(task="emotion", lang="es")

    df['Sentimiento'] = df['Texto_Incidencia'].apply(lambda x: analyzer_s.predict(str(x)).output)
    df['Emocion'] = df['Texto_Incidencia'].apply(lambda x: analyzer_e.predict(str(x)).output)

    def asignar_prioridad(fila):
        """
        He diseñado esta heurística de triaje basándome en que la ira (anger) o el
        sentimiento puramente negativo son los mejores predictores de una crisis de marca.
        """
        if fila['Emocion'] == 'anger' or fila['Sentimiento'] == 'NEG':
            return 'ALTA (Urgente)'
        return 'MEDIA' if fila['Sentimiento'] == 'NEU' else 'BAJA'

    df['Prioridad'] = df.apply(asignar_prioridad, axis=1)
    df['Estado'] = 'Abierto'

    df.to_csv(ruta_out, index=False)
    print(f"Analisis completado: {len(df)} registros procesados.")

if __name__ == "__main__":
    enriquecer_correos()