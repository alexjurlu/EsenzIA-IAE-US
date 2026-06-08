import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def procesar_texto_nlp(texto):
    if not isinstance(texto, str): return ""
    tokens = word_tokenize(texto.lower(), language='spanish')
    stop_words = set(stopwords.words('spanish'))
    negaciones = {"no", "nunca", "jamas", "tampoco", "nadie", "ningun", "ninguna", "nada"}
    stop_words = stop_words - negaciones
    signos = set(string.punctuation) | {"¿", "¡", "...", "''", "``"}
    tokens_filtrados = [w for w in tokens if w not in stop_words and w not in signos and not w.isdigit()]
    stemmer = SnowballStemmer('spanish')
    return " ".join([stemmer.stem(w) for w in tokens_filtrados])

def ejecutar_nlp_entrenamiento():
    ruta_in = "data/dataset_maestro.csv"
    ruta_out = "data/dataset_nlp.csv"
    if os.path.exists(ruta_in):
        df = pd.read_csv(ruta_in)
        df['Texto_Procesado'] = df['Texto_Incidencia'].apply(procesar_texto_nlp)
        df.to_csv(ruta_out, index=False)
        print(f"Dataset procesado: {len(df)} registros.")

if __name__ == "__main__":
    ejecutar_nlp_entrenamiento()
