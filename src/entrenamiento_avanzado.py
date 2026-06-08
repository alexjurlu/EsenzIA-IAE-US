import pandas as pd
import numpy as np
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuración de reproducibilidad para asegurar que mis resultados son consistentes
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def vectorizar_word2vec(textos, modelo_w2v):
    """
    He implementado esta función para convertir los correos en vectores densos 
    promediando las palabras. Es la base de mi comparación con Word2Vec.
    """
    vectores = []
    for texto in textos:
        palabras = str(texto).split()
        vectores_palabras = [modelo_w2v.wv[p] for p in palabras if p in modelo_w2v.wv]
        if vectores_palabras:
            vectores.append(np.mean(vectores_palabras, axis=0))
        else:
            vectores.append(np.zeros(modelo_w2v.vector_size))
    return np.array(vectores)

def entrenar_modelos():
    """
    Este es mi script principal de entrenamiento. Aquí orquesto la comparativa
    entre los 8 escenarios solicitados, aplicando estratificación para manejar
    el desequilibrio de las clases minoritarias.
    """
    ruta_in = "data/dataset_nlp.csv"
    if not os.path.exists(ruta_in):
        print(f"Error: No se encuentra {ruta_in}")
        return

    df = pd.read_csv(ruta_in).dropna(subset=['Texto_Procesado', 'Categoria_Origen'])
    
    # Análisis estadístico básico para Streamlit (sobre texto original)
    stats_primitivas = {
        'total_filas': len(df),
        'volumen_texto_total': int(df['Texto_Incidencia'].apply(len).sum()),
        'media_palabras_por_correo': float(df['Texto_Incidencia'].apply(lambda x: len(str(x).split())).mean()),
        'distribucion_categorias': df['Categoria_Origen'].value_counts().to_dict()
    }

    # Estratificación para manejar clases pequeñas
    X = df['Texto_Procesado']
    y = df['Categoria_Origen']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    labels = sorted(y.unique().tolist())
    label_to_id = {label: i for i, label in enumerate(labels)}
    y_train_id = np.array([label_to_id[val] for val in y_train])
    y_test_id = np.array([label_to_id[val] for val in y_test])

    resultados = {}

    # --- ESCENARIOS SIN WORD2VEC (Bag of Words) ---
    bow = CountVectorizer(max_features=1000)
    X_train_bow = bow.fit_transform(X_train_raw).toarray()
    X_test_bow = bow.transform(X_test_raw).toarray()

    modelos_ml = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(probability=True, kernel='linear'),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    for nombre, modelo in modelos_ml.items():
        modelo.fit(X_train_bow, y_train)
        y_pred = modelo.predict(X_test_bow)
        
        resultados[f"{nombre} (Sin Word2Vec)"] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'cm': confusion_matrix(y_test, y_pred, labels=labels).tolist(),
            'y_pred': y_pred.tolist(),
            'usando_w2v': False
        }

    # LSTM sin Word2Vec (Embedding Keras estándar)
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(X_train_raw)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=50)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=50)

    def crear_lstm_simple(vocab_size, output_dim):
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=50))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    lstm_s = crear_lstm_simple(1000, len(labels))
    lstm_s.fit(X_train_seq, y_train_id, epochs=10, batch_size=32, verbose=0)
    y_pred_ls_prob = lstm_s.predict(X_test_seq)
    y_pred_ls = [labels[i] for i in np.argmax(y_pred_ls_prob, axis=1)]

    resultados["LSTM (Sin Word2Vec)"] = {
        'accuracy': accuracy_score(y_test, y_pred_ls),
        'recall': recall_score(y_test, y_pred_ls, average='macro'),
        'cm': confusion_matrix(y_test, y_pred_ls, labels=labels).tolist(),
        'y_pred': y_pred_ls,
        'usando_w2v': False
    }

    # --- ESCENARIOS CON WORD2VEC ---
    corpus = [str(text).split() for text in X_train_raw]
    w2v_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    
    X_train_w2v = vectorizar_word2vec(X_train_raw, w2v_model)
    X_test_w2v = vectorizar_word2vec(X_test_raw, w2v_model)

    modelos_ml_w2v = {
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, kernel='linear'),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    for nombre, modelo in modelos_ml_w2v.items():
        modelo.fit(X_train_w2v, y_train)
        y_pred = modelo.predict(X_test_w2v)
        
        resultados[f"{nombre} (Con Word2Vec)"] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'cm': confusion_matrix(y_test, y_pred, labels=labels).tolist(),
            'y_pred': y_pred.tolist(),
            'usando_w2v': True
        }

    # LSTM con Word2Vec
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    def crear_lstm_w2v(vocab_size, output_dim, embedding_matrix):
        model = Sequential()
        model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=50, trainable=False))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    lstm_w = crear_lstm_w2v(vocab_size, len(labels), embedding_matrix)
    lstm_w.fit(X_train_seq, y_train_id, epochs=10, batch_size=32, verbose=0)
    y_pred_lw_prob = lstm_w.predict(X_test_seq)
    y_pred_lw = [labels[i] for i in np.argmax(y_pred_lw_prob, axis=1)]

    resultados["LSTM (Con Word2Vec)"] = {
        'accuracy': accuracy_score(y_test, y_pred_lw),
        'recall': recall_score(y_test, y_pred_lw, average='macro'),
        'cm': confusion_matrix(y_test, y_pred_lw, labels=labels).tolist(),
        'y_pred': y_pred_lw,
        'usando_w2v': True
    }

    # Guardar todo
    os.makedirs('modelos', exist_ok=True)
    with open('modelos/comparativa_modelos.pkl', 'wb') as f:
        pickle.dump({
            'resultados': resultados,
            'labels': labels,
            'stats_primitivas': stats_primitivas,
            'y_test': y_test.tolist()
        }, f)

    print("Entrenamiento completado. Resultados guardados en modelos/comparativa_modelos.pkl")

if __name__ == "__main__":
    entrenar_modelos()
