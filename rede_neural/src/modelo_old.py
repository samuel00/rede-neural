import json
import pickle
import random

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

from rede_neural.src.preprocessamento import preprocessar_texto

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer_path = "modelos/tokenizer_contratos.pkl"
modelo_path = "modelos/modelo_contratos_lstm.h5"

script_dir = Path(__file__).resolve().parent

# Caminho para o JSON no diretório irmão
json_path = script_dir.parent / 'dados' / 'contratos_treinamento.json'

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def carregar_dados_treinamento():
    with open(json_path, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    textos = [d["texto"] for d in dados]
    rotulos = [d["rotulo"] for d in dados]
    return textos, rotulos


def treinar_e_salvar_modelo():
    textos, rotulos = carregar_dados_treinamento()

    # Split ANTES do fit_on_texts
    X_train, X_val, y_train, y_val = train_test_split(
        textos, rotulos, test_size=0.2, random_state=42
    )

    # Tokenizer só vê os dados de treino
    tokenizer.fit_on_texts(X_train)
    salvar_tokenizer(tokenizer, tokenizer_path)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)

    X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=100, padding='post')

    modelo = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    modelo.fit(
        X_train_pad,
        np.array(y_train),
        validation_data=(X_val_pad, np.array(y_val)),
        epochs=10,
        verbose=2
    )

    # Avaliação
    predicoes_val = modelo.predict(X_val_pad)
    predicoes_binarias = (predicoes_val > 0.5).astype(int)

    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_val, predicoes_binarias, digits=4))

    # Salvar modelo
    if not os.path.exists("modelos"):
        os.makedirs("modelos")
    modelo.save(modelo_path)

    return modelo


def carregar_modelo():
    if not os.path.exists(modelo_path):
        return treinar_e_salvar_modelo()
    return tf.keras.models.load_model(modelo_path)

def dividir_em_trechos(texto, tamanho=100):
    palavras = texto.split()
    return [" ".join(palavras[i:i + tamanho]) for i in range(0, len(palavras), tamanho)]


def verificar_contrato(texto):
    texto = preprocessar_texto(texto)
    print(f"[DEBUG] Texto pré-processado (resumo): {texto[:300]}...")

    if not texto.strip():
        print("[AVISO] Texto vazio após pré-processamento.")
        return False

    tokenizer = carregar_tokenizer(tokenizer_path)
    modelo = carregar_modelo()

    trechos = dividir_em_trechos(texto, tamanho=100)
    trechos_reais = 0

    votos_positivos = 0
    for i, trecho in enumerate(trechos):
        if len(trecho.split()) < 20:
            print(f"[DEBUG] Trecho {i + 1}: ignorado (muito curto)")
            continue
        trechos_reais += 1
        seq = tokenizer.texts_to_sequences([trecho])
        if not seq or not any(seq[0]):
            print(f"[DEBUG] Trecho {i + 1}: ignorado (sem tokens reconhecidos)")
            continue

        pad_seq = pad_sequences(seq, maxlen=100, padding='post')
        pred = modelo.predict(pad_seq)[0][0]
        print(f"[DEBUG] Trecho {i + 1}: Predição bruta = {pred:.6f}")

        if pred > 0.7:
            votos_positivos += 1

    print(f"[INFO] Total de trechos positivos: {votos_positivos} de {len(trechos)}")

    if trechos_reais == 1:
        return votos_positivos >= 1  # texto curto
    else:
        return votos_positivos >= 2  # texto longo



def salvar_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)


def carregar_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
