import os
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

from rede_neural.src.preprocessamento import preprocessar_texto

# Caminhos de modelo e tokenizer
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "modelos" / "modelo_bert.pkl"
BERT_NAME = "neuralmind/bert-base-portuguese-cased"

# Inicializa tokenizer e modelo BERT
bert_tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
bert_model = BertModel.from_pretrained(BERT_NAME)

# Função para dividir em trechos
def dividir_em_trechos(texto, tamanho=100):
    palavras = texto.split()
    return [" ".join(palavras[i:i + tamanho]) for i in range(0, len(palavras), tamanho)]

# Função para gerar embeddings com BERT
@torch.no_grad()
def gerar_embedding(texto):
    inputs = bert_tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token
    return embedding

# Treinamento inicial e salvamento
def treinar_modelo_bert(json_path):
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        dados = json.load(f)

    textos = [preprocessar_texto(d["texto"]) for d in dados]
    rotulos = [d["rotulo"] for d in dados]

    embeddings = []
    for texto in tqdm(textos, desc="Gerando embeddings BERT"):
        emb = gerar_embedding(texto)
        embeddings.append(emb)

    X = np.array(embeddings)
    y = np.array(rotulos)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\n--- Relatório de Classificação (validação) ---")
    print(classification_report(y_val, y_pred, digits=4))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Modelo salvo em: {MODEL_PATH}")

# Carregar modelo salvo
def carregar_modelo_bert():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo BERT ainda não foi treinado. Execute treinar_modelo_bert primeiro.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Verifica se é contrato com votação por trecho
def verificar_contrato_com_bert(texto, limiar=0.7, minimo_votos=1):
    texto = "CONTRATO DE LOCAÇÃO DE IMÓVEL RESIDENCIAL Pelo presente instrumento particular, as partes: LOCADOR: João da Silva, brasileiro, solteiro, FPC n° 00000000000, residente à Rua das Flores, n° 100, Bairro Centro, Cidade XYZ; LOCATÁRIO: Maria Oliveira, brasileira, solteira, FPC n° 11111111111, residente na Rua dos Lírios, n° 200, Bairro Jardim, Cidade XYZ; Têm entre si, justo e contratado o seguinte: CLÁUSULA PRIMEIRA - DO OBJETO: O LOCADOR dá em locação ao LOCATÁRIO o imóvel residencial situado na Rua das Acácias, n° 300, Bairro Primavera, Cidade XYZ, exclusivamente para fins residenciais. CLÁUSULA SEGUNDA - DO PRAZO: O prazo da locação será de 12 (doze) meses, com início em 01 de maio de 2024 e término em 30 de abril de 2025. CLÁUSULA TERCEIRA - DO VALOR: O aluguel mensal será de R$ 1.500,00 (mil e quinhentos reais), com vencimento no dia 5 (cinco) de cada mês. CLÁUSULA QUARTA - DOS ENCARGOS: Serão de responsabilidade do LOCATÁRIO todas as despesas de consumo (água, luz, gás, internet) e taxa de condomínio, se houver. CLÁUSULA QUINTA - DA CONSERVAÇÃO: O LOCATÁRIO compromete-se a manter o imóvel em perfeito estado de conservação e a devolvê- lo nas mesmas condições ao final do contrato. E por estarem de pleno acordo, assinam este instrumento em duas vias de igual teor e forma. Cidade XYZ, 01 de maio de 2024. João da Silva - LOCADOR Maria Oliveira - LOCATÁRIA"
    texto = preprocessar_texto(texto)
    print(f"[DEBUG] Texto pré-processado (resumo): {texto[:300]}...")
    trechos = dividir_em_trechos(texto, tamanho=100)
    modelo = carregar_modelo_bert()

    votos_positivos = 0
    for i, trecho in enumerate(trechos):
        if len(trecho.split()) < 20:
            continue
        emb = gerar_embedding(trecho)
        prob = modelo.predict_proba([emb])[0][1]
        print(f"[DEBUG] Trecho {i + 1}: Predição BERT = {prob:.6f}")
        if prob > limiar:
            votos_positivos += 1

    print(f"[INFO] Total de trechos positivos (BERT): {votos_positivos} de {len(trechos)}")
    return votos_positivos >= minimo_votos
