import os
import shutil

from rede_neural.src.extrator_texto import extrair_texto_pdf
from rede_neural.src.bert_classificador import verificar_contrato_com_bert
from rede_neural.src.modelo_old import carregar_modelo, verificar_contrato

DIRETORIO_ORIGEM = "/home/samuel/Documents/rede_neural/origem"
DIRETORIO_DESTINO = "/home/samuel/Documents/rede_neural/destino"

def texto_legivel(texto):
    palavras = texto.split()
    return len([p for p in palavras if len(p) > 3]) >= 20  # ex: pelo menos 20 palavras “úteis”

def processar_documentos(modo_modelo="lstm"):  # lstm ou bert
    if not os.path.exists(DIRETORIO_DESTINO):
        os.makedirs(DIRETORIO_DESTINO)

    if modo_modelo == "lstm":
        carregar_modelo()  # só carrega se for LSTM

    for arquivo in os.listdir(DIRETORIO_ORIGEM):
        caminho_arquivo = os.path.join(DIRETORIO_ORIGEM, arquivo)

        if not arquivo.lower().endswith(".pdf"):
            continue

        texto = extrair_texto_pdf(caminho_arquivo)

        if not texto_legivel(texto):
            print(f"[AVISO] {arquivo} – Texto muito corrompido, não será classificado")
            continue

        print(f"Arquivo: {arquivo}")

        if modo_modelo == "lstm":
            resultado = verificar_contrato(texto)
        elif modo_modelo == "bert":
            resultado = verificar_contrato_com_bert(texto)
        else:
            raise ValueError("Modo inválido: use 'lstm' ou 'bert'")

        if resultado:
            destino = os.path.join(DIRETORIO_DESTINO, arquivo)
            shutil.move(caminho_arquivo, destino)
            print(f"Status: Movido ✅")
        else:
            print(f"Status: Ignorado ❌")

        print("#" * 120)

    print("Processamento concluído.")
