import re

import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import cv2
from spellchecker import SpellChecker


from pytesseract import pytesseract

spell = SpellChecker(language="pt")

def corrigir_acentos_simples(texto):
    palavras = texto.split()
    corrigidas = [spell.correction(palavra) or palavra for palavra in palavras]
    return " ".join(corrigidas)

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)  # remove pontuação
    texto = re.sub(r"\s+", " ", texto)  # remove espaços duplicados
    texto = re.sub(r"\d{1,2}[a-z]{3,}", "", texto)  # remover datas tipo "7june"
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)  # remove pontuação especial
    texto = re.sub(r"\s+", " ", texto)
    # Correção ortográfica (leve)
    texto = corrigir_acentos_simples(texto)  # usa spellchecker
    return texto.strip()


def extrair_texto_pdf(caminho_arquivo):
    texto_extraido = ""
    texto_limpo = ""
    imagens = convert_from_path(caminho_arquivo)

    for i, imagem in enumerate(imagens):
        # Converter imagem PIL para formato OpenCV (NumPy array)
        imagem_cv = cv2.cvtColor(np.array(imagem), cv2.COLOR_RGB2BGR)

        # Pré-processamento: converter para escala de cinza
        imagem_cinza = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2GRAY)
        imagem_cinza = cv2.medianBlur(imagem_cinza, 3)   # remove ruído

        # Binarizar (isso ajuda o OCR a detectar melhor os textos)
        _, imagem_bin = cv2.threshold(imagem_cinza, 150, 255, cv2.THRESH_BINARY)

        # Converter de volta para formato PIL para passar ao pytesseract
        imagem_final = Image.fromarray(imagem_bin)

        # Extrair texto com pytesseract
        texto = pytesseract.image_to_string(imagem_final, lang="por")

        texto_extraido += f"\n--- Página {i + 1} ---\n{texto}"
        #texto_limpo = limpar_texto(texto_extraido)

    return texto_extraido
