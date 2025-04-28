import nltk
from nltk.corpus import stopwords
import re

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

def corrigir_valores_monetarios(texto):
    """
    Corrige padrões como 'rs 1 500 00' para 'R$ 1500,00'
    """
    padrao = re.compile(r"\brs\s+((?:\d{1,3}\s*)+)\b", flags=re.IGNORECASE)

    def formatar(match):
        numeros = match.group(1).strip().replace(" ", "")
        if len(numeros) > 2:
            return f"R$ {numeros[:-2]},{numeros[-2:]}"
        return f"R$ {numeros}"

    return padrao.sub(formatar, texto)

def limpar_ocr(texto):
    """
    Corrige ruídos comuns de OCR usando regex genéricos
    """
    substituicoes = {
        r"\bcl[ \-_]?usula\b": "cláusula",
        r"\bim[ \-_]?vel\b": "imóvel",
        r"\bcondi[ \-_]?es\b": "condições",
        r"\bt[ \-_]?rmino\b": "término",
        r"\bgu[ \-_]?a\b": "água",
        r"\bloca[ \-_]?[cç]ao\b": "locação",
        r"\bdevolv[ \-_]?lo\b": "devolvê-lo",
        r"\bresponsabilidade\b": "responsabilidade",
        r"\bresiden[ \-_]?cia[ls]?\b": "residência",
    }

    for padrao, subst in substituicoes.items():
        texto = re.sub(padrao, subst, texto, flags=re.IGNORECASE)
    return texto

def preprocessar_texto(texto):
    #texto = " ".join([palavra for palavra in texto.lower().split() if palavra not in STOPWORDS])
    return texto.strip()

