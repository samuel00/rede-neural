import os
import shutil
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


import fitz  # PyMuPDF para PDFs
import docx
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

# Diretórios
DIRETORIO_ORIGEM = "documentos"
DIRETORIO_DESTINO = "contratos"

# Criar diretório de destino se não existir
if not os.path.exists(DIRETORIO_DESTINO):
    os.makedirs(DIRETORIO_DESTINO)


def extrair_texto_pdf(caminho_arquivo):
    """Extrai texto de um arquivo PDF."""
    texto = ""
    try:
        with fitz.open(caminho_arquivo) as doc:
            for pagina in doc:
                texto += pagina.get_text()
    except Exception as e:
        print(f"Erro ao ler PDF {caminho_arquivo}: {e}")
    return texto


def extrair_texto_docx(caminho_arquivo):
    """Extrai texto de um arquivo DOCX."""
    texto = ""
    try:
        doc = docx.Document(caminho_arquivo)
        texto = "\n".join(paragrafo.text for paragrafo in doc.paragraphs)
    except Exception as e:
        print(f"Erro ao ler DOCX {caminho_arquivo}: {e}")
    return texto


def preprocessar_texto(texto):
    """Pré-processa o texto removendo stopwords e convertendo para minúsculas."""
    return " ".join([palavra for palavra in texto.lower().split() if palavra not in STOPWORDS])


# Conjunto de treinamento manual (idealmente, use mais dados!)
DADOS_TREINAMENTO = [
    ("Este contrato de negociação de cotas de consórcio estabelece as regras...", 1),
    ("Contrato de adesão ao consórcio do banco XYZ...", 1),
    ("Nota fiscal referente à compra de equipamentos...", 0),
    ("Orçamento de um carro financiado...", 0),
    ("Termos e condições de um aplicativo...", 0),
]

# Separar textos e rótulos
textos, rotulos = zip(*DADOS_TREINAMENTO)

# Tokenização
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(textos)
sequences = tokenizer.texts_to_sequences(textos)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

# Criar modelo de rede neural
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária (1 = contrato, 0 = outro)
])

modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo.fit(padded_sequences, np.array(rotulos), epochs=10, verbose=2)

# Salvar o modelo treinado
modelo.save("modelo_contratos_nn.h5")


def verificar_contrato(texto):
    """Usa o modelo de rede neural para prever se o documento é um contrato."""
    texto = preprocessar_texto(texto)
    seq = tokenizer.texts_to_sequences([texto])
    pad_seq = pad_sequences(seq, maxlen=100, padding='post')
    modelo = tf.keras.models.load_model("modelo_contratos_nn.h5")
    predicao = modelo.predict(pad_seq)[0][0]
    return predicao > 0.5  # Se for maior que 0.5, classifica como contrato


# Processar arquivos
for arquivo in os.listdir(DIRETORIO_ORIGEM):
    caminho_arquivo = os.path.join(DIRETORIO_ORIGEM, arquivo)

    if arquivo.lower().endswith(".pdf"):
        texto = extrair_texto_pdf(caminho_arquivo)
    elif arquivo.lower().endswith(".docx"):
        texto = extrair_texto_docx(caminho_arquivo)
    else:
        continue  # Ignorar arquivos de outros formatos

    if verificar_contrato(texto):
        destino = os.path.join(DIRETORIO_DESTINO, arquivo)
        shutil.move(caminho_arquivo, destino)
        print(f"Movido: {arquivo}")

print("Processamento concluído.")
