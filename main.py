import pytesseract
from pdf2image import convert_from_path

# Se precisar, defina o caminho do Tesseract no Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extrair_texto_imagem_pdf(caminho_pdf):
    """Extrai texto de imagens dentro de um PDF usando OCR."""
    texto_extraido = ""

    # Converter PDF em imagens
    imagens = convert_from_path(caminho_pdf)

    for i, imagem in enumerate(imagens):
        texto = pytesseract.image_to_string(imagem, lang="por")  # Define o idioma para português
        texto_extraido += f"\n--- Página {i+1} ---\n{texto}"

    return texto_extraido.strip()

# Testando com um PDF
if __name__ == '__main__':
    caminho_pdf = "contrato.pdf"
    texto = extrair_texto_imagem_pdf(caminho_pdf)
    print(texto)
