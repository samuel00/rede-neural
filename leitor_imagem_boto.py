import boto3


def extrair_texto_imagem_pdf(caminho_pdf):

    client = boto3.client("textract")

    with open("contrato.pdf", "rb") as file:
        response = client.analyze_document(
            Document={"Bytes": file.read()},
            FeatureTypes=["TABLES", "FORMS"]
    )

# Extrai o texto
    texto = " ".join([block["Text"] for block in response["Blocks"] if block["BlockType"] == "WORD"])
    return texto

if __name__ == '__main__':
    caminho_pdf = "contrato.pdf"
    texto = extrair_texto_imagem_pdf(caminho_pdf)
    print(texto)
