from rede_neural.src.bert_classificador import treinar_modelo_bert
from pathlib import Path

if __name__ == "__main__":
    caminho_json = Path(__file__).resolve().parent.parent / "dados" / "contratos_treinamento.json"
    print(f"Treinando modelo BERT com base: {caminho_json}")
    treinar_modelo_bert(str(caminho_json))
