import json

from pathlib import Path

script_dir = Path(__file__).resolve().parent

def handle_text():
    # Suponha que o JSON esteja carregado na variável `dados`
    print(script_dir.parent / 'dados' / 'contratos_treinamento.json')
    with open(script_dir.parent /'rede_neural' / 'dados' / 'ocr_itau.json', 'r', encoding='utf-8') as f:
        dados = json.load(f)

    # Acessa o array de palavras dentro de response[0]['only_text']
    only_text = dados['data']['response'][0]['only_text']

    # Junta todas as palavras com espaço entre elas
    texto_unico = ' '.join(only_text)

    return texto_unico

if __name__ == "__main__":
    print(handle_text())