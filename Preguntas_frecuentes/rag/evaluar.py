import json
from pathlib import Path

from cargar import cargar_pdf
from limpiar import limpiar
from trocear import trocear
from embeddings import cargar_modelo, crear_embeddings
from indice import buscar_exacto

"""
Mis resultados
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:04<00:00,  1.85it/s]
 k | Recall@k
 1 |  62.8%
 2 |  75.6%
 3 |  83.7%
 4 |  84.9%
 5 |  89.5%
 6 |  89.5%
 7 |  91.9%
 8 |  91.9%
 9 |  91.9%
10 |  91.9%
 """
def cargar_preguntas(ruta):
    """cargo las prguntas"""
    preguntas = []
    caso = None
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            caso = json.loads(linea)
            preguntas.append((caso["question"], caso["expected_article_ids"]))
    return preguntas


def recall_por_k(preguntas, chunks, vectores, modelo, k_maximo=10):
    """Para cada k de 1 a k_maximo, % de preguntas cuyo artículo esperado sale en el top-k."""
    aciertos = [0] * (k_maximo + 1)
    v_pregunta = None
    resultados = []
    ids_encontrados = []

    for pregunta, ids_esperados in preguntas:
        v_pregunta = modelo.encode(["query: " + pregunta], normalize_embeddings=True)[0]
        resultados = buscar_exacto(v_pregunta, vectores, k_maximo)

        ids_encontrados = []
        for similitud, posicion in resultados:
            ids_encontrados.append(chunks[posicion].id)

        for k in range(1, k_maximo + 1):
            if ids_esperados[0] in ids_encontrados[:k]:
                aciertos[k] += 1

    print(" k | Recall@k")
    for k in range(1, k_maximo + 1):
        print(f"{k:2} | {aciertos[k] / len(preguntas):6.1%}")


def main():
    chunks = []
    lineas = []
    modelo = None
    vectores = None
    preguntas = []

    # Lo que ya tienes: PDFs → chunks → vectores
    for ruta in sorted(Path("../datos").glob("*.pdf")):
        lineas = cargar_pdf(ruta)
        lineas = limpiar(lineas)
        chunks += trocear(lineas, ruta.name)

    modelo = cargar_modelo()
    vectores = crear_embeddings(chunks, modelo)

    # Lo nuevo: medir
    preguntas = cargar_preguntas("../evaluacion/casos_desarrollo.jsonl")
    recall_por_k(preguntas, chunks, vectores, modelo)


if __name__ == "__main__":
    main()