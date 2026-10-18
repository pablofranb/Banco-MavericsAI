import json
from pathlib import Path

from Preguntas_frecuentes.rag.cargar import cargar_pdf
from Preguntas_frecuentes.rag.limpiar import limpiar
from Preguntas_frecuentes.rag.trocear import trocear
from Preguntas_frecuentes.rag.embeddings import cargar_modelo, crear_embeddings
from Preguntas_frecuentes.rag.indice import buscar_exacto

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


def recall_por_k(ruta_preguntas, chunks, vectores, modelo, k_maximo=10):
    vectores_preguntas = []
    recalls = []
    ids_encontrados = []
    preguntas = []
    # 1. Convertir cada pregunta en vector
    preguntas =cargar_preguntas(ruta_preguntas)
    for pregunta  in preguntas:
        v = modelo.encode(["query: " + pregunta[1]], normalize_embeddings=True)[0]
        vectores_preguntas.append(v)

    # 2. Para cada k, recorrer las preguntas y contar cuántas aciertan
    for k in range(1, k_maximo + 1):
        aciertos = 0
        for i in range(len(preguntas)):
            esperado = preguntas[i][1][0] #pillo la pregunta, la segunda tupla con los id y como alguno tiene 2 cojo el texto de la primera posicion
            resultados = buscar_exacto(vectores_preguntas[i], vectores, k)
            for resultado in resultados:
                ids_encontrados.append(chunks[resultado[1]].id)
            if esperado in ids_encontrados:
                aciertos += 1

        recalls.append(aciertos / len(preguntas))         # el Recall de este k
        print(f"k={k:2} | {aciertos} de {len(preguntas)} | {recalls[-1]:.1%}")

    return recalls


def main():
    chunks = []
    lineas = []
    modelo = None
    vectores = None
    preguntas = []

    for ruta in sorted(Path("../datos").glob("*.pdf")):
        lineas = cargar_pdf(ruta)
        lineas = limpiar(lineas)
        chunks += trocear(lineas, ruta.name)

    modelo = cargar_modelo()
    vectores = crear_embeddings(chunks, modelo)
    ruta = "../evaluacion/casos_desarrollo.jsonl"
    recall_por_k(ruta, chunks, vectores, modelo)


if __name__ == "__main__":
    main()