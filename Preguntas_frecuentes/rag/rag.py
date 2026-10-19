"""
rag.py — Programa principal del RAG de preguntas frecuentes.

Une las piezas (cargar, limpiar, trocear, embeddings, índice) en el orden correcto.
Es lo que importan evaluar.py, el router y la app de Streamlit.
"""
from pathlib import Path

from cargar import cargar_pdf
from limpiar import limpiar
from trocear import trocear
from embeddings import crear_embeddings
from indice import guardar_indice, cargar_indice
from indice import buscar_exacto       


def preparar_indice(carpeta_pdfs, carpeta_indice, modelo, nombre_modelo):
    """Prepara el índice para poder buscar.

    1. Si el índice ya existe en carpeta_indice, lo carga (rápido).
    2. Si no existe, lo crea a partir de los PDFs (cargar, limpiar, trocear y
       embeddings) y lo guarda para la próxima vez.
    Devuelve los chunks y los vectores, en el mismo orden.
    """
    chunks = []
    vectores = None
    lineas = []

    # 1. Si ya existe, cargarlo
    if (Path(carpeta_indice) / "chunks.json").exists():
        print("Cargando índice guardado...")
        return cargar_indice(carpeta_indice, nombre_modelo)

    # 2. Si no existe, crearlo a partir de los PDFs, PDF a PDF
    print("No hay índice: creándolo a partir de los PDFs...")
    for ruta in sorted(Path(carpeta_pdfs).glob("*.pdf")):
        lineas = cargar_pdf(ruta)
        lineas = limpiar(lineas)
        chunks += trocear(lineas, ruta.name)

    vectores = crear_embeddings(chunks, modelo)
    guardar_indice(carpeta_indice, chunks, vectores, nombre_modelo)
    return chunks, vectores

def recuperar(pregunta, chunks, vectores, modelo, k=5, umbral=0.82):
    """Busca los chunks más relevantes para la pregunta.

    Devuelve una lista de (chunk, similitud), de mayor a menor similitud.
    Si ni el mejor resultado llega al umbral, la pregunta está fuera de tema
    y devuelve una lista vacía (así no se llama al LLM).
    """
    encontrados = []

    v_pregunta = modelo.encode(["query: " + pregunta], normalize_embeddings=True)[0]
    resultados = buscar_exacto(v_pregunta, vectores, k)
    # 3. Umbral: si el mejor no llega, fuera de tema
    mejor_similitud = resultados[0][0]
    if mejor_similitud < umbral:
        return []

    # 4. Devolver los chunks con su similitud
    for similitud, posicion in resultados:
        encontrados.append((chunks[posicion], similitud))
    return encontrados