import time
import numpy as np
import faiss

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
def buscar_exacto(v_pregunta, vectores, k=5):
    """Compara la pregunta con cada chunk y devuelve los k más parecidos.
    Devuelve una lista de (posición, similitud).
    Ya he normalizado los vectores por lo que para calcular el angulo se hace solo con es producto escalar"""
    similitudes = []

    for posicion in range(len(vectores)):
        similitud = np.dot(vectores[posicion], v_pregunta) 
        similitudes.append((similitud, posicion)) 

    similitudes.sort(reverse=True)                     
    return similitudes[:k]


def guardar_indice(carpeta, chunks, vectores, nombre_modelo):
    """Guarda los vectores y los chunks en disco."""
    lista_chunks = []
    
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)                 
    #guardo los vectores 
    np.save(carpeta / "vectores.npy", vectores)
    #y los chunks como diccionarios json en la carpeta 
    for chunk in chunks:
        lista_chunks.append(asdict(chunk))
    datos = {"modelo": nombre_modelo, "chunks": lista_chunks}
    with open(carpeta / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def cargar_indice(carpeta, nombre_modelo):
    """Carga los vectores y los chunks guardados. Comprueba que son del mismo modelo."""
    carpeta = Path(carpeta)

    with open(carpeta / "chunks.json", encoding="utf-8") as f:
        datos = json.load(f)

    if datos["modelo"] != nombre_modelo:
        raise ValueError(f"El índice se creó con {datos['modelo']}, no con {nombre_modelo}")

    chunks = []
    for diccionario in datos["chunks"]:
        chunks.append(Chunk(**diccionario))

    vectores = np.load(carpeta / "vectores.npy")
    return chunks, vectores