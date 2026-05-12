from backend.services.knowledge_loader import cargar_documentos
from backend.services.vector_store import (
    get_client,
    get_collection,
    chunk_text,
    embed_texts,
    COLLECTION_NAME
)


def construir_base_conocimiento():
    client = get_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection()
    documentos = cargar_documentos()

    textos = []
    ids = []
    metadatos = []

    contador_global = 1

    for doc in documentos:
        chunks = chunk_text(doc["text"], chunk_size=500, overlap=100)

        for i, chunk in enumerate(chunks, start=1):
            textos.append(chunk)
            ids.append(f"{doc['source']}_{i}")
            metadatos.append({
                "source": doc["source"],
                "type": doc["type"],
                "chunk": i
            })
            contador_global += 1

    if not textos:
        raise ValueError("No hay textos para indexar en la base vectorial.")

    embeddings = embed_texts(textos)

    collection.add(
        ids=ids,
        documents=textos,
        metadatas=metadatos,
        embeddings=embeddings
    )

    print("Base vectorial creada correctamente.")
    print(f"Documentos indexados: {len(documentos)}")
    print(f"Fragmentos almacenados: {len(textos)}")


if __name__ == "__main__":
    construir_base_conocimiento()