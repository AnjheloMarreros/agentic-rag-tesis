from pathlib import Path
import re

from backend.services.knowledge_loader import cargar_documentos
from backend.services.vector_store import (
    get_client,
    get_collection,
    chunk_text,
    embed_texts,
    COLLECTION_NAME,
)


def _safe_id_component(value: str) -> str:
    value = value.strip()
    value = value.replace("\\", "/")
    value = value.replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "documento"


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

    for doc in documentos:
        source_id = _safe_id_component(doc["source"])
        chunks = chunk_text(doc["text"], chunk_size=500, overlap=100)

        for i, chunk in enumerate(chunks, start=1):
            if not chunk or not chunk.strip():
                continue

            textos.append(chunk)
            ids.append(f"{source_id}_chunk_{i}")
            metadatos.append(
                {
                    "source": doc["source"],
                    "type": doc["type"],
                    "chunk": i,
                }
            )

    if not textos:
        raise ValueError("No hay textos para indexar en la base vectorial.")

    embeddings = embed_texts(textos)

    collection.add(
        ids=ids,
        documents=textos,
        metadatas=metadatos,
        embeddings=embeddings,
    )

    print("Base vectorial creada correctamente.")
    print(f"Documentos indexados: {len(documentos)}")
    print(f"Fragmentos almacenados: {len(textos)}")


if __name__ == "__main__":
    construir_base_conocimiento()