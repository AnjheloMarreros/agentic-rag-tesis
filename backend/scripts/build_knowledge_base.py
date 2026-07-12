from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Optional

from backend.services.knowledge_loader import cargar_documentos
from backend.services.vector_store import (
    get_client,
    get_collection,
    chunk_text,
    embed_texts,
    COLLECTION_NAME,
)

CASE_ID_PATTERN = re.compile(r"(caso_\d{3})", re.IGNORECASE)


def _safe_id_component(value: str) -> str:
    value = value.strip()
    value = value.replace("\\", "/")
    value = value.replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "documento"


def _extraer_case_id(source: str, scope: Optional[str] = None) -> str | None:
    texto = f"{source or ''} {scope or ''}".strip()
    match = CASE_ID_PATTERN.search(texto)
    if match:
        return match.group(1).lower()
    return None


def _metadata_base(doc: dict[str, Any]) -> dict[str, Any]:
    source = str(doc.get("source", "")).strip()
    doc_type = str(doc.get("type", "")).strip()
    scope = str(doc.get("scope", "knowledge")).strip()

    metadata: dict[str, Any] = {
        "source": source,
        "type": doc_type,
        "scope": scope,
    }

    case_id = _extraer_case_id(source, scope)
    if case_id:
        metadata["case_id"] = case_id

    if "title" in doc and isinstance(doc["title"], str) and doc["title"].strip():
        metadata["title"] = doc["title"].strip()

    return metadata


def construir_base_conocimiento(include_pedagogical: bool = False):
    client = get_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection()
    documentos = cargar_documentos(include_pedagogical=include_pedagogical)

    textos: list[str] = []
    ids: list[str] = []
    metadatos: list[dict[str, Any]] = []

    for doc in documentos:
        source = str(doc.get("source", "")).strip()
        if not source:
            continue

        source_id = _safe_id_component(source)
        chunks = chunk_text(str(doc.get("text", "")), chunk_size=500, overlap=100)

        if not chunks:
            continue

        base_metadata = _metadata_base(doc)

        for i, chunk in enumerate(chunks, start=1):
            if not chunk or not chunk.strip():
                continue

            textos.append(chunk)
            ids.append(f"{source_id}_chunk_{i}")

            chunk_metadata = dict(base_metadata)
            chunk_metadata["chunk"] = i
            chunk_metadata["chunk_count"] = len(chunks)
            metadatos.append(chunk_metadata)

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
    print(f"Fragmentos con case_id: {sum(1 for m in metadatos if m.get('case_id'))}")


if __name__ == "__main__":
    construir_base_conocimiento()