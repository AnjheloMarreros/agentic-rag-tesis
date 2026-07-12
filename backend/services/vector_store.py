from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "conocimiento_juridico"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_client() -> PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return PersistentClient(path=str(CHROMA_DIR))


def get_collection(collection_name: str | None = None):
    client = get_client()
    return client.get_or_create_collection(name=collection_name or COLLECTION_NAME)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    texto = " ".join((text or "").split()).strip()
    if not texto:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks: List[str] = []
    inicio = 0

    while inicio < len(texto):
        fin = min(inicio + chunk_size, len(texto))
        chunks.append(texto[inicio:fin])
        if fin >= len(texto):
            break
        inicio = max(0, fin - overlap)

    return chunks


def embed_texts(texts: List[str]):
    model = get_model()
    textos = [t or "" for t in texts]
    return model.encode(textos, normalize_embeddings=True).tolist()


def embed_query(query: str):
    model = get_model()
    return model.encode([query or ""], normalize_embeddings=True).tolist()[0]


def _tiene_resultados(resultado: dict[str, Any]) -> bool:
    documentos = resultado.get("documents")
    if not isinstance(documentos, list):
        return False

    for bloque in documentos:
        if isinstance(bloque, list) and len(bloque) > 0:
            return True

    return False


def _query_collection(
    *,
    collection,
    query_embedding,
    n_results: int,
    where: dict[str, Any] | None = None,
):
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    return collection.query(**query_kwargs)


def buscar_similares(
    query: str,
    n_results: int = 3,
    case_id: str | None = None,
    source_type: str | None = None,
):
    collection = get_collection()
    query_embedding = embed_query(query)

    filtros: dict[str, Any] = {}
    if case_id:
        filtros["case_id"] = str(case_id).strip()
    if source_type:
        filtros["source_type"] = str(source_type).strip()

    try:
        if filtros:
            resultado_filtrado = _query_collection(
                collection=collection,
                query_embedding=query_embedding,
                n_results=n_results,
                where=filtros,
            )
            if _tiene_resultados(resultado_filtrado):
                return resultado_filtrado

        return _query_collection(
            collection=collection,
            query_embedding=query_embedding,
            n_results=n_results,
            where=None,
        )
    except Exception:
        # Fallback conservador: si falla el filtro o la consulta, mantenemos el flujo previo.
        try:
            return _query_collection(
                collection=collection,
                query_embedding=query_embedding,
                n_results=n_results,
                where=None,
            )
        except Exception:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }


def agregar_documentos(
    documentos: List[str],
    metadatas: List[dict[str, Any]] | None = None,
    ids: List[str] | None = None,
):
    if not documentos:
        return

    collection = get_collection()
    embeddings = embed_texts(documentos)

    if ids is None:
        ids = [f"doc_{uuid4().hex[:12]}_{i}" for i in range(len(documentos))]

    if metadatas is None:
        metadatas = [{} for _ in documentos]

    collection.add(
        documents=documentos,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )