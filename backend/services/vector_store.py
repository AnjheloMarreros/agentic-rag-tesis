from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "conocimiento_juridico"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_client() -> PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    texto = " ".join((text or "").split()).strip()
    if not texto:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0.")
    if overlap < 0:
        raise ValueError("overlap no puede ser negativo.")
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size.")

    chunks: List[str] = []
    inicio = 0
    largo = len(texto)

    while inicio < largo:
        fin = min(inicio + chunk_size, largo)
        chunks.append(texto[inicio:fin])

        if fin >= largo:
            break

        inicio = fin - overlap

    return chunks


def embed_texts(texts: List[str]):
    valid_texts = [(text or "").strip() for text in texts if (text or "").strip()]
    if not valid_texts:
        return []

    model = get_model()
    return model.encode(valid_texts, normalize_embeddings=True).tolist()


def embed_query(query: str):
    texto = (query or "").strip()
    if not texto:
        raise ValueError("query no puede estar vacía.")

    model = get_model()
    return model.encode([texto], normalize_embeddings=True).tolist()[0]


def buscar_similares(query: str, n_results: int = 3):
    if n_results <= 0:
        raise ValueError("n_results debe ser mayor que 0.")

    collection = get_collection()
    query_embedding = embed_query(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )