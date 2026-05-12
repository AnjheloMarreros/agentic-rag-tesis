from pathlib import Path
from typing import List

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "conocimiento_juridico"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    texto = " ".join(text.split()).strip()
    if not texto:
        return []

    chunks = []
    inicio = 0

    while inicio < len(texto):
        fin = min(inicio + chunk_size, len(texto))
        chunks.append(texto[inicio:fin])

        if fin == len(texto):
            break

        inicio = fin - overlap

    return chunks


def embed_texts(texts: List[str]):
    model = get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str):
    model = get_model()
    return model.encode([query], normalize_embeddings=True).tolist()[0]


def buscar_similares(query: str, n_results: int = 3):
    collection = get_collection()
    query_embedding = embed_query(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )