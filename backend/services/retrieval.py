from backend.services.vector_store import buscar_similares


def recuperar_contexto(consulta: str, n_resultados: int = 3):
    return buscar_similares(consulta, n_resultados)