from __future__ import annotations

from backend.services.vector_store import buscar_similares


def recuperar_contexto(
    consulta: str,
    n_resultados: int = 3,
    case_id: str | None = None,
    source_type: str | None = None,
):
    return buscar_similares(
        consulta,
        n_results=n_resultados,
        case_id=case_id,
        source_type=source_type,
    )