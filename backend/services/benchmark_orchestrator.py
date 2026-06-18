from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph
from backend.services.langchain_bridge import ejecutar_evaluacion_langchain
from backend.services.logs import registrar_evento


def _extraer_resumen(resultado: dict[str, Any]) -> dict[str, Any]:
    evaluacion = resultado.get("evaluacion", {}) if isinstance(resultado, dict) else {}
    semantica = resultado.get("evaluacion_semantica", {}) if isinstance(resultado, dict) else {}

    return {
        "puntaje_total": evaluacion.get("puntaje_total", 0),
        "nivel_global": evaluacion.get("nivel_global", "N/D"),
        "puntaje_rubrica": evaluacion.get("puntaje_rubrica", 0),
        "puntaje_semantico": evaluacion.get("puntaje_semantico", 0),
        "indice_relevancia_caso": semantica.get("indice_relevancia_caso", 0.0),
        "indice_relevancia_lexica": semantica.get("indice_relevancia_lexica", 0.0),
    }


def _ejecutar_seguro(nombre: str, fn, *args, **kwargs) -> dict[str, Any]:
    try:
        resultado = fn(*args, **kwargs)
        return {
            "ok": True,
            "pipeline": nombre,
            "resultado_final": resultado,
            "resumen": _extraer_resumen(resultado if isinstance(resultado, dict) else {}),
        }
    except Exception as exc:
        return {
            "ok": False,
            "pipeline": nombre,
            "error": str(exc),
            "resultado_final": None,
            "resumen": {},
        }


def ejecutar_benchmark_dual(
    caso_id: str,
    tipo_entrada_original: str,
    texto_procesado: str,
    benchmark_id: str | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    benchmark_id = benchmark_id or f"bm_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex[:8]}"
    sample_id = sample_id or f"{caso_id}_{uuid4().hex[:8]}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_langgraph = executor.submit(
            _ejecutar_seguro,
            "langgraph",
            ejecutar_evaluacion_langgraph,
            caso_id,
            "texto",
            texto_procesado,
            "",
            benchmark_id,
            sample_id,
        )
        future_langchain = executor.submit(
            _ejecutar_seguro,
            "langchain",
            ejecutar_evaluacion_langchain,
            caso_id,
            "texto",
            texto_procesado,
            "",
            benchmark_id,
            sample_id,
        )

        resultado_langgraph = future_langgraph.result()
        resultado_langchain = future_langchain.result()

    payload = {
        "benchmark_id": benchmark_id,
        "sample_id": sample_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "caso_id": caso_id,
        "tipo_entrada_original": tipo_entrada_original,
        "texto_procesado": texto_procesado,
        "langgraph": resultado_langgraph,
        "langchain": resultado_langchain,
    }

    registrar_evento(
        "benchmark_dual_ejecucion",
        payload,
    )

    return payload