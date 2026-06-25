from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph
from backend.services.langchain_bridge import ejecutar_evaluacion_langchain
from backend.services.logs import registrar_evento


def _float_or_none(valor: Any) -> float | None:
    try:
        if valor is None:
            return None
        return float(valor)
    except Exception:
        return None


def _promedio(valores: list[Any]) -> float:
    numeros: list[float] = []
    for valor in valores:
        numero = _float_or_none(valor)
        if numero is not None:
            numeros.append(numero)

    if not numeros:
        return 0.0

    return round(sum(numeros) / len(numeros), 4)


def _nivel_global(puntaje: float) -> str:
    if puntaje >= 85:
        return "Excelente"
    if puntaje >= 70:
        return "Alto"
    if puntaje >= 55:
        return "Medio"
    return "Bajo"


def _extraer_resumen(resultado: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(resultado, dict):
        return {
            "ok": False,
            "pipeline": "N/D",
            "puntaje_total": 0.0,
            "nivel_global": "N/D",
            "puntaje_rubrica": 0.0,
            "puntaje_semantico": 0.0,
            "indice_relevancia_caso": 0.0,
            "indice_relevancia_lexica": 0.0,
            "error": "Resultado inválido",
        }

    cuerpo = resultado.get("resultado_final")
    if not isinstance(cuerpo, dict):
        cuerpo = resultado

    evaluacion = cuerpo.get("evaluacion") if isinstance(cuerpo.get("evaluacion"), dict) else {}
    semantica = (
        cuerpo.get("evaluacion_semantica")
        if isinstance(cuerpo.get("evaluacion_semantica"), dict)
        else {}
    )

    resumen = resultado.get("resumen") if isinstance(resultado.get("resumen"), dict) else {}

    return {
        "ok": bool(resultado.get("ok", False)),
        "pipeline": resultado.get("pipeline", cuerpo.get("pipeline", "N/D")),
        "puntaje_total": _float_or_none(
            evaluacion.get("puntaje_total")
            or cuerpo.get("puntaje_total")
            or resumen.get("puntaje_total")
        ) or 0.0,
        "nivel_global": evaluacion.get("nivel_global") or cuerpo.get("nivel_global") or "N/D",
        "puntaje_rubrica": _float_or_none(
            evaluacion.get("puntaje_rubrica")
            or cuerpo.get("puntaje_rubrica")
            or resumen.get("puntaje_rubrica")
        ) or 0.0,
        "puntaje_semantico": _float_or_none(
            evaluacion.get("puntaje_semantico")
            or cuerpo.get("puntaje_semantico")
            or resumen.get("puntaje_semantico")
        ) or 0.0,
        "indice_relevancia_caso": _float_or_none(
            evaluacion.get("indice_relevancia_caso")
            or semantica.get("indice_relevancia_caso")
            or cuerpo.get("indice_relevancia_caso")
            or resumen.get("indice_relevancia_caso")
        ) or 0.0,
        "indice_relevancia_lexica": _float_or_none(
            semantica.get("indice_relevancia_lexica")
            or cuerpo.get("indice_relevancia_lexica")
            or resumen.get("indice_relevancia_lexica")
        ) or 0.0,
        "error": resultado.get("error", ""),
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


def _construir_resultado_benchmark(
    caso_id: str,
    tipo_entrada_original: str,
    texto_procesado: str,
    benchmark_id: str,
    sample_id: str,
    resultado_langgraph: dict[str, Any],
    resultado_langchain: dict[str, Any],
) -> dict[str, Any]:
    resumen_langgraph = _extraer_resumen(resultado_langgraph)
    resumen_langchain = _extraer_resumen(resultado_langchain)

    exitosos = [
        resumen
        for resumen in (resumen_langgraph, resumen_langchain)
        if resumen.get("ok")
    ]

    puntaje_total = _promedio([r.get("puntaje_total") for r in exitosos])
    puntaje_rubrica = _promedio([r.get("puntaje_rubrica") for r in exitosos])
    puntaje_semantico = _promedio([r.get("puntaje_semantico") for r in exitosos])
    relevancia_caso = _promedio([r.get("indice_relevancia_caso") for r in exitosos])
    relevancia_lexica = _promedio([r.get("indice_relevancia_lexica") for r in exitosos])

    benchmark_evaluacion = {
        "puntaje_total": puntaje_total,
        "nivel_global": _nivel_global(puntaje_total),
        "puntaje_rubrica": puntaje_rubrica,
        "puntaje_semantico": puntaje_semantico,
        "indice_relevancia_caso": relevancia_caso,
        "indice_relevancia_lexica": relevancia_lexica,
        "resumen": (
            f"Benchmark comparativo ejecutado para {caso_id}. "
            f"LangGraph: {resumen_langgraph.get('puntaje_total', 0.0)}; "
            f"LangChain: {resumen_langchain.get('puntaje_total', 0.0)}."
        ),
    }

    benchmark_retroalimentacion = {
        "estado": "comparado",
        "resumen": benchmark_evaluacion["resumen"],
        "observaciones": [
            f"LangGraph: {resumen_langgraph.get('nivel_global', 'N/D')} "
            f"({resumen_langgraph.get('puntaje_total', 0.0)}).",
            f"LangChain: {resumen_langchain.get('nivel_global', 'N/D')} "
            f"({resumen_langchain.get('puntaje_total', 0.0)}).",
        ],
        "recomendaciones": [
            "Compara ambos pipelines con el mismo benchmark_id y sample_id.",
            "Usa estos resultados para documentar la corrida en la tesis.",
        ],
    }

    benchmark_resultado_final = {
        "caso_id": caso_id,
        "sample_id": sample_id,
        "benchmark_id": benchmark_id,
        "modo": "Benchmark",
        "pipeline": "benchmark",
        "tipo_entrada": tipo_entrada_original,
        "entrada": texto_procesado,
        "comparativa": {
            "langgraph": resumen_langgraph,
            "langchain": resumen_langchain,
        },
        "evaluacion": benchmark_evaluacion,
        "retroalimentacion": benchmark_retroalimentacion,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "ok": bool(resumen_langgraph.get("ok") or resumen_langchain.get("ok")),
        "status": "completed" if (resumen_langgraph.get("ok") or resumen_langchain.get("ok")) else "failed",
        "pipeline": "benchmark",
        "benchmark_id": benchmark_id,
        "sample_id": sample_id,
        "caso_id": caso_id,
        "tipo_entrada_original": tipo_entrada_original,
        "texto_procesado": texto_procesado,
        "benchmark": {
            "ok": True,
            "pipeline": "benchmark",
            "benchmark_id": benchmark_id,
            "sample_id": sample_id,
            "caso_id": caso_id,
            "tipo_entrada_original": tipo_entrada_original,
            "texto_procesado": texto_procesado,
            "resultado_final": benchmark_resultado_final,
            "resumen": benchmark_evaluacion,
        },
        "langgraph": resultado_langgraph,
        "langchain": resultado_langchain,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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

    payload = _construir_resultado_benchmark(
        caso_id=caso_id,
        tipo_entrada_original=tipo_entrada_original,
        texto_procesado=texto_procesado,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
        resultado_langgraph=resultado_langgraph,
        resultado_langchain=resultado_langchain,
    )

    registrar_evento(
        "benchmark_dual_ejecucion",
        payload,
    )

    return payload