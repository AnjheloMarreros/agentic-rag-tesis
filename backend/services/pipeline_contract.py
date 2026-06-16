from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.services.logs import registrar_evento


def asegurar_lista(valor: Any) -> list[str]:
    if valor is None:
        return []
    if isinstance(valor, list):
        salida: list[str] = []
        for item in valor:
            texto = str(item).strip()
            if texto:
                salida.append(texto)
        return salida
    texto = str(valor).strip()
    return [texto] if texto else []


def extraer_fragmentos_texto(contexto_recuperado: list[dict[str, Any]] | None) -> list[str]:
    fragmentos: list[str] = []

    for item in contexto_recuperado or []:
        if isinstance(item, dict):
            fragmento = str(item.get("fragmento", "")).strip()
            if fragmento:
                fragmentos.append(fragmento)
        elif isinstance(item, str):
            texto = item.strip()
            if texto:
                fragmentos.append(texto)

    return fragmentos


def construir_prompt_caso(caso: dict[str, Any]) -> str:
    partes = [
        str(caso.get("titulo", "")).strip(),
        str(caso.get("enunciado", "")).strip(),
        " ".join(asegurar_lista(caso.get("contexto") or caso.get("context"))).strip(),
        " ".join(asegurar_lista(caso.get("instrucciones"))).strip(),
    ]
    return "\n".join([p for p in partes if p]).strip()


def construir_retroalimentacion(
    evaluacion: dict[str, Any],
    evaluacion_semantica: dict[str, Any],
) -> dict[str, Any]:
    resumen_rubrica = str(evaluacion.get("resumen", "")).strip()
    resumen_semantico = str(evaluacion_semantica.get("resumen", "")).strip()

    observaciones: list[str] = []
    recomendaciones: list[str] = []

    for item in evaluacion.get("criterios", []):
        nombre = item.get("nombre", item.get("clave", "Criterio"))
        observacion = str(item.get("observacion", "")).strip()
        recomendacion = str(item.get("recomendacion", "")).strip()
        if observacion:
            observaciones.append(f"Rúbrica - {nombre}: {observacion}")
        if recomendacion:
            recomendaciones.append(f"Rúbrica - {nombre}: {recomendacion}")

    for item in evaluacion_semantica.get("criterios", []):
        nombre = item.get("nombre", item.get("clave", "Criterio"))
        observacion = str(item.get("observacion", "")).strip()
        recomendacion = str(item.get("recomendacion", "")).strip()
        if observacion:
            observaciones.append(f"Semántica - {nombre}: {observacion}")
        if recomendacion:
            recomendaciones.append(f"Semántica - {nombre}: {recomendacion}")

    return {
        "estado": "evaluado",
        "resumen": " ".join([t for t in [resumen_rubrica, resumen_semantico] if t]).strip(),
        "observaciones": observaciones,
        "recomendaciones": recomendaciones,
    }


def construir_resultado_final(
    caso_id: str,
    tipo_entrada: str,
    caso: dict[str, Any],
    texto_procesado: str,
    evaluacion: dict[str, Any],
    evaluacion_semantica: dict[str, Any],
    contexto_recuperado: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    caso_publico = {
        "id": caso.get("id", caso_id),
        "titulo": caso.get("titulo", ""),
        "curso": caso.get("curso", ""),
        "enunciado": caso.get("enunciado", ""),
    }

    return {
        "caso_id": caso_id,
        "modo": "LangGraph + evaluación semántica",
        "tipo_entrada": tipo_entrada,
        "caso": caso_publico,
        "entrada_estudiante": texto_procesado,
        "contexto_recuperado": contexto_recuperado or [],
        "evaluacion": {
            "puntaje_total": evaluacion.get("puntaje_total", 0),
            "nivel_global": evaluacion.get("nivel_global", "N/D"),
            "resumen": evaluacion.get("resumen", ""),
            "criterios": evaluacion.get("criterios", []),
            "recomendaciones_generales": evaluacion.get("recomendaciones_generales", []),
        },
        "evaluacion_semantica": {
            "puntaje_total": evaluacion_semantica.get("puntaje_total", 0),
            "nivel_global": evaluacion_semantica.get("nivel_global", "N/D"),
            "resumen": evaluacion_semantica.get("resumen", ""),
            "similitud_caso": evaluacion_semantica.get("similitud_caso", 0.0),
            "similitud_contexto": evaluacion_semantica.get("similitud_contexto", 0.0),
            "criterios": evaluacion_semantica.get("criterios", []),
            "observaciones": evaluacion_semantica.get("observaciones", []),
            "recomendaciones": evaluacion_semantica.get("recomendaciones", []),
        },
    }


def registrar_ejecucion_pipeline(
    pipeline: str,
    caso_id: str,
    tipo_entrada: str,
    texto_procesado: str,
    caso: dict[str, Any],
    contexto_recuperado: list[dict[str, Any]],
    evaluacion: dict[str, Any],
    evaluacion_semantica: dict[str, Any],
    resultado_final: dict[str, Any],
) -> None:
    registrar_evento(
        f"evaluacion_{pipeline}",
        {
            "sample_id": f"{caso_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "case_id": caso_id,
            "caso_id": caso_id,
            "user_input": construir_prompt_caso(caso),
            "response": texto_procesado,
            "retrieved_contexts": extraer_fragmentos_texto(contexto_recuperado),
            "reference_contexts": asegurar_lista(caso.get("reference_contexts") or caso.get("contexto") or caso.get("context")),
            "reference": str(caso.get("reference_answer", "") or "").strip(),
            "source_type": tipo_entrada,
            "rubric_result": evaluacion,
            "semantic_result": evaluacion_semantica,
            "puntaje_total": evaluacion.get("puntaje_total", 0),
            "puntaje_semantico": evaluacion_semantica.get("puntaje_total", 0),
            "fuentes_recuperadas": len(contexto_recuperado or []),
            "resultado_final": resultado_final,
        },
    )