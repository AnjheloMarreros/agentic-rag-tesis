from __future__ import annotations

from pathlib import Path
from typing import Any
import json

try:
    # En algunas versiones funciona aquí
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
except ImportError:
    # Fallback por si la instalación expone los objetos en otro sitio
    from ragas import EvaluationDataset, SingleTurnSample  # type: ignore

from backend.services.case_loader import cargar_caso


BASE_DIR = Path(__file__).resolve().parents[2]
LOG_FILE = BASE_DIR / "logs" / "eventos.jsonl"


def _leer_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    registros: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError:
                continue
    return registros


def _asegurar_lista(valor: Any) -> list[str]:
    if valor is None:
        return []

    if isinstance(valor, list):
        salida: list[str] = []
        for item in valor:
            if isinstance(item, str):
                texto = item.strip()
                if texto:
                    salida.append(texto)
            elif isinstance(item, dict):
                if "fragmento" in item:
                    texto = str(item.get("fragmento", "")).strip()
                    if texto:
                        salida.append(texto)
                else:
                    texto = str(item).strip()
                    if texto:
                        salida.append(texto)
            else:
                texto = str(item).strip()
                if texto:
                    salida.append(texto)
        return salida

    if isinstance(valor, str):
        texto = valor.strip()
        return [texto] if texto else []

    texto = str(valor).strip()
    return [texto] if texto else []


def _extraer_evento_y_payload(registro: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Soporta formatos de log como:
    - {"tipo": "...", "datos": {...}}
    - {"evento": "...", "data": {...}}
    - {"event_type": "...", "payload": {...}}
    - {"tipo": "...", ...campos_directos...}
    """
    if not isinstance(registro, dict):
        return "", {}

    tipo = (
        registro.get("tipo")
        or registro.get("evento")
        or registro.get("event_type")
        or registro.get("event")
        or ""
    )
    tipo = str(tipo).strip()

    payload = (
        registro.get("datos")
        or registro.get("data")
        or registro.get("payload")
        or registro.get("detalle")
        or registro.get("info")
    )

    if isinstance(payload, dict):
        return tipo, payload

    payload_directo = {
        k: v
        for k, v in registro.items()
        if k not in {"tipo", "evento", "event_type", "event", "timestamp", "fecha", "datetime"}
    }
    return tipo, payload_directo


def _construir_prompt_caso(caso: dict[str, Any]) -> str:
    partes = [
        str(caso.get("titulo", "")).strip(),
        str(caso.get("enunciado", "")).strip(),
        " ".join(_asegurar_lista(caso.get("contexto") or caso.get("context"))).strip(),
        " ".join(_asegurar_lista(caso.get("instrucciones"))).strip(),
    ]
    partes = [p for p in partes if p]
    return "\n".join(partes).strip()


def _construir_user_input(payload: dict[str, Any], caso: dict[str, Any]) -> str:
    """
    En tu flujo real, user_input debe representar el prompt del caso.
    Si el log ya lo contiene, se usa ese valor.
    """
    user_input_guardado = str(payload.get("user_input", "") or "").strip()
    respuesta_estudiante = str(payload.get("response", "") or "").strip()

    if user_input_guardado:
        return user_input_guardado

    prompt_caso = _construir_prompt_caso(caso)
    if respuesta_estudiante:
        return f"{prompt_caso}\n\nRespuesta del estudiante:\n{respuesta_estudiante}".strip()

    return prompt_caso


def _construir_respuesta_agente(payload: dict[str, Any]) -> str:
    """
    Intenta construir una respuesta evaluable para RAGAS a partir del log.
    Prioridad:
    1) texto final del agente si existiera
    2) resumen de rúbrica
    3) resumen semántico
    4) retroalimentación consolidada
    5) respuesta del estudiante como último recurso
    """
    candidatos_directos = [
        payload.get("assistant_response"),
        payload.get("final_response"),
        payload.get("feedback_text"),
        payload.get("salida"),
        payload.get("respuesta_agente"),
        payload.get("texto_feedback"),
    ]
    for candidato in candidatos_directos:
        if isinstance(candidato, str) and candidato.strip():
            return candidato.strip()

    partes: list[str] = []

    rubric_result = payload.get("rubric_result") or {}
    if isinstance(rubric_result, dict):
        resumen = str(rubric_result.get("resumen", "")).strip()
        if resumen:
            partes.append(resumen)

        criterios = rubric_result.get("criterios", []) or []
        for criterio in criterios:
            if not isinstance(criterio, dict):
                continue
            nombre = str(criterio.get("nombre", criterio.get("clave", "Criterio"))).strip()
            observacion = str(criterio.get("observacion", "")).strip()
            recomendacion = str(criterio.get("recomendacion", "")).strip()
            puntaje = criterio.get("puntaje", None)
            nivel = str(criterio.get("nivel", "")).strip()

            bloque = []
            if nombre:
                bloque.append(nombre)
            if puntaje is not None:
                bloque.append(f"puntaje={puntaje}")
            if nivel:
                bloque.append(f"nivel={nivel}")
            if observacion:
                bloque.append(f"observación: {observacion}")
            if recomendacion:
                bloque.append(f"recomendación: {recomendacion}")

            if bloque:
                partes.append(" | ".join(bloque))

    semantic_result = payload.get("semantic_result") or {}
    if isinstance(semantic_result, dict):
        resumen = str(semantic_result.get("resumen", "")).strip()
        if resumen:
            partes.append(resumen)

        criterios = semantic_result.get("criterios", []) or []
        for criterio in criterios:
            if not isinstance(criterio, dict):
                continue
            nombre = str(criterio.get("nombre", criterio.get("clave", "Criterio"))).strip()
            observacion = str(criterio.get("observacion", "")).strip()
            recomendacion = str(criterio.get("recomendacion", "")).strip()
            puntaje = criterio.get("puntaje", None)
            nivel = str(criterio.get("nivel", "")).strip()

            bloque = []
            if nombre:
                bloque.append(nombre)
            if puntaje is not None:
                bloque.append(f"puntaje={puntaje}")
            if nivel:
                bloque.append(f"nivel={nivel}")
            if observacion:
                bloque.append(f"observación: {observacion}")
            if recomendacion:
                bloque.append(f"recomendación: {recomendacion}")

            if bloque:
                partes.append(" | ".join(bloque))

    retroalimentacion = payload.get("retroalimentacion") or {}
    if isinstance(retroalimentacion, dict):
        resumen = str(retroalimentacion.get("resumen", "")).strip()
        if resumen:
            partes.append(resumen)

        observaciones = retroalimentacion.get("observaciones", [])
        if observaciones:
            obs = [str(x).strip() for x in observaciones if str(x).strip()]
            if obs:
                partes.append("Observaciones:\n- " + "\n- ".join(obs))

        recomendaciones = retroalimentacion.get("recomendaciones", [])
        if recomendaciones:
            rec = [str(x).strip() for x in recomendaciones if str(x).strip()]
            if rec:
                partes.append("Recomendaciones:\n- " + "\n- ".join(rec))

    resultado_final = payload.get("resultado_final") or {}
    if isinstance(resultado_final, dict) and not partes:
        eval_final = resultado_final.get("evaluacion", {})
        retro_final = resultado_final.get("retroalimentacion", {})

        if isinstance(eval_final, dict):
            resumen = str(eval_final.get("resumen", "")).strip()
            if resumen:
                partes.append(resumen)

        if isinstance(retro_final, dict):
            resumen = str(retro_final.get("resumen", "")).strip()
            if resumen:
                partes.append(resumen)

    if partes:
        return "\n\n".join(partes).strip()

    response_fallback = payload.get("response")
    if isinstance(response_fallback, str) and response_fallback.strip():
        return response_fallback.strip()

    return ""


def _crear_dataset(samples: list[SingleTurnSample]):
    """
    RAGAS 0.3.7 puede exponer el dataset de distintas formas según instalación.
    Probamos varias opciones para máxima compatibilidad.
    """
    if hasattr(EvaluationDataset, "from_list"):
        try:
            return EvaluationDataset.from_list(samples)  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        return EvaluationDataset(samples=samples)  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        return EvaluationDataset(samples)  # type: ignore[call-arg]
    except Exception:
        pass

    raise RuntimeError("No se pudo construir el EvaluationDataset de RAGAS.")


def build_dataset_from_logs(
    event_types: tuple[str, ...] = ("evaluacion_langgraph",),
):
    eventos = _leer_jsonl(LOG_FILE)
    samples: list[SingleTurnSample] = []

    for registro in eventos:
        tipo_evento, payload = _extraer_evento_y_payload(registro)
        if event_types and tipo_evento not in event_types:
            continue

        case_id = (
            payload.get("case_id")
            or payload.get("caso_id")
            or payload.get("id_caso")
            or ""
        )
        case_id = str(case_id).strip()
        if not case_id:
            continue

        try:
            caso = cargar_caso(case_id)
        except Exception:
            continue

        retrieved_contexts = _asegurar_lista(
            payload.get("retrieved_contexts")
            or payload.get("contexto_recuperado")
            or payload.get("contexts")
        )
        if not retrieved_contexts:
            continue

        reference_contexts = _asegurar_lista(
            payload.get("reference_contexts")
            or caso.get("reference_contexts")
            or caso.get("contexto")
            or caso.get("context")
        )

        reference_answer = (
            payload.get("reference")
            or payload.get("reference_answer")
            or caso.get("reference_answer")
            or ""
        )
        reference_answer = str(reference_answer).strip() or None

        response = _construir_respuesta_agente(payload)
        if not response:
            continue

        user_input = _construir_user_input(payload, caso)
        if not user_input:
            continue

        samples.append(
            SingleTurnSample(
                user_input=user_input,
                retrieved_contexts=retrieved_contexts,
                reference_contexts=reference_contexts,
                response=response,
                reference=reference_answer,
            )
        )

    if not samples:
        raise ValueError(
            "No hay muestras suficientes en logs/eventos.jsonl para construir el dataset RAGAS."
        )

    return _crear_dataset(samples)