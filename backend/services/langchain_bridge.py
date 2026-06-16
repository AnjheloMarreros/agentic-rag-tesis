from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.runnables import RunnableLambda

from backend.services.audio_handler import transcribir_audio
from backend.services.case_loader import cargar_caso
from backend.services.input_handler import normalizar_texto
from backend.services.pipeline_contract import (
    construir_resultado_final,
    construir_retroalimentacion,
    registrar_ejecucion_pipeline,
)
from backend.services.retrieval import recuperar_contexto
from backend.services.rubric_evaluator import evaluar_respuesta_con_rubrica
from backend.services.rubric_loader import cargar_rubrica
from backend.services.semantic_judge import evaluar_semantica


class EvaluacionState(TypedDict, total=False):
    caso_id: str
    tipo_entrada: str
    texto: str
    ruta_audio: str
    texto_procesado: str
    caso: dict[str, Any]
    contexto_recuperado: list[dict[str, Any]]
    evaluacion_rubrica: dict[str, Any]
    evaluacion_semantica: dict[str, Any]
    evaluacion: dict[str, Any]
    retroalimentacion: dict[str, Any]
    resultado_final: dict[str, Any]


def _merge_state(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    nuevo = dict(state)
    nuevo.update(updates)
    return nuevo


def _float_safe(valor: Any, default: float = 0.0) -> float:
    try:
        if valor is None:
            return default
        return float(valor)
    except Exception:
        return default


def _nivel_global(puntaje: float) -> str:
    if puntaje >= 85:
        return "Excelente"
    if puntaje >= 70:
        return "Alto"
    if puntaje >= 55:
        return "Medio"
    return "Bajo"


def _nivel_1_5(puntaje: float) -> str:
    if puntaje <= 1:
        return "Muy bajo"
    if puntaje <= 2:
        return "Bajo"
    if puntaje <= 3:
        return "Medio"
    if puntaje <= 4:
        return "Alto"
    return "Excelente"


def _construir_evaluacion_consolidada(
    evaluacion_rubrica: dict[str, Any],
    evaluacion_semantica: dict[str, Any],
) -> dict[str, Any]:
    puntaje_rubrica = _float_safe(evaluacion_rubrica.get("puntaje_total", 0.0))
    puntaje_semantico = _float_safe(evaluacion_semantica.get("puntaje_total", 0.0))
    similitud_caso = _float_safe(evaluacion_semantica.get("similitud_caso", 0.0))
    similitud_contexto = _float_safe(evaluacion_semantica.get("similitud_contexto", 0.0))
    relevancia_lexica = _float_safe(evaluacion_semantica.get("indice_relevancia_lexica", 0.0))
    relevancia_caso = _float_safe(evaluacion_semantica.get("indice_relevancia_caso", 0.0))

    relevancia_total = max(relevancia_caso, similitud_caso, similitud_contexto, relevancia_lexica)
    puntaje_relevancia = round(relevancia_total * 100.0, 1)

    puntaje_total = round(
        (puntaje_rubrica * 0.25) + (puntaje_semantico * 0.55) + (puntaje_relevancia * 0.20),
        1,
    )

    if relevancia_lexica < 0.05 and relevancia_total < 0.50:
        puntaje_total = min(puntaje_total, 10.0)
    elif relevancia_lexica < 0.08 and relevancia_total < 0.55:
        puntaje_total = min(puntaje_total, 20.0)
    elif relevancia_lexica < 0.12 and relevancia_total < 0.62:
        puntaje_total = min(puntaje_total, 30.0)
    elif relevancia_lexica < 0.15 and relevancia_total < 0.70:
        puntaje_total = min(puntaje_total, 40.0)
    elif relevancia_lexica < 0.18 and relevancia_total < 0.78:
        puntaje_total = min(puntaje_total, 50.0)

    puntaje_total = max(0.0, min(100.0, puntaje_total))
    nivel_global = _nivel_global(puntaje_total)

    criterio_relevancia = {
        "clave": "relevancia_caso",
        "nombre": "Relevancia con el caso",
        "peso": 0.20,
        "puntaje": max(1, min(5, int(round(relevancia_total * 5)))),
        "nivel": _nivel_1_5(relevancia_total * 5),
        "observacion": (
            "La respuesta está alineada con el problema del caso."
            if relevancia_total >= 0.60
            else "La respuesta se aleja del caso planteado."
        ),
        "recomendacion": (
            "Mantén el foco en los hechos y la cuestión jurídica específica del caso."
            if relevancia_total < 0.60
            else "La relevancia con el caso es adecuada."
        ),
    }

    criterios = [criterio_relevancia] + list(evaluacion_rubrica.get("criterios", []))

    recomendaciones_generales = list(evaluacion_rubrica.get("recomendaciones_generales", []))
    if relevancia_total < 0.60:
        recomendaciones_generales.insert(
            0,
            "Ajusta la respuesta al problema concreto del caso antes de desarrollar la argumentación.",
        )

    resumen = (
        f"Tu respuesta obtuvo {puntaje_total}% de coherencia global. "
        f"El nivel global es {nivel_global}. "
        f"Relevancia con el caso: {round(relevancia_total * 100, 1)}%."
    )

    return {
        "puntaje_total": puntaje_total,
        "nivel_global": nivel_global,
        "resumen": resumen,
        "puntaje_rubrica": puntaje_rubrica,
        "puntaje_semantico": puntaje_semantico,
        "indice_relevancia_caso": round(relevancia_total, 4),
        "indice_relevancia_lexica": round(relevancia_lexica, 4),
        "criterios": criterios,
        "recomendaciones_generales": recomendaciones_generales,
    }


def procesar_entrada(state: EvaluacionState) -> EvaluacionState:
    tipo = state["tipo_entrada"].lower().strip()

    if tipo == "texto":
        texto = normalizar_texto(state.get("texto", ""))
    elif tipo == "audio":
        texto = normalizar_texto(transcribir_audio(state["ruta_audio"]))
    else:
        raise ValueError("tipo_entrada debe ser 'texto' o 'audio'.")

    if not texto:
        raise ValueError("No se pudo obtener texto válido desde la entrada.")

    return _merge_state(state, {"texto_procesado": texto})


def cargar_caso_node(state: EvaluacionState) -> EvaluacionState:
    caso = cargar_caso(state["caso_id"])
    return _merge_state(state, {"caso": caso})


def recuperar_contexto_node(state: EvaluacionState) -> EvaluacionState:
    caso = state["caso"]
    texto = state["texto_procesado"]

    consulta = (
        f"{caso.get('titulo', '')} "
        f"{caso.get('enunciado', '')} "
        f"{' '.join(caso.get('contexto', []))} "
        f"{' '.join(caso.get('instrucciones', []))} "
        f"{texto}"
    ).strip()

    resultado = recuperar_contexto(consulta, 3)

    documentos = resultado.get("documents", [[]])
    metadatos = resultado.get("metadatas", [[]])
    distancias = resultado.get("distances", [[]])

    docs = documentos[0] if documentos and len(documentos) > 0 else []
    metas = metadatos[0] if metadatos and len(metadatos) > 0 else []
    dists = distancias[0] if distancias and len(distancias) > 0 else []

    fuentes: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        fuentes.append(
            {
                "fragmento": doc,
                "metadatos": metas[i] if i < len(metas) else {},
                "distancia": dists[i] if i < len(dists) else None,
            }
        )

    return _merge_state(state, {"contexto_recuperado": fuentes})


def evaluar_semantica_node(state: EvaluacionState) -> EvaluacionState:
    caso = state["caso"]
    respuesta = state["texto_procesado"]
    contexto = state.get("contexto_recuperado", [])

    evaluacion_semantica = evaluar_semantica(
        respuesta=respuesta,
        caso=caso,
        contexto_recuperado=contexto,
    )

    return _merge_state(state, {"evaluacion_semantica": evaluacion_semantica})


def evaluar_rubrica_node(state: EvaluacionState) -> EvaluacionState:
    caso = state["caso"]
    texto = state["texto_procesado"]
    contexto = state.get("contexto_recuperado", [])

    rubrica = cargar_rubrica()

    evaluacion_rubrica = evaluar_respuesta_con_rubrica(
        respuesta=texto,
        caso=caso,
        fuentes=contexto,
        rubrica=rubrica,
    )

    return _merge_state(state, {"evaluacion_rubrica": evaluacion_rubrica})


def compilar_resultado_node(state: EvaluacionState) -> EvaluacionState:
    caso = state["caso"]
    evaluacion_rubrica = state.get("evaluacion_rubrica", {})
    evaluacion_semantica = state.get("evaluacion_semantica", {})
    contexto_recuperado = state.get("contexto_recuperado", [])

    retroalimentacion = construir_retroalimentacion(
        evaluacion_rubrica,
        evaluacion_semantica,
    )

    if _float_safe(evaluacion_semantica.get("indice_relevancia_lexica", 0.0)) < 0.10:
        retroalimentacion.setdefault("alertas", [])
        retroalimentacion["alertas"].append(
            "La respuesta parece poco alineada con el caso, aunque use términos jurídicos."
        )

    resultado = construir_resultado_final(
        caso_id=state["caso_id"],
        tipo_entrada=state.get("tipo_entrada", "texto"),
        caso=caso,
        texto_procesado=state["texto_procesado"],
        evaluacion=evaluacion_rubrica,
        evaluacion_semantica=evaluacion_semantica,
        contexto_recuperado=contexto_recuperado,
    )

    evaluacion_consolidada = _construir_evaluacion_consolidada(
        evaluacion_rubrica=evaluacion_rubrica,
        evaluacion_semantica=evaluacion_semantica,
    )

    resultado["evaluacion_rubrica"] = evaluacion_rubrica
    resultado["evaluacion"] = evaluacion_consolidada
    resultado["retroalimentacion"] = retroalimentacion
    resultado["modo"] = "LangChain + evaluación semántica"

    registrar_ejecucion_pipeline(
        pipeline="langchain",
        caso_id=state["caso_id"],
        tipo_entrada=state.get("tipo_entrada", "texto"),
        texto_procesado=state["texto_procesado"],
        caso=caso,
        contexto_recuperado=contexto_recuperado,
        evaluacion=evaluacion_consolidada,
        evaluacion_semantica=evaluacion_semantica,
        resultado_final=resultado,
    )

    return _merge_state(state, {"resultado_final": resultado})


pipeline = (
    RunnableLambda(procesar_entrada)
    | RunnableLambda(cargar_caso_node)
    | RunnableLambda(recuperar_contexto_node)
    | RunnableLambda(evaluar_semantica_node)
    | RunnableLambda(evaluar_rubrica_node)
    | RunnableLambda(compilar_resultado_node)
)


def ejecutar_evaluacion_langchain(
    caso_id: str,
    tipo_entrada: str,
    texto: str = "",
    ruta_audio: str = "",
):
    estado_inicial: EvaluacionState = {
        "caso_id": caso_id,
        "tipo_entrada": tipo_entrada,
        "texto": texto,
        "ruta_audio": ruta_audio,
    }

    estado_final = pipeline.invoke(estado_inicial)
    return estado_final.get("resultado_final", estado_final)