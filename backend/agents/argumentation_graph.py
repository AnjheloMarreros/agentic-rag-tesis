from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from backend.services.semantic_judge import evaluar_semantica

from backend.services.case_loader import cargar_caso
from backend.services.preprocess import normalizar_texto, extraer_texto_pdf
from backend.services.audio_handler import transcribir_audio
from backend.services.retrieval import recuperar_contexto
from backend.services.logs import registrar_evento
from backend.services.rubric_loader import cargar_rubrica
from backend.services.rubric_evaluator import evaluar_respuesta_con_rubrica


class EvaluacionState(TypedDict, total=False):
    caso_id: str
    tipo_entrada: str
    texto: str
    ruta_pdf: str
    ruta_audio: str
    texto_procesado: str
    caso: dict[str, Any]
    contexto_recuperado: list[dict[str, Any]]
    evaluacion: dict[str, Any]
    #
    evaluacion_semantica: dict[str, Any]
    #
    retroalimentacion: dict[str, Any]
    resultado_final: dict[str, Any]


def procesar_entrada(state: EvaluacionState) -> EvaluacionState:
    tipo = state["tipo_entrada"].lower().strip()

    if tipo == "texto":
        texto = normalizar_texto(state.get("texto", ""))
    elif tipo == "pdf":
        texto = extraer_texto_pdf(state["ruta_pdf"])
    elif tipo == "audio":
        texto = transcribir_audio(state["ruta_audio"])
    else:
        raise ValueError("tipo_entrada debe ser 'texto', 'pdf' o 'audio'.")

    if not texto:
        raise ValueError("No se pudo obtener texto válido desde la entrada.")

    return {"texto_procesado": texto}


def cargar_caso_node(state: EvaluacionState) -> EvaluacionState:
    caso = cargar_caso(state["caso_id"])
    return {"caso": caso}


def recuperar_contexto_node(state: EvaluacionState) -> EvaluacionState:
    caso = state["caso"]
    texto = state["texto_procesado"]

    consulta = (
        f"{caso['titulo']} "
        f"{caso['enunciado']} "
        f"{' '.join(caso.get('contexto', []))} "
        f"{' '.join(caso.get('instrucciones', []))} "
        f"{texto}"
    )

    resultado = recuperar_contexto(consulta, 3)

    documentos = resultado.get("documents", [[]])
    metadatos = resultado.get("metadatas", [[]])
    distancias = resultado.get("distances", [[]])

    docs = documentos[0] if documentos and len(documentos) > 0 else []
    metas = metadatos[0] if metadatos and len(metadatos) > 0 else []
    dists = distancias[0] if distancias and len(distancias) > 0 else []

    fuentes = []
    for i, doc in enumerate(docs):
        fuentes.append(
            {
                "fragmento": doc,
                "metadatos": metas[i] if i < len(metas) else {},
                "distancia": dists[i] if i < len(dists) else None,
            }
        )

    return {"contexto_recuperado": fuentes}


def evaluar_rubrica_node(state: EvaluacionState) -> EvaluacionState:
    caso = state["caso"]
    texto = state["texto_procesado"]
    contexto = state.get("contexto_recuperado", [])

    rubrica = cargar_rubrica()

    evaluacion = evaluar_respuesta_con_rubrica(
        respuesta=texto,
        caso=caso,
        fuentes=contexto,
        rubrica=rubrica,
    )

    observaciones = []
    recomendaciones = []

    # Tomamos la salida por criterio para construir una retroalimentación legible.
    for item in evaluacion.get("criterios", []):
        observaciones.append(
            f"{item['nombre']}: {item['observacion']}"
        )
        recomendaciones.append(
            f"{item['nombre']}: {item['recomendacion']}"
        )

    retroalimentacion = {
        "estado": "evaluado",
        "resumen": evaluacion.get("resumen", ""),
        "observaciones": observaciones,
        "recomendaciones": recomendaciones,
    }

    return {
        "evaluacion": evaluacion,
        "retroalimentacion": retroalimentacion,
    }


def compilar_resultado_node(state: EvaluacionState) -> EvaluacionState:
    evaluacion = state["evaluacion"]
    retroalimentacion = state["retroalimentacion"]

    resultado = {
        "caso_id": state["caso_id"],
        "modo": "LangGraph",
        "evaluacion": {
            "puntaje_total": evaluacion.get("puntaje_total", 0),
            "nivel_global": evaluacion.get("nivel_global", "N/D"),
            "resumen": evaluacion.get("resumen", ""),
            "criterios": evaluacion.get("criterios", []),
            "recomendaciones_generales": evaluacion.get("recomendaciones_generales", []),
        },
        #
        "evaluacion_semantica": state.get("evaluacion_semantica", {}),
        #
        "retroalimentacion": retroalimentacion,
    }

    registrar_evento(
        "evaluacion_langgraph",
        {
            "caso_id": state["caso_id"],
            "longitud_texto": len(state["texto_procesado"]),
            "puntaje_total": evaluacion.get("puntaje_total", 0),
            "nivel_global": evaluacion.get("nivel_global", "N/D"),
            "fuentes_recuperadas": len(state.get("contexto_recuperado", [])),
        },
    )

    return {"resultado_final": resultado}

#####################################################
def evaluar_semantica_node(state: EvaluacionState) -> EvaluacionState:
    caso = state["caso"]
    respuesta = state["texto_procesado"]
    contexto = state.get("contexto_recuperado", [])

    evaluacion_semantica = evaluar_semantica(
        respuesta=respuesta,
        caso=caso,
        contexto_recuperado=contexto,
    )

    return {
        "evaluacion_semantica": evaluacion_semantica
    }
#####################################################


builder = StateGraph(EvaluacionState)
builder.add_node("procesar_entrada", procesar_entrada)
builder.add_node("cargar_caso", cargar_caso_node)
builder.add_node("recuperar_contexto", recuperar_contexto_node)
builder.add_node("evaluar_rubrica", evaluar_rubrica_node)
#
builder.add_node("evaluar_semantica", evaluar_semantica_node)
#
builder.add_node("compilar_resultado", compilar_resultado_node)

builder.add_edge(START, "procesar_entrada")
builder.add_edge("procesar_entrada", "cargar_caso")
builder.add_edge("cargar_caso", "recuperar_contexto")
#builder.add_edge("recuperar_contexto", "evaluar_rubrica")

builder.add_edge("recuperar_contexto", "evaluar_semantica")
builder.add_edge("evaluar_semantica", "evaluar_rubrica")

#builder.add_edge("evaluar_rubrica", "compilar_resultado")
builder.add_edge("compilar_resultado", END)

graph = builder.compile()


def ejecutar_evaluacion_langgraph(
    caso_id: str,
    tipo_entrada: str,
    texto: str = "",
    ruta_pdf: str = "",
    ruta_audio: str = "",
):
    estado_inicial: EvaluacionState = {
        "caso_id": caso_id,
        "tipo_entrada": tipo_entrada,
        "texto": texto,
        "ruta_pdf": ruta_pdf,
        "ruta_audio": ruta_audio,
    }

    estado_final = graph.invoke(estado_inicial)

    # Devuelve directamente el resultado limpio, no el estado interno del grafo.
    return estado_final.get("resultado_final", estado_final)