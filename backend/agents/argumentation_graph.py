from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from backend.services.case_loader import cargar_caso
from backend.services.preprocess import normalizar_texto, extraer_texto_pdf
from backend.services.audio_handler import transcribir_audio
from backend.services.retrieval import recuperar_contexto
from backend.services.feedback import generar_retroalimentacion
from backend.services.logs import registrar_evento


class EvaluacionState(TypedDict, total=False):
    caso_id: str
    tipo_entrada: str
    texto: str
    ruta_pdf: str
    ruta_audio: str
    texto_procesado: str
    caso: dict[str, Any]
    contexto_recuperado: list[dict[str, Any]]
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
        f"{' '.join(caso.get('instrucciones', []))} "
        f"{texto}"
    )

    resultado = recuperar_contexto(consulta, 3)
    documentos = resultado.get("documents", [[]])[0]
    metadatos = resultado.get("metadatas", [[]])[0]
    distancias = resultado.get("distances", [[]])[0]

    fuentes = []
    for i, doc in enumerate(documentos):
        fuentes.append({
            "fragmento": doc,
            "metadatos": metadatos[i] if i < len(metadatos) else {},
            "distancia": distancias[i] if i < len(distancias) else None
        })

    return {"contexto_recuperado": fuentes}


def generar_retroalimentacion_node(state: EvaluacionState) -> EvaluacionState:
    texto = state["texto_procesado"]
    contexto = state.get("contexto_recuperado", [])

    feedback = generar_retroalimentacion(texto)

    if contexto:
        feedback["observaciones"].insert(
            0,
            "Se recuperó contexto jurídico relevante desde la base vectorial."
        )
        feedback["recomendaciones"].append(
            "Vincula tu respuesta con el contexto recuperado."
        )
    else:
        feedback["observaciones"].append(
            "No se recuperó suficiente contexto jurídico."
        )

    return {"retroalimentacion": feedback}


def compilar_resultado_node(state: EvaluacionState) -> EvaluacionState:
    resultado = {
        "caso_id": state["caso_id"],
        "modo": "LangGraph",
        "entrada": state["texto_procesado"],
        "contexto_recuperado": state.get("contexto_recuperado", []),
        "retroalimentacion": state["retroalimentacion"]
    }

    registrar_evento(
        "evaluacion_langgraph",
        {
            "caso_id": state["caso_id"],
            "longitud_texto": len(state["texto_procesado"]),
            "fuentes_recuperadas": len(state.get("contexto_recuperado", []))
        }
    )

    return {"resultado_final": resultado}


builder = StateGraph(EvaluacionState)
builder.add_node("procesar_entrada", procesar_entrada)
builder.add_node("cargar_caso", cargar_caso_node)
builder.add_node("recuperar_contexto", recuperar_contexto_node)
builder.add_node("generar_retroalimentacion", generar_retroalimentacion_node)
builder.add_node("compilar_resultado", compilar_resultado_node)

builder.add_edge(START, "procesar_entrada")
builder.add_edge("procesar_entrada", "cargar_caso")
builder.add_edge("cargar_caso", "recuperar_contexto")
builder.add_edge("recuperar_contexto", "generar_retroalimentacion")
builder.add_edge("generar_retroalimentacion", "compilar_resultado")
builder.add_edge("compilar_resultado", END)

graph = builder.compile()


def ejecutar_evaluacion_langgraph(
    caso_id: str,
    tipo_entrada: str,
    texto: str = "",
    ruta_pdf: str = "",
    ruta_audio: str = ""
):
    estado_inicial: EvaluacionState = {
        "caso_id": caso_id,
        "tipo_entrada": tipo_entrada,
        "texto": texto,
        "ruta_pdf": ruta_pdf,
        "ruta_audio": ruta_audio
    }

    return graph.invoke(estado_inicial)