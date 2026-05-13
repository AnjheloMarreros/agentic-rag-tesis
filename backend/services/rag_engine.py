#Se unen sercicios de caso, respuesta del estudiante,
#búsqueda vectorial, retroalimentación, log del evento

from backend.services.case_loader import cargar_caso
from backend.services.retrieval import recuperar_contexto
from backend.services.feedback import generar_retroalimentacion
from backend.services.logs import registrar_evento


def extraer_fuentes(resultado_busqueda: dict) -> list:
    documentos = resultado_busqueda.get("documents", [[]])
    metadatos = resultado_busqueda.get("metadatas", [[]])
    distancias = resultado_busqueda.get("distances", [[]])

    docs = documentos[0] if documentos and len(documentos) > 0 else []
    metas = metadatos[0] if metadatos and len(metadatos) > 0 else []
    dists = distancias[0] if distancias and len(distancias) > 0 else []

    fuentes = []

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None

        fuentes.append({
            "fragmento": doc,
            "metadatos": meta,
            "distancia": dist
        })

    return fuentes


def evaluar_respuesta_con_rag(caso_id: str, respuesta: str) -> dict:
    caso = cargar_caso(caso_id)

    consulta = (
        f"{caso['titulo']} "
        f"{caso['enunciado']} "
        f"{' '.join(caso.get('instrucciones', []))} "
        f"{respuesta}"
    )

    resultado_busqueda = recuperar_contexto(consulta, 3)
    fuentes = extraer_fuentes(resultado_busqueda)

    feedback_base = generar_retroalimentacion(respuesta)

    if fuentes:
        feedback_base["observaciones"].insert(
            0,
            "Se recuperó contexto jurídico relevante desde la base vectorial."
        )
        feedback_base["recomendaciones"].append(
            "Asegúrate de vincular tu respuesta con el contexto recuperado."
        )
    else:
        feedback_base["observaciones"].append(
            "No se recuperó contexto suficiente en esta consulta."
        )
        feedback_base["recomendaciones"].append(
            "Prueba usando más palabras clave jurídicas del caso."
        )

    registrar_evento(
        "evaluacion_rag",
        {
            "caso_id": caso_id,
            "respuesta_longitud": len(respuesta),
            "fuentes_recuperadas": len(fuentes)
        }
    )

    return {
        "caso": {
            "id": caso["id"],
            "titulo": caso["titulo"],
            "enunciado": caso["enunciado"]
        },
        "entrada_estudiante": respuesta,
        "contexto_recuperado": fuentes,
        "retroalimentacion": feedback_base,
        "modo": "RAG básico local"
    }