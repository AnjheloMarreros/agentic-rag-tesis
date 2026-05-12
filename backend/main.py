from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.services.case_loader import cargar_caso
from backend.services.preprocess import normalizar_texto
from backend.services.feedback import generar_retroalimentacion
from backend.services.logs import registrar_evento

app = FastAPI(
    title="Agentic RAG Tesis MVP",
    version="0.1.0",
    description="Primera versión del backend para casos simulados y retroalimentación básica."
)


class RespuestaEstudiante(BaseModel):
    caso_id: str = "caso_001"
    respuesta: str


@app.get("/")
def raiz():
    return {
        "mensaje": "API funcionando correctamente",
        "endpoint_principal": "/caso/caso_001",
        "docs": "/docs"
    }


@app.get("/caso/{caso_id}")
def obtener_caso(caso_id: str):
    try:
        return cargar_caso(caso_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.post("/evaluar")
def evaluar_respuesta(payload: RespuestaEstudiante):
    texto_limpio = normalizar_texto(payload.respuesta)
    feedback = generar_retroalimentacion(texto_limpio)

    registrar_evento(
        "evaluacion",
        {
            "caso_id": payload.caso_id,
            "longitud_respuesta": len(texto_limpio)
        }
    )

    return {
        "caso_id": payload.caso_id,
        "respuesta_limpia": texto_limpio,
        "retroalimentacion": feedback
    }