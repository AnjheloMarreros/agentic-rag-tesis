from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from backend.services.case_loader import cargar_caso
from backend.services.preprocess import normalizar_texto
from backend.services.feedback import generar_retroalimentacion
from backend.services.logs import registrar_evento
from backend.services.retrieval import recuperar_contexto

app = FastAPI(
    title="Agentic RAG Tesis MVP",
    version="0.2.0",
    description="Backend con casos simulados, retroalimentación básica y búsqueda vectorial."
)


class RespuestaEstudiante(BaseModel):
    caso_id: str = "caso_001"
    respuesta: str


@app.get("/")
def raiz():
    return {
        "mensaje": "API funcionando correctamente",
        "endpoints": ["/caso/caso_001", "/evaluar", "/buscar"],
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


@app.get("/buscar")
def buscar(consulta: str = Query(..., min_length=3), n: int = 3):
    try:
        return recuperar_contexto(consulta, n)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))