from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
from tempfile import NamedTemporaryFile

from backend.services.case_loader import cargar_caso
from backend.services.preprocess import normalizar_texto
from backend.services.feedback import generar_retroalimentacion
from backend.services.logs import registrar_evento
from backend.services.retrieval import recuperar_contexto
from backend.services.rag_engine import evaluar_respuesta_con_rag
from backend.services.input_handler import extraer_texto_pdf


app = FastAPI(
    title="Agentic RAG Tesis MVP",
    version="0.4.0",
    description="Backend con casos simulados, retroalimentación básica, búsqueda vectorial, RAG básico y entrada real de texto/PDF."
)


class RespuestaEstudiante(BaseModel):
    caso_id: str = "caso_001"
    respuesta: str


@app.get("/")
def raiz():
    return {
        "mensaje": "API funcionando correctamente",
        "endpoints": [
            "/caso/caso_001",
            "/evaluar",
            "/evaluar-entrada",
            "/buscar",
            "/evaluar-rag"
        ],
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
        "evaluacion_basica",
        {
            "caso_id": payload.caso_id,
            "longitud_respuesta": len(texto_limpio)
        }
    )

    return {
        "caso_id": payload.caso_id,
        "respuesta_limpia": texto_limpio,
        "retroalimentacion": feedback,
        "modo": "reglas básicas"
    }


@app.get("/buscar")
def buscar(consulta: str = Query(..., min_length=3), n: int = 3):
    try:
        return recuperar_contexto(consulta, n)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-rag")
def evaluar_rag(payload: RespuestaEstudiante):
    try:
        return evaluar_respuesta_con_rag(
            caso_id=payload.caso_id,
            respuesta=payload.respuesta
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-entrada")
async def evaluar_entrada(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    archivo_pdf: UploadFile | None = File(None)
):
    tipo_entrada = tipo_entrada.lower().strip()

    if tipo_entrada not in ("texto", "pdf"):
        raise HTTPException(
            status_code=400,
            detail="tipo_entrada debe ser 'texto' o 'pdf'."
        )

    if tipo_entrada == "texto":
        contenido = normalizar_texto(texto)

        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="Debes escribir un texto para evaluar."
            )

        return evaluar_respuesta_con_rag(caso_id=caso_id, respuesta=contenido)

    if archivo_pdf is None:
        raise HTTPException(
            status_code=400,
            detail="Debes subir un archivo PDF."
        )

    suffix = Path(archivo_pdf.filename).suffix or ".pdf"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        contenido_bytes = await archivo_pdf.read()
        temp_file.write(contenido_bytes)

    try:
        contenido = extraer_texto_pdf(temp_path)

        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="No se pudo extraer texto del PDF."
            )

        return evaluar_respuesta_con_rag(caso_id=caso_id, respuesta=contenido)

    finally:
        if temp_path.exists():
            temp_path.unlink()