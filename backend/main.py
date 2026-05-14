from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from backend.services.case_loader import cargar_caso
from backend.services.preprocess import normalizar_texto, extraer_texto_pdf
from backend.services.feedback import generar_retroalimentacion
from backend.services.logs import registrar_evento
from backend.services.retrieval import recuperar_contexto
from backend.services.rag_engine import evaluar_respuesta_con_rag
from backend.services.audio_handler import transcribir_audio
from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph


app = FastAPI(
    title="Agentic RAG Tesis MVP",
    version="0.5.0",
    description="Backend con casos simulados, PDF, audio, búsqueda vectorial y RAG básico."
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
            "/buscar",
            "/evaluar-rag",
            "/evaluar-entrada"
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
    archivo_pdf: UploadFile | None = File(None),
    archivo_audio: UploadFile | None = File(None)
):
    tipo_entrada = tipo_entrada.lower().strip()

    if tipo_entrada not in ("texto", "pdf", "audio"):
        raise HTTPException(
            status_code=400,
            detail="tipo_entrada debe ser 'texto', 'pdf' o 'audio'."
        )

    if tipo_entrada == "texto":
        contenido = normalizar_texto(texto)

        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="Debes escribir un texto para evaluar."
            )

        return evaluar_respuesta_con_rag(caso_id=caso_id, respuesta=contenido)

    if tipo_entrada == "pdf":
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

    if archivo_audio is None:
        raise HTTPException(
            status_code=400,
            detail="Debes subir un archivo de audio."
        )

    suffix = Path(archivo_audio.filename).suffix or ".mp3"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        contenido_bytes = await archivo_audio.read()
        temp_file.write(contenido_bytes)

    try:
        contenido = transcribir_audio(str(temp_path))

        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="No se pudo transcribir el audio."
            )

        return evaluar_respuesta_con_rag(caso_id=caso_id, respuesta=contenido)

    finally:
        if temp_path.exists():
            temp_path.unlink()
            
@app.post("/evaluar-langgraph")
async def evaluar_langgraph(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    #archivo_pdf: UploadFile | None = File(None),
    #archivo_audio: UploadFile | None = File(None)
    archivo_pdf: Optional[UploadFile] | None = File(None),
    archivo_audio: Optional[UploadFile] | None = File(None)
):
    tipo_entrada = tipo_entrada.lower().strip()

    ruta_pdf = ""
    ruta_audio = ""

    if tipo_entrada == "pdf":
        if archivo_pdf is None:
            raise HTTPException(status_code=400, detail="Debes subir un PDF.")
        suffix = Path(archivo_pdf.filename).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await archivo_pdf.read())
            ruta_pdf = temp_file.name

    elif tipo_entrada == "audio":
        if archivo_audio is None:
            raise HTTPException(status_code=400, detail="Debes subir un audio.")
        suffix = Path(archivo_audio.filename).suffix or ".mp3"
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await archivo_audio.read())
            ruta_audio = temp_file.name

    elif tipo_entrada == "texto":
        texto = normalizar_texto(texto)
        if not texto:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")
    else:
        raise HTTPException(status_code=400, detail="tipo_entrada inválido.")

    try:
        return ejecutar_evaluacion_langgraph(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto,
            ruta_pdf=ruta_pdf,
            ruta_audio=ruta_audio
        )
    finally:
        if ruta_pdf and Path(ruta_pdf).exists():
            Path(ruta_pdf).unlink()
        if ruta_audio and Path(ruta_audio).exists():
            Path(ruta_audio).unlink()