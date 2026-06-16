from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph
from backend.services.audio_handler import transcribir_audio
from backend.services.case_loader import cargar_caso, listar_casos
from backend.services.feedback import generar_retroalimentacion
from backend.services.input_handler import normalizar_texto
from backend.services.logs import registrar_evento
from backend.services.rag_engine import evaluar_respuesta_con_rag
from backend.services.retrieval import recuperar_contexto

from backend.services.langchain_bridge import ejecutar_evaluacion_langchain
from backend.services.benchmark_orchestrator import ejecutar_benchmark_dual


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Agentic RAG Tesis MVP",
    version="0.7.0",
    description="Backend con casos simulados, texto, audio, búsqueda vectorial, RAG y evaluación híbrida.",
)


class RespuestaEstudiante(BaseModel):
    caso_id: str = "caso_001"
    respuesta: str


async def _guardar_audio_temporal(archivo: UploadFile) -> Path:
    suffix = Path(archivo.filename or "audio.mp3").suffix or ".mp3"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        contenido_bytes = await archivo.read()
        temp_file.write(contenido_bytes)
    return temp_path


def _registrar_respuesta_recibida(
    caso_id: str,
    tipo_entrada: str,
    texto: str,
    ruta_audio: str = "",
) -> None:
    registrar_evento(
        "respuesta_estudiante_recibida",
        {
            "caso_id": caso_id,
            "tipo_entrada": tipo_entrada,
            "texto": texto,
            "ruta_audio": ruta_audio,
        },
    )


@app.get("/", response_class=HTMLResponse)
def raiz(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.get("/casos")
def obtener_casos():
    return listar_casos()


@app.get("/caso/{caso_id}")
def obtener_caso(caso_id: str):
    try:
        return cargar_caso(caso_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.post("/evaluar")
def evaluar_respuesta(payload: RespuestaEstudiante):
    texto_limpio = normalizar_texto(payload.respuesta)

    _registrar_respuesta_recibida(
        caso_id=payload.caso_id,
        tipo_entrada="texto",
        texto=texto_limpio,
    )

    feedback = generar_retroalimentacion(texto_limpio)

    registrar_evento(
        "evaluacion_basica",
        {
            "caso_id": payload.caso_id,
            "longitud_respuesta": len(texto_limpio),
        },
    )

    return {
        "caso_id": payload.caso_id,
        "respuesta_limpia": texto_limpio,
        "retroalimentacion": feedback,
        "modo": "reglas básicas",
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
            respuesta=payload.respuesta,
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
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()

    if tipo_entrada not in ("texto", "audio"):
        raise HTTPException(
            status_code=400,
            detail="tipo_entrada debe ser 'texto' o 'audio'.",
        )

    if tipo_entrada == "texto":
        contenido = normalizar_texto(texto)

        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="Debes escribir un texto para evaluar.",
            )

        _registrar_respuesta_recibida(
            caso_id=caso_id,
            tipo_entrada="texto",
            texto=contenido,
        )

        return evaluar_respuesta_con_rag(caso_id=caso_id, respuesta=contenido)

    if archivo_audio is None:
        raise HTTPException(
            status_code=400,
            detail="Debes subir un archivo de audio.",
        )

    temp_path = await _guardar_audio_temporal(archivo_audio)
    try:
        contenido = normalizar_texto(transcribir_audio(str(temp_path)))

        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="No se pudo transcribir el audio.",
            )

        _registrar_respuesta_recibida(
            caso_id=caso_id,
            tipo_entrada="audio",
            texto=contenido,
            ruta_audio=str(temp_path),
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
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()

    ruta_audio = ""

    if tipo_entrada == "texto":
        texto = normalizar_texto(texto)
        if not texto:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")

        _registrar_respuesta_recibida(
            caso_id=caso_id,
            tipo_entrada="texto",
            texto=texto,
        )

    elif tipo_entrada == "audio":
        if archivo_audio is None:
            raise HTTPException(status_code=400, detail="Debes subir un audio.")

        temp_path = await _guardar_audio_temporal(archivo_audio)
        ruta_audio = str(temp_path)

        try:
            texto = normalizar_texto(transcribir_audio(str(temp_path)))
            if not texto:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo transcribir el audio.",
                )

            _registrar_respuesta_recibida(
                caso_id=caso_id,
                tipo_entrada="audio",
                texto=texto,
                ruta_audio=ruta_audio,
            )
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    else:
        raise HTTPException(status_code=400, detail="tipo_entrada inválido.")

    try:
        return ejecutar_evaluacion_langgraph(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto,
            ruta_audio=ruta_audio,
        )
    finally:
        if ruta_audio and Path(ruta_audio).exists():
            Path(ruta_audio).unlink()


@app.post("/evaluar-langchain")
async def evaluar_langchain(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()

    ruta_audio = ""

    if tipo_entrada == "texto":
        texto = normalizar_texto(texto)
        if not texto:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")

        _registrar_respuesta_recibida(
            caso_id=caso_id,
            tipo_entrada="texto",
            texto=texto,
        )

    elif tipo_entrada == "audio":
        if archivo_audio is None:
            raise HTTPException(status_code=400, detail="Debes subir un audio.")

        temp_path = await _guardar_audio_temporal(archivo_audio)
        ruta_audio = str(temp_path)

        try:
            texto = normalizar_texto(transcribir_audio(str(temp_path)))
            if not texto:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo transcribir el audio.",
                )

            _registrar_respuesta_recibida(
                caso_id=caso_id,
                tipo_entrada="audio",
                texto=texto,
                ruta_audio=ruta_audio,
            )
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    else:
        raise HTTPException(status_code=400, detail="tipo_entrada inválido.")

    try:
        return ejecutar_evaluacion_langchain(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto,
            ruta_audio=ruta_audio,
        )
    finally:
        if ruta_audio and Path(ruta_audio).exists():
            Path(ruta_audio).unlink()


@app.post("/benchmark/evaluar")
async def evaluar_benchmark(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()

    if tipo_entrada not in ("texto", "audio"):
        raise HTTPException(
            status_code=400,
            detail="tipo_entrada debe ser 'texto' o 'audio'.",
        )

    if tipo_entrada == "texto":
        texto_procesado = normalizar_texto(texto)
        if not texto_procesado:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")

    else:
        if archivo_audio is None:
            raise HTTPException(status_code=400, detail="Debes subir un audio.")

        temp_path = await _guardar_audio_temporal(archivo_audio)
        try:
            texto_procesado = normalizar_texto(transcribir_audio(str(temp_path)))
            if not texto_procesado:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo transcribir el audio.",
                )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    _registrar_respuesta_recibida(
        caso_id=caso_id,
        tipo_entrada=tipo_entrada,
        texto=texto_procesado,
    )

    return ejecutar_benchmark_dual(
        caso_id=caso_id,
        tipo_entrada_original=tipo_entrada,
        texto_procesado=texto_procesado,
    )


@app.get("/api/ragas/live")
def api_ragas_live():
    try:
        from backend.services.ragas_runner import run_ragas_live_evaluation
        return run_ragas_live_evaluation()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))