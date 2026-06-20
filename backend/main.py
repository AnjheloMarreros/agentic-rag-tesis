from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel

from backend.services.audio_handler import transcribir_audio
from backend.services.case_loader import cargar_caso, listar_casos
from backend.services.feedback import generar_retroalimentacion
from backend.services.input_handler import normalizar_texto
from backend.services.logs import registrar_evento
from backend.services.rag_engine import evaluar_respuesta_con_rag
from backend.services.retrieval import recuperar_contexto


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

app = FastAPI(
    title="Agentic RAG Tesis MVP",
    version="0.8.0",
    description="Backend con casos, texto, audio, benchmark dual, LangGraph, LangChain y RAGAS.",
)


class RespuestaEstudiante(BaseModel):
    caso_id: str = "caso_001"
    respuesta: str


def _guardar_audio_temporal(archivo: UploadFile) -> Path:
    suffix = Path(archivo.filename or "audio.mp3").suffix or ".mp3"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        contenido_bytes = archivo.file.read()
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


def _transcribir_y_normalizar_audio(archivo_audio: UploadFile) -> tuple[str, Path]:
    temp_path = _guardar_audio_temporal(archivo_audio)
    try:
        contenido = normalizar_texto(transcribir_audio(str(temp_path)))
        return contenido, temp_path
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


@app.get("/")
def raiz():
    return {
        "message": "Agentic RAG API funcionando",
        "endpoints": [
            "/casos",
            "/caso/{caso_id}",
            "/evaluar",
            "/evaluar-entrada",
            "/evaluar-langgraph",
            "/evaluar-langchain",
            "/benchmark/evaluar",
            "/api/ragas/live",
            "/api/ragas/benchmark-daily",
            "/api/ragas/benchmark-daily/{job_id}",
        ],
    }


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


@app.post("/transcribir")
async def transcribir(archivo_audio: UploadFile = File(...)):
    try:
        texto, temp_path = _transcribir_y_normalizar_audio(archivo_audio)
        return {
            "ok": True,
            "texto_transcrito": texto,
        }
    finally:
        try:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


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

        from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph

        return ejecutar_evaluacion_langgraph(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=contenido,
        )

    if archivo_audio is None:
        raise HTTPException(
            status_code=400,
            detail="Debes subir un archivo de audio.",
        )

    temp_path = None
    try:
        contenido, temp_path = _transcribir_y_normalizar_audio(archivo_audio)

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

        from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph

        return ejecutar_evaluacion_langgraph(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=contenido,
        )
    finally:
        try:
            if temp_path and temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


@app.post("/evaluar-langgraph")
async def evaluar_langgraph(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()
    texto_procesado = ""

    if tipo_entrada == "texto":
        texto_procesado = normalizar_texto(texto)
        if not texto_procesado:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")

        _registrar_respuesta_recibida(
            caso_id=caso_id,
            tipo_entrada="texto",
            texto=texto_procesado,
        )

    elif tipo_entrada == "audio":
        if archivo_audio is None:
            raise HTTPException(status_code=400, detail="Debes subir un audio.")

        temp_path = None
        try:
            texto_procesado, temp_path = _transcribir_y_normalizar_audio(archivo_audio)
            if not texto_procesado:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo transcribir el audio.",
                )

            _registrar_respuesta_recibida(
                caso_id=caso_id,
                tipo_entrada="audio",
                texto=texto_procesado,
                ruta_audio=str(temp_path),
            )
        finally:
            try:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
    else:
        raise HTTPException(status_code=400, detail="tipo_entrada inválido.")

    try:
        from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph

        return ejecutar_evaluacion_langgraph(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto_procesado,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-langchain")
async def evaluar_langchain(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()
    texto_procesado = ""

    if tipo_entrada == "texto":
        texto_procesado = normalizar_texto(texto)
        if not texto_procesado:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")

        _registrar_respuesta_recibida(
            caso_id=caso_id,
            tipo_entrada="texto",
            texto=texto_procesado,
        )

    elif tipo_entrada == "audio":
        if archivo_audio is None:
            raise HTTPException(status_code=400, detail="Debes subir un audio.")

        temp_path = None
        try:
            texto_procesado, temp_path = _transcribir_y_normalizar_audio(archivo_audio)
            if not texto_procesado:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo transcribir el audio.",
                )

            _registrar_respuesta_recibida(
                caso_id=caso_id,
                tipo_entrada="audio",
                texto=texto_procesado,
                ruta_audio=str(temp_path),
            )
        finally:
            try:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
    else:
        raise HTTPException(status_code=400, detail="tipo_entrada inválido.")

    try:
        from backend.services.langchain_bridge import ejecutar_evaluacion_langchain

        return ejecutar_evaluacion_langchain(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto_procesado,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/benchmark/evaluar")
async def evaluar_benchmark(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()
    texto_procesado = ""

    if tipo_entrada == "texto":
        texto_procesado = normalizar_texto(texto)
        if not texto_procesado:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")

    elif tipo_entrada == "audio":
        if archivo_audio is None:
            raise HTTPException(status_code=400, detail="Debes subir un audio.")

        temp_path = None
        try:
            texto_procesado, temp_path = _transcribir_y_normalizar_audio(archivo_audio)
            if not texto_procesado:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo transcribir el audio.",
                )

            _registrar_respuesta_recibida(
                caso_id=caso_id,
                tipo_entrada="audio",
                texto=texto_procesado,
                ruta_audio=str(temp_path),
            )
        finally:
            try:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
    else:
        raise HTTPException(status_code=400, detail="tipo_entrada inválido.")

    _registrar_respuesta_recibida(
        caso_id=caso_id,
        tipo_entrada=tipo_entrada,
        texto=texto_procesado,
    )

    try:
        from backend.services.benchmark_orchestrator import ejecutar_benchmark_dual

        return ejecutar_benchmark_dual(
            caso_id=caso_id,
            tipo_entrada_original=tipo_entrada,
            texto_procesado=texto_procesado,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/ragas/live")
def api_ragas_live():
    try:
        from backend.services.ragas_runner import run_ragas_live_evaluation

        return run_ragas_live_evaluation()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/ragas/benchmark-daily")
def api_ragas_benchmark_daily_start():
    from backend.services.benchmark_ragas_runner import start_daily_benchmark_job

    return start_daily_benchmark_job()


@app.get("/api/ragas/benchmark-daily/{job_id}")
def api_ragas_benchmark_daily_status(job_id: str):
    from backend.services.benchmark_ragas_runner import get_daily_benchmark_job

    return get_daily_benchmark_job(job_id)