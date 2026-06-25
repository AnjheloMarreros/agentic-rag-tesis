from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from backend.services.audio_handler import transcribir_audio
from backend.services.case_loader import cargar_caso, listar_casos
from backend.services.feedback import generar_retroalimentacion
from backend.services.input_handler import normalizar_texto
from backend.services.logs import registrar_evento
from backend.services.result_store import (
    append_result,
    load_results,
    render_result_detail_html,
    render_results_csv,
    render_results_html,
    render_results_jsonl,
)

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


def _generar_identificadores_benchmark(
    caso_id: str,
    benchmark_id: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> tuple[str, str]:
    benchmark_id_final = (
        benchmark_id.strip()
        if isinstance(benchmark_id, str) and benchmark_id.strip()
        else f"bm_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex[:8]}"
    )
    sample_id_final = (
        sample_id.strip()
        if isinstance(sample_id, str) and sample_id.strip()
        else f"{caso_id}_{uuid4().hex[:8]}"
    )
    return benchmark_id_final, sample_id_final


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
    benchmark_id: str = "",
    sample_id: str = "",
) -> None:
    payload = {
        "caso_id": caso_id,
        "tipo_entrada": tipo_entrada,
        "texto": texto,
        "ruta_audio": ruta_audio,
    }
    if benchmark_id:
        payload["benchmark_id"] = benchmark_id
    if sample_id:
        payload["sample_id"] = sample_id

    registrar_evento(
        "respuesta_estudiante_recibida",
        payload,
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


def _guardar_resultado_en_historial(
    *,
    caso_id: str,
    benchmark_id: str,
    sample_id: str,
    pipeline: str,
    answer: str,
    resultado: Any,
) -> None:
    try:
        if isinstance(resultado, list):
            for item in resultado:
                if isinstance(item, dict):
                    _guardar_resultado_en_historial(
                        caso_id=caso_id,
                        benchmark_id=benchmark_id,
                        sample_id=sample_id,
                        pipeline=str(item.get("pipeline", pipeline)),
                        answer=answer,
                        resultado=item,
                    )
            return

        if not isinstance(resultado, dict):
            return

        evaluacion = resultado.get("evaluacion") or {}
        if not isinstance(evaluacion, dict):
            evaluacion = {}

        evaluacion_semantica = resultado.get("evaluacion_semantica") or {}
        if not isinstance(evaluacion_semantica, dict):
            evaluacion_semantica = {}

        evaluacion_rubrica = resultado.get("evaluacion_rubrica") or {}
        if not isinstance(evaluacion_rubrica, dict):
            evaluacion_rubrica = {}

        retroalimentacion = resultado.get("retroalimentacion") or {}
        if not isinstance(retroalimentacion, dict):
            retroalimentacion = {}

        caso = resultado.get("caso")
        if not isinstance(caso, dict):
            caso = None

        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": caso_id,
            "caso_id": caso_id,
            "benchmark_id": benchmark_id,
            "sample_id": sample_id,
            "pipeline": pipeline,
            "answer": answer,
            "case": caso,
            "score_total": evaluacion.get("puntaje_total") or resultado.get("puntaje_total"),
            "score_semantic": evaluacion_semantica.get("puntaje_total") or resultado.get("puntaje_semantico"),
            "score_rubric": evaluacion_rubrica.get("puntaje_total"),
            "retrieval_score": (
                evaluacion.get("indice_relevancia_caso")
                or evaluacion_semantica.get("indice_relevancia_caso")
                or resultado.get("indice_relevancia_caso")
            ),
            "feedback": retroalimentacion.get("resumen") or evaluacion.get("resumen") or "",
            "response_json": resultado,
        }

        append_result(registro)
    except Exception:
        pass


def _guardar_resultado_benchmark(
    *,
    caso_id: str,
    benchmark_id: str,
    sample_id: str,
    pipeline: str,
    answer: str,
    resultado: Any,
) -> None:
    _guardar_resultado_en_historial(
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
        pipeline=pipeline,
        answer=answer,
        resultado=resultado,
    )


def _guardar_resultado_ragas(resultado: Any, benchmark_id: str = "") -> None:
    try:
        if not isinstance(resultado, dict):
            return

        summary = resultado.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}

        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": "",
            "caso_id": "",
            "benchmark_id": benchmark_id or resultado.get("benchmark_id_used") or resultado.get("benchmark_id") or "",
            "sample_id": "",
            "pipeline": "ragas",
            "status": resultado.get("status", ""),
            "provider": resultado.get("provider", ""),
            "metrics_used": resultado.get("metrics_used", []),
            "num_samples": resultado.get("num_samples"),
            "input_samples": resultado.get("input_samples"),
            "has_retrieved_contexts": resultado.get("has_retrieved_contexts"),
            "faithfulness": summary.get("faithfulness"),
            "answer_relevancy": summary.get("answer_relevancy"),
            "summary": summary,
            "rows": resultado.get("rows", []),
            "output_csv": resultado.get("output_csv"),
            "response_json": resultado,
        }

        append_result(registro)
    except Exception:
        pass


@app.get("/")
def raiz():
    return {
        "message": "Agentic RAG API funcionando",
        "endpoints": [
            "/casos",
            "/caso/{caso_id}",
            "/benchmark/start",
            "/evaluar",
            "/evaluar-entrada",
            "/evaluar-langgraph",
            "/evaluar-langchain",
            "/benchmark/evaluar",
            "/resultados",
            "/debug/logs-status",
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


@app.post("/benchmark/start")
def benchmark_start(caso_id: str = Query("caso_001", min_length=1)):
    benchmark_id, sample_id = _generar_identificadores_benchmark(caso_id)

    registrar_evento(
        "benchmark_inicio",
        {
            "caso_id": caso_id,
            "benchmark_id": benchmark_id,
            "sample_id": sample_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "ok": True,
        "caso_id": caso_id,
        "benchmark_id": benchmark_id,
        "sample_id": sample_id,
        "mensaje": "Benchmark iniciado correctamente.",
    }


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
        from backend.services.retrieval import recuperar_contexto
        return recuperar_contexto(consulta, n)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-rag")
def evaluar_rag(payload: RespuestaEstudiante):
    try:
        from backend.services.rag_engine import evaluar_respuesta_con_rag
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
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()
    benchmark_id_final, sample_id_final = _generar_identificadores_benchmark(
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
    )

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
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
        )

        from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph

        resultado = ejecutar_evaluacion_langgraph(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=contenido,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
        )

        _guardar_resultado_benchmark(
            caso_id=caso_id,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
            pipeline="langgraph",
            answer=contenido,
            resultado=resultado,
        )

        return resultado

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
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
        )

        from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph

        resultado = ejecutar_evaluacion_langgraph(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=contenido,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
        )

        _guardar_resultado_benchmark(
            caso_id=caso_id,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
            pipeline="langgraph",
            answer=contenido,
            resultado=resultado,
        )

        return resultado
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
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()
    benchmark_id_final, sample_id_final = _generar_identificadores_benchmark(
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
    )
    texto_procesado = ""

    if tipo_entrada == "texto":
        texto_procesado = normalizar_texto(texto)
        if not texto_procesado:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")

        _registrar_respuesta_recibida(
            caso_id=caso_id,
            tipo_entrada="texto",
            texto=texto_procesado,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
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
                benchmark_id=benchmark_id_final,
                sample_id=sample_id_final,
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

        resultado = ejecutar_evaluacion_langgraph(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto_procesado,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
        )

        _guardar_resultado_benchmark(
            caso_id=caso_id,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
            pipeline="langgraph",
            answer=texto_procesado,
            resultado=resultado,
        )

        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-langchain")
async def evaluar_langchain(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()
    benchmark_id_final, sample_id_final = _generar_identificadores_benchmark(
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
    )
    texto_procesado = ""

    if tipo_entrada == "texto":
        texto_procesado = normalizar_texto(texto)
        if not texto_procesado:
            raise HTTPException(status_code=400, detail="Debes escribir texto.")

        _registrar_respuesta_recibida(
            caso_id=caso_id,
            tipo_entrada="texto",
            texto=texto_procesado,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
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
                benchmark_id=benchmark_id_final,
                sample_id=sample_id_final,
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

        resultado = ejecutar_evaluacion_langchain(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto_procesado,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
        )

        _guardar_resultado_benchmark(
            caso_id=caso_id,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
            pipeline="langchain",
            answer=texto_procesado,
            resultado=resultado,
        )

        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/benchmark/evaluar")
async def evaluar_benchmark(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form(...),
    texto: str = Form(""),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: Optional[UploadFile] = File(None),
):
    tipo_entrada = tipo_entrada.lower().strip()
    benchmark_id_final, sample_id_final = _generar_identificadores_benchmark(
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
    )
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
                benchmark_id=benchmark_id_final,
                sample_id=sample_id_final,
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
        benchmark_id=benchmark_id_final,
        sample_id=sample_id_final,
    )

    try:
        from backend.services.benchmark_orchestrator import ejecutar_benchmark_dual

        resultado = ejecutar_benchmark_dual(
            caso_id=caso_id,
            tipo_entrada_original=tipo_entrada,
            texto_procesado=texto_procesado,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
        )

        _guardar_resultado_benchmark(
            caso_id=caso_id,
            benchmark_id=benchmark_id_final,
            sample_id=sample_id_final,
            pipeline="benchmark",
            answer=texto_procesado,
            resultado=resultado,
        )

        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/resultados")
def obtener_resultados(
    case_id: str | None = None,
    caso_id: str | None = None,
    benchmark_id: str | None = None,
    sample_id: str | None = None,
    pipeline: str | None = None,
    limit: int | None = None,
    format: str = "html",
    detail_index: int | None = None,
):
    rows = load_results(
        case_id=case_id,
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
        pipeline=pipeline,
        limit=limit,
    )

    if detail_index is not None:
        if detail_index < 0 or detail_index >= len(rows):
            raise HTTPException(status_code=404, detail="No existe ese registro.")
        return HTMLResponse(render_result_detail_html(rows[detail_index]))

    fmt = format.lower().strip()

    if fmt == "json":
        return {
            "ok": True,
            "count": len(rows),
            "rows": rows,
        }

    if fmt == "csv":
        return Response(
            content=render_results_csv(rows),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'inline; filename="resultados.csv"'},
        )

    if fmt == "jsonl":
        return Response(
            content=render_results_jsonl(rows),
            media_type="application/jsonl; charset=utf-8",
            headers={"Content-Disposition": 'inline; filename="resultados.jsonl"'},
        )

    return HTMLResponse(render_results_html(rows))


@app.get("/debug/logs-status")
def debug_logs_status(benchmark_id: Optional[str] = Query(None, min_length=1)):
    from backend.services.logs import ARCHIVO_LOG

    if not ARCHIVO_LOG.exists():
        return {
            "ok": False,
            "exists": False,
            "path": str(ARCHIVO_LOG),
            "line_count": 0,
            "benchmark_id": benchmark_id,
            "message": "El archivo de logs no existe.",
        }

    try:
        with open(ARCHIVO_LOG, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as error:
        return {
            "ok": False,
            "exists": True,
            "path": str(ARCHIVO_LOG),
            "line_count": 0,
            "benchmark_id": benchmark_id,
            "message": "No se pudo leer el archivo de logs.",
            "detail": str(error),
        }

    events = []
    matching = []

    for raw in lines[-20:]:
        try:
            item = json.loads(raw)
        except Exception:
            continue

        tipo = item.get("tipo", "")
        datos = item.get("datos", {})
        event = {
            "tipo": tipo,
            "benchmark_id": datos.get("benchmark_id", ""),
            "sample_id": datos.get("sample_id", ""),
            "caso_id": datos.get("caso_id", ""),
        }
        events.append(event)

        if benchmark_id and datos.get("benchmark_id") == benchmark_id:
            matching.append(event)

    return {
        "ok": True,
        "exists": True,
        "path": str(ARCHIVO_LOG),
        "line_count": len(lines),
        "last_events": events,
        "matching_benchmark_events": matching,
        "benchmark_id": benchmark_id,
    }


@app.get("/api/ragas/live")
def api_ragas_live(benchmark_id: Optional[str] = Query(None, min_length=1)):
    try:
        from backend.services.ragas_runner import run_ragas_live_evaluation

        resultado = run_ragas_live_evaluation(benchmark_id=benchmark_id)

        try:
            _guardar_resultado_ragas(resultado, benchmark_id=benchmark_id or "")
        except Exception:
            pass

        return resultado
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