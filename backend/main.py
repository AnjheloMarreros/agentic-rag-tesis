from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import HTMLResponse

from backend.services.case_loader import cargar_caso, listar_casos
from backend.services.deepgram_stt import transcribir_audio_deepgram
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
    version="0.9.0",
    description="Backend con casos, voz, benchmark dual, LangGraph, LangChain y RAGAS.",
)


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


def _registrar_respuesta_recibida(
    caso_id: str,
    tipo_entrada: str,
    texto: str,
    benchmark_id: str = "",
    sample_id: str = "",
) -> None:
    payload = {
        "caso_id": caso_id,
        "tipo_entrada": tipo_entrada,
        "texto": texto,
    }
    if benchmark_id:
        payload["benchmark_id"] = benchmark_id
    if sample_id:
        payload["sample_id"] = sample_id

    registrar_evento(
        "respuesta_estudiante_recibida",
        payload,
    )


def _resultado_cuerpo(resultado: Any) -> dict[str, Any]:
    if not isinstance(resultado, dict):
        return {}

    resultado_final = resultado.get("resultado_final")
    if isinstance(resultado_final, dict):
        return resultado_final

    return resultado


def _extraer_componentes_benchmark(resultado: Any) -> dict[str, dict[str, Any]]:
    componentes: dict[str, dict[str, Any]] = {}

    if not isinstance(resultado, dict):
        return componentes

    candidatos = [resultado]

    resultado_final = resultado.get("resultado_final")
    if isinstance(resultado_final, dict):
        candidatos.append(resultado_final)

    response_json = resultado.get("response_json")
    if isinstance(response_json, dict):
        candidatos.append(response_json)

    for candidato in candidatos:
        for clave in ("benchmark", "langgraph", "langchain"):
            valor = candidato.get(clave)
            if isinstance(valor, dict):
                componentes[clave] = valor

    return componentes


async def _transcribir_audio_recibido(archivo_audio: UploadFile) -> str:
    audio_bytes = await archivo_audio.read()
    try:
        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="El audio está vacío.",
            )

        texto = await asyncio.to_thread(
            transcribir_audio_deepgram,
            audio_bytes,
            archivo_audio.content_type,
            archivo_audio.filename,
        )

        return texto.strip()
    finally:
        try:
            await archivo_audio.close()
        except Exception:
            pass


async def _obtener_texto_procesado(
    tipo_entrada: str,
    texto: str,
    archivo_audio: Optional[UploadFile] = None,
) -> tuple[str, str]:
    tipo = (tipo_entrada or "").lower().strip()

    if tipo == "texto":
        contenido = normalizar_texto(texto)
        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="Debes escribir un texto para evaluar.",
            )
        return "texto", contenido

    if tipo == "audio":
        if archivo_audio is None:
            raise HTTPException(
                status_code=400,
                detail="Debes grabar y enviar un audio.",
            )

        contenido = await _transcribir_audio_recibido(archivo_audio)
        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="No se pudo transcribir el audio.",
            )

        return "audio", contenido

    raise HTTPException(
        status_code=400,
        detail="tipo_entrada debe ser 'texto' o 'audio'.",
    )


def _primer_valor_no_nulo(*valores):
    for valor in valores:
        if valor is not None and valor != "":
            return valor
    return None


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

        outer = resultado
        body = _resultado_cuerpo(resultado)

        evaluacion = body.get("evaluacion") or {}
        if not isinstance(evaluacion, dict):
            evaluacion = {}

        evaluacion_semantica = body.get("evaluacion_semantica") or {}
        if not isinstance(evaluacion_semantica, dict):
            evaluacion_semantica = {}

        evaluacion_rubrica = body.get("evaluacion_rubrica") or {}
        if not isinstance(evaluacion_rubrica, dict):
            evaluacion_rubrica = {}

        retroalimentacion = body.get("retroalimentacion") or {}
        if not isinstance(retroalimentacion, dict):
            retroalimentacion = {}

        caso = body.get("caso") or outer.get("caso")
        if not isinstance(caso, dict):
            caso = None

        summary = outer.get("summary") if isinstance(outer.get("summary"), dict) else {}
        if not isinstance(summary, dict):
            summary = {}

        faithfulness = (
            outer.get("faithfulness")
            if outer.get("faithfulness") is not None
            else summary.get("faithfulness")
        )
        answer_relevancy = (
            outer.get("answer_relevancy")
            if outer.get("answer_relevancy") is not None
            else summary.get("answer_relevancy")
        )

        score_rubric = _primer_valor_no_nulo(
            evaluacion_rubrica.get("puntaje_total"),
            body.get("puntaje_rubrica"),
            outer.get("puntaje_rubrica"),
        )
        if score_rubric is None:
            score_rubric = _primer_valor_no_nulo(
                evaluacion.get("puntaje_total"),
                body.get("puntaje_total"),
                outer.get("puntaje_total"),
            )

        score_semantic = _primer_valor_no_nulo(
            evaluacion_semantica.get("puntaje_total"),
            body.get("puntaje_semantico"),
            outer.get("puntaje_semantico"),
        )

        score_consolidado = _primer_valor_no_nulo(
            body.get("puntaje_total_consolidado"),
            outer.get("puntaje_total_consolidado"),
        )

        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": caso_id,
            "caso_id": caso_id,
            "benchmark_id": benchmark_id,
            "sample_id": sample_id,
            "pipeline": pipeline,
            "answer": answer,
            "case": caso,
            "score_total": score_rubric,
            "score_rubric": score_rubric,
            "score_semantic": score_semantic,
            "score_total_consolidado": score_consolidado,
            "relevance_case": (
                evaluacion.get("indice_relevancia_caso")
                or evaluacion_semantica.get("indice_relevancia_caso")
                or body.get("indice_relevancia_caso")
                or outer.get("indice_relevancia_caso")
            ),
            "relevance_lexica": (
                evaluacion_semantica.get("indice_relevancia_lexica")
                or body.get("indice_relevancia_lexica")
                or outer.get("indice_relevancia_lexica")
            ),
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "feedback": (
                evaluacion_rubrica.get("resumen")
                or retroalimentacion.get("resumen")
                or evaluacion.get("resumen")
                or ""
            ),
            "response_json": outer,
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
    try:
        componentes = _extraer_componentes_benchmark(resultado)

        if componentes:
            if "benchmark" in componentes:
                _guardar_resultado_en_historial(
                    caso_id=caso_id,
                    benchmark_id=benchmark_id,
                    sample_id=sample_id,
                    pipeline="benchmark",
                    answer=answer,
                    resultado=componentes["benchmark"],
                )
            elif pipeline == "benchmark":
                _guardar_resultado_en_historial(
                    caso_id=caso_id,
                    benchmark_id=benchmark_id,
                    sample_id=sample_id,
                    pipeline="benchmark",
                    answer=answer,
                    resultado=resultado,
                )

            if "langgraph" in componentes:
                _guardar_resultado_en_historial(
                    caso_id=caso_id,
                    benchmark_id=benchmark_id,
                    sample_id=sample_id,
                    pipeline="langgraph",
                    answer=answer,
                    resultado=componentes["langgraph"],
                )

            if "langchain" in componentes:
                _guardar_resultado_en_historial(
                    caso_id=caso_id,
                    benchmark_id=benchmark_id,
                    sample_id=sample_id,
                    pipeline="langchain",
                    answer=answer,
                    resultado=componentes["langchain"],
                )
            return

        _guardar_resultado_en_historial(
            caso_id=caso_id,
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            pipeline=pipeline,
            answer=answer,
            resultado=resultado,
        )
    except Exception:
        pass


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
            "benchmark_id": benchmark_id
            or resultado.get("benchmark_id_used")
            or resultado.get("benchmark_id")
            or "",
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


async def _evaluar_langgraph_base(
    *,
    caso_id: str,
    tipo_entrada: str,
    texto: str,
    benchmark_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    archivo_audio: Optional[UploadFile] = None,
) -> dict[str, Any]:
    benchmark_id_final, sample_id_final = _generar_identificadores_benchmark(
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
    )

    tipo_normalizado, texto_procesado = await _obtener_texto_procesado(
        tipo_entrada=tipo_entrada,
        texto=texto,
        archivo_audio=archivo_audio,
    )

    _registrar_respuesta_recibida(
        caso_id=caso_id,
        tipo_entrada=tipo_normalizado,
        texto=texto_procesado,
        benchmark_id=benchmark_id_final,
        sample_id=sample_id_final,
    )

    from backend.agents.argumentation_graph import ejecutar_evaluacion_langgraph

    resultado = ejecutar_evaluacion_langgraph(
        caso_id=caso_id,
        tipo_entrada=tipo_normalizado,
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


async def _evaluar_langchain_base(
    *,
    caso_id: str,
    tipo_entrada: str,
    texto: str,
    benchmark_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    archivo_audio: Optional[UploadFile] = None,
) -> dict[str, Any]:
    benchmark_id_final, sample_id_final = _generar_identificadores_benchmark(
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
    )

    tipo_normalizado, texto_procesado = await _obtener_texto_procesado(
        tipo_entrada=tipo_entrada,
        texto=texto,
        archivo_audio=archivo_audio,
    )

    _registrar_respuesta_recibida(
        caso_id=caso_id,
        tipo_entrada=tipo_normalizado,
        texto=texto_procesado,
        benchmark_id=benchmark_id_final,
        sample_id=sample_id_final,
    )

    from backend.services.langchain_bridge import ejecutar_evaluacion_langchain

    resultado = ejecutar_evaluacion_langchain(
        caso_id=caso_id,
        tipo_entrada=tipo_normalizado,
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


async def _evaluar_benchmark_base(
    *,
    caso_id: str,
    tipo_entrada: str,
    texto: str,
    benchmark_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    archivo_audio: Optional[UploadFile] = None,
) -> dict[str, Any]:
    benchmark_id_final, sample_id_final = _generar_identificadores_benchmark(
        caso_id=caso_id,
        benchmark_id=benchmark_id,
        sample_id=sample_id,
    )

    tipo_normalizado, texto_procesado = await _obtener_texto_procesado(
        tipo_entrada=tipo_entrada,
        texto=texto,
        archivo_audio=archivo_audio,
    )

    _registrar_respuesta_recibida(
        caso_id=caso_id,
        tipo_entrada=tipo_normalizado,
        texto=texto_procesado,
        benchmark_id=benchmark_id_final,
        sample_id=sample_id_final,
    )

    from backend.services.benchmark_orchestrator import ejecutar_benchmark_dual

    resultado = ejecutar_benchmark_dual(
        caso_id=caso_id,
        tipo_entrada_original=tipo_normalizado,
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


@app.get("/")
def raiz():
    return {
        "message": "Agentic RAG API funcionando",
        "endpoints": [
            "/casos",
            "/caso/{caso_id}",
            "/benchmark/start",
            "/evaluar-entrada",
            "/evaluar-langgraph",
            "/evaluar-langgraph-voz",
            "/evaluar-langchain",
            "/evaluar-langchain-voz",
            "/benchmark/evaluar",
            "/benchmark/evaluar-voz",
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


@app.post("/evaluar-entrada")
async def evaluar_entrada(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form("texto"),
    texto: str = Form(""),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: Optional[UploadFile] = File(None),
):
    try:
        resultado = await _evaluar_langgraph_base(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto,
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            archivo_audio=archivo_audio,
        )
        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-langgraph")
async def evaluar_langgraph(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form("texto"),
    texto: str = Form(""),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: Optional[UploadFile] = File(None),
):
    try:
        resultado = await _evaluar_langgraph_base(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto,
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            archivo_audio=archivo_audio,
        )
        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-langgraph-voz")
async def evaluar_langgraph_voz(
    caso_id: str = Form("caso_001"),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: UploadFile = File(...),
):
    try:
        resultado = await _evaluar_langgraph_base(
            caso_id=caso_id,
            tipo_entrada="audio",
            texto="",
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            archivo_audio=archivo_audio,
        )
        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-langchain")
async def evaluar_langchain(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form("texto"),
    texto: str = Form(""),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: Optional[UploadFile] = File(None),
):
    try:
        resultado = await _evaluar_langchain_base(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto,
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            archivo_audio=archivo_audio,
        )
        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/evaluar-langchain-voz")
async def evaluar_langchain_voz(
    caso_id: str = Form("caso_001"),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: UploadFile = File(...),
):
    try:
        resultado = await _evaluar_langchain_base(
            caso_id=caso_id,
            tipo_entrada="audio",
            texto="",
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            archivo_audio=archivo_audio,
        )
        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/benchmark/evaluar")
async def evaluar_benchmark(
    caso_id: str = Form("caso_001"),
    tipo_entrada: str = Form("texto"),
    texto: str = Form(""),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: Optional[UploadFile] = File(None),
):
    try:
        resultado = await _evaluar_benchmark_base(
            caso_id=caso_id,
            tipo_entrada=tipo_entrada,
            texto=texto,
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            archivo_audio=archivo_audio,
        )
        return resultado
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/benchmark/evaluar-voz")
async def evaluar_benchmark_voz(
    caso_id: str = Form("caso_001"),
    benchmark_id: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    archivo_audio: UploadFile = File(...),
):
    try:
        resultado = await _evaluar_benchmark_base(
            caso_id=caso_id,
            tipo_entrada="audio",
            texto="",
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            archivo_audio=archivo_audio,
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