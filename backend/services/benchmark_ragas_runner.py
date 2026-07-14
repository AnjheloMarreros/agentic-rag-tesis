from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from pprint import pformat
from typing import Any

from dotenv import load_dotenv

from backend.services.ragas_dataset_builder import (
    build_dataset_from_logs,
    find_latest_benchmark_id,
)

# CHANGED: importamos el histórico para que el resultado de RAGAS quede visible
# en /resultados?pipeline=ragas&format=json.
from backend.services.result_store import append_result

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
RESULTS_DIR = BASE_DIR / "data" / "evals" / "ragas_results"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


DEBUG_RAGAS = _env_flag("RAGAS_DEBUG", "0")


def _debug_print(title: str, payload: Any | None = None) -> None:
    if not DEBUG_RAGAS:
        return

    print(f"[RAGAS-DEBUG] {title}")
    if payload is None:
        return

    try:
        print(pformat(payload, width=140, sort_dicts=False))
    except Exception:
        print(str(payload))


def _clean_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_jsonable(v) for v in value]
    if isinstance(value, (int, float)):
        try:
            numeric = float(value)
        except Exception:
            return value
        return numeric if isfinite(numeric) else None
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, (int, float)):
                numeric = float(item)
                return numeric if isfinite(numeric) else None
            return item
        except Exception:
            return value
    return value


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        numeric = float(value)
        if not isfinite(numeric):
            return None
        return numeric
    except Exception:
        return None


def _mean_safe(values: list[Any]) -> float:
    nums: list[float] = []
    for value in values:
        numeric = _to_float(value)
        if numeric is not None:
            nums.append(numeric)
    return round(sum(nums) / len(nums), 4) if nums else 0.0


def _dataset_samples(dataset: Any) -> list[Any]:
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        try:
            return list(samples)
        except Exception:
            pass

    try:
        return list(dataset.to_list())  # type: ignore[attr-defined]
    except Exception:
        return []


def _sample_has_contexts(sample: Any) -> bool:
    if isinstance(sample, dict):
        return bool(sample.get("retrieved_contexts") or sample.get("contexto_recuperado"))

    return bool(getattr(sample, "retrieved_contexts", None))


def _sample_preview(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        return {
            "user_input": sample.get("user_input"),
            "response": sample.get("response"),
            "retrieved_contexts": sample.get("retrieved_contexts"),
            "reference_contexts": sample.get("reference_contexts"),
            "reference": sample.get("reference"),
        }

    preview: dict[str, Any] = {}
    for attr in ("user_input", "response", "retrieved_contexts", "reference_contexts", "reference"):
        if hasattr(sample, attr):
            try:
                preview[attr] = getattr(sample, attr)
            except Exception:
                preview[attr] = "<unreadable>"
    return preview


def _build_models():
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", os.getenv("RAGAS_MODEL", "llama-3.3-70b-versatile")).strip()
    embedding_model = os.getenv(
        "RAGAS_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()

    if not groq_api_key:
        raise RuntimeError("No se encontró GROQ_API_KEY en el entorno.")

    try:
        from ragas.llms.base import LangchainLLMWrapper
    except Exception:
        from ragas.llms import LangchainLLMWrapper  # type: ignore

    try:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
    except Exception:
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore

    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    llm = ChatGroq(
        model=groq_model,
        api_key=groq_api_key,
        temperature=0,
    )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def _build_metric_list(llm, embeddings, include_faithfulness: bool = True):
    from ragas.metrics import Faithfulness, ResponseRelevancy

    metrics = []
    if include_faithfulness:
        metrics.append(Faithfulness(llm=llm))

    metrics.append(
        ResponseRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=int(os.getenv("RAGAS_STRICTNESS", "1")),
        )
    )
    return metrics


def _evaluate_dataset(dataset, metrics, llm, embeddings):
    from ragas import evaluate
    from ragas.run_config import RunConfig

    run_config = RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT", "180")),
        max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "2")),
        max_wait=int(os.getenv("RAGAS_MAX_WAIT", "10")),
        max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "1")),
        seed=int(os.getenv("RAGAS_SEED", "42")),
    )

    try:
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=DEBUG_RAGAS,
            run_config=run_config,
        )
    except TypeError:
        try:
            return evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                show_progress=False,
                raise_exceptions=DEBUG_RAGAS,
                run_config=run_config,
            )
        except TypeError:
            return evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                show_progress=False,
                raise_exceptions=DEBUG_RAGAS,
            )


def _empty_report(
    *,
    status: str,
    benchmark_id: str | None,
    message: str,
    detail: str,
    input_samples: int = 0,
    num_samples: int = 0,
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": status,
        "provider": "groq",
        "benchmark_id_used": benchmark_id,
        "message": message,
        "detail": detail,
        "summary": {},
        "num_samples": num_samples,
        "input_samples": input_samples,
        "rows": [],
        "output_csv": None,
    }


def _persist_ragas_report(report: dict[str, Any]) -> None:
    # CHANGED: guardamos el resultado final para que /resultados?pipeline=ragas
    # pueda mostrar el reporte sin hacerlo manualmente.
    try:
        summary = report.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}

        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": "",
            "caso_id": "",
            "benchmark_id": report.get("benchmark_id_used", "") or "",
            "sample_id": "",
            "pipeline": "ragas",
            "status": report.get("status", ""),
            "provider": report.get("provider", ""),
            "metrics_used": report.get("metrics_used", []),
            "num_samples": report.get("num_samples"),
            "input_samples": report.get("input_samples"),
            "has_retrieved_contexts": report.get("has_retrieved_contexts"),
            "faithfulness": summary.get("faithfulness"),
            "answer_relevancy": summary.get("answer_relevancy"),
            "summary": summary,
            "rows": report.get("rows", []),
            "output_csv": report.get("output_csv"),
            "response_json": report,
        }

        append_result(registro)
    except Exception:
        # No rompemos el benchmark si el histórico falla.
        pass


def run_ragas_live_evaluation(benchmark_id: str | None = None) -> dict[str, Any]:
    selected_benchmark_id = benchmark_id or find_latest_benchmark_id()

    _debug_print("selected_benchmark_id", selected_benchmark_id)

    dataset = build_dataset_from_logs(benchmark_id=selected_benchmark_id)
    samples = _dataset_samples(dataset)

    _debug_print("samples_loaded", len(samples))
    if DEBUG_RAGAS:
        for i, sample in enumerate(samples[:5], start=1):
            _debug_print(f"sample_preview_{i}", _sample_preview(sample))

    if not samples:
        return _empty_report(
            status="empty_dataset",
            benchmark_id=selected_benchmark_id,
            message=(
                "No se encontraron registros compatibles para RAGAS. "
                "El dataset quedó vacío porque no hubo eventos válidos para la corrida seleccionada."
            ),
            detail="build_dataset_from_logs() devolvió 0 muestras.",
        )

    include_faithfulness = any(_sample_has_contexts(sample) for sample in samples)
    _debug_print("has_retrieved_contexts", include_faithfulness)

    llm, embeddings = _build_models()

    print("=" * 60)
    print("BENCHMARK_ID =", selected_benchmark_id)
    print("GROQ_MODEL =", os.getenv("GROQ_MODEL") or os.getenv("RAGAS_MODEL"))
    print("LLM =", type(llm))
    print("EMBEDDINGS =", type(embeddings))
    print("SAMPLES =", len(samples))
    print("HAS_CONTEXTS =", include_faithfulness)
    print("RAGAS_DEBUG =", DEBUG_RAGAS)
    print("=" * 60)

    metrics = _build_metric_list(llm, embeddings, include_faithfulness=include_faithfulness)
    _debug_print("metrics_used", [metric.__class__.__name__ for metric in metrics])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = _evaluate_dataset(dataset, metrics, llm, embeddings)
    except Exception as exc:
        print("[RAGAS-ERROR] evaluate() falló")
        print(f"[RAGAS-ERROR] benchmark_id={selected_benchmark_id}")
        print(f"[RAGAS-ERROR] type={type(exc).__name__}")
        print(f"[RAGAS-ERROR] message={exc}")
        print(traceback.format_exc())

        message = str(exc).lower()
        if "resource_exhausted" in message or "quota" in message or "429" in message:
            return _empty_report(
                status="quota_exhausted",
                benchmark_id=selected_benchmark_id,
                message=(
                    "El proveedor de LLM no tiene cuota disponible para continuar "
                    "con la evaluación RAGAS en este momento."
                ),
                detail=str(exc),
                input_samples=len(samples),
            )

        if DEBUG_RAGAS:
            raise

        return _empty_report(
            status="evaluation_error",
            benchmark_id=selected_benchmark_id,
            message="RAGAS falló durante la evaluación.",
            detail=str(exc),
            input_samples=len(samples),
        )

    try:
        df = result.to_pandas()
    except Exception as exc:
        print("[RAGAS-ERROR] to_pandas() falló")
        print(f"[RAGAS-ERROR] benchmark_id={selected_benchmark_id}")
        print(f"[RAGAS-ERROR] type={type(exc).__name__}")
        print(f"[RAGAS-ERROR] message={exc}")
        print(traceback.format_exc())

        if DEBUG_RAGAS:
            raise

        return _empty_report(
            status="result_to_dataframe_error",
            benchmark_id=selected_benchmark_id,
            message="RAGAS devolvió un resultado que no pudo convertirse a tabla.",
            detail=str(exc),
            input_samples=len(samples),
        )

    if DEBUG_RAGAS:
        _debug_print("df_columns", list(df.columns))
        _debug_print("df_head", df.head(5).to_dict(orient="records"))

    if df.empty:
        return _empty_report(
            status="empty_result",
            benchmark_id=selected_benchmark_id,
            message="RAGAS devolvió un resultado vacío.",
            detail="to_pandas() produjo un DataFrame vacío.",
            input_samples=len(samples),
        )

    output_csv = RESULTS_DIR / "ragas_latest.csv"
    df.to_csv(output_csv, index=False)

    summary: dict[str, float] = {}
    metric_columns = {
        "faithfulness": ("faithfulness",),
        "answer_relevancy": ("answer_relevancy", "response_relevancy"),
    }

    for key, candidates in metric_columns.items():
        summary[key] = 0.0
        for col in candidates:
            if col in df.columns:
                summary[key] = _mean_safe(df[col].tolist())
                break

    if not include_faithfulness:
        summary["faithfulness"] = 0.0

    cumple_objetivo_general = (
        summary.get("faithfulness", 0.0) >= 0.85
        and summary.get("answer_relevancy", 0.0) >= 0.80
    )

    report = {
        "ok": True,
        "status": "completed" if include_faithfulness else "completed_without_faithfulness",
        "provider": "groq",
        "benchmark_id_used": selected_benchmark_id,
        "metrics_used": ["faithfulness", "answer_relevancy"] if include_faithfulness else ["answer_relevancy"],
        "summary": summary,
        "cumple_objetivo_general": cumple_objetivo_general,
        "num_samples": int(len(df)),
        "input_samples": int(len(samples)),
        "has_retrieved_contexts": include_faithfulness,
        "output_csv": str(output_csv),
        "rows": df.to_dict(orient="records"),
    }

    report = _clean_jsonable(report)

    for metric_name, metric_value in report["summary"].items():
        if metric_value is None:
            report["summary"][metric_name] = 0.0

    output_json = RESULTS_DIR / "ragas_latest.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    # CHANGED: persistimos también la corrida en el histórico consultable.
    _persist_ragas_report(report)

    return report