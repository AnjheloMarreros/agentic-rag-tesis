from __future__ import annotations

from csv import DictWriter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
import json
import os
from uuid import uuid4

from ragas import evaluate
from ragas.run_config import RunConfig

try:
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
except Exception:  # pragma: no cover
    from ragas import EvaluationDataset, SingleTurnSample  # type: ignore

try:
    from ragas.llms import LangchainLLMWrapper
except Exception:  # pragma: no cover
    from ragas.llms.base import LangchainLLMWrapper  # type: ignore

try:
    from ragas.embeddings import LangchainEmbeddingsWrapper
except Exception:  # pragma: no cover
    from ragas.embeddings.base import LangchainEmbeddingsWrapper  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[2]

LOG_FILES = [
    BASE_DIR / "logs" / "eventos.jsonl",
    BASE_DIR / "data" / "logs" / "eventos.jsonl",
    BASE_DIR / "data" / "logs" / "langgraph" / "eventos.jsonl",
    BASE_DIR / "data" / "logs" / "langchain" / "eventos.jsonl",
]

OUTPUT_DIR = BASE_DIR / "data" / "evals" / "ragas_benchmark"

PIPELINES = {
    "langgraph": "evaluacion_langgraph",
    "langchain": "evaluacion_langchain",
}

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = Lock()


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
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return None
        return numeric
    if hasattr(value, "item"):
        try:
            return _clean_jsonable(value.item())
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
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
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


def _job_set(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        current = JOBS.get(job_id, {})
        current.update(updates)
        JOBS[job_id] = current


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        cleaned: list[str] = []
        for item in value:
            text = _normalize_text(item)
            if text:
                cleaned.append(text)
        return cleaned
    text = _normalize_text(value)
    return [text] if text else []


def _extract_payload(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("datos", row)
    return data if isinstance(data, dict) else {}


def _iter_events(event_types: set[str] | None = None):
    for path in LOG_FILES:
        for row in _load_jsonl(path):
            tipo = _normalize_text(row.get("tipo") or row.get("event_type") or row.get("event"))
            payload = _extract_payload(row)

            if not payload:
                continue
            if event_types and tipo not in event_types:
                continue

            pipeline = _normalize_text(payload.get("pipeline"))
            if not pipeline:
                if "langgraph" in tipo:
                    pipeline = "langgraph"
                elif "langchain" in tipo:
                    pipeline = "langchain"

            if not pipeline:
                continue

            yield {
                "tipo": tipo,
                "pipeline": pipeline,
                "payload": payload,
                "row": row,
            }


def _build_models():
    model_name = os.getenv("RAGAS_MODEL", "llama-3.3-70b-versatile").strip()
    embedding_name = os.getenv(
        "RAGAS_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ).strip()

    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("No se encontró GROQ_API_KEY en el entorno.")

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=embedding_name)

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def _build_metric_list(llm: Any, embeddings: Any):
    from ragas.metrics import Faithfulness, ResponseRelevancy

    return [
        Faithfulness(llm=llm),
        ResponseRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=int(os.getenv("RAGAS_STRICTNESS", "3")),
        ),
    ]


def _evaluate_dataset(dataset, metrics, llm, embeddings):
    run_config = RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT", "180")),
        max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "2")),
        max_wait=int(os.getenv("RAGAS_MAX_WAIT", "10")),
        max_workers=1,
        seed=int(os.getenv("RAGAS_SEED", "42")),
    )

    try:
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
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
                raise_exceptions=False,
                run_config=run_config,
            )
        except TypeError:
            return evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                show_progress=False,
                raise_exceptions=False,
            )


def _build_sample_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    user_input = _normalize_text(
        payload.get("user_input")
        or payload.get("entrada_estudiante")
        or payload.get("texto_procesado")
        or payload.get("question")
        or payload.get("entrada")
        or payload.get("input")
    )

    response = _normalize_text(
        payload.get("response")
        or payload.get("respuesta")
        or payload.get("texto_respuesta")
        or payload.get("answer")
    )

    retrieved_contexts = _normalize_list(
        payload.get("retrieved_contexts")
        or payload.get("contexto_recuperado")
    )

    reference_contexts = _normalize_list(
        payload.get("reference_contexts")
        or payload.get("reference_context")
        or payload.get("contexto_referencia")
    )

    reference = _normalize_text(
        payload.get("reference")
        or payload.get("reference_answer")
        or payload.get("respuesta_referencia")
    )

    sample_id = _normalize_text(payload.get("sample_id"))
    benchmark_id = _normalize_text(payload.get("benchmark_id"))

    return {
        "user_input": user_input,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
        "reference_contexts": reference_contexts,
        "reference": reference,
        "sample_id": sample_id,
        "benchmark_id": benchmark_id,
    }


def _make_dataset_one(sample_kwargs: dict[str, Any]):
    ragas_kwargs: dict[str, Any] = {
        "user_input": sample_kwargs["user_input"],
        "response": sample_kwargs["response"],
        "retrieved_contexts": sample_kwargs["retrieved_contexts"],
    }

    if sample_kwargs.get("reference_contexts"):
        ragas_kwargs["reference_contexts"] = sample_kwargs["reference_contexts"]
    if sample_kwargs.get("reference"):
        ragas_kwargs["reference"] = sample_kwargs["reference"]

    sample = SingleTurnSample(**ragas_kwargs)

    try:
        return EvaluationDataset(samples=[sample])
    except Exception:
        return EvaluationDataset.from_list([sample])  # type: ignore[attr-defined]


def _safe_evaluate_sample(sample_payload: dict[str, Any], metrics, llm, embeddings) -> dict[str, Any]:
    try:
        if not sample_payload.get("user_input") or not sample_payload.get("response"):
            return {
                "ok": False,
                "error": "missing_user_input_or_response",
                **sample_payload,
            }

        dataset = _make_dataset_one(sample_payload)
        result = _evaluate_dataset(dataset, metrics, llm, embeddings)
        df = result.to_pandas()

        if df.empty:
            return {
                "ok": False,
                "error": "empty_result",
                **sample_payload,
            }

        row = df.iloc[0].to_dict()
        row.update(sample_payload)
        row["ok"] = True
        return _clean_jsonable(row)

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            **sample_payload,
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = DictWriter(f, fieldnames=["ok"])
            writer.writeheader()
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_clean_jsonable(row))


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "faithfulness": _mean_safe([row.get("faithfulness") for row in rows]),
        "answer_relevancy": _mean_safe([row.get("answer_relevancy") for row in rows]),
    }


def _empty_report(
    *,
    pipeline_name: str,
    event_type: str,
    status: str,
    message: str,
    num_failed: int = 0,
    failed_rows: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    output_csv: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": status,
        "pipeline": pipeline_name,
        "event_type": event_type,
        "summary": {},
        "num_samples": 0,
        "num_failed": num_failed,
        "failed_rows": failed_rows or [],
        "rows": rows or [],
        "output_csv": output_csv,
        "message": message,
    }


def _run_one_pipeline(
    pipeline_name: str,
    event_type: str,
    llm: Any,
    embeddings: Any,
    out_dir: Path,
) -> dict[str, Any]:
    pipeline_dir = out_dir / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for item in _iter_events(event_types={event_type}):
        if item["pipeline"] != pipeline_name:
            continue
        candidates.append(_build_sample_from_payload(item["payload"]))

    if not candidates:
        report = _empty_report(
            pipeline_name=pipeline_name,
            event_type=event_type,
            status="empty_dataset",
            message="No se encontraron eventos válidos para esta corrida.",
        )

        with open(pipeline_dir / f"{pipeline_name}_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(_clean_jsonable(report), f, ensure_ascii=False, indent=2, allow_nan=False)

        return report

    metrics = _build_metric_list(llm, embeddings)

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for sample_payload in candidates:
        row = _safe_evaluate_sample(sample_payload, metrics, llm, embeddings)
        if row.get("ok"):
            rows.append(row)
        else:
            failed.append(row)

    if not rows:
        report = _empty_report(
            pipeline_name=pipeline_name,
            event_type=event_type,
            status="no_valid_rows",
            message="No se pudieron evaluar filas válidas para esta corrida.",
            num_failed=len(failed),
            failed_rows=failed,
        )

        with open(pipeline_dir / f"{pipeline_name}_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(_clean_jsonable(report), f, ensure_ascii=False, indent=2, allow_nan=False)

        return report

    summary = _summarize_rows(rows)

    output_csv = pipeline_dir / f"{pipeline_name}_benchmark.csv"
    _write_csv(output_csv, rows)

    report = {
        "ok": True,
        "status": "completed",
        "pipeline": pipeline_name,
        "event_type": event_type,
        "summary": summary,
        "num_samples": int(len(rows)),
        "num_failed": int(len(failed)),
        "failed_rows": failed,
        "output_csv": str(output_csv),
        "rows": rows,
    }

    report = _clean_jsonable(report)

    with open(pipeline_dir / f"{pipeline_name}_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    return report


def run_daily_benchmark_ragas_reports() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_DIR / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    llm, embeddings = _build_models()

    reports: dict[str, Any] = {}
    for pipeline_name, event_type in PIPELINES.items():
        reports[pipeline_name] = _run_one_pipeline(
            pipeline_name=pipeline_name,
            event_type=event_type,
            llm=llm,
            embeddings=embeddings,
            out_dir=batch_dir,
        )

    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(batch_dir),
        "reports": reports,
        "delta": {},
    }

    langgraph_summary = reports.get("langgraph", {}).get("summary", {}) or {}
    langchain_summary = reports.get("langchain", {}).get("summary", {}) or {}

    for metric in ("faithfulness", "answer_relevancy"):
        g = float(langgraph_summary.get(metric, 0.0) or 0.0)
        c = float(langchain_summary.get(metric, 0.0) or 0.0)
        comparison["delta"][metric] = round(g - c, 4)

    comparison_path = batch_dir / "comparison_benchmark.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(_clean_jsonable(comparison), f, ensure_ascii=False, indent=2, allow_nan=False)

    summary_csv = batch_dir / "comparison_benchmark.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=["pipeline", "metric", "value"])
        writer.writeheader()
        for pipeline_name in ("langgraph", "langchain"):
            summary = reports.get(pipeline_name, {}).get("summary", {}) or {}
            for metric in ("faithfulness", "answer_relevancy"):
                writer.writerow(
                    {
                        "pipeline": pipeline_name,
                        "metric": metric,
                        "value": summary.get(metric, 0.0),
                    }
                )

    return comparison


def _run_daily_job(job_id: str) -> None:
    _job_set(job_id, status="running", started_at=datetime.now(timezone.utc).isoformat())
    try:
        result = run_daily_benchmark_ragas_reports()
        _job_set(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            result=result,
        )
    except Exception as exc:
        _job_set(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


def start_daily_benchmark_job() -> dict[str, Any]:
    job_id = uuid4().hex
    _job_set(job_id, status="queued", created_at=datetime.now(timezone.utc).isoformat())
    Thread(target=_run_daily_job, args=(job_id,), daemon=True).start()
    return {"ok": True, "job_id": job_id, "status": "queued"}


def get_daily_benchmark_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        return JOBS.get(job_id, {"ok": False, "status": "not_found", "job_id": job_id})


def main() -> int:
    result = run_daily_benchmark_ragas_reports()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())