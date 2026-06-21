from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from datetime import datetime, timezone
import json

BASE_DIR = Path(__file__).resolve().parents[2]

try:
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
except Exception:  # pragma: no cover
    from ragas import EvaluationDataset, SingleTurnSample  # type: ignore


DEFAULT_LOG_FILES = [
    BASE_DIR / "logs" / "eventos.jsonl",
    BASE_DIR / "data" / "logs" / "eventos.jsonl",
    BASE_DIR / "data" / "logs" / "langgraph" / "eventos.jsonl",
    BASE_DIR / "data" / "logs" / "langchain" / "eventos.jsonl",
]

ALLOWED_EVENT_TYPES = {"evaluacion_langgraph", "evaluacion_langchain"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _context_item_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in (
            "fragmento",
            "texto",
            "text",
            "content",
            "context",
            "contexto",
            "chunk",
            "document",
            "documento",
            "respuesta_contexto",
        ):
            candidate = _normalize_text(value.get(key))
            if candidate:
                return candidate
        return _normalize_text(value)
    return _normalize_text(value)


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (_context_item_to_text(x) for x in value) if item]
    if isinstance(value, tuple):
        return [item for item in (_context_item_to_text(x) for x in value) if item]
    text = _context_item_to_text(value)
    return [text] if text else []


def _parse_timestamp(value: Any) -> datetime:
    fallback = datetime.min.replace(tzinfo=timezone.utc)
    text = _normalize_text(value)
    if not text:
        return fallback

    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return fallback


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
                if isinstance(item, dict):
                    rows.append(item)
            except Exception:
                continue
    return rows


def _iter_candidate_events(log_files: Iterable[Path]):
    for path in log_files:
        for row in _load_jsonl(path):
            event_name = (
                _normalize_text(row.get("tipo"))
                or _normalize_text(row.get("event_type"))
                or _normalize_text(row.get("event"))
                or _normalize_text(row.get("pipeline"))
            )

            data = row.get("datos", row)
            if not isinstance(data, dict):
                continue

            payload_pipeline = _normalize_text(data.get("pipeline")) or event_name
            benchmark_id = _normalize_text(
                data.get("benchmark_id")
                or row.get("benchmark_id")
                or data.get("run_id")
                or row.get("run_id")
            )
            timestamp = _parse_timestamp(row.get("timestamp") or data.get("timestamp"))

            yield {
                "event_name": event_name,
                "payload_pipeline": payload_pipeline,
                "benchmark_id": benchmark_id,
                "timestamp": timestamp,
                "payload": data,
            }


def find_latest_benchmark_id(log_files: list[str | Path] | None = None) -> str | None:
    files = [Path(p) for p in (log_files or DEFAULT_LOG_FILES)]

    latest_benchmark_id: str | None = None
    latest_timestamp = datetime.min.replace(tzinfo=timezone.utc)

    for item in _iter_candidate_events(files):
        if item["event_name"] not in ALLOWED_EVENT_TYPES:
            continue

        benchmark_id = item["benchmark_id"]
        if not benchmark_id:
            continue

        if item["timestamp"] >= latest_timestamp:
            latest_timestamp = item["timestamp"]
            latest_benchmark_id = benchmark_id

    return latest_benchmark_id


def _build_sample_from_payload(payload: dict[str, Any]) -> SingleTurnSample:
    user_input = _normalize_text(payload.get("user_input"))
    response = _normalize_text(payload.get("response"))
    retrieved_contexts = _normalize_list(payload.get("retrieved_contexts"))
    reference_contexts = _normalize_list(payload.get("reference_contexts"))
    reference = _normalize_text(payload.get("reference"))

    sample_kwargs: dict[str, Any] = {
        "user_input": user_input,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
    }

    if reference_contexts:
        sample_kwargs["reference_contexts"] = reference_contexts
    if reference:
        sample_kwargs["reference"] = reference

    return SingleTurnSample(**sample_kwargs)


def build_dataset_from_logs(
    log_files: list[str | Path] | None = None,
    benchmark_id: str | None = None,
):
    """
    Construye un dataset compatible con RAGAS únicamente desde eventos de evaluación
    reales (evaluacion_langgraph / evaluacion_langchain).

    Si benchmark_id es None, se toma la corrida más reciente encontrada en los logs.
    """
    files = [Path(p) for p in (log_files or DEFAULT_LOG_FILES)]
    selected_benchmark_id = benchmark_id or find_latest_benchmark_id(files)

    samples: list[SingleTurnSample] = []

    for item in _iter_candidate_events(files):
        event_name = item["event_name"]
        payload = item["payload"]
        payload_benchmark_id = item["benchmark_id"]

        if event_name not in ALLOWED_EVENT_TYPES:
            continue

        if selected_benchmark_id and payload_benchmark_id != selected_benchmark_id:
            continue

        user_input = _normalize_text(payload.get("user_input"))
        response = _normalize_text(payload.get("response"))
        retrieved_contexts = _normalize_list(payload.get("retrieved_contexts"))

        if not user_input or not response or not retrieved_contexts:
            continue

        samples.append(_build_sample_from_payload(payload))

    if not samples:
        return EvaluationDataset(samples=[])

    try:
        return EvaluationDataset(samples=samples)
    except Exception:
        return EvaluationDataset.from_list(samples)  # type: ignore[attr-defined]