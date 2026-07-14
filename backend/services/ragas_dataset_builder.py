from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
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
            except Exception:
                continue

            if isinstance(item, dict):
                rows.append(item)

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

            benchmark_id = _normalize_text(
                data.get("benchmark_id")
                or row.get("benchmark_id")
                or data.get("run_id")
                or row.get("run_id")
            )

            yield {
                "event_name": event_name,
                "benchmark_id": benchmark_id,
                "timestamp": _parse_timestamp(row.get("timestamp") or data.get("timestamp")),
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


def _build_user_input(payload: dict[str, Any]) -> str:
    caso = payload.get("caso")
    if isinstance(caso, dict):
        enunciado = _normalize_text(caso.get("enunciado"))
        if enunciado:
            return enunciado

        titulo = _normalize_text(caso.get("titulo"))
        if titulo:
            return titulo

    enunciado = _normalize_text(payload.get("enunciado"))
    if enunciado:
        return enunciado

    candidate = _normalize_text(payload.get("user_input"))
    if candidate:
        return candidate

    texto_caso = _normalize_text(payload.get("texto_caso"))
    if texto_caso:
        return texto_caso

    return ""


def _build_sample_from_payload(payload: dict[str, Any]) -> SingleTurnSample:
    user_input = _build_user_input(payload)

    response = _normalize_text(
        payload.get("response")
        or payload.get("entrada")
        or payload.get("texto_procesado")
        or payload.get("entrada_estudiante")
        or payload.get("texto")
    )

    retrieved_contexts = _normalize_list(
        payload.get("retrieved_contexts")
        or payload.get("contexto_recuperado")
    )

    reference_contexts = _normalize_list(
        payload.get("reference_contexts")
        or payload.get("referencia_contextos")
        or payload.get("contexto_referencia")
        or payload.get("reference_context")
    )

    reference = _normalize_text(
        payload.get("reference")
        or payload.get("reference_answer")
        or payload.get("respuesta_referencia")
    )

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
    files = [Path(p) for p in (log_files or DEFAULT_LOG_FILES)]
    selected_benchmark_id = benchmark_id or find_latest_benchmark_id(files)

    samples: list[SingleTurnSample] = []

    for item in _iter_candidate_events(files):
        if item["event_name"] not in ALLOWED_EVENT_TYPES:
            continue

        if selected_benchmark_id and item["benchmark_id"] != selected_benchmark_id:
            continue

        payload = item["payload"]
        if not isinstance(payload, dict):
            continue

        user_input = _build_user_input(payload)
        response = _normalize_text(
            payload.get("response")
            or payload.get("entrada")
            or payload.get("texto_procesado")
            or payload.get("entrada_estudiante")
            or payload.get("texto")
        )
        retrieved_contexts = _normalize_list(
            payload.get("retrieved_contexts")
            or payload.get("contexto_recuperado")
        )

        if not user_input or not response or not retrieved_contexts:
            continue

        samples.append(_build_sample_from_payload(payload))

    if not samples:
        return EvaluationDataset(samples=[])

    try:
        return EvaluationDataset(samples=samples)
    except Exception:
        return EvaluationDataset.from_list(samples)