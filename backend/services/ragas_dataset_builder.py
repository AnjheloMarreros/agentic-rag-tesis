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

# CHANGED: agregamos benchmark_dual_ejecucion para poder reconstruir ambas muestras
# desde una sola corrida comparativa, sin pedirle al alumno que responda dos veces.
ALLOWED_EVENT_TYPES = {"evaluacion_langgraph", "evaluacion_langchain", "benchmark_dual_ejecucion"}


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


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _normalize_text(value)
        if text:
            return text
    return ""


def _merge_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    # CHANGED: fusiona payloads sin perder información útil y sin reemplazar
    # valores válidos por vacíos.
    merged: dict[str, Any] = {}

    for payload in payloads:
        if not isinstance(payload, dict):
            continue

        for key, value in payload.items():
            if value is None:
                continue

            if isinstance(value, str) and not value.strip():
                continue

            if isinstance(value, (list, tuple, dict)) and len(value) == 0:
                continue

            merged[key] = value

    return merged


def _expand_benchmark_dual_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    CHANGED:
    Si el evento es benchmark_dual_ejecucion, extraemos dos payloads:
    uno para langgraph y otro para langchain, reutilizando una sola entrada.
    """
    if _normalize_text(payload.get("pipeline")) != "benchmark" and _normalize_text(payload.get("event_type")) != "benchmark_dual_ejecucion":
        return [payload]

    benchmark_block = payload.get("benchmark")
    comparative: dict[str, Any] = {}

    if isinstance(benchmark_block, dict):
        resultado_final = benchmark_block.get("resultado_final")
        if isinstance(resultado_final, dict):
            comparative = resultado_final.get("comparativa") or {}

    # Fallbacks por si la estructura viene en otro lugar
    if not comparative:
        outer_result = payload.get("result")
        if isinstance(outer_result, dict):
            reports = outer_result.get("reports")
            if isinstance(reports, dict):
                comparative = reports

    langgraph_payload = _merge_payloads(
        payload,
        comparative.get("langgraph", {}) if isinstance(comparative, dict) else {},
        payload.get("langgraph", {}) if isinstance(payload.get("langgraph"), dict) else {},
    )
    langgraph_payload["pipeline"] = "langgraph"
    langgraph_payload["source_event"] = "benchmark_dual_ejecucion"

    langchain_payload = _merge_payloads(
        payload,
        comparative.get("langchain", {}) if isinstance(comparative, dict) else {},
        payload.get("langchain", {}) if isinstance(payload.get("langchain"), dict) else {},
    )
    langchain_payload["pipeline"] = "langchain"
    langchain_payload["source_event"] = "benchmark_dual_ejecucion"

    expanded: list[dict[str, Any]] = []
    if langgraph_payload:
        expanded.append(langgraph_payload)
    if langchain_payload:
        expanded.append(langchain_payload)

    return expanded or [payload]


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

        # CHANGED: si viene una corrida comparativa, la dividimos en dos muestras:
        # una para LangGraph y otra para LangChain, usando la misma respuesta/entrada.
        expanded_payloads = _expand_benchmark_dual_payload(payload)

        for expanded in expanded_payloads:
            if not isinstance(expanded, dict):
                continue

            user_input = _build_user_input(expanded)
            response = _normalize_text(
                expanded.get("response")
                or expanded.get("entrada")
                or expanded.get("texto_procesado")
                or expanded.get("entrada_estudiante")
                or expanded.get("texto")
            )
            retrieved_contexts = _normalize_list(
                expanded.get("retrieved_contexts")
                or expanded.get("contexto_recuperado")
            )

            if not user_input or not response or not retrieved_contexts:
                continue

            samples.append(_build_sample_from_payload(expanded))

    if not samples:
        return EvaluationDataset(samples=[])

    try:
        return EvaluationDataset(samples=samples)
    except Exception:
        return EvaluationDataset.from_list(samples)