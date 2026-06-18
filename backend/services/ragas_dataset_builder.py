from __future__ import annotations

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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_normalize_text(item) for item in value if _normalize_text(item)]
    if isinstance(value, tuple):
        return [_normalize_text(item) for item in value if _normalize_text(item)]
    text = _normalize_text(value)
    return [text] if text else []


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


def _iter_candidate_events(
    log_files: Iterable[Path],
    event_types: set[str] | None = None,
    pipelines: set[str] | None = None,
):
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

            if event_types and event_name not in event_types:
                continue
            if pipelines and payload_pipeline not in pipelines:
                continue

            yield event_name, payload_pipeline, data


def _build_sample_from_payload(payload: dict[str, Any]) -> SingleTurnSample:
    user_input = _normalize_text(
        payload.get("user_input")
        or payload.get("entrada_estudiante")
        or payload.get("texto")
        or payload.get("response")
        or payload.get("respuesta")
    )

    response = _normalize_text(
        payload.get("response")
        or payload.get("respuesta")
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

    # Campos opcionales útiles para futuras métricas.
    if reference_contexts:
        sample_kwargs["reference_contexts"] = reference_contexts
    if reference:
        sample_kwargs["reference"] = reference

    return SingleTurnSample(**sample_kwargs)


def build_dataset_from_logs(
    log_files: list[str | Path] | None = None,
    event_types: Iterable[str] | None = None,
    pipelines: Iterable[str] | None = None,
):
    """
    Construye un dataset compatible con RAGAS desde logs JSONL.

    event_types:
        Filtra por el tipo de evento guardado en el log.
        Ejemplo: "evaluacion_langgraph", "evaluacion_langchain".

    pipelines:
        Filtra por el campo interno pipeline.
        Ejemplo: "langgraph", "langchain".
    """
    files = [Path(p) for p in (log_files or DEFAULT_LOG_FILES)]
    event_types_set = {str(x).strip() for x in event_types} if event_types else None
    pipelines_set = {str(x).strip() for x in pipelines} if pipelines else None

    samples: list[SingleTurnSample] = []

    for _, _, payload in _iter_candidate_events(
        files,
        event_types=event_types_set,
        pipelines=pipelines_set,
    ):
        user_input = _normalize_text(
            payload.get("user_input")
            or payload.get("entrada_estudiante")
            or payload.get("texto")
            or payload.get("response")
            or payload.get("respuesta")
        )
        response = _normalize_text(
            payload.get("response")
            or payload.get("respuesta")
            or payload.get("entrada_estudiante")
            or payload.get("texto")
        )
        retrieved_contexts = _normalize_list(
            payload.get("retrieved_contexts")
            or payload.get("contexto_recuperado")
        )

        if not user_input or not response:
            continue
        if not retrieved_contexts:
            # Faithfulness necesita contexto recuperado.
            continue

        samples.append(_build_sample_from_payload(payload))

    if not samples:
        # Dataset vacío pero válido.
        return EvaluationDataset(samples=[])

    try:
        return EvaluationDataset(samples=samples)
    except Exception:
        # Compatibilidad con otras firmas.
        return EvaluationDataset.from_list(samples)  # type: ignore[attr-defined]