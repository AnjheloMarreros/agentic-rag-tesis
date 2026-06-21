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


def _extract_nested_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Soporta:
    - eventos planos: evaluacion_langgraph / evaluacion_langchain
    - evento agregado: benchmark_dual_ejecucion con langgraph/langchain anidados
    """
    nested: list[dict[str, Any]] = []

    for pipeline_name in ("langgraph", "langchain"):
        sub = payload.get(pipeline_name)
        if isinstance(sub, dict):
            resultado = sub.get("resultado_final")
            if isinstance(resultado, dict):
                nested.append(
                    {
                        "user_input": resultado.get("texto_caso") or payload.get("texto_procesado") or "",
                        "response": resultado.get("entrada") or resultado.get("entrada_estudiante") or "",
                        "retrieved_contexts": resultado.get("contexto_recuperado") or [],
                        "reference_contexts": resultado.get("caso", {}).get("contexto", []),
                        "reference": "",
                        "sample_id": payload.get("sample_id", ""),
                        "benchmark_id": payload.get("benchmark_id", ""),
                    }
                )

    return nested


def _build_sample_from_payload(payload: dict[str, Any]) -> SingleTurnSample:
    user_input = _normalize_text(
        payload.get("user_input")
        or payload.get("entrada_estudiante")
        or payload.get("texto")
        or payload.get("response")
        or payload.get("respuesta")
        or payload.get("texto_procesado")
        or payload.get("question")
        or payload.get("entrada")
        or payload.get("input")
    )

    response = _normalize_text(
        payload.get("response")
        or payload.get("respuesta")
        or payload.get("entrada_estudiante")
        or payload.get("texto")
        or payload.get("texto_respuesta")
        or payload.get("answer")
    )

    retrieved_contexts = _normalize_list(
        payload.get("retrieved_contexts")
        or payload.get("contexto_recuperado")
        or payload.get("fuentes")
        or payload.get("contexto")
        or payload.get("context")
        or payload.get("documents")
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
    event_types: Iterable[str] | None = None,
    pipelines: Iterable[str] | None = None,
):
    files = [Path(p) for p in (log_files or DEFAULT_LOG_FILES)]
    event_types_set = {str(x).strip() for x in event_types} if event_types else None
    pipelines_set = {str(x).strip() for x in pipelines} if pipelines else None

    samples: list[SingleTurnSample] = []

    for event_name, pipeline_name, payload in _iter_candidate_events(
        files,
        event_types=event_types_set,
        pipelines=pipelines_set,
    ):
        # Caso 1: evento plano ya compatible
        user_input = _normalize_text(
            payload.get("user_input")
            or payload.get("entrada_estudiante")
            or payload.get("texto")
            or payload.get("response")
            or payload.get("respuesta")
            or payload.get("texto_procesado")
            or payload.get("question")
            or payload.get("entrada")
            or payload.get("input")
        )
        response = _normalize_text(
            payload.get("response")
            or payload.get("respuesta")
            or payload.get("entrada_estudiante")
            or payload.get("texto")
            or payload.get("texto_respuesta")
            or payload.get("answer")
        )
        retrieved_contexts = _normalize_list(
            payload.get("retrieved_contexts")
            or payload.get("contexto_recuperado")
            or payload.get("fuentes")
            or payload.get("contexto")
            or payload.get("context")
            or payload.get("documents")
        )

        if user_input and response:
            samples.append(_build_sample_from_payload(payload))
            continue

        # Caso 2: benchmark_dual_ejecucion con payload anidado
        if event_name == "benchmark_dual_ejecucion" or pipeline_name == "benchmark_dual_ejecucion":
            for nested_payload in _extract_nested_payloads(payload):
                nested_user_input = _normalize_text(nested_payload.get("user_input"))
                nested_response = _normalize_text(nested_payload.get("response"))
                nested_contexts = _normalize_list(nested_payload.get("retrieved_contexts"))

                if not nested_user_input or not nested_response:
                    continue

                # Para Faithfulness, solo dejamos pasar si hay contexto.
                if not nested_contexts:
                    continue

                samples.append(_build_sample_from_payload(nested_payload))

    if not samples:
        return EvaluationDataset(samples=[])

    try:
        return EvaluationDataset(samples=samples)
    except Exception:
        return EvaluationDataset.from_list(samples)  # type: ignore[attr-defined]