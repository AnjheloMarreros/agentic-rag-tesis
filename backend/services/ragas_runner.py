from __future__ import annotations

from pathlib import Path
from typing import Any
from math import isfinite
import json
import os

from dotenv import load_dotenv

from backend.services.ragas_dataset_builder import build_dataset_from_logs

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
RESULTS_DIR = BASE_DIR / "data" / "evals" / "ragas_results"


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
    for v in values:
        numeric = _to_float(v)
        if numeric is not None:
            nums.append(numeric)
    return round(sum(nums) / len(nums), 4) if nums else 0.0


def _build_models():
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    embedding_model = os.getenv(
        "RAGAS_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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

    base_llm = ChatGroq(
        model=groq_model,
        api_key=groq_api_key,
        temperature=0,
    )

    base_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
    )

    return LangchainLLMWrapper(base_llm), LangchainEmbeddingsWrapper(base_embeddings)


def _load_metric_symbols():
    try:
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy
        return Faithfulness, AnswerRelevancy
    except Exception:
        from ragas.metrics import faithfulness, answer_relevancy
        return faithfulness, answer_relevancy


def _metric_instance(symbol: Any):
    if isinstance(symbol, type):
        return symbol()
    return symbol


def _build_metric_list(llm, embeddings):
    Faithfulness, AnswerRelevancy = _load_metric_symbols()

    faithfulness_metric = _metric_instance(Faithfulness)
    answer_relevancy_metric = _metric_instance(AnswerRelevancy)

    try:
        if hasattr(faithfulness_metric, "llm"):
            faithfulness_metric.llm = llm
    except Exception:
        pass

    try:
        if hasattr(answer_relevancy_metric, "llm"):
            answer_relevancy_metric.llm = llm
    except Exception:
        pass

    try:
        if hasattr(answer_relevancy_metric, "embeddings"):
            answer_relevancy_metric.embeddings = embeddings
    except Exception:
        pass

    try:
        if hasattr(answer_relevancy_metric, "strictness"):
            answer_relevancy_metric.strictness = 1
    except Exception:
        pass

    return [
        faithfulness_metric,
        answer_relevancy_metric,
    ]


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


def run_ragas_live_evaluation() -> dict[str, Any]:
    dataset = build_dataset_from_logs()
    llm, embeddings = _build_models()

    print("=" * 60)
    print("GROQ_MODEL =", os.getenv("GROQ_MODEL"))
    print("LLM =", type(llm))
    print("EMBEDDINGS =", type(embeddings))
    print("=" * 60)

    metrics = _build_metric_list(llm, embeddings)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = _evaluate_dataset(dataset, metrics, llm, embeddings)
    except Exception as exc:
        message = str(exc)
        lower_message = message.lower()

        if "resource_exhausted" in lower_message or "quota" in lower_message or "429" in lower_message:
            return {
                "ok": False,
                "status": "quota_exhausted",
                "provider": "groq",
                "message": (
                    "El proveedor de LLM no tiene cuota disponible para continuar "
                    "con la evaluación RAGAS en este momento."
                ),
                "detail": message,
                "summary": {},
                "num_samples": 0,
                "rows": [],
                "output_csv": None,
            }

        raise

    df = result.to_pandas()

    output_csv = RESULTS_DIR / "ragas_latest.csv"
    df.to_csv(output_csv, index=False)

    metric_columns = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy", "response_relevancy"],
    }

    summary: dict[str, float] = {}
    for key, candidates in metric_columns.items():
        found_value = None
        for col in candidates:
            if col in df.columns:
                found_value = _mean_safe(df[col].tolist())
                break
        summary[key] = found_value if found_value is not None else 0.0

    cumple_objetivo_general = (
        summary.get("faithfulness", 0.0) >= 0.85
        and summary.get("answer_relevancy", 0.0) >= 0.80
    )

    report = {
        "ok": True,
        "status": "completed",
        "provider": "groq",
        "summary": summary,
        "cumple_objetivo_general": cumple_objetivo_general,
        "num_samples": int(len(df)),
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

    return report