from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.services.ragas_dataset_builder import build_dataset_from_logs
from backend.services.ragas_runner import (
    _build_models,
    _build_metric_list,
    _clean_jsonable,
    _evaluate_dataset,
    _mean_safe,
)


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data" / "evals" / "ragas_daily"

PIPELINES = {
    "langgraph": "evaluacion_langgraph",
    "langchain": "evaluacion_langchain",
}


def _dataset_size(dataset: Any) -> int:
    try:
        return len(dataset)
    except Exception:
        try:
            return len(getattr(dataset, "samples", []))
        except Exception:
            return 0


def _summarize_df(df) -> dict[str, float]:
    summary: dict[str, float] = {}

    metric_columns = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy", "response_relevancy"],
    }

    for key, candidates in metric_columns.items():
        found_value = None
        for col in candidates:
            if col in df.columns:
                found_value = _mean_safe(df[col].tolist())
                break
        summary[key] = found_value if found_value is not None else 0.0

    return summary


def _run_one_pipeline(
    pipeline_name: str,
    event_type: str,
    llm: Any,
    embeddings: Any,
    out_dir: Path,
) -> dict[str, Any]:
    pipeline_dir = out_dir / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset_from_logs(event_types=(event_type,))
    sample_count = _dataset_size(dataset)

    if sample_count == 0:
        report = {
            "ok": False,
            "status": "empty_dataset",
            "pipeline": pipeline_name,
            "event_type": event_type,
            "message": f"No se encontraron eventos para {pipeline_name}.",
            "summary": {},
            "num_samples": 0,
            "rows": [],
            "output_csv": None,
        }

        with open(pipeline_dir / f"{pipeline_name}_daily.json", "w", encoding="utf-8") as f:
            json.dump(_clean_jsonable(report), f, ensure_ascii=False, indent=2, allow_nan=False)

        return report

    metrics = _build_metric_list(llm, embeddings)

    result = _evaluate_dataset(dataset, metrics, llm, embeddings)
    df = result.to_pandas()

    output_csv = pipeline_dir / f"{pipeline_name}_daily.csv"
    df.to_csv(output_csv, index=False)

    summary = _summarize_df(df)

    report = {
        "ok": True,
        "status": "completed",
        "pipeline": pipeline_name,
        "event_type": event_type,
        "summary": summary,
        "num_samples": int(len(df)),
        "output_csv": str(output_csv),
        "rows": df.to_dict(orient="records"),
    }

    report = _clean_jsonable(report)

    with open(pipeline_dir / f"{pipeline_name}_daily.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    return report


def run_daily_ragas_reports() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        "timestamp": datetime.now().isoformat(timespec="seconds"),
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

    comparison_path = batch_dir / "comparison_daily.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(_clean_jsonable(comparison), f, ensure_ascii=False, indent=2, allow_nan=False)

    summary_csv = batch_dir / "comparison_daily.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "metric", "value"])
        for pipeline_name in ("langgraph", "langchain"):
            summary = reports.get(pipeline_name, {}).get("summary", {}) or {}
            for metric in ("faithfulness", "answer_relevancy"):
                writer.writerow([pipeline_name, metric, summary.get(metric, 0.0)])

    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Runner diario de RAGAS para comparar LangGraph vs LangChain."
    )
    _ = parser.parse_args()

    result = run_daily_ragas_reports()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())