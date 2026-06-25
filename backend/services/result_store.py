from __future__ import annotations

import csv
import html
import io
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional


RESULTS_BUCKET = os.getenv("RESULTS_BUCKET", "").strip()
RESULTS_OBJECT = os.getenv("RESULTS_OBJECT", "results/index.jsonl").strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _get_storage_client():
    try:
        from google.cloud import storage
    except Exception as exc:
        raise RuntimeError(
            "Falta instalar 'google-cloud-storage' en requirements.txt."
        ) from exc
    return storage.Client()


def _get_blob():
    if not RESULTS_BUCKET:
        raise RuntimeError(
            "RESULTS_BUCKET no está configurado. Debes definir el bucket de Cloud Storage."
        )

    storage = _get_storage_client()
    client = storage
    bucket = client.bucket(RESULTS_BUCKET)
    return bucket.blob(RESULTS_OBJECT)


def append_result(record: dict[str, Any]) -> None:
    payload = _json_safe(dict(record))
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    line = json.dumps(payload, ensure_ascii=False)

    blob = _get_blob()
    current = ""

    for attempt in range(5):
        try:
            if blob.exists():
                current = blob.download_as_text(encoding="utf-8").strip()
            else:
                current = ""

            if current:
                current += "\n"
            current += line + "\n"

            blob.upload_from_string(
                current,
                content_type="application/jsonl; charset=utf-8",
            )
            return
        except Exception:
            if attempt >= 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def load_results(
    case_id: Optional[str] = None,
    caso_id: Optional[str] = None,
    benchmark_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    pipeline: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    blob = _get_blob()

    if not blob.exists():
        return []

    text = blob.download_as_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            item = json.loads(raw_line)
        except Exception:
            continue

        row_case_id = item.get("case_id") or item.get("caso_id")

        if case_id and row_case_id != case_id:
            continue
        if caso_id and row_case_id != caso_id:
            continue
        if benchmark_id and item.get("benchmark_id") != benchmark_id:
            continue
        if sample_id and item.get("sample_id") != sample_id:
            continue
        if pipeline and item.get("pipeline") != pipeline:
            continue

        rows.append(item)

    rows.sort(key=lambda x: x.get("timestamp", ""))

    if limit is not None and limit > 0:
        rows = rows[-limit:]

    return rows


def _row_value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


def render_results_html(rows: list[dict[str, Any]]) -> str:
    def esc(value: Any) -> str:
        if value is None:
            return ""
        return html.escape(str(value))

    cards: list[str] = []

    for row in rows:
        answer = esc(_row_value(row, "answer", "entrada", "input", "response"))
        feedback = esc(_row_value(row, "feedback", "retroalimentacion"))
        score_total = esc(_row_value(row, "score_total", "puntaje_total"))
        score_semantic = esc(_row_value(row, "score_semantic", "puntaje_semantico"))
        score_rubric = esc(_row_value(row, "score_rubric", "puntaje_rubrica"))
        case_id = esc(_row_value(row, "case_id", "caso_id"))
        sample_id = esc(_row_value(row, "sample_id"))
        pipeline = esc(_row_value(row, "pipeline"))
        benchmark_id = esc(_row_value(row, "benchmark_id"))
        timestamp = esc(_row_value(row, "timestamp"))

        cards.append(
            f"""
            <tr>
              <td>{timestamp}</td>
              <td>{case_id}</td>
              <td>{sample_id}</td>
              <td>{pipeline}</td>
              <td>{benchmark_id}</td>
              <td>{score_total}</td>
              <td>{score_semantic}</td>
              <td>{score_rubric}</td>
              <td style="max-width:340px;white-space:pre-wrap">{answer}</td>
              <td style="max-width:360px;white-space:pre-wrap">{feedback}</td>
            </tr>
            """
        )

    body = "\n".join(cards) if cards else """
        <tr><td colspan="10" style="text-align:center;padding:24px;">Sin resultados.</td></tr>
    """

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Resultados</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 0;
          padding: 24px;
          background: #f8fafc;
          color: #111827;
        }}
        h1 {{
          margin: 0 0 8px 0;
          font-size: 28px;
        }}
        .sub {{
          margin: 0 0 20px 0;
          color: #6b7280;
        }}
        .table-wrap {{
          overflow-x: auto;
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }}
        th, td {{
          padding: 12px 14px;
          border-bottom: 1px solid #e5e7eb;
          vertical-align: top;
          text-align: left;
        }}
        th {{
          background: #f3f4f6;
          position: sticky;
          top: 0;
          z-index: 1;
        }}
        tr:hover td {{
          background: #fafafa;
        }}
      </style>
    </head>
    <body>
      <h1>Resultados de evaluaciones</h1>
      <p class="sub">Total de registros: {len(rows)}</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Fecha</th>
              <th>Caso</th>
              <th>Sample</th>
              <th>Pipeline</th>
              <th>Benchmark</th>
              <th>Puntaje total</th>
              <th>Puntaje semántico</th>
              <th>Puntaje rúbrica</th>
              <th>Respuesta</th>
              <th>Retroalimentación</th>
            </tr>
          </thead>
          <tbody>
            {body}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """


def render_results_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "timestamp",
            "case_id",
            "sample_id",
            "pipeline",
            "benchmark_id",
            "score_total",
            "score_semantic",
            "score_rubric",
            "answer",
            "feedback",
        ]
    )

    for row in rows:
        writer.writerow(
            [
                _row_value(row, "timestamp"),
                _row_value(row, "case_id", "caso_id"),
                _row_value(row, "sample_id"),
                _row_value(row, "pipeline"),
                _row_value(row, "benchmark_id"),
                _row_value(row, "score_total", "puntaje_total"),
                _row_value(row, "score_semantic", "puntaje_semantico"),
                _row_value(row, "score_rubric", "puntaje_rubrica"),
                _row_value(row, "answer", "entrada", "input", "response"),
                _row_value(row, "feedback", "retroalimentacion"),
            ]
        )

    return output.getvalue()


def render_results_jsonl(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(json.dumps(_json_safe(row), ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")