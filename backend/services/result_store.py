from __future__ import annotations

import csv
import html
import io
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from google.cloud import storage


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
    return storage.Client()


def _get_blob():
    if not RESULTS_BUCKET:
        raise RuntimeError(
            "RESULTS_BUCKET no está configurado. Debes definir el bucket de Cloud Storage."
        )

    client = _get_storage_client()
    bucket = client.bucket(RESULTS_BUCKET)
    return bucket.blob(RESULTS_OBJECT)


def append_result(record: dict[str, Any]) -> None:
    payload = _json_safe(dict(record))
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    line = json.dumps(payload, ensure_ascii=False)

    blob = _get_blob()

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


def _pick(row: dict[str, Any], *paths: Any, default: str = "") -> Any:
    for path in paths:
        if isinstance(path, str):
            value = row.get(path)
            if value not in (None, "", [], {}):
                return value
            continue

        if isinstance(path, tuple):
            current: Any = row
            ok = True
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    ok = False
                    break
                current = current[key]
            if ok and current not in (None, "", [], {}):
                return current

    return default


def _row_value(row: dict[str, Any], *keys: Any) -> str:
    value = _pick(row, *keys, default="")
    if value is None:
        return ""
    return str(value)


def render_result_detail_html(row: dict[str, Any]) -> str:
    pretty = json.dumps(_json_safe(row), ensure_ascii=False, indent=2)

    def esc(value: Any) -> str:
        if value is None:
            return ""
        return html.escape(str(value))

    case_id = esc(_row_value(row, "case_id", "caso_id"))
    sample_id = esc(_row_value(row, "sample_id"))
    benchmark_id = esc(_row_value(row, "benchmark_id"))
    pipeline = esc(_row_value(row, "pipeline"))
    timestamp = esc(_row_value(row, "timestamp"))

    score_total = esc(
        _row_value(
            row,
            "score_total",
            ("evaluacion", "puntaje_total"),
            ("evaluacion_semantica", "puntaje_total"),
            ("summary", "score_total"),
        )
    )
    score_semantic = esc(
        _row_value(
            row,
            "score_semantic",
            ("evaluacion_semantica", "puntaje_total"),
            ("summary", "puntaje_semantico"),
        )
    )
    score_rubric = esc(
        _row_value(
            row,
            "score_rubric",
            ("evaluacion_rubrica", "puntaje_total"),
            ("summary", "puntaje_rubrica"),
        )
    )
    relevance_case = esc(
        _row_value(
            row,
            "indice_relevancia_caso",
            ("evaluacion", "indice_relevancia_caso"),
            ("evaluacion_semantica", "indice_relevancia_caso"),
            ("summary", "indice_relevancia_caso"),
        )
    )
    relevance_lexica = esc(
        _row_value(
            row,
            "indice_relevancia_lexica",
            ("evaluacion_semantica", "indice_relevancia_lexica"),
            ("summary", "indice_relevancia_lexica"),
        )
    )
    faithfulness = esc(
        _row_value(
            row,
            "faithfulness",
            ("summary", "faithfulness"),
            ("response_json", "summary", "faithfulness"),
        )
    )
    answer_relevancy = esc(
        _row_value(
            row,
            "answer_relevancy",
            ("summary", "answer_relevancy"),
            ("response_json", "summary", "answer_relevancy"),
        )
    )

    answer = esc(_row_value(row, "answer", "entrada", "input", "response"))
    feedback = esc(_row_value(row, "feedback", "retroalimentacion"))

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Detalle del resultado</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 0;
          padding: 24px;
          background: #f8fafc;
          color: #111827;
        }}
        .card {{
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          padding: 20px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          margin-bottom: 18px;
        }}
        .grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 12px;
        }}
        .item {{
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          padding: 12px;
          background: #fafafa;
        }}
        .label {{
          font-size: 12px;
          color: #6b7280;
          margin-bottom: 4px;
        }}
        .value {{
          font-size: 14px;
          word-break: break-word;
        }}
        pre {{
          white-space: pre-wrap;
          word-break: break-word;
          background: #0f172a;
          color: #e2e8f0;
          padding: 16px;
          border-radius: 12px;
          overflow-x: auto;
        }}
        a {{
          color: #2563eb;
          text-decoration: none;
        }}
      </style>
    </head>
    <body>
      <p><a href="/resultados?format=html">← Volver a resultados</a></p>

      <div class="card">
        <h1>Detalle del resultado</h1>
        <div class="grid">
          <div class="item"><div class="label">Fecha</div><div class="value">{timestamp}</div></div>
          <div class="item"><div class="label">Caso</div><div class="value">{case_id}</div></div>
          <div class="item"><div class="label">Sample</div><div class="value">{sample_id}</div></div>
          <div class="item"><div class="label">Benchmark</div><div class="value">{benchmark_id}</div></div>
          <div class="item"><div class="label">Pipeline</div><div class="value">{pipeline}</div></div>
          <div class="item"><div class="label">Puntaje total</div><div class="value">{score_total}</div></div>
          <div class="item"><div class="label">Puntaje semántico</div><div class="value">{score_semantic}</div></div>
          <div class="item"><div class="label">Puntaje rúbrica</div><div class="value">{score_rubric}</div></div>
          <div class="item"><div class="label">Relevancia con el caso</div><div class="value">{relevance_case}</div></div>
          <div class="item"><div class="label">Relevancia léxica</div><div class="value">{relevance_lexica}</div></div>
          <div class="item"><div class="label">Faithfulness</div><div class="value">{faithfulness}</div></div>
          <div class="item"><div class="label">Answer relevancy</div><div class="value">{answer_relevancy}</div></div>
        </div>
      </div>

      <div class="card">
        <h2>Respuesta</h2>
        <p style="white-space:pre-wrap">{answer}</p>
      </div>

      <div class="card">
        <h2>Retroalimentación</h2>
        <p style="white-space:pre-wrap">{feedback}</p>
      </div>

      <div class="card">
        <h2>Registro completo</h2>
        <pre>{html.escape(pretty)}</pre>
      </div>
    </body>
    </html>
    """


def render_results_html(rows: list[dict[str, Any]]) -> str:
    def esc(value: Any) -> str:
        if value is None:
            return ""
        return html.escape(str(value))

    cards: list[str] = []

    for idx, row in enumerate(rows):
        answer = esc(_row_value(row, "answer", "entrada", "input", "response"))
        feedback = esc(_row_value(row, "feedback", "retroalimentacion"))
        score_total = esc(
            _row_value(
                row,
                "score_total",
                ("evaluacion", "puntaje_total"),
                ("summary", "puntaje_total"),
            )
        )
        score_semantic = esc(
            _row_value(
                row,
                "score_semantic",
                ("evaluacion_semantica", "puntaje_total"),
                ("summary", "puntaje_semantico"),
            )
        )
        score_rubric = esc(
            _row_value(
                row,
                "score_rubric",
                ("evaluacion_rubrica", "puntaje_total"),
                ("summary", "puntaje_rubrica"),
            )
        )
        relevance_case = esc(
            _row_value(
                row,
                "indice_relevancia_caso",
                ("evaluacion", "indice_relevancia_caso"),
                ("evaluacion_semantica", "indice_relevancia_caso"),
                ("summary", "indice_relevancia_caso"),
            )
        )
        relevance_lexica = esc(
            _row_value(
                row,
                "indice_relevancia_lexica",
                ("evaluacion_semantica", "indice_relevancia_lexica"),
                ("summary", "indice_relevancia_lexica"),
            )
        )
        faithfulness = esc(
            _row_value(
                row,
                "faithfulness",
                ("summary", "faithfulness"),
                ("response_json", "summary", "faithfulness"),
            )
        )
        answer_relevancy = esc(
            _row_value(
                row,
                "answer_relevancy",
                ("summary", "answer_relevancy"),
                ("response_json", "summary", "answer_relevancy"),
            )
        )
        case_id = esc(_row_value(row, "case_id", "caso_id"))
        sample_id = esc(_row_value(row, "sample_id"))
        pipeline = esc(_row_value(row, "pipeline"))
        benchmark_id = esc(_row_value(row, "benchmark_id"))
        timestamp = esc(_row_value(row, "timestamp"))

        detail_link = f"/resultados?detail_index={idx}"

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
              <td>{relevance_case}</td>
              <td>{relevance_lexica}</td>
              <td>{faithfulness}</td>
              <td>{answer_relevancy}</td>
              <td style="max-width:340px;white-space:pre-wrap">{answer}</td>
              <td style="max-width:360px;white-space:pre-wrap">{feedback}</td>
              <td><a href="{detail_link}">Ver detalle</a></td>
            </tr>
            """
        )

    body = "\n".join(cards) if cards else """
        <tr><td colspan="15" style="text-align:center;padding:24px;">Sin resultados.</td></tr>
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
              <th>Relevancia caso</th>
              <th>Relevancia léxica</th>
              <th>Faithfulness</th>
              <th>Answer relevancy</th>
              <th>Respuesta</th>
              <th>Retroalimentación</th>
              <th>Detalle</th>
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
            "relevance_case",
            "relevance_lexica",
            "faithfulness",
            "answer_relevancy",
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
                _row_value(row, "score_total", ("evaluacion", "puntaje_total")),
                _row_value(row, "score_semantic", ("evaluacion_semantica", "puntaje_total")),
                _row_value(row, "score_rubric", ("evaluacion_rubrica", "puntaje_total")),
                _row_value(
                    row,
                    "indice_relevancia_caso",
                    ("evaluacion", "indice_relevancia_caso"),
                    ("evaluacion_semantica", "indice_relevancia_caso"),
                ),
                _row_value(
                    row,
                    "indice_relevancia_lexica",
                    ("evaluacion_semantica", "indice_relevancia_lexica"),
                ),
                _row_value(row, "faithfulness", ("summary", "faithfulness")),
                _row_value(row, "answer_relevancy", ("summary", "answer_relevancy")),
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