from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return dict(json.load(f))


def eval_summary_row(name: str, output_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    swm_results = dict(payload.get("swm_results", {}))
    diagnostics = dict(payload.get("planning_diagnostics", {}))
    return {
        "name": str(name),
        "env_id": str(payload.get("env_id", "")),
        "checkpoint_epoch": payload.get("checkpoint_epoch", ""),
        "episodes": int(payload.get("episodes", 0)),
        "goal_offset": int(payload.get("goal_offset", 0)),
        "success_rate": float(swm_results.get("success_rate", float("nan"))),
        "plans": int(diagnostics.get("plans", 0)),
        "steps": int(diagnostics.get("steps", 0)),
        "bits_used_total": int(diagnostics.get("bits_used_total", 0)),
        "plan_time_total_sec": float(diagnostics.get("plan_time_total_sec", 0.0)),
        "output_json": str(output_path),
    }


def write_summary_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "env_id",
        "checkpoint_epoch",
        "episodes",
        "goal_offset",
        "success_rate",
        "plans",
        "steps",
        "bits_used_total",
        "plan_time_total_sec",
        "output_json",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_review_html(path: str | Path, title: str, rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "name",
        "env_id",
        "success_rate",
        "episodes",
        "plans",
        "bits_used_total",
        "plan_time_total_sec",
        "output_json",
    ]
    table_rows = []
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header, "")
            if header == "output_json" and value:
                label = html.escape(Path(str(value)).name)
                value_html = f"<a href='{html.escape(str(value))}'>{label}</a>"
            else:
                value_html = html.escape(str(value))
            cells.append(f"<td>{value_html}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    media_links: list[str] = []
    for payload in outputs:
        for video in payload.get("videos", []):
            path_text = str(video)
            media_links.append(f"<li><a href='{html.escape(path_text)}'>{html.escape(path_text)}</a></li>")

    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; }}
    th {{ background: #f0f4f8; font-weight: 600; }}
    code {{ background: #f0f4f8; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>SWM evaluation results with HUDM planning diagnostics. Environment-specific hand-coded task metrics are intentionally absent.</p>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
  <h2>Media</h2>
  <ul>{''.join(media_links) if media_links else '<li>No media recorded.</li>'}</ul>
</body>
</html>
"""
    out.write_text(body, encoding="utf-8")
