from __future__ import annotations

import csv
import html
import json
import mimetypes
import os
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from scripts import planning_media


REVIEW_MEDIA = (
    "closed_loop_replay",
    "planner_view_replay",
    "predicted_backend_replay",
    "gt_replay",
)

STATIC_FILES = (
    "per_rollout.csv",
    "summary.csv",
    "summary.json",
    "selected_rollouts.json",
    "experiment_resolved.json",
    "experiment_report.html",
)


@dataclass(frozen=True)
class ExperimentReviewData:
    run_dir: str
    experiment_name: str
    baseline_variant: str
    summary_rows: list[dict[str, str]]
    rows: list[dict[str, str]]
    rows_by_key: dict[tuple[str, str], dict[str, str]]


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_experiment_review_data(run_dir: str) -> ExperimentReviewData:
    root = os.path.abspath(run_dir)
    summary_rows = _read_csv_rows(os.path.join(root, "summary.csv"))
    rows = _read_csv_rows(os.path.join(root, "per_rollout.csv"))
    if len(rows) <= 0:
        raise FileNotFoundError(f"No experiment rows found under {root}")

    experiment_name = os.path.basename(root)
    baseline_variant = ""
    resolved_path = os.path.join(root, "experiment_resolved.json")
    if os.path.isfile(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            resolved = json.load(f)
        experiment_name = str(resolved.get("name", experiment_name))
        baseline_variant = str(resolved.get("baseline", baseline_variant))

    rows_by_key = {
        (str(row.get("variant_name", "")), str(row.get("rollout_id", ""))): row
        for row in rows
    }
    return ExperimentReviewData(
        run_dir=root,
        experiment_name=experiment_name,
        baseline_variant=baseline_variant,
        summary_rows=summary_rows,
        rows=rows,
        rows_by_key=rows_by_key,
    )


def _safe_join(root: str, rel_path: str) -> str:
    candidate = os.path.abspath(os.path.join(root, rel_path))
    root_abs = os.path.abspath(root)
    if candidate != root_abs and not candidate.startswith(root_abs + os.sep):
        raise ValueError(f"Path escapes experiment root: {rel_path}")
    return candidate


def trace_dir_for_row(experiment_root: str, row: dict[str, Any]) -> str:
    return os.path.join(
        os.path.abspath(experiment_root),
        "traces",
        str(row.get("variant_name", "")),
        str(row.get("rollout_id", "")),
    )


def resolve_row(data: ExperimentReviewData, variant_name: str, rollout_id: str) -> dict[str, str]:
    key = (str(variant_name), str(rollout_id))
    row = data.rows_by_key.get(key)
    if row is None:
        raise KeyError(f"Unknown run: {variant_name}/{rollout_id}")
    return row


def _html_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns)
    body_parts: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{row.get(key, '')}</td>" for _, key in columns)
        body_parts.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_parts)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _artifact_links_html(run_dir: str, experiment_root: str) -> str:
    links = []
    for filename in ("trace.json", "trace.npz", "metadata.json", "run.log", "index.html"):
        path = os.path.join(run_dir, filename)
        if os.path.isfile(path):
            rel_path = os.path.relpath(path, experiment_root)
            links.append(f"<li><a href='/files/{quote(rel_path)}'>{html.escape(filename)}</a></li>")
    return "".join(links) if links else "<li class='muted'>No run-local files found.</li>"


def _media_path_for_name(run_dir: str, media_name: str) -> str | None:
    for filename in planning_media.MEDIA_ALIASES.get(media_name, []):
        path = os.path.join(run_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def _media_section_html(experiment_root: str, row: dict[str, str]) -> str:
    run_dir = trace_dir_for_row(experiment_root, row)
    pieces: list[str] = []
    variant = str(row.get("variant_name", ""))
    rollout_id = str(row.get("rollout_id", ""))
    for media_name in REVIEW_MEDIA:
        media_path = _media_path_for_name(run_dir, media_name)
        if media_path is None:
            button = (
                f"<a class='button' href='/render?variant={quote(variant)}&rollout_id={quote(rollout_id)}"
                f"&media={quote(media_name)}'>render</a>"
            )
            pieces.append(
                "<section class='card'>"
                f"<h3>{html.escape(media_name)}</h3>"
                "<p class='muted'>Not generated yet.</p>"
                f"{button}"
                "</section>"
            )
            continue
        rel_path = os.path.relpath(media_path, experiment_root)
        pieces.append(
            "<section class='card'>"
            f"<h3>{html.escape(media_name)}</h3>"
            f"<video controls preload='metadata' src='/files/{quote(rel_path)}'></video>"
            "</section>"
        )
    return "\n".join(pieces)


def render_media_for_run(
    experiment_root: str,
    *,
    variant_name: str,
    rollout_id: str,
    media: list[str],
) -> tuple[list[str], list[str]]:
    outputs: list[str] = []
    errors: list[str] = []
    for media_name in media:
        try:
            outputs.extend(
                planning_media.render_media(
                    experiment_root,
                    schedule=variant_name,
                    rollout_id=rollout_id,
                    media=[media_name],
                )
            )
        except Exception as exc:
            errors.append(f"{media_name}: {type(exc).__name__}: {exc}")
    return outputs, errors


def _notice_html(message: str | None, *, kind: str = "info") -> str:
    if not message:
        return ""
    return f"<section class='card notice {html.escape(kind)}'>{html.escape(message)}</section>"


def build_summary_page(data: ExperimentReviewData, *, notice: str | None = None) -> str:
    summary_rows = [
        {
            **{key: html.escape(str(value)) for key, value in row.items()},
        }
        for row in data.summary_rows
    ]
    detail_rows = []
    for row in data.rows:
        variant = str(row.get("variant_name", ""))
        rollout_id = str(row.get("rollout_id", ""))
        detail_rows.append(
            {
                "variant_name": html.escape(variant),
                "rollout_id": html.escape(rollout_id),
                "success": html.escape(str(row.get("success", ""))),
                "termination_reason": html.escape(str(row.get("termination_reason", ""))),
                "final_pos_diff": html.escape(str(row.get("final_pos_diff", ""))),
                "final_coverage": html.escape(str(row.get("final_coverage", ""))),
                "detail": (
                    f"<a href='/run?variant={quote(variant)}&rollout_id={quote(rollout_id)}'>review</a>"
                ),
            }
        )

    static_links = []
    for filename in STATIC_FILES:
        path = os.path.join(data.run_dir, filename)
        if os.path.isfile(path):
            static_links.append(f"<li><a href='/files/{quote(filename)}'>{html.escape(filename)}</a></li>")
    static_links_html = "".join(static_links) if static_links else "<li class='muted'>No experiment files found.</li>"

    summary_table = _html_table(
        summary_rows,
        [
            ("Variant", "variant_name"),
            ("Rollouts", "n_rollouts"),
            ("Success Rate", "success_rate"),
            ("Mean Final Pos", "mean_final_pos_diff"),
            ("Mean Coverage", "mean_final_coverage"),
            ("Mean Bits", "mean_bits_used_total"),
            ("Mean Plan Time", "mean_plan_time_total_sec"),
        ],
    )
    detail_table = _html_table(
        detail_rows,
        [
            ("Variant", "variant_name"),
            ("Rollout", "rollout_id"),
            ("Success", "success"),
            ("Termination", "termination_reason"),
            ("Final Pos", "final_pos_diff"),
            ("Final Coverage", "final_coverage"),
            ("Review", "detail"),
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(data.experiment_name)} review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f7fb; color: #18212b; }}
    h1, h2 {{ margin-bottom: 12px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .notice.info {{ border-left: 4px solid #0f62fe; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e7ebf1; text-align: left; }}
    th {{ background: #f2f5fa; }}
    .table-wrap {{ overflow: auto; max-height: 60vh; }}
    .muted {{ color: #52606d; }}
    a {{ color: #0f62fe; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  {_notice_html(notice)}
  <section class="card">
    <h1>{html.escape(data.experiment_name)}</h1>
    <p>Baseline variant: <strong>{html.escape(data.baseline_variant or "")}</strong></p>
    <p class="muted">This reviewer renders replay media on demand from saved traces. It is intended for experiments run with <code>plan.artifacts.save: false</code>.</p>
  </section>
  <section class="card">
    <h2>Experiment Files</h2>
    <ul>{static_links_html}</ul>
  </section>
  <section class="card">
    <h2>Variant Summary</h2>
    <div class="table-wrap">{summary_table}</div>
  </section>
  <section class="card">
    <h2>Runs</h2>
    <div class="table-wrap">{detail_table}</div>
  </section>
</body>
</html>
"""


def _metric_rows(row: dict[str, str]) -> list[dict[str, str]]:
    skip = {"run_dir", "trace_json", "trace_npz", "run_log"}
    return [
        {"label": html.escape(key), "value": html.escape(str(value))}
        for key, value in row.items()
        if key not in skip
    ]


def _read_log_tail(path: str, *, max_lines: int = 120) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:]) if lines else ""


def build_run_page(
    data: ExperimentReviewData,
    row: dict[str, str],
    *,
    notice: str | None = None,
    errors: list[str] | None = None,
) -> str:
    run_dir = trace_dir_for_row(data.run_dir, row)
    variant = str(row.get("variant_name", ""))
    rollout_id = str(row.get("rollout_id", ""))
    metric_table = _html_table(_metric_rows(row), [("Metric", "label"), ("Value", "value")])
    log_tail = _read_log_tail(os.path.join(run_dir, "run.log"))
    log_html = ""
    if log_tail is not None:
        log_html = (
            "<details class='card'><summary>Run Log Tail</summary>"
            f"<pre>{html.escape(log_tail)}</pre></details>"
        )
    error_html = ""
    if errors:
        error_html = _notice_html(" | ".join(errors), kind="error")

    render_all_href = (
        f"/render?variant={quote(variant)}&rollout_id={quote(rollout_id)}"
        + "".join(f"&media={quote(media_name)}" for media_name in REVIEW_MEDIA)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(variant)} / {html.escape(rollout_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f7fb; color: #18212b; }}
    .card {{ background: white; border-radius: 12px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .button {{ display: inline-block; padding: 9px 12px; border-radius: 8px; background: #eef2f7; color: #18212b; text-decoration: none; margin-right: 8px; }}
    .button.primary {{ background: #0f62fe; color: white; }}
    .notice.info {{ border-left: 4px solid #0f62fe; }}
    .notice.error {{ border-left: 4px solid #da1e28; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e7ebf1; text-align: left; }}
    th {{ background: #f2f5fa; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
    video {{ width: 100%; max-height: 420px; background: #111; border-radius: 8px; }}
    .muted {{ color: #52606d; }}
    a {{ color: #0f62fe; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <p><a href="/">&larr; Back to experiment review</a></p>
  {_notice_html(notice)}
  {error_html}
  <section class="card">
    <h1>{html.escape(variant)} / {html.escape(rollout_id)}</h1>
    <p class="muted">Trace dir: <code>{html.escape(os.path.relpath(run_dir, data.run_dir))}</code></p>
    <p>
      <a class="button primary" href="{render_all_href}">Render all review media</a>
    </p>
  </section>
  <section class="card">
    <h2>Metrics</h2>
    {metric_table}
  </section>
  <section class="card">
    <h2>Artifacts</h2>
    <ul>{_artifact_links_html(run_dir, data.run_dir)}</ul>
  </section>
  <section>
    <h2>Media</h2>
    <div class="grid">
      {_media_section_html(data.run_dir, row)}
    </div>
  </section>
  {log_html}
</body>
</html>
"""


class ExperimentReviewApp:
    def __init__(self, run_dir: str):
        self.data = load_experiment_review_data(run_dir)

    def summary_page(self, *, notice: str | None = None) -> str:
        return build_summary_page(self.data, notice=notice)

    def run_page(
        self,
        *,
        variant_name: str,
        rollout_id: str,
        notice: str | None = None,
        errors: list[str] | None = None,
    ) -> str:
        row = resolve_row(self.data, variant_name, rollout_id)
        return build_run_page(self.data, row, notice=notice, errors=errors)


def make_review_handler(app: ExperimentReviewApp):
    class ReviewHandler(BaseHTTPRequestHandler):
        review_app = app

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"", "/"}:
                    self._send_html(self.review_app.summary_page())
                    return
                if parsed.path == "/run":
                    params = parse_qs(parsed.query)
                    variant = params.get("variant", [""])[0]
                    rollout_id = params.get("rollout_id", [""])[0]
                    self._send_html(self.review_app.run_page(variant_name=variant, rollout_id=rollout_id))
                    return
                if parsed.path == "/render":
                    params = parse_qs(parsed.query)
                    variant = params.get("variant", [""])[0]
                    rollout_id = params.get("rollout_id", [""])[0]
                    media = [item for item in params.get("media", []) if item]
                    if len(media) <= 0:
                        media = list(REVIEW_MEDIA)
                    outputs, errors = render_media_for_run(
                        self.review_app.data.run_dir,
                        variant_name=variant,
                        rollout_id=rollout_id,
                        media=media,
                    )
                    notice = f"Rendered {len(outputs)} media artifact(s)." if len(errors) <= 0 else None
                    self._send_html(
                        self.review_app.run_page(
                            variant_name=variant,
                            rollout_id=rollout_id,
                            notice=notice,
                            errors=errors,
                        )
                    )
                    return
                if parsed.path.startswith("/files/"):
                    rel_path = unquote(parsed.path[len("/files/") :])
                    self._serve_file(rel_path)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except KeyError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # pragma: no cover - defensive server path
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send_html(self, text: str, *, status: int = HTTPStatus.OK) -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_file(self, rel_path: str) -> None:
            path = _safe_join(self.review_app.data.run_dir, rel_path)
            if not os.path.isfile(path):
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            ctype, _ = mimetypes.guess_type(path)
            ctype = ctype or "application/octet-stream"
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ReviewHandler


def create_review_server(run_dir: str, *, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    app = ExperimentReviewApp(run_dir)
    handler_cls = make_review_handler(app)
    return ThreadingHTTPServer((host, port), handler_cls)


def serve_review(run_dir: str, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = create_review_server(run_dir, host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    display_host = "127.0.0.1" if bound_host in {"0.0.0.0", ""} else str(bound_host)
    print(f"[experiment-review] serving {os.path.abspath(run_dir)}")
    print(f"[experiment-review] open http://{display_host}:{bound_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
