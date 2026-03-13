from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Sequence


MEDIA_FILES = (
    "planned.mp4",
    "planner_view.mp4",
    "gt.mp4",
    "closed_loop_replay.mp4",
    "planner_view_replay.mp4",
    "gt_replay.mp4",
    "predicted_backend_replay.mp4",
)

TEXT_FILES = (
    "metadata.json",
    "trace.json",
    "trace.npz",
    "run.log",
    "pos_diffs_angle_diffs_eef_diffs.png",
)


def _fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _html_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns)
    body_parts: list[str] = []
    for row in rows:
        cells = "".join(
            f"<td>{row.get(key, '')}</td>"
            for _, key in columns
        )
        body_parts.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_parts)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _read_log_tail(run_dir: str, max_lines: int = 120) -> str | None:
    path = os.path.join(run_dir, "run.log")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:]) if lines else ""


def _write_run_detail_page(run_dir: str, row: dict[str, Any], experiment_root: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    rel_back = os.path.relpath(experiment_root, run_dir)
    media_tags: list[str] = []
    for filename in MEDIA_FILES:
        path = os.path.join(run_dir, filename)
        if os.path.isfile(path):
            media_tags.append(
                "\n".join(
                    [
                        f"<section class='card'><h3>{html.escape(filename)}</h3>",
                        f"<video controls preload='metadata' src='{html.escape(filename)}'></video>",
                        "</section>",
                    ]
                )
            )
    if not media_tags:
        media_html = (
            "<p class='muted'>No media artifacts were saved for this run. "
            "Set <code>plan.artifacts.save: true</code> in the experiment plan if you want embedded rollout videos.</p>"
        )
    else:
        media_html = "<div class='grid'>" + "\n".join(media_tags) + "</div>"

    file_links = []
    for filename in TEXT_FILES:
        if os.path.isfile(os.path.join(run_dir, filename)):
            file_links.append(f"<li><a href='{html.escape(filename)}'>{html.escape(filename)}</a></li>")
    log_tail = _read_log_tail(run_dir)
    metrics_rows = [
        {"label": html.escape(key), "value": html.escape(_fmt_value(value))}
        for key, value in row.items()
        if key not in {"run_dir", "trace_json", "trace_npz"}
    ]
    metrics_table = _html_table(metrics_rows, [("Metric", "label"), ("Value", "value")])
    log_html = ""
    if log_tail is not None:
        log_html = (
            "<details class='card'><summary>Run Log Tail</summary>"
            f"<pre>{html.escape(log_tail)}</pre></details>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(str(row.get("variant_name", "run")))} | {html.escape(str(row.get("rollout_id", "")))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f7fb; color: #18212b; }}
    a {{ color: #0f62fe; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .card {{ background: white; border-radius: 12px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e7ebf1; text-align: left; }}
    th {{ background: #f2f5fa; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
    video {{ width: 100%; max-height: 420px; background: #111; border-radius: 8px; }}
    .muted {{ color: #52606d; }}
  </style>
</head>
<body>
  <p><a href="{html.escape(rel_back)}/experiment_report.html">&larr; Back to experiment report</a></p>
  <section class="card">
    <h1>{html.escape(str(row.get("variant_name", "variant")))} / {html.escape(str(row.get("rollout_id", "")))}</h1>
    <p class="muted">Run directory: <code>{html.escape(os.path.relpath(run_dir, experiment_root))}</code></p>
  </section>
  <section class="card">
    <h2>Metrics</h2>
    {metrics_table}
  </section>
  <section class="card">
    <h2>Artifacts</h2>
    <ul>
      {''.join(file_links)}
    </ul>
  </section>
  <section class="card">
    <h2>Media</h2>
    {media_html}
  </section>
  {log_html}
</body>
</html>
"""
    with open(os.path.join(run_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_text)


def write_experiment_report(
    run_dir: str,
    summary_rows: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    experiment_name: str,
    baseline_variant: str,
    summary_plot_files: Sequence[str] = (),
) -> str:
    run_dir = os.path.abspath(run_dir)
    for row in rows:
        _write_run_detail_page(str(row["run_dir"]), row, run_dir)

    summary_cards = {
        "Variants": len(summary_rows),
        "Runs": len(rows),
        "Baseline": baseline_variant,
    }
    if rows:
        success_rate = sum(int(bool(row.get("success", 0))) for row in rows) / max(1, len(rows))
        summary_cards["Overall Success"] = f"{success_rate:.3f}"

    summary_card_html = "".join(
        f"<div class='card stat'><div class='label'>{html.escape(str(label))}</div><div class='value'>{html.escape(str(value))}</div></div>"
        for label, value in summary_cards.items()
    )
    summary_table = _html_table(
        [
            {
                **{key: html.escape(_fmt_value(value)) for key, value in row.items()},
            }
            for row in summary_rows
        ],
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

    detail_rows = []
    for row in rows:
        rel_run_dir = os.path.relpath(str(row["run_dir"]), run_dir)
        detail_rows.append(
            {
                "variant_name": html.escape(str(row.get("variant_name", ""))),
                "rollout_id": html.escape(str(row.get("rollout_id", ""))),
                "success": html.escape(_fmt_value(row.get("success"))),
                "termination_reason": html.escape(str(row.get("termination_reason", ""))),
                "final_pos_diff": html.escape(_fmt_value(row.get("final_pos_diff"))),
                "final_coverage": html.escape(_fmt_value(row.get("final_coverage"))),
                "bits_used_total": html.escape(_fmt_value(row.get("bits_used_total"))),
                "plan_time_total_sec": html.escape(_fmt_value(row.get("plan_time_total_sec"))),
                "detail": f"<a href='{html.escape(rel_run_dir)}/index.html'>open</a>",
            }
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
            ("Bits", "bits_used_total"),
            ("Plan Time (s)", "plan_time_total_sec"),
            ("Detail", "detail"),
        ],
    )

    plot_tags = []
    for filename in summary_plot_files:
        if os.path.isfile(os.path.join(run_dir, filename)):
            plot_tags.append(
                f"<figure class='card'><img src='{html.escape(filename)}' alt='{html.escape(filename)}'><figcaption>{html.escape(filename)}</figcaption></figure>"
            )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(experiment_name)} report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f7fb; color: #18212b; }}
    h1, h2 {{ margin-bottom: 12px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 16px; }}
    .stat .label {{ color: #52606d; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat .value {{ font-size: 24px; font-weight: 600; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e7ebf1; text-align: left; }}
    th {{ background: #f2f5fa; position: sticky; top: 0; }}
    .table-wrap {{ overflow: auto; max-height: 60vh; }}
    .toolbar {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }}
    input {{ padding: 8px 10px; border: 1px solid #ccd5df; border-radius: 8px; min-width: 260px; }}
    img {{ width: 100%; height: auto; border-radius: 8px; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    a {{ color: #0f62fe; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <section class="card">
    <h1>{html.escape(experiment_name)}</h1>
    <p>Baseline variant: <strong>{html.escape(baseline_variant)}</strong></p>
    <p>For on-demand replay rendering, run <code>python3 scripts/experiment_review.py --run-dir {html.escape(run_dir)}</code>.</p>
  </section>
  <section class="stats">
    {summary_card_html}
  </section>
  <section class="card">
    <h2>Variant Summary</h2>
    <div class="table-wrap">{summary_table}</div>
  </section>
  <section class="card">
    <h2>Run Details</h2>
    <div class="toolbar">
      <label for="runFilter">Filter</label>
      <input id="runFilter" type="search" placeholder="variant, rollout, termination...">
    </div>
    <div class="table-wrap" id="runTableWrap">{detail_table}</div>
  </section>
  <section class="plot-grid">
    {''.join(plot_tags)}
  </section>
  <script>
    const input = document.getElementById('runFilter');
    const rows = Array.from(document.querySelectorAll('#runTableWrap tbody tr'));
    input.addEventListener('input', () => {{
      const q = input.value.trim().toLowerCase();
      for (const row of rows) {{
        const match = row.textContent.toLowerCase().includes(q);
        row.style.display = match ? '' : 'none';
      }}
    }});
  </script>
</body>
</html>
"""
    out_path = os.path.join(run_dir, "experiment_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    with open(os.path.join(run_dir, "report_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "baseline_variant": baseline_variant,
                "report_file": "experiment_report.html",
                "summary_plot_files": list(summary_plot_files),
            },
            f,
            indent=2,
        )
    return out_path
