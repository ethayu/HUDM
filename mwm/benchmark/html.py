from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from mwm.benchmark.analysis import env_label, float_metric, outcome_rows, paired_rows, role_label, sorted_rows


def _short(value: Any, chars: int = 10) -> str:
    text = str(value or "")
    return text[:chars] if text else ""


def _pct(value: Any) -> str:
    val = float_metric(value, float("nan"))
    return "n/a" if math.isnan(val) else f"{val:.1f}%"


def _num(value: Any) -> str:
    val = float_metric(value, float("nan"))
    if math.isnan(val):
        return "n/a"
    if abs(val) >= 1_000_000_000:
        return f"{val / 1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:.1f}K"
    return f"{val:.0f}"


def _seconds(value: Any) -> str:
    val = float_metric(value, float("nan"))
    if math.isnan(val):
        return "n/a"
    return f"{val:.1f}s" if val < 120 else f"{val / 60:.1f}m"

def _href(path_text: Any, base_dir: Path) -> str:
    text = str(path_text or "")
    if not text:
        return ""
    path = Path(text)
    try:
        if path.is_absolute():
            return Path(path).relative_to(base_dir.resolve()).as_posix()
        resolved = (Path.cwd() / path).resolve()
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except (ValueError, OSError):
        parts = path.parts
        if len(parts) >= 2 and parts[0] == "rollouts" and parts[1] == base_dir.name:
            return Path(*parts[2:]).as_posix()
        return path.as_posix()


def _link(path_text: Any, label: str, base_dir: Path) -> str:
    href = _href(path_text, base_dir)
    if not href:
        return html.escape(label)
    return f"<a href='{html.escape(href)}'>{html.escape(label)}</a>"


def _rollout_review_href(run_dir: Path, episode_index: int) -> str:
    return f"rollouts/{html.escape(run_dir.name)}/episode_{int(episode_index):04d}.html"


def _payload_rollouts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rollouts = payload.get("review_rollouts", [])
    return [dict(item) for item in rollouts if isinstance(item, dict)]


def _review_media_items(payload: dict[str, Any], base_dir: Path) -> list[str]:
    items: list[str] = []
    media = payload.get("review_media", {}).get("rollouts", {})
    if not isinstance(media, dict):
        return items
    for rollout_key, entries in sorted(media.items()):
        if not isinstance(entries, dict):
            continue
        for kind, entry in sorted(entries.items()):
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            href = _href(entry["path"], base_dir)
            label = f"{rollout_key} {str(kind).replace('_', ' ')}"
            items.append(
                "<li>"
                f"<a href='{html.escape(href)}'>{html.escape(label)}</a>"
                f"<video controls preload='metadata' src='{html.escape(href)}'></video>"
                "</li>"
            )
    return items

def _benchmark_status_cards(rows: list[dict[str, Any]], plots: list[str], expected_cells: int | None = None) -> tuple[str, list[str]]:
    envs = sorted({str(row.get("env_id", "")) for row in rows})
    seeds = sorted({int(row.get("seed", 0)) for row in rows})
    roles = sorted({str(row.get("role", "")) for row in rows})
    pairs = paired_rows(rows)
    expected = int(expected_cells) if expected_cells is not None else len(envs) * len(seeds) * len(roles)
    shared_pairs = sum(1 for pair in pairs if pair["same_manifest"])
    min_episodes = min((int(row.get("episodes", 0)) for row in rows), default=0)
    warnings: list[str] = []
    if len(rows) != expected:
        warnings.append(f"Observed {len(rows)} runs but matrix implies {expected}.")
    if shared_pairs != len(pairs):
        warnings.append("At least one baseline/MWM pair does not share a manifest hash.")
    if min_episodes and min_episodes < 10:
        warnings.append(f"Each run has as few as {min_episodes} episodes; treat this as a quick benchmark, not a high-confidence study.")
    for outcome in outcome_rows(rows):
        env_rows = [r for r in rows if str(r.get("env_id", "")) == outcome["env_id"]]
        if env_rows and all(float_metric(r.get("success_rate")) == 0.0 for r in env_rows):
            warnings.append(f"{env_label(outcome['env_id'])} has zero success for every role and seed.")
    cards = [
        ("Runs", f"{len(rows)}/{expected}", "matrix cells present"),
        ("Pairs", f"{shared_pairs}/{len(pairs)}", "shared manifests"),
        ("Plots", str(len(plots)), "embedded figures"),
        ("Seeds", ", ".join(str(seed) for seed in seeds), "per environment"),
    ]
    html_cards = "".join(
        f"<div class='card'><div class='card-label'>{html.escape(label)}</div><div class='card-value'>{html.escape(value)}</div><div class='card-note'>{html.escape(note)}</div></div>"
        for label, value, note in cards
    )
    return html_cards, warnings


def write_review_html(
    path: str | Path,
    title: str,
    rows: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    plots: list[str] | None = None,
    expected_cells: int | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_dir = out.parent.resolve()
    rows = sorted_rows(rows)
    plots = plots or sorted(str(p) for p in (out.parent / "plots").glob("*.png"))
    status_html, warnings = _benchmark_status_cards(rows, plots, expected_cells=expected_cells)
    warning_html = "".join(f"<li>{html.escape(w)}</li>" for w in warnings) or "<li>No structural warnings detected.</li>"

    outcome_html = []
    for row in outcome_rows(rows):
        delta = float_metric(row["delta_success"], float("nan"))
        delta_class = "good" if delta > 0 else "bad" if delta < 0 else "muted"
        outcome_html.append(
            "<tr>"
            f"<td>{html.escape(env_label(row['env_id']))}</td>"
            f"<td>{html.escape(role_label(str(row['comparison_role'])))}</td>"
            f"<td>{_pct(row['baseline_success'])}</td>"
            f"<td>{_pct(row['comparison_success'])}</td>"
            f"<td class='{delta_class}'>{delta:+.1f} pp</td>"
            f"<td>{_seconds(row['baseline_wall'])}</td>"
            f"<td>{_seconds(row['comparison_wall'])}</td>"
            f"<td>{_num(row['baseline_compute'])}</td>"
            f"<td>{_num(row['comparison_compute'])}</td>"
            f"<td>{int(row['same_manifests'])}/{int(row['pairs'])}</td>"
            "</tr>"
        )

    plot_cards = []
    for plot in plots:
        href = _href(plot, base_dir)
        label = Path(str(plot)).stem.replace("_", " ")
        plot_cards.append(
            f"<figure><a href='{html.escape(href)}'><img src='{html.escape(href)}' alt='{html.escape(label)}'></a><figcaption>{html.escape(label)}</figcaption></figure>"
        )

    pair_html = []
    for pair in paired_rows(rows):
        pair_html.append(
            "<tr>"
            f"<td>{html.escape(env_label(pair['env_id']))}</td>"
            f"<td>{int(pair['seed'])}</td>"
            f"<td>{html.escape(role_label(str(pair['comparison_role'])))}</td>"
            f"<td>{_pct(pair['baseline'].get('success_rate'))}</td>"
            f"<td>{_pct(pair['comparison'].get('success_rate'))}</td>"
            f"<td>{float_metric(pair['delta_success']):+.1f} pp</td>"
            f"<td>{float_metric(pair['wall_ratio'], float('nan')):.2f}x</td>"
            f"<td>{float_metric(pair['compute_ratio'], float('nan')):.2f}x</td>"
            f"<td>{'yes' if pair['same_manifest'] else 'no'}</td>"
            "</tr>"
        )

    detail_rows = []
    payload_by_run_dir: dict[str, dict[str, Any]] = {}
    for payload in outputs:
        resolved_path = payload.get("config", {}).get("resolved_path") if isinstance(payload.get("config"), dict) else None
        if resolved_path:
            payload_by_run_dir[str(Path(str(resolved_path)).parent)] = payload
    for row_idx, row in enumerate(rows):
        run_dir = Path(str(row.get("output_json", ""))).parent
        payload = payload_by_run_dir.get(str(run_dir)) or (outputs[row_idx] if row_idx < len(outputs) else {})
        rollout_links = []
        for rollout in _payload_rollouts(payload):
            if "episode_index" not in rollout:
                continue
            idx = int(rollout["episode_index"])
            rollout_links.append(f"<a href='{_rollout_review_href(run_dir, idx)}'>rollout {idx}</a>")
        if not rollout_links and int(row.get("episodes", 0)) > 0:
            rollout_links.append(f"<a href='{_rollout_review_href(run_dir, 0)}'>rollouts</a>")
        links = " ".join(
            _link(run_dir / name, label, base_dir)
            for name, label in (
                ("eval.json", "eval"),
                ("summary.json", "summary"),
                ("planning_diagnostics.json", "diagnostics"),
                ("episode_traces.jsonl", "episodes"),
                ("resolved_config.yaml", "config"),
                ("run.log", "log"),
            )
        )
        if rollout_links:
            links = " ".join([links, *rollout_links])
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(env_label(str(row.get('env_id', ''))))}</td>"
            f"<td>{int(row.get('seed', 0))}</td>"
            f"<td>{html.escape(role_label(str(row.get('role', ''))))}</td>"
            f"<td>{_pct(row.get('success_rate'))}</td>"
            f"<td>{int(row.get('episodes', 0))}</td>"
            f"<td>{_seconds(row.get('wall_time_sec'))}</td>"
            f"<td>{_num(row.get('bits_used_total'))}</td>"
            f"<td>{int(row.get('plans', 0))}</td>"
            f"<td><code>{html.escape(_short(row.get('manifest_sha256')))}</code></td>"
            f"<td><code>{html.escape(_short(row.get('config_sha256')))}</code></td>"
            f"<td>{links}</td>"
            "</tr>"
        )

    media_links: list[str] = []
    for payload in outputs:
        for video in payload.get("videos", []):
            path_text = str(video)
            media_links.append(f"<li>{_link(path_text, path_text, base_dir)}</li>")
        media_links.extend(_review_media_items(payload, base_dir))

    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; color: #172026; background: #f6f8fa; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 32px; }}
    h1 {{ margin: 0 0 8px; }}
    h2 {{ margin-top: 32px; }}
    a {{ color: #0b5cad; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .lede {{ color: #52616b; margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px 16px; }}
    .card-label {{ color: #627282; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .card-value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    .card-note {{ color: #627282; font-size: 13px; }}
    .panel {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 18px; margin-top: 16px; }}
    .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    video {{ display: block; max-width: 520px; width: 100%; margin-top: 8px; background: #000; }}
    figcaption {{ color: #52616b; font-size: 13px; margin-top: 8px; text-transform: capitalize; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f7; font-weight: 650; }}
    code {{ background: #eef3f7; padding: 2px 4px; border-radius: 4px; }}
    details {{ margin-top: 18px; }}
    summary {{ cursor: pointer; font-weight: 650; }}
    .good {{ color: #0f7b3f; font-weight: 650; }}
    .bad {{ color: #b42318; font-weight: 650; }}
    .muted {{ color: #627282; }}
    .notes li, .warnings li {{ margin: 6px 0; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="lede">Static benchmark review for converted upstream Le-WM, retrained identity-parity Le-WM, and scheduled MWM.</p>

  <h2>Benchmark Status</h2>
  <section class="cards">{status_html}</section>
  <section class="panel">
    <strong>Review warnings</strong>
    <ul class="warnings">{warning_html}</ul>
  </section>

  <h2>Outcome Summary</h2>
  <section class="panel">
    <table>
      <thead><tr><th>env</th><th>comparison role</th><th>upstream success</th><th>comparison success</th><th>delta</th><th>upstream wall</th><th>comparison wall</th><th>upstream compute</th><th>comparison compute</th><th>shared manifests</th></tr></thead>
      <tbody>{''.join(outcome_html)}</tbody>
    </table>
  </section>

  <h2>Plots</h2>
  <section class="plots">{''.join(plot_cards)}</section>

  <h2>Paired Seed Comparison</h2>
  <section class="panel">
    <table>
      <thead><tr><th>env</th><th>seed</th><th>comparison role</th><th>upstream success</th><th>comparison success</th><th>delta</th><th>wall ratio</th><th>compute ratio</th><th>same manifest</th></tr></thead>
      <tbody>{''.join(pair_html)}</tbody>
    </table>
  </section>

  <details open>
    <summary>Run Drilldown</summary>
    <section class="panel">
      <table>
        <thead><tr><th>env</th><th>seed</th><th>role</th><th>success</th><th>episodes</th><th>wall</th><th>compute</th><th>plans</th><th>manifest</th><th>config</th><th>artifacts</th></tr></thead>
        <tbody>{''.join(detail_rows)}</tbody>
      </table>
    </section>
  </details>

  <h2>Review Notes</h2>
  <section class="panel">
    <ul class="notes">
      <li>Confirm the Benchmark Status has no structural blockers before interpreting model quality.</li>
      <li>Use paired seed deltas before drawing conclusions from aggregate means.</li>
      <li>Investigate runs with slow wall-time, unexpected compute, or zero-success patterns via the drilldown links.</li>
      <li>Record whether this is a quick smoke-scale benchmark or a final report-scale benchmark.</li>
    </ul>
  </section>

  <h2>Media</h2>
  <section class="panel"><ul>{''.join(media_links) if media_links else '<li>No media recorded.</li>'}</ul></section>
</main>
</body>
</html>
"""
    out.write_text(body, encoding="utf-8")


__all__ = ["write_review_html"]
