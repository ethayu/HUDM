from __future__ import annotations

import html
import math
from pathlib import Path
import shlex
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


def _href_is_file(href: str, base_dir: Path) -> bool:
    candidate = (base_dir / href).resolve()
    try:
        candidate.relative_to(base_dir.resolve())
    except ValueError:
        return False
    return candidate.is_file()


def _rollout_review_href(run_dir: Path, episode_index: int) -> str:
    return f"rollouts/{html.escape(run_dir.name)}/episode_{int(episode_index):04d}.html"


def _payload_rollouts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rollouts = payload.get("review_rollouts", [])
    return [dict(item) for item in rollouts if isinstance(item, dict)]


def _review_media_items(payload: dict[str, Any], base_dir: Path, run_label: str) -> list[str]:
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
            if not _href_is_file(href, base_dir):
                continue
            label = f"{run_label} · {rollout_key} · {str(kind).replace('_', ' ')}"
            items.append(
                "<li>"
                f"<a href='{html.escape(href)}'>{html.escape(label)}</a>"
                f"<video controls preload='metadata' src='{html.escape(href)}'></video>"
                "</li>"
            )
    return items


def _rollout_review_group(
    row: dict[str, Any],
    payload: dict[str, Any],
    run_dir: Path,
) -> str:
    rollouts = _payload_rollouts(payload)
    successes = sum(1 for rollout in rollouts if rollout.get("success") is True)
    failures = sum(1 for rollout in rollouts if rollout.get("success") is False)
    unknown = len(rollouts) - successes - failures
    recorded = payload.get("review_media", {}).get("rollouts", {})
    recorded = recorded if isinstance(recorded, dict) else {}
    episode_links: list[str] = []
    for rollout in rollouts:
        if "episode_index" not in rollout:
            continue
        idx = int(rollout["episode_index"])
        success = rollout.get("success")
        status = "success" if success is True else "failure" if success is False else "unknown"
        media_count = len(recorded.get(f"episode_{idx:04d}", {}))
        media_badge = f"<span class='media-count' title='{media_count} recorded video(s)'>▶ {media_count}</span>" if media_count else ""
        title = (
            f"Episode {idx}; {status}; dataset episode {rollout.get('dataset_episode', 'n/a')}; "
            f"steps {rollout.get('start_step', 'n/a')}→{rollout.get('goal_step', 'n/a')}"
        )
        episode_links.append(
            f"<a class='episode-link {status}' data-rollout-link data-status='{status}' "
            f"data-has-media='{'true' if media_count else 'false'}' "
            f"href='{_rollout_review_href(run_dir, idx)}' title='{html.escape(title)}'>"
            f"<span>{idx}</span>{media_badge}</a>"
        )
    role = role_label(str(row.get("role", "")))
    name = str(row.get("name") or run_dir.name)
    count_note = f"{len(rollouts)} episodes · {failures} failed · {successes} succeeded"
    if unknown:
        count_note += f" · {unknown} unknown"
    episode_html = "".join(episode_links) or '<span class="muted">No replay traces recorded for this run.</span>'
    return (
        "<details class='run-rollouts' open>"
        f"<summary><span>{html.escape(name)}</span> <span class='muted'>({html.escape(role)}) — {html.escape(count_note)}</span></summary>"
        f"<div class='episode-links'>{episode_html}</div>"
        "</details>"
    )

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
    rollout_groups: list[str] = []
    payload_by_run_dir: dict[str, dict[str, Any]] = {}
    for payload in outputs:
        resolved_path = payload.get("config", {}).get("resolved_path") if isinstance(payload.get("config"), dict) else None
        if resolved_path:
            payload_by_run_dir[str(Path(str(resolved_path)).parent)] = payload
    for row_idx, row in enumerate(rows):
        run_dir = Path(str(row.get("output_json", ""))).parent
        payload = payload_by_run_dir.get(str(run_dir)) or (outputs[row_idx] if row_idx < len(outputs) else {})
        rollout_groups.append(_rollout_review_group(row, payload, run_dir))
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
    for payload_idx, payload in enumerate(outputs):
        run_label = str(payload.get("benchmark_name") or payload.get("role") or f"run {payload_idx}")
        for video in payload.get("videos", []):
            path_text = str(video)
            media_links.append(f"<li>{_link(path_text, f'{run_label} · {Path(path_text).name}', base_dir)}</li>")
        media_links.extend(_review_media_items(payload, base_dir, run_label))

    serve_command = (
        "python -m mwm.benchmark.render_review "
        f"{shlex.quote(str(base_dir))} --serve"
    )

    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
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
    .table-scroll {{ width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }}
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
    .review-mode {{ border-left: 4px solid #0b5cad; }}
    .review-mode.connected {{ border-left-color: #0f7b3f; }}
    .mode-badge {{ display: inline-block; border-radius: 999px; background: #e7f0fa; color: #0b5cad; font-size: 12px; font-weight: 700; padding: 4px 9px; text-transform: uppercase; letter-spacing: .03em; }}
    .review-mode.connected .mode-badge {{ background: #e8f5ed; color: #0f7b3f; }}
    .command {{ display: block; width: fit-content; max-width: 100%; overflow-x: auto; margin-top: 10px; padding: 8px 10px; }}
    .command[hidden] {{ display: none; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 14px 0; }}
    .filter-bar button {{ border: 1px solid #9fb3c8; background: #fff; border-radius: 999px; padding: 6px 11px; cursor: pointer; }}
    .filter-bar button.active {{ background: #0b5cad; border-color: #0b5cad; color: white; }}
    .run-rollouts {{ border-top: 1px solid #d9e2ec; padding: 14px 0 4px; }}
    .run-rollouts:first-of-type {{ border-top: 0; }}
    .episode-links {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(58px, 1fr)); gap: 7px; margin-top: 12px; }}
    .episode-link {{ display: flex; justify-content: space-between; align-items: center; gap: 3px; border: 1px solid #cbd5df; border-radius: 6px; padding: 7px 8px; background: #f8fafc; color: #172026; font-variant-numeric: tabular-nums; }}
    .episode-link:hover {{ text-decoration: none; border-color: #0b5cad; box-shadow: 0 0 0 1px #0b5cad; }}
    .episode-link.failure {{ background: #fff1f0; border-color: #f2b8b5; color: #8f1912; font-weight: 700; }}
    .episode-link.success {{ background: #f0faf4; border-color: #b8dfc7; color: #146c3a; }}
    .media-count {{ color: #52616b; font-size: 10px; white-space: nowrap; }}
    .episode-link[hidden] {{ display: none; }}
    @media (max-width: 700px) {{
      main {{ padding: 18px 12px; }}
      h1 {{ font-size: 26px; }}
      .plots {{ grid-template-columns: 1fr; }}
      .panel {{ padding: 12px; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .episode-links {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
      video {{ max-width: 100%; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="lede">Benchmark results with paired comparisons, run artifacts, and rollout-level review.</p>

  <section id="review-mode" class="panel review-mode" aria-live="polite">
    <span class="mode-badge" id="mode-badge">Static report</span>
    <strong id="mode-title">Start the interactive server to inspect individual rollouts.</strong>
    <p id="mode-copy">The aggregate report works as a file. Episode pages and on-demand rendering require the local review server.</p>
    <code class="command" id="serve-command">{html.escape(serve_command)}</code>
  </section>

  <h2>Benchmark Status</h2>
  <section class="cards">{status_html}</section>
  <section class="panel">
    <strong>Review warnings</strong>
    <ul class="warnings">{warning_html}</ul>
  </section>

  <h2>Outcome Summary</h2>
  <section class="panel">
    <div class="table-scroll"><table>
        <thead><tr><th>env</th><th>comparison role</th><th>upstream success</th><th>comparison success</th><th>delta</th><th>upstream wall</th><th>comparison wall</th><th>upstream compute</th><th>comparison compute</th><th>shared manifests</th></tr></thead>
        <tbody>{''.join(outcome_html)}</tbody>
    </table></div>
  </section>

  <h2>Plots</h2>
  <section class="plots">{''.join(plot_cards)}</section>

  <h2>Paired Seed Comparison</h2>
  <section class="panel">
    <div class="table-scroll"><table>
        <thead><tr><th>env</th><th>seed</th><th>comparison role</th><th>upstream success</th><th>comparison success</th><th>delta</th><th>wall ratio</th><th>compute ratio</th><th>same manifest</th></tr></thead>
        <tbody>{''.join(pair_html)}</tbody>
    </table></div>
  </section>

  <h2>Rollout Review</h2>
  <section class="panel">
    <p><strong>Start with failures</strong> to find systematic mistakes, then sample successes to check that the score matches visually plausible behavior. Aligned episode numbers use the shared manifest, so compare the same number across runs.</p>
    <div class="filter-bar" aria-label="Filter rollout episodes">
      <span class="muted">Show:</span>
      <button class="active" type="button" data-filter="all">All</button>
      <button type="button" data-filter="failure">Failures</button>
      <button type="button" data-filter="success">Successes</button>
      <button type="button" data-filter="media">With media</button>
    </div>
    {''.join(rollout_groups)}
  </section>

  <details open>
    <summary>Run Drilldown</summary>
    <section class="panel">
      <div class="table-scroll"><table>
        <thead><tr><th>env</th><th>seed</th><th>role</th><th>success</th><th>episodes</th><th>wall</th><th>compute</th><th>plans</th><th>manifest</th><th>config</th><th>artifacts</th></tr></thead>
        <tbody>{''.join(detail_rows)}</tbody>
      </table></div>
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
<script>
let interactiveReview = false;
const mode = document.getElementById('review-mode');
fetch('/api/status', {{cache: 'no-store'}})
  .then((response) => {{ if (!response.ok) throw new Error('not review server'); return response.json(); }})
  .then(() => {{
    interactiveReview = true;
    mode.classList.add('connected');
    document.getElementById('mode-badge').textContent = 'Interactive server connected';
    document.getElementById('mode-title').textContent = 'Rollout pages and on-demand rendering are ready.';
    document.getElementById('mode-copy').textContent = 'Open an episode below. Failure episodes are highlighted in red; recorded media has a play-count badge.';
    document.getElementById('serve-command').hidden = true;
  }})
  .catch(() => {{}});

document.querySelectorAll('[data-rollout-link]').forEach((link) => {{
  link.addEventListener('click', (event) => {{
    if (interactiveReview) return;
    event.preventDefault();
    mode.scrollIntoView({{behavior: 'smooth', block: 'center'}});
    mode.focus?.();
  }});
}});

document.querySelectorAll('[data-filter]').forEach((button) => {{
  button.addEventListener('click', () => {{
    const filter = button.dataset.filter;
    document.querySelectorAll('[data-filter]').forEach((item) => item.classList.toggle('active', item === button));
    document.querySelectorAll('.episode-link').forEach((link) => {{
      link.hidden = filter !== 'all' && (filter === 'media' ? link.dataset.hasMedia !== 'true' : link.dataset.status !== filter);
    }});
  }});
}});
</script>
</body>
</html>
"""
    out.write_text(body, encoding="utf-8")


__all__ = ["write_review_html"]
