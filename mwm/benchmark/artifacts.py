from __future__ import annotations

import csv
import hashlib
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


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_summary_row(name: str, output_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    swm_results = dict(payload.get("swm_results", {}))
    diagnostics = dict(payload.get("planning_diagnostics", {}))
    manifest = dict(payload.get("manifest", {}))
    config = dict(payload.get("config", {}))
    level_counts = diagnostics.get("schedule_level_counts", {})
    return {
        "name": str(name),
        "env_id": str(payload.get("env_id", "")),
        "checkpoint_epoch": payload.get("checkpoint_epoch", ""),
        "checkpoint_run_dir": str(payload.get("checkpoint_run_dir", "")),
        "config_sha256": str(config.get("sha256", "")),
        "manifest_sha256": str(manifest.get("manifest_sha256", "")),
        "manifest_file_sha256": str(manifest.get("sha256", "")),
        "episodes": int(payload.get("episodes", 0)),
        "goal_offset": int(payload.get("goal_offset", 0)),
        "success_rate": float(swm_results.get("success_rate", float("nan"))),
        "plans": int(diagnostics.get("plans", 0)),
        "steps": int(diagnostics.get("steps", 0)),
        "bits_used_total": int(diagnostics.get("bits_used_total", 0)),
        "plan_time_total_sec": float(diagnostics.get("plan_time_total_sec", 0.0)),
        "wall_time_sec": float(payload.get("wall_time_sec", 0.0)),
        "schedule_level_counts": json.dumps(_jsonable(level_counts), sort_keys=True),
        "schedule": str(payload.get("schedule", "")),
        "role": str(payload.get("role", "")),
        "seed": int(payload.get("seed", payload.get("eval_seed", 0))),
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
        "checkpoint_run_dir",
        "config_sha256",
        "manifest_sha256",
        "manifest_file_sha256",
        "episodes",
        "goal_offset",
        "success_rate",
        "plans",
        "steps",
        "bits_used_total",
        "plan_time_total_sec",
        "wall_time_sec",
        "schedule_level_counts",
        "schedule",
        "role",
        "seed",
        "output_json",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[Any]) -> float:
    vals = [_float(v, float("nan")) for v in values]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _short(value: Any, chars: int = 10) -> str:
    text = str(value or "")
    return text[:chars] if text else ""


def _pct(value: Any) -> str:
    val = _float(value, float("nan"))
    return "n/a" if np.isnan(val) else f"{val:.1f}%"


def _num(value: Any) -> str:
    val = _float(value, float("nan"))
    if np.isnan(val):
        return "n/a"
    if abs(val) >= 1_000_000_000:
        return f"{val / 1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:.1f}K"
    return f"{val:.0f}"


def _seconds(value: Any) -> str:
    val = _float(value, float("nan"))
    if np.isnan(val):
        return "n/a"
    return f"{val:.1f}s" if val < 120 else f"{val / 60:.1f}m"


def _role_label(role: str) -> str:
    labels = {
        "upstream_lewm_converted": "Upstream Le-WM",
        "retrained_lewm_single": "Retrained Le-WM",
        "mwm_scheduled": "MWM scheduled",
    }
    return labels.get(str(role), str(role))


def _env_label(env_id: str) -> str:
    return str(env_id).removeprefix("swm/").removesuffix("-v1")


def _sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    role_order = {"upstream_lewm_converted": 0, "retrained_lewm_single": 1, "mwm_scheduled": 2}
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("env_id", "")),
            int(r.get("seed", 0)),
            role_order.get(str(r.get("role", "")), 99),
            str(r.get("name", "")),
        ),
    )


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


def _row_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (str(row.get("env_id", "")), int(row.get("seed", 0)), str(row.get("role", ""))): row
        for row in rows
    }


def _comparison_roles(rows: Iterable[dict[str, Any]]) -> list[str]:
    role_order = {"retrained_lewm_single": 0, "mwm_scheduled": 1}
    roles = {
        str(row.get("role", ""))
        for row in rows
        if str(row.get("role", "")) and str(row.get("role", "")) != "upstream_lewm_converted"
    }
    return sorted(roles, key=lambda role: (role_order.get(role, 99), role))


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = _row_index(rows)
    pairs: list[dict[str, Any]] = []
    env_seed = sorted({(env, seed) for env, seed, _ in indexed})
    comparison_roles = _comparison_roles(rows)
    for env_id, seed in env_seed:
        baseline = indexed.get((env_id, seed, "upstream_lewm_converted"))
        if not baseline:
            continue
        for role in comparison_roles:
            comparison = indexed.get((env_id, seed, role))
            if not comparison:
                continue
            base_success = _float(baseline.get("success_rate"), float("nan"))
            comparison_success = _float(comparison.get("success_rate"), float("nan"))
            base_wall = _float(baseline.get("wall_time_sec"), float("nan"))
            comparison_wall = _float(comparison.get("wall_time_sec"), float("nan"))
            base_bits = _float(baseline.get("bits_used_total"), float("nan"))
            comparison_bits = _float(comparison.get("bits_used_total"), float("nan"))
            pairs.append(
                {
                    "env_id": env_id,
                    "seed": seed,
                    "baseline": baseline,
                    "comparison": comparison,
                    "comparison_role": role,
                    "mwm": comparison,
                    "delta_success": comparison_success - base_success,
                    "wall_ratio": comparison_wall / base_wall if base_wall > 0 else float("nan"),
                    "compute_ratio": comparison_bits / base_bits if base_bits > 0 else float("nan"),
                    "same_manifest": str(baseline.get("manifest_sha256", ""))
                    == str(comparison.get("manifest_sha256", "")),
                }
            )
    return pairs


def _outcome_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = _paired_rows(rows)
    envs = sorted({str(row.get("env_id", "")) for row in rows})
    comparison_roles = _comparison_roles(rows)
    out: list[dict[str, Any]] = []
    for env_id in envs:
        env_rows = [r for r in rows if str(r.get("env_id", "")) == env_id]
        base = [r for r in env_rows if str(r.get("role", "")) == "upstream_lewm_converted"]
        for role in comparison_roles:
            comparison = [r for r in env_rows if str(r.get("role", "")) == role]
            if not comparison:
                continue
            env_pairs = [p for p in pairs if p["env_id"] == env_id and p["comparison_role"] == role]
            base_success = _mean(r.get("success_rate") for r in base)
            comparison_success = _mean(r.get("success_rate") for r in comparison)
            out.append(
                {
                    "env_id": env_id,
                    "comparison_role": role,
                    "baseline_success": base_success,
                    "comparison_success": comparison_success,
                    "delta_success": comparison_success - base_success,
                    "baseline_wall": _mean(r.get("wall_time_sec") for r in base),
                    "comparison_wall": _mean(r.get("wall_time_sec") for r in comparison),
                    "baseline_compute": _mean(r.get("bits_used_total") for r in base),
                    "comparison_compute": _mean(r.get("bits_used_total") for r in comparison),
                    "same_manifests": sum(1 for p in env_pairs if p["same_manifest"]),
                    "pairs": len(env_pairs),
                }
            )
    return out


def _gate_cards(rows: list[dict[str, Any]], plots: list[str], expected_cells: int | None = None) -> tuple[str, list[str]]:
    envs = sorted({str(row.get("env_id", "")) for row in rows})
    seeds = sorted({int(row.get("seed", 0)) for row in rows})
    roles = sorted({str(row.get("role", "")) for row in rows})
    pairs = _paired_rows(rows)
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
    for outcome in _outcome_rows(rows):
        env_rows = [r for r in rows if str(r.get("env_id", "")) == outcome["env_id"]]
        if env_rows and all(_float(r.get("success_rate")) == 0.0 for r in env_rows):
            warnings.append(f"{_env_label(outcome['env_id'])} has zero success for every role and seed.")
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
    rows = _sorted_rows(rows)
    plots = plots or sorted(str(p) for p in (out.parent / "plots").glob("*.png"))
    gate_html, warnings = _gate_cards(rows, plots, expected_cells=expected_cells)
    warning_html = "".join(f"<li>{html.escape(w)}</li>" for w in warnings) or "<li>No structural warnings detected.</li>"

    outcome_html = []
    for row in _outcome_rows(rows):
        delta = _float(row["delta_success"], float("nan"))
        delta_class = "good" if delta > 0 else "bad" if delta < 0 else "muted"
        outcome_html.append(
            "<tr>"
            f"<td>{html.escape(_env_label(row['env_id']))}</td>"
            f"<td>{html.escape(_role_label(str(row['comparison_role'])))}</td>"
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
    for pair in _paired_rows(rows):
        pair_html.append(
            "<tr>"
            f"<td>{html.escape(_env_label(pair['env_id']))}</td>"
            f"<td>{int(pair['seed'])}</td>"
            f"<td>{html.escape(_role_label(str(pair['comparison_role'])))}</td>"
            f"<td>{_pct(pair['baseline'].get('success_rate'))}</td>"
            f"<td>{_pct(pair['comparison'].get('success_rate'))}</td>"
            f"<td>{_float(pair['delta_success']):+.1f} pp</td>"
            f"<td>{_float(pair['wall_ratio'], float('nan')):.2f}x</td>"
            f"<td>{_float(pair['compute_ratio'], float('nan')):.2f}x</td>"
            f"<td>{'yes' if pair['same_manifest'] else 'no'}</td>"
            "</tr>"
        )

    detail_rows = []
    for row in rows:
        run_dir = Path(str(row.get("output_json", ""))).parent
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
            f"<td>{html.escape(_env_label(str(row.get('env_id', ''))))}</td>"
            f"<td>{int(row.get('seed', 0))}</td>"
            f"<td>{html.escape(_role_label(str(row.get('role', ''))))}</td>"
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
  <p class="lede">Static benchmark review for converted upstream Le-WM, retrained single-fidelity Le-WM, and scheduled MWM.</p>

  <h2>Gate Status</h2>
  <section class="cards">{gate_html}</section>
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
      <li>Confirm the Gate Status has no structural blockers before interpreting model quality.</li>
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


def write_metrics_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def write_run_sidecars(run_dir: str | Path, row: dict[str, Any], payload: dict[str, Any]) -> None:
    root = Path(run_dir)
    write_json(root / "summary.json", {"run": row})
    write_json(root / "dependencies.json", dict(payload.get("dependencies", {})))
    write_json(root / "planning_diagnostics.json", dict(payload.get("planning_diagnostics", {})))


def write_per_env_table(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("env_id", "")), str(row.get("role", ""))), []).append(row)
    out_rows = []
    for (env_id, role), group in sorted(grouped.items()):
        rates = [float(r.get("success_rate", float("nan"))) for r in group]
        out_rows.append(
            {
                "env_id": env_id,
                "role": role,
                "runs": len(group),
                "mean_success_rate": float(np.nanmean(rates)) if rates else float("nan"),
                "seeds": ",".join(str(r.get("seed", "")) for r in sorted(group, key=lambda x: int(x.get("seed", 0)))),
            }
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["env_id", "role", "runs", "mean_success_rate", "seeds"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    return str(out)


def write_default_plots(output_dir: str | Path, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = _sorted_rows(rows)
    plots: list[str] = []
    role_order = {"upstream_lewm_converted": 0, "retrained_lewm_single": 1, "mwm_scheduled": 2}
    colors = {
        "upstream_lewm_converted": "#2f6fbb",
        "retrained_lewm_single": "#7a5fb4",
        "mwm_scheduled": "#d76f1f",
    }

    def _save(fig: Any, name: str) -> None:
        path = root / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots.append(str(path))

    def _roles() -> list[str]:
        roles = sorted({str(row.get("role", "")) for row in rows if str(row.get("role", ""))}, key=lambda role: role_order.get(role, 99))
        return roles or [""]

    def _envs() -> list[str]:
        return sorted({str(row.get("env_id", "")) for row in rows if str(row.get("env_id", ""))})

    def _scatter(x_key: str, y_key: str, name: str, xlabel: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        roles = _roles()
        for role in roles:
            role_rows = [row for row in rows if str(row.get("role", "")) == role]
            x = [_float(row.get(x_key), float("nan")) for row in role_rows]
            y = [_float(row.get(y_key), float("nan")) for row in role_rows]
            ax.scatter(x, y, label=_role_label(role) if role else "runs", color=colors.get(role), alpha=0.85)
            if len(rows) <= 18:
                for row, x_val, y_val in zip(role_rows, x, y):
                    if not np.isnan(x_val) and not np.isnan(y_val):
                        label = f"{_env_label(str(row.get('env_id', '')))} s{int(row.get('seed', 0))}"
                        ax.annotate(label, (x_val, y_val), fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("success rate (%)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, name)

    _scatter("bits_used_total", "success_rate", "success_vs_compute.png", "latent work bits")
    _scatter("wall_time_sec", "success_rate", "success_vs_wall_time.png", "wall time (sec)")

    envs = _envs()
    roles = _roles()
    if envs and any(roles):
        fig, ax = plt.subplots(figsize=(max(6, len(envs) * 2.2), 4.2))
        x = np.arange(len(envs), dtype=float)
        width = min(0.34, 0.76 / max(1, len(roles)))
        offsets = (np.arange(len(roles), dtype=float) - (len(roles) - 1) / 2.0) * width
        for idx, role in enumerate(roles):
            means = [
                _mean(row.get("success_rate") for row in rows if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == role)
                for env_id in envs
            ]
            centers = x + offsets[idx]
            ax.bar(centers, means, width=width, label=_role_label(role), color=colors.get(role), alpha=0.82)
            for env_idx, env_id in enumerate(envs):
                seed_rows = [
                    row
                    for row in rows
                    if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == role
                ]
                jitter = np.linspace(-width * 0.25, width * 0.25, max(1, len(seed_rows)))
                for j_idx, row in enumerate(seed_rows):
                    y = _float(row.get("success_rate"), float("nan"))
                    if not np.isnan(y):
                        ax.scatter(centers[env_idx] + jitter[j_idx], y, color="#172026", s=22, zorder=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([_env_label(env_id) for env_id in envs])
        ax.set_ylabel("success rate (%)")
        ax.set_title("Mean success by environment and role")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, "success_by_env_role.png")

    pairs = _paired_rows(rows)
    if pairs:
        multiple_roles = len({str(pair["comparison_role"]) for pair in pairs}) > 1
        labels = [
            f"{_env_label(pair['env_id'])} s{pair['seed']}"
            + (f"\n{_role_label(str(pair['comparison_role']))}" if multiple_roles else "")
            for pair in pairs
        ]
        deltas = [_float(pair["delta_success"], float("nan")) for pair in pairs]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.72), 4.2))
        x = np.arange(len(labels), dtype=float)
        bar_colors = ["#0f7b3f" if delta > 0 else "#b42318" if delta < 0 else "#627282" for delta in deltas]
        ax.axhline(0, color="#172026", linewidth=1)
        ax.bar(x, deltas, color=bar_colors, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("comparison - upstream success (percentage points)")
        ax.set_title("Paired success delta by seed and role")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, "paired_success_delta.png")

        ratio_labels = list(labels)
        wall = [_float(pair["wall_ratio"], float("nan")) for pair in pairs]
        compute = [_float(pair["compute_ratio"], float("nan")) for pair in pairs]
        fig, ax = plt.subplots(figsize=(max(7, len(ratio_labels) * 0.72), 4.2))
        x = np.arange(len(ratio_labels), dtype=float)
        width = 0.36
        ax.axhline(1.0, color="#172026", linewidth=1)
        ax.bar(x - width / 2, wall, width=width, label="wall-time ratio", color="#6b7f2a", alpha=0.82)
        ax.bar(x + width / 2, compute, width=width, label="compute ratio", color="#7a4e9f", alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(ratio_labels, rotation=35, ha="right")
        ax.set_ylabel("comparison / upstream")
        ax.set_title("Efficiency ratios by paired seed and role")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, "efficiency_ratios.png")

    level_totals: dict[str, int] = {}
    for row in rows:
        try:
            counts = json.loads(str(row.get("schedule_level_counts", "{}")))
        except json.JSONDecodeError:
            counts = {}
        for level, count in counts.items():
            level_totals[str(level)] = level_totals.get(str(level), 0) + int(count)
    if level_totals:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = sorted(
            level_totals,
            key=lambda x: (0, int(str(x))) if str(x).isdigit() else (1, str(x)),
        )
        ax.bar(labels, [level_totals[k] for k in labels])
        ax.set_xlabel("base fidelity level")
        ax.set_ylabel("CEM cost calls")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, "schedule_level_usage.png")

        grouped_counts: dict[tuple[str, str], dict[str, int]] = {}
        for row in rows:
            env_id = str(row.get("env_id", ""))
            role = str(row.get("role", ""))
            try:
                counts = json.loads(str(row.get("schedule_level_counts", "{}")))
            except json.JSONDecodeError:
                counts = {}
            slot = grouped_counts.setdefault((env_id, role), {})
            for level, count in counts.items():
                slot[str(level)] = slot.get(str(level), 0) + int(count)
        if grouped_counts:
            groups = sorted(grouped_counts, key=lambda item: (item[0], role_order.get(item[1], 99), item[1]))
            labels = [f"{_env_label(env_id)}\n{_role_label(role)}" for env_id, role in groups]
            levels = sorted(
                {level for counts in grouped_counts.values() for level in counts},
                key=lambda x: (0, int(str(x))) if str(x).isdigit() else (1, str(x)),
            )
            fig, ax = plt.subplots(figsize=(max(7, len(groups) * 0.86), 4.4))
            x = np.arange(len(groups), dtype=float)
            bottom = np.zeros(len(groups), dtype=float)
            palette = ["#2f6fbb", "#d76f1f", "#0f7b3f", "#7a4e9f", "#64748b", "#c2410c"]
            for idx, level in enumerate(levels):
                vals = np.array([grouped_counts[group].get(level, 0) for group in groups], dtype=float)
                ax.bar(x, vals, bottom=bottom, label=str(level), color=palette[idx % len(palette)], alpha=0.84)
                bottom += vals
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel("CEM cost calls")
            ax.set_title("Schedule level usage by environment and role")
            ax.legend(title="level", loc="best", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            _save(fig, "schedule_usage_by_role.png")
    return plots
