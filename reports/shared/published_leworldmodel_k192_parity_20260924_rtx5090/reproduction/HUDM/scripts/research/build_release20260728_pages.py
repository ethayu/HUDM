from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.parse import unquote, urlsplit


DEFAULT_REPORTS_ROOT = Path(
    "/ceph/projects/dineshj/lab/ethanyu/HUDM/reports/research"
)
DEFAULT_ANALYSIS_DIR = DEFAULT_REPORTS_ROOT / (
    "release20260728_schedule_screening/five_anchor_homogeneous_analysis"
)

BENCHMARKS = (
    (
        "cube-25",
        "OGB Cube",
        25,
        "release20260728_dense_ogb_cube_all_fidelity_schedules",
    ),
    (
        "cube-50",
        "OGB Cube",
        50,
        "release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules",
    ),
    (
        "pusht-25",
        "PushT",
        25,
        "release20260728_dense_pusht_all_fidelity_schedules",
    ),
    (
        "pusht-50",
        "PushT",
        50,
        "release20260728_dense_pusht_goal50_plan50_execute20_all_fidelity_schedules",
    ),
    (
        "reacher-25",
        "Reacher",
        25,
        "release20260728_dense_reacher_all_fidelity_schedules",
    ),
    (
        "reacher-50",
        "Reacher",
        50,
        "release20260728_dense_reacher_goal50_plan50_execute20_all_fidelity_schedules",
    ),
    (
        "tworoom-25",
        "TwoRoom",
        25,
        "release20260728_dense_tworoom_all_fidelity_schedules",
    ),
    (
        "tworoom-50",
        "TwoRoom",
        50,
        "release20260728_dense_tworoom_goal50_plan50_execute20_all_fidelity_schedules",
    ),
)


STYLE = """
:root {
  color-scheme: light;
  --ink: #18221c;
  --muted: #5b675f;
  --paper: #f5f3ec;
  --card: #fffefa;
  --line: #d9d8ce;
  --green: #145c3d;
  --green-soft: #e4f0e8;
  --orange: #b85b20;
  --blue: #225c8a;
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--paper); font: 15px/1.55 Inter, ui-sans-serif, system-ui, sans-serif; }
a { color: var(--blue); text-decoration-thickness: .08em; text-underline-offset: .16em; }
main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 42px 0 72px; }
header { margin-bottom: 30px; }
.eyebrow { color: var(--green); font-size: 12px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
h1 { font-size: clamp(34px, 6vw, 68px); letter-spacing: -.045em; line-height: .98; margin: 10px 0 14px; max-width: 900px; }
h2 { font-size: 25px; letter-spacing: -.02em; margin: 44px 0 14px; }
h3 { font-size: 18px; margin: 0 0 8px; }
p { max-width: 820px; }
.lede { color: var(--muted); font-size: 18px; }
.notice { background: var(--green-soft); border-left: 4px solid var(--green); padding: 14px 18px; margin: 22px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(235px, 1fr)); gap: 14px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: 0 2px 10px rgb(34 45 37 / 4%); }
.card p { color: var(--muted); margin: 5px 0 13px; }
.metric { font-size: 31px; font-weight: 780; letter-spacing: -.04em; }
.metric small { color: var(--muted); font-size: 14px; font-weight: 600; letter-spacing: 0; }
.bar { height: 7px; background: #e4e3da; border-radius: 99px; overflow: hidden; margin: 12px 0; }
.bar > span { display: block; height: 100%; background: var(--green); }
.pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 12px; padding: 3px 9px; }
.pill.complete { background: var(--green-soft); border-color: #b9d2c1; color: var(--green); }
.table-wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; vertical-align: top; }
th { background: #ebeae2; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
tr:last-child td { border-bottom: 0; }
.num { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.positive { color: var(--green); font-weight: 700; }
.negative { color: #9a3326; font-weight: 700; }
.links { display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 18px; }
.finding { border-top: 1px solid var(--line); padding: 18px 0; }
.finding:first-child { border-top: 0; }
code { background: #eae9e0; border-radius: 4px; padding: 2px 5px; }
footer { border-top: 1px solid var(--line); color: var(--muted); margin-top: 52px; padding-top: 18px; }
@media (max-width: 640px) { main { width: min(100% - 20px, 1180px); padding-top: 26px; } }
"""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_public_summary_csv(source: Path, target: Path) -> None:
    """Copy aggregate metrics without publishing internal cell locations."""
    with source.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {source}")
        fieldnames = [name for name in reader.fieldnames if name != "output_json"]
        rows = list(reader)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as target_handle:
        writer = csv.DictWriter(target_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def public_analysis_value(value: Any) -> Any:
    """Replace cluster-local path prefixes while preserving analysis content."""
    prefixes = (
        "/ceph/projects/dineshj/lab/ethanyu/HUDM/",
        "/vast/projects/dineshj/lab/ethanyu/code/HUDM/",
    )
    if isinstance(value, str):
        for prefix in prefixes:
            if value.startswith(prefix):
                return "artifact://" + value.removeprefix(prefix)
        return value
    if isinstance(value, list):
        return [public_analysis_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            public_key = public_analysis_value(key)
            if public_key in result:
                raise ValueError(f"Path sanitization produced duplicate key: {public_key}")
            result[public_key] = public_analysis_value(item)
        return result
    return value


class _ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self,
        _tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.references.append(value)


def validate_site(output_dir: Path) -> dict[str, int]:
    """Enforce Pages limits, public-path hygiene, and local-link integrity."""
    prohibited = (
        output_dir / "functions",
        output_dir / "_worker.js",
        output_dir / ".env",
        output_dir / ".dev.vars",
    )
    present = [str(path) for path in prohibited if path.exists()]
    if present:
        raise ValueError(f"Unexpected executable or secret-bearing publish paths: {present}")

    symlinks = [path for path in output_dir.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Pages bundle must not contain symlinks: {symlinks[:5]}")
    files = [path for path in output_dir.rglob("*") if path.is_file()]
    if len(files) > 20_000:
        raise ValueError(f"Pages file limit exceeded: {len(files)}")
    oversized = [path for path in files if path.stat().st_size > 25 * 1024 * 1024]
    if oversized:
        raise ValueError(f"Pages 25 MiB per-file limit exceeded: {oversized[:5]}")

    forbidden_bytes = (
        b"/ceph/projects/dineshj/lab/ethanyu/",
        b"/vast/projects/dineshj/lab/ethanyu/",
        b"eval.json",
        b"CLOUDFLARE_API_TOKEN",
    )
    leaked: list[str] = []
    for path in files:
        payload = path.read_bytes()
        if any(marker in payload for marker in forbidden_bytes):
            leaked.append(str(path))
    if leaked:
        raise ValueError(f"Internal paths or secret identifiers found in publish bundle: {leaked[:5]}")

    missing: list[str] = []
    reference_count = 0
    for html_path in output_dir.rglob("*.html"):
        parser = _ReferenceParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        for reference in parser.references:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            reference_count += 1
            reference_path = unquote(parsed.path)
            if reference_path.startswith("/"):
                target = output_dir / reference_path.lstrip("/")
            else:
                target = html_path.parent / reference_path
            if target.is_dir() or reference_path.endswith("/"):
                target /= "index.html"
            if not target.is_file():
                missing.append(f"{html_path.relative_to(output_dir)} -> {reference}")
    if missing:
        raise ValueError(f"Broken internal links: {missing[:10]}")
    return {
        "file_count": len(files),
        "internal_reference_count": reference_count,
        "max_file_bytes": max(path.stat().st_size for path in files),
    }


def page(title: str, body: str, *, depth: int = 0) -> str:
    prefix = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="{prefix}assets/site.css">
</head>
<body>
<main>{body}</main>
</body>
</html>
"""


def public_review_html(source: Path, *, title: str) -> str:
    html = source.read_text(encoding="utf-8")
    html = html.replace(" — Live</title>", " — Published snapshot</title>")
    html = html.replace(" — Live</h1>", " — Published snapshot</h1>")
    html = html.replace("Live benchmark snapshot:", "Published benchmark snapshot:")
    html = re.sub(
        r"\s*<section id=\"review-mode\".*?</section>\s*",
        "\n  <section class='panel review-mode'><span class='mode-badge'>Published aggregate snapshot</span>"
        "<strong> Scientific cell artifacts remain on PARCC and are not copied to this public site.</strong>"
        "</section>\n",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"\s*<h2>Rollout Review</h2>.*?</main>",
        "\n  <section class='panel'><strong>Public snapshot scope</strong>"
        "<p class='muted'>This page contains aggregate tables and Pareto plots only. "
        "Per-cell traces and internal paths are intentionally excluded.</p>"
        "<p><a href='../../index.html'>← Back to the release dashboard</a></p></section>\n</main>",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(r"\s*<script>.*?</script>\s*", "\n", html, flags=re.DOTALL)
    html = html.replace(
        "<body>\n<main>",
        f"<body>\n<main>\n  <p><a href='../../index.html'>← Release dashboard</a></p>\n"
        f"  <p class='muted'>Public export: {escape(title)}</p>",
    )
    return html


def fmt_percent(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def fmt_pp(value: float) -> str:
    return f"{100.0 * value:+.1f} pp"


def diagnostic_page(analysis: dict[str, Any], built_at: str) -> str:
    design = analysis["design_validation"]
    audit = analysis["artifact_audit"]
    summaries = sorted(
        analysis["schedule_summaries"],
        key=lambda row: float(row["mean_success_regret_to_stratum_best"]),
    )
    contrasts = analysis["stage_contrasts"]
    frontier = sorted(
        (row for row in analysis["frontier_stability"] if row["empirical_frontier"]),
        key=lambda row: (
            row["env_id"],
            int(row["goal_offset"]),
            float(row["dynamics_flops_per_episode"]),
        ),
    )

    schedule_rows = "".join(
        "<tr>"
        f"<td><strong>{escape(row['policy_id'])}</strong><br>{escape(row['schedule'])}</td>"
        f"<td class='num'>{fmt_percent(float(row['mean_success_rate_fraction']), 2)}</td>"
        f"<td class='num'>{float(row['mean_dynamics_flops_per_episode']) / 1e12:.3f}</td>"
        f"<td class='num'>{int(row['empirical_frontier_cells'])}</td>"
        f"<td class='num'>{float(row['frontier_selection_frequency_mass']):.2f}</td>"
        f"<td class='num'>{100 * float(row['mean_success_regret_to_stratum_best']):.2f} pp</td>"
        "</tr>"
        for row in summaries
    )
    contrast_rows = "".join(
        "<tr>"
        f"<td>{escape(row['contrast'])}</td>"
        f"<td class='num {'positive' if float(row['mean_paired_success_delta_fraction']) > 0 else 'negative'}'>"
        f"{fmt_pp(float(row['mean_paired_success_delta_fraction']))}</td>"
        f"<td class='num'>[{100 * float(row['fixed_matrix_paired_episode_cluster_bootstrap_95_interval'][0]):+.1f}, "
        f"{100 * float(row['fixed_matrix_paired_episode_cluster_bootstrap_95_interval'][1]):+.1f}] pp</td>"
        f"<td class='num'>{float(row['mean_cost_ratio']):.3f}×</td>"
        f"<td class='num'>{int(row['matched_cell_pairs'])}</td>"
        "</tr>"
        for row in contrasts
    )
    frontier_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['env_id']).replace('swm/', '').replace('DMControl', ''))}</td>"
        f"<td class='num'>{int(row['goal_offset'])}</td>"
        f"<td>{escape(row['policy_id'])}/{escape(row['anchor_id'])}</td>"
        f"<td class='num'>{fmt_percent(float(row['success_rate_fraction']))}</td>"
        f"<td class='num'>{float(row['dynamics_flops_per_episode']) / 1e12:.4f}</td>"
        f"<td class='num'>{float(row['paired_trajectory_cluster_bootstrap_frontier_selection_frequency']):.2f}</td>"
        "</tr>"
        for row in frontier
    )

    body = f"""
<header>
  <p><a href="../index.html">← Release dashboard</a></p>
  <div class="eyebrow">Complete · homogeneous screen50 diagnostic</div>
  <h1>Where fidelity schedules pay off</h1>
  <p class="lede">Five planner anchors across eight task/goal matrices. This page separates
  high-success schedules from schedules that sit on the Pareto frontier merely because they are cheapest.</p>
</header>
<section class="grid">
  <div class="card"><div class="metric">{int(design['rows']):,}</div><p>source-label cells audited</p></div>
  <div class="card"><div class="metric">{len(frontier)}</div><p>empirical Pareto cells out of 560 operational cells</p></div>
  <div class="card"><div class="metric">14</div><p>runtime-distinct policies from 26 YAML labels</p></div>
  <div class="card"><div class="metric">{int(audit['full_archive_loads']):,}</div><p>archives fully materialized and integrity checked</p></div>
</section>
<section class="notice"><strong>Practical reading:</strong> P08 is the broad efficiency knee, P02 is
the strongest general success/compute compromise, and P19 buys essentially no average success over P02
despite substantially more compute. P23 is the cost floor—not the universal winner.</section>

<h2>What to scale next</h2>
<div class="card">
  <p><strong>Scale a stability envelope, not only the 42 exact frontier points.</strong> Use the exact
  frontier plus near-frontier/bootstrap-stable neighbors, retain P02 and P08 as global controls, and
  include P23 only as the cheap boundary. Concentrate planner work at a1–a3; only keep a4/a5 champions
  as saturation controls.</p>
  <p>First reuse any already-completed 250-episode cells from the full matrices. For missing cells,
  run paired 250-episode confirmation and technical replicates before opening a local population ×
  iteration neighborhood. This protects against finite-panel frontier flips and the observed runtime-equivalent
  execution variance.</p>
</div>

<h2>Cross-task operational schedules</h2>
<div class="table-wrap"><table>
  <thead><tr><th>Policy</th><th class="num">Mean success</th><th class="num">TFLOPs/episode</th>
  <th class="num">Exact frontier cells</th><th class="num">Selection mass</th><th class="num">Mean regret</th></tr></thead>
  <tbody>{schedule_rows}</tbody>
</table></div>

<h2>Matched stage contrasts</h2>
<p>Conditional paired-trajectory bootstrap intervals hold matrix, planner anchor, and episode identities fixed.</p>
<div class="table-wrap"><table>
  <thead><tr><th>Contrast</th><th class="num">Success Δ</th><th class="num">95% interval</th>
  <th class="num">Cost ratio</th><th class="num">Pairs</th></tr></thead>
  <tbody>{contrast_rows}</tbody>
</table></div>

<h2>Exact empirical frontier</h2>
<p>A cheapest point can be nondominated even at poor success. Selection frequency is a conditional
bootstrap frequency, not a posterior probability.</p>
<div class="table-wrap"><table>
  <thead><tr><th>Environment</th><th class="num">Goal</th><th>Policy/anchor</th>
  <th class="num">Success</th><th class="num">TFLOPs/episode</th><th class="num">Bootstrap frequency</th></tr></thead>
  <tbody>{frontier_rows}</tbody>
</table></div>

<h2>Environment-specific result</h2>
<div class="card">
  <div class="finding"><h3>PushT</h3><p>Spend fidelity before action commitment. Fine→base rollout
  gains +10.2/+4.2 pp and CEM base→fine gains +9.2/+8.9 pp, while MPC coarse→fine loses
  11.3/4.0 pp at goal 25/50.</p></div>
  <div class="finding"><h3>TwoRoom</h3><p>Cheap global reasoning works. MPC coarse→fine gains
  +6.0/+7.4 pp while saving compute; extra early rollout fidelity is neutral or slightly harmful.</p></div>
  <div class="finding"><h3>OGB Cube</h3><p>Coarse outer scheduling helps, plausibly because the
  5-D action search benefits from cheap broad coverage. Dynamic-MPC CEM base→fine is slightly harmful.</p></div>
  <div class="finding"><h3>Reacher</h3><p>Early coarsening is unsafe despite smooth, low-dimensional
  dynamics. Fixed fidelity leads average success; MPC coarse→fine loses 5.9/3.8 pp.</p></div>
</div>

<h2>Downloads and provenance</h2>
<div class="links">
  <a href="data/report.md">Full report</a>
  <a href="data/analysis.json">Analysis JSON</a>
  <a href="data/pareto_frontier.csv">Pareto CSV</a>
  <a href="data/frontier_stability.csv">Frontier stability CSV</a>
  <a href="data/stage_contrasts.csv">Stage contrasts CSV</a>
  <a href="data/schedule_summary.csv">Schedule summary CSV</a>
</div>
<footer>Built {escape(built_at)}. Strict audit passed with zero scientific invariant violations.
Checkpoint path and epoch were verified; historical checkpoint bytes were not content-hashed.</footer>
"""
    return page("HUDM five-anchor schedule diagnostic", body, depth=1)


def index_page(statuses: list[dict[str, Any]], built_at: str) -> str:
    completed = sum(int(row["completed_cells"]) for row in statuses)
    expected = sum(int(row["expected_cells"]) for row in statuses)
    cards = "".join(
        f"""<article class="card">
  <span class="pill">Goal {row['goal_offset']}</span>
  <h3>{escape(row['environment'])}</h3>
  <div class="metric">{int(row['completed_cells']):,}<small> / {int(row['expected_cells']):,}</small></div>
  <div class="bar"><span style="width:{100 * int(row['completed_cells']) / int(row['expected_cells']):.2f}%"></span></div>
  <p>{100 * int(row['completed_cells']) / int(row['expected_cells']):.1f}% complete · frozen public snapshot</p>
  <a href="benchmarks/{escape(row['slug'])}/index.html">Open aggregate viewer →</a>
</article>"""
        for row in statuses
    )
    body = f"""
<header>
  <div class="eyebrow">HUDM · release 2026-07-28</div>
  <h1>Fidelity schedule benchmark</h1>
  <p class="lede">Durable public snapshots for four environments, two planning lengths, and the
  completed five-anchor schedule diagnostic.</p>
</header>
<section class="notice"><strong>The diagnostic is complete.</strong> The full 24,960-cell sweep remains
in progress. Interpret the eight full-matrix frontiers as provisional; use the homogeneous diagnostic
for current schedule conclusions.</section>
<section class="grid">
  <div class="card"><div class="metric">{completed:,}<small> / {expected:,}</small></div>
  <p>full-matrix cells currently represented</p><div class="bar"><span style="width:{100 * completed / expected:.2f}%"></span></div></div>
  <div class="card"><div class="metric">1,040</div><p>homogeneous diagnostic cells</p><span class="pill complete">strict audit passed</span></div>
  <div class="card"><div class="metric">42</div><p>exact operational Pareto cells</p><a href="diagnostic/index.html">Read the diagnostic →</a></div>
</section>

<h2>Benchmark snapshots</h2>
<section class="grid">{cards}</section>

<h2>Decision summary</h2>
<div class="card">
  <p><strong>P08</strong> is the broad efficiency knee. <strong>P02</strong> is the best general
  success/compute compromise. <strong>P19</strong> has nearly identical mean success to P02 at much
  higher cost. <strong>P23</strong> frequently appears on the frontier because it is cheapest, not because
  it is reliably successful.</p>
  <p><a href="diagnostic/index.html">Open the complete scientific analysis →</a></p>
</div>
<footer>Published {escape(built_at)}. This is a static, reproducible Pages export and does not depend on
PARCC viewer processes or a temporary Cloudflare tunnel.</footer>
"""
    return page("HUDM release20260728 benchmark", body)


def copy_diagnostic_data(
    analysis_dir: Path,
    output_dir: Path,
    analysis: dict[str, Any],
) -> None:
    data_dir = output_dir / "diagnostic/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "report.md",
        "pareto_frontier.csv",
        "frontier_stability.csv",
        "stage_contrasts.csv",
        "stage_contrast_strata.csv",
        "schedule_summary.csv",
        "paired_cell_comparisons.csv",
    )
    for name in names:
        source = analysis_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, data_dir / name)
    public_analysis = public_analysis_value(analysis)
    public_analysis["public_export"] = {
        "cluster_paths_sanitized": True,
        "scientific_payload_unchanged": True,
        "source_input_sha256": analysis["input_sha256"],
    }
    write_text(
        data_dir / "analysis.json",
        json.dumps(public_analysis, indent=2, sort_keys=True) + "\n",
    )


def build_site(reports_root: Path, analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    built_at = datetime.now().astimezone().isoformat(timespec="seconds")
    write_text(output_dir / "assets/site.css", STYLE.strip() + "\n")

    statuses: list[dict[str, Any]] = []
    for slug, environment, goal_offset, directory_name in BENCHMARKS:
        source_root = reports_root / directory_name
        summary = load_json(source_root / "summary.live.json")
        viewer_dir = output_dir / f"benchmarks/{slug}"
        viewer_dir.mkdir(parents=True, exist_ok=True)
        title = f"{environment} · Goal {goal_offset}"
        write_text(
            viewer_dir / "index.html",
            public_review_html(source_root / "review.live.html", title=title),
        )
        plot_source = source_root / "plots/live"
        plot_target = viewer_dir / "plots/live"
        plot_target.mkdir(parents=True, exist_ok=True)
        for name in ("pareto.html", "success_vs_compute.png", "success_vs_wall_time.png", "strategy_legend.png"):
            shutil.copy2(plot_source / name, plot_target / name)
        public_status = {
            "slug": slug,
            "environment": environment,
            "goal_offset": goal_offset,
            "status": summary["status"],
            "complete": bool(summary["complete"]),
            "generated_at": summary["generated_at"],
            "fingerprint": summary["fingerprint"],
            "completed_cells": int(summary["completed_cells"]),
            "expected_cells": int(summary["expected_cells"]),
            "missing_cells": int(summary["missing_cells"]),
        }
        write_text(
            viewer_dir / "summary.live.json",
            json.dumps(public_status, indent=2, sort_keys=True) + "\n",
        )
        write_public_summary_csv(
            source_root / "summary.live.csv",
            viewer_dir / "summary.live.csv",
        )
        shutil.copy2(
            source_root / "per_env_summary.live.csv",
            viewer_dir / "per_env_summary.live.csv",
        )
        statuses.append(public_status)

    analysis = load_json(analysis_dir / "analysis.json")
    if analysis.get("artifact_audit", {}).get("status") != "passed":
        raise ValueError("Refusing to publish a diagnostic whose strict artifact audit did not pass.")
    copy_diagnostic_data(analysis_dir, output_dir, analysis)
    write_text(output_dir / "diagnostic/index.html", diagnostic_page(analysis, built_at))
    write_text(output_dir / "index.html", index_page(statuses, built_at))
    write_text(
        output_dir / "_headers",
        "/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: no-referrer\n  Cache-Control: public, max-age=300\n\n/assets/*\n  Cache-Control: public, max-age=86400\n",
    )
    write_text(
        output_dir / "404.html",
        page(
            "Not found · HUDM benchmark",
            "<header><div class='eyebrow'>404</div><h1>Page not found</h1>"
            "<p><a href='/'>Return to the benchmark dashboard</a></p></header>",
        ),
    )
    manifest = {
        "schema_version": "hudm.release20260728.pages/v1",
        "built_at": built_at,
        "completed_cells": sum(row["completed_cells"] for row in statuses),
        "expected_cells": sum(row["expected_cells"] for row in statuses),
        "benchmarks": statuses,
        "analysis_input_sha256": analysis["input_sha256"],
        "analysis_schema_version": analysis["schema_version"],
        "artifact_audit_status": analysis["artifact_audit"]["status"],
    }
    write_text(output_dir / "build-manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["pages_preflight"] = validate_site(output_dir)
    write_text(output_dir / "build-manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_site(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the durable release20260728 Cloudflare Pages bundle.")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_site(
        args.reports_root.resolve(),
        args.analysis_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
