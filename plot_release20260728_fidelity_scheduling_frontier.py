"""Build the "winning adaptive schedule vs. fixed-K" frontier plot (same style
as pusht's rollouts/pusht_release20260728_goal25_horizon5_100ep_all_configs.png)
for a given env + goal-offset variant, from its release20260728 all-fidelity
schedule sweep report:

- every individual changing-fidelity (adaptive) schedule is plotted as a thin
  gray background cloud (context, not the headline),
- one bold blue step line: the pooled Pareto frontier across all adaptive
  schedules combined -- "the best any adaptive schedule achieves at this bits
  budget" -- since no single fixed schedule wins everywhere,
- the four low-K fixed baselines (K=96/120/144/168) are muted to a single
  thin dashed tan style,
- the K=192 baseline (full base dimensionality) is drawn bold black, on top
  of everything else.

Stray leftover report directories whose "schedule" text doesn't match any
run currently declared in the sweep config (renamed/removed runs across
earlier config edits) are dropped; dense schedules whose MPC/CEM/Rollout text
never contains a "->" are dropped as uninformative duplicates of a
single-K baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/aurora/HUDM-mwm-ethan")

CLOUD_GRAY = "#b7bcc2"
FRONTIER_BLUE = "#2a5f9e"
LOWK_TAN = "#c9a06a"
BASELINE_BLACK = "#111214"


def declared_schedules(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text())
    return [str(run["schedule"]) for run in cfg["runs"]]


def load_rows(output_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(output_dir.iterdir()):
        summary = d / "summary.json"
        if not d.is_dir() or not summary.exists():
            continue
        rows.append(json.loads(summary.read_text())["run"])
    return rows


def pareto_step(xs: list[float], ys: list[float]):
    order = np.argsort(xs)
    xs_s, ys_s = np.array(xs)[order], np.array(ys)[order]
    px, py = [xs_s[0]], [ys_s[0]]
    best = ys_s[0]
    for x, y in zip(xs_s[1:], ys_s[1:]):
        if y >= best:
            best = y
            px.append(x)
            py.append(y)
    return px, py


def baseline_k(label: str) -> int:
    try:
        return int(label.split("K=")[1].split()[0].rstrip("|"))
    except (IndexError, ValueError):
        return 0


def make_plot(config_path: Path, output_dir: Path, env_title: str, goal_offset: int, episodes: int,
              horizon: int, out_path: Path, x_clip: float | None = None) -> None:
    rows = load_rows(output_dir)
    if not rows:
        raise SystemExit(f"no summary.json files found under {output_dir}")

    active = set(declared_schedules(config_path))

    by_schedule: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        schedule = str(row.get("schedule", row.get("base_name", "")))
        if schedule not in active:
            continue
        if schedule not in by_schedule:
            by_schedule[schedule] = []
            order.append(schedule)
        by_schedule[schedule].append(row)

    dense_labels = [s for s in order if not s.strip().startswith("single K=") and "->" in s]
    baseline_labels = [s for s in order if s.strip().startswith("single K=")]
    baseline_labels.sort(key=baseline_k, reverse=True)
    dropped = [s for s in order if s not in dense_labels and s not in baseline_labels]

    if 192 not in {baseline_k(l) for l in baseline_labels}:
        raise SystemExit(f"no K=192 baseline found among {baseline_labels}")

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    max_x = 0.0

    for label in dense_labels:
        pts = by_schedule[label]
        xs = [r["bits_used_total"] / r["episodes"] / 1e6 for r in pts]
        ys = [r["success_rate"] for r in pts]
        px, py = pareto_step(xs, ys)
        ax.scatter(xs, ys, color=CLOUD_GRAY, s=16, alpha=0.35, linewidths=0, zorder=1)
        ax.step(px, py, where="post", color=CLOUD_GRAY, linewidth=0.9, alpha=0.55, zorder=1)
        max_x = max(max_x, max(xs))

    pooled_x = [r["bits_used_total"] / r["episodes"] / 1e6 for label in dense_labels for r in by_schedule[label]]
    pooled_y = [r["success_rate"] for label in dense_labels for r in by_schedule[label]]
    fx, fy = pareto_step(pooled_x, pooled_y)
    ax.step(fx, fy, where="post", color=FRONTIER_BLUE, linewidth=3.2, zorder=4,
            label="Winning adaptive schedules (Pareto frontier, changing fidelity)")
    ax.scatter(fx, fy, color=FRONTIER_BLUE, s=46, zorder=5, edgecolors="white", linewidths=1.0)

    for i, label in enumerate([l for l in baseline_labels if baseline_k(l) != 192]):
        pts = by_schedule[label]
        xs = [r["bits_used_total"] / r["episodes"] / 1e6 for r in pts]
        ys = [r["success_rate"] for r in pts]
        px, py = pareto_step(xs, ys)
        ax.scatter(xs, ys, color=LOWK_TAN, s=16, alpha=0.45, linewidths=0, zorder=1)
        ax.step(px, py, where="post", color=LOWK_TAN, linewidth=1.1, linestyle="--", alpha=0.7, zorder=1,
                label="Fixed K < 192 (96/120/144/168)" if i == 0 else None)
        max_x = max(max_x, max(xs))

    baseline_192 = next(l for l in baseline_labels if baseline_k(l) == 192)
    pts = by_schedule[baseline_192]
    xs = [r["bits_used_total"] / r["episodes"] / 1e6 for r in pts]
    ys = [r["success_rate"] for r in pts]
    px, py = pareto_step(xs, ys)
    ax.step(px, py, where="post", color=BASELINE_BLACK, linewidth=2.6, zorder=6,
            label="Fixed K=192 (baseline, full width)")
    ax.scatter(xs, ys, color=BASELINE_BLACK, s=44, marker="s", zorder=7, edgecolors="white", linewidths=1.0)
    max_x = max(max_x, max(xs))

    if x_clip is not None:
        ax.set_xlim(0, x_clip)
        clip_note = f"; x clipped at {x_clip:.0f}M bits/ep"
    else:
        ax.set_xlim(0, max_x * 1.03)
        clip_note = ""

    ax.set_xlabel("Bits per episode (×10⁶)", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title(
        f"{env_title}  goal_offset={goal_offset}  {episodes} episodes  horizon {horizon}\n"
        f"adaptive fidelity scheduling vs. fixed-K{clip_note}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9.5, loc="lower right", frameon=True, facecolor="white",
              edgecolor="0.8", framealpha=0.95)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}  (dense schedules: {len(dense_labels)}, baselines: {len(baseline_labels)}, "
          f"dropped fixed-fidelity: {len(dropped)})")
    if dropped:
        print("  dropped:", dropped)


ENVS = [
    ("ogb_cube", "OGB Cube"),
    ("tworoom", "TwoRoom"),
    ("reacher", "Reacher"),
]

if __name__ == "__main__":
    failures = []
    for key, title in ENVS:
        for tag, name_suffix, goal_offset, horizon in [
            ("goal25", "", 25, 2),
            ("goal50", "_goal50_plan100_execute20", 50, 4),
        ]:
            cfg_name = f"release20260728_dense_{key}{name_suffix}_all_fidelity_schedules"
            try:
                make_plot(
                    config_path=REPO / f"configs/research/{cfg_name}.yaml",
                    output_dir=REPO / f"reports/research/{cfg_name}",
                    env_title=title,
                    goal_offset=goal_offset,
                    episodes=100,
                    horizon=horizon,
                    out_path=REPO / f"rollouts/{key}_release20260728_{tag}_100ep_all_configs.png",
                )
            except SystemExit as exc:
                print(f"SKIPPED {key} {tag}: {exc}")
                failures.append((key, tag, str(exc)))
    if failures:
        print("\nfailed/skipped combos:", failures)
