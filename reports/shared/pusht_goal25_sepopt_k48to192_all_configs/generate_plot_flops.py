"""Regenerate plot_flops.png for PushT goal25 (horizon 5, 100 episodes) on
REAL audited dynamics FLOPs (dynamics_flops_total) -- same layout and style
as ../ogb_cube_goal25_sepopt_k48to192_all_configs/generate_plot_flops.py.

Series (all natively audited, no reconstruction):
- Adaptive schedules, sepopt K48-192 checkpoint (572 cells):
  data/sepopt_k48to192_adaptive_summary.json
- Individually-trained fixed-K, K=120-192 (88 cells; K=96 dropped to match
  the K>=120 range of the dense-sliced series, as in the PushT goal50 report):
  data/fixed_k192_baseline_summary.json + data/fixed_k_lt192_baseline_summary.json
- Dense-sliced fixed-level, K>=120 (100 cells):
  data/sepopt_k{120,144,168,192}_fixed_cem_grid_summary.json

Usage: python generate_plot_flops.py  (writes plot_flops.png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EPISODES = 100

PURPLE = "#8e5fd1"
ORANGE = "#e07b1a"
TEAL = "#1f9e8e"
GRID = "#d9d9d6"


def gflops_per_ep(run: dict) -> float:
    episodes = run.get("episodes") or EPISODES
    return run.get("dynamics_flops_total", 0) / episodes / 1e9


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts_sorted = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier = []
    best_sr = -1.0
    for gflops, sr in pts_sorted:
        if sr > best_sr:
            frontier.append((gflops, sr))
            best_sr = sr
    return frontier


def main() -> None:
    def runs(name: str) -> list[dict]:
        return json.load(open(HERE / "data" / name))["runs"]

    adaptive_runs = runs("sepopt_k48to192_adaptive_summary.json")
    fixedk_runs = [r for r in runs("fixed_k192_baseline_summary.json") + runs("fixed_k_lt192_baseline_summary.json")
                   if "k96" not in r["base_name"]]
    dense_runs = [r for k in (120, 144, 168, 192) for r in runs(f"sepopt_k{k}_fixed_cem_grid_summary.json")]
    a_shown = a_total = len(adaptive_runs)
    f_shown = f_total = len(fixedk_runs)
    d_shown = d_total = len(dense_runs)

    adaptive_pts = [(gflops_per_ep(r), r["success_rate"]) for r in adaptive_runs]
    fixedk_pts = [(gflops_per_ep(r), r["success_rate"]) for r in fixedk_runs]
    dense_pts = [(gflops_per_ep(r), r["success_rate"]) for r in dense_runs]

    adaptive_frontier = pareto_frontier(adaptive_pts)
    fixedk_frontier = pareto_frontier(fixedk_pts)
    dense_frontier = pareto_frontier(dense_pts)

    def draw_series(target_ax, marker_scale=1.0):
        # Dimmed all-cells scatter (same visual language across all three series).
        target_ax.scatter(*zip(*dense_pts), s=14 * marker_scale, color=TEAL, alpha=0.25,
                           marker="s", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*fixedk_pts), s=14 * marker_scale, color=ORANGE, alpha=0.25,
                           marker="^", zorder=3, linewidths=0)
        target_ax.scatter(*zip(*adaptive_pts), s=14 * marker_scale, color=PURPLE, alpha=0.25,
                           zorder=4, linewidths=0)
        # Frontier lines + solid markers.
        target_ax.plot(*zip(*dense_frontier), color=TEAL, lw=2, zorder=5)
        target_ax.scatter(*zip(*dense_frontier), s=40 * marker_scale, color=TEAL, marker="s",
                           zorder=6, linewidths=0)
        target_ax.plot(*zip(*fixedk_frontier), color=ORANGE, lw=2, zorder=7)
        target_ax.scatter(*zip(*fixedk_frontier), s=45 * marker_scale, color=ORANGE, marker="^",
                           zorder=8, linewidths=0)
        target_ax.plot(*zip(*adaptive_frontier), color=PURPLE, lw=2, zorder=9)
        target_ax.scatter(*zip(*adaptive_frontier), s=45 * marker_scale, color=PURPLE, zorder=10, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    ax.scatter([], [], s=45, color=PURPLE, linewidths=0,
               label="MWM (scheduled) (Pareto frontier)")
    ax.scatter([], [], s=14, color=PURPLE, alpha=0.25, linewidths=0,
               label="MWM (scheduled) (all cells)")
    ax.scatter([], [], s=45, color=ORANGE, marker="^", linewidths=0,
               label="Single-$d$ (Pareto frontier)")
    ax.scatter([], [], s=14, color=ORANGE, alpha=0.25, marker="^", linewidths=0,
               label="Single-$d$ (all cells)")
    ax.scatter([], [], s=40, color=TEAL, marker="s", linewidths=0,
               label="MWM (fixed) (Pareto frontier)")
    ax.scatter([], [], s=14, color=TEAL, alpha=0.25, marker="s", linewidths=0,
               label="MWM (fixed) (all cells)")

    ax.set_xlabel("Audited dynamics GFLOPs per episode", fontsize=18)
    ax.set_ylabel("Success rate (%)", fontsize=18)
    title = r"PushT  $\Delta$=25"
    ax.set_title(title, fontsize=20, fontweight="bold")
    all_success_rates = [sr for _, sr in dense_pts + fixedk_pts + adaptive_pts]
    y_min = max(0, min(all_success_rates) - 5)
    y_max = min(102, max(all_success_rates) + 5)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95)

    # No inset here: on this log-scale x-axis, a zoomed inset of the low-cost
    # region visually collides with the outer plot's own real low-cost data
    # (which is also compressed into that corner by the log scale), making
    # the frontier look like it dips when it doesn't. See git history for
    # the removed inset code if this needs revisiting with a better placement.
    fig.tight_layout()

    out = HERE / "plot_flops.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
