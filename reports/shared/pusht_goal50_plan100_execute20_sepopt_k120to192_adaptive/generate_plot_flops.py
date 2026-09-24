"""Regenerate plot_flops.png -- same structure as generate_plot.py, but using
REAL audited dynamics FLOPs (dynamics_flops_total) as the x-axis instead of
the bits/latent-work proxy (bits_used_total).

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


def load(name: str) -> dict:
    return json.load(open(HERE / "data" / name))


def main() -> None:
    adaptive = load("adaptive_summary.json")
    singlek = load("singlek_summary.json")
    dense_levels = load("dense_checkpoint_individual_levels_summary.json")

    a_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    a_target = adaptive.get("runs_target", len(adaptive["runs"]))

    # Individually-trained fixed-K: drop K=96 (K=48/72 have no
    # individually-trained checkpoint at all). Still present, unfiltered, in
    # data/singlek_summary.json.
    singlek = {**singlek, "runs": [r for r in singlek["runs"] if r["base_name"].split("_", 1)[0] != "31"]}
    s_done = s_target = len(singlek["runs"])
    d_done = dense_levels.get("runs_completed", len(dense_levels["runs"]))
    d_target = dense_levels.get("runs_target", len(dense_levels["runs"]))

    adaptive_pts = [(gflops_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    singlek_pts = [(gflops_per_ep(r), r["success_rate"]) for r in singlek["runs"]]
    dense_pts = [(gflops_per_ep(r), r["success_rate"]) for r in dense_levels["runs"]]

    adaptive_frontier = pareto_frontier(adaptive_pts)
    singlek_frontier = pareto_frontier(singlek_pts)
    dense_frontier = pareto_frontier(dense_pts)

    def draw_series(target_ax, marker_scale=1.0):
        target_ax.scatter(*zip(*dense_pts), s=14 * marker_scale, color=TEAL, alpha=0.25,
                           marker="s", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*singlek_pts), s=14 * marker_scale, color=ORANGE, alpha=0.25,
                           marker="^", zorder=3, linewidths=0)
        target_ax.scatter(*zip(*adaptive_pts), s=14 * marker_scale, color=PURPLE, alpha=0.25,
                           zorder=4, linewidths=0)
        target_ax.plot(*zip(*dense_frontier), color=TEAL, lw=2, zorder=5)
        target_ax.scatter(*zip(*dense_frontier), s=40 * marker_scale, color=TEAL, marker="s",
                           zorder=6, linewidths=0)
        target_ax.plot(*zip(*singlek_frontier), color=ORANGE, lw=2, zorder=7)
        target_ax.scatter(*zip(*singlek_frontier), s=45 * marker_scale, color=ORANGE, marker="^",
                           zorder=8, linewidths=0)
        target_ax.plot(*zip(*adaptive_frontier), color=PURPLE, lw=2, zorder=9)
        target_ax.scatter(*zip(*adaptive_frontier), s=45 * marker_scale, color=PURPLE, marker="D",
                           zorder=10, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    ax.scatter([], [], s=45, color=PURPLE, marker="D", linewidths=0,
               label=f"K120-192 floor: MWM (scheduled) (Pareto frontier, {a_done}/{a_target})")
    ax.scatter([], [], s=14, color=PURPLE, alpha=0.25, linewidths=0,
               label="K120-192 floor: MWM (scheduled) (all cells)")
    ax.scatter([], [], s=45, color=ORANGE, marker="^", linewidths=0,
               label=f"Baseline (Pareto frontier, {s_done}/{s_target})")
    ax.scatter([], [], s=14, color=ORANGE, alpha=0.25, marker="^", linewidths=0,
               label="Baseline (all cells)")
    ax.scatter([], [], s=40, color=TEAL, marker="s", linewidths=0,
               label=f"MWM (fixed), K>=120 (Pareto frontier, {d_done}/{d_target})")
    ax.scatter([], [], s=14, color=TEAL, alpha=0.25, marker="s", linewidths=0,
               label="MWM (fixed), K>=120 (all cells)")

    ax.set_xlabel("Audited dynamics GFLOPs per episode", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_title(
        r"PushT  $\Delta$=50",
        fontsize=16, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()

    out = HERE / "plot_flops.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
