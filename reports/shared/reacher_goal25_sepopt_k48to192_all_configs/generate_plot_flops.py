"""Regenerate plot_flops.png -- same structure as generate_plot.py, but using
REAL audited dynamics FLOPs (dynamics_flops_total) as the x-axis instead of
the bits/latent-work proxy (bits_used_total).

All three plotted series (k96to192_slice_summary.json,
individual_fixed_k_baselines_summary.json,
dense_checkpoint_individual_levels_summary.json) carry real, natively
audited dynamics_flops_total -- no post-hoc reconstruction needed. The full
K48-192 adaptive-schedule series (data/sepopt_k48to192_adaptive_summary.json)
is excluded from both plots because every one of its 418 cells has
dynamics_flops_total=0 (never reconstructed).

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

ADAPTIVE_PREFIXES = {"02", "05", "06", "07", "08", "11", "12", "13", "14", "17",
                     "18", "19", "20", "21", "22", "23", "24", "25", "26"}


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
    singlek = load("individual_fixed_k_baselines_summary.json")
    k96to192 = load("k96to192_slice_summary.json")
    dense_levels = load("dense_checkpoint_individual_levels_summary.json")

    k96_runs = [r for r in k96to192["runs"] if r["base_name"].split("_", 1)[0] in ADAPTIVE_PREFIXES]
    k96_pts = [(gflops_per_ep(r), r["success_rate"]) for r in k96_runs]
    k96_done, k96_target = len(k96_runs), 19 * 22

    singlek_pts = [(gflops_per_ep(r), r["success_rate"]) for r in singlek["runs"]]
    s_done = s_target = len(singlek["runs"])

    # Dense-sliced: only K>=96 levels shown, matching the K96-192 range of
    # the other two series (level0_k48/level1_k72 excluded from the plot;
    # still present in data/dense_checkpoint_individual_levels_summary.json).
    dense_runs = [r for r in dense_levels["runs"]
                  if r["base_name"].split("_", 1)[0] not in {"level0", "level1"}]
    dense_pts = [(gflops_per_ep(r), r["success_rate"]) for r in dense_runs]
    d_done = d_target = len(dense_pts)

    k96_frontier = pareto_frontier(k96_pts)
    singlek_frontier = pareto_frontier(singlek_pts)
    dense_frontier = pareto_frontier(dense_pts)

    def draw_series(target_ax, marker_scale=1.0):
        target_ax.scatter(*zip(*dense_pts), s=14 * marker_scale, color=TEAL, alpha=0.25,
                           marker="s", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*singlek_pts), s=14 * marker_scale, color=ORANGE, alpha=0.25,
                           marker="^", zorder=3, linewidths=0)
        target_ax.scatter(*zip(*k96_pts), s=14 * marker_scale, color=PURPLE, alpha=0.25,
                           zorder=4, linewidths=0)
        target_ax.plot(*zip(*dense_frontier), color=TEAL, lw=2, zorder=5)
        target_ax.scatter(*zip(*dense_frontier), s=40 * marker_scale, color=TEAL, marker="s",
                           zorder=6, linewidths=0)
        target_ax.plot(*zip(*singlek_frontier), color=ORANGE, lw=2, zorder=7)
        target_ax.scatter(*zip(*singlek_frontier), s=45 * marker_scale, color=ORANGE, marker="^",
                           zorder=8, linewidths=0)
        target_ax.plot(*zip(*k96_frontier), color=PURPLE, lw=2, zorder=9)
        target_ax.scatter(*zip(*k96_frontier), s=45 * marker_scale, color=PURPLE, marker="D",
                           zorder=10, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    ax.scatter([], [], s=45, color=PURPLE, marker="D", linewidths=0,
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
    ax.set_title(
        r"Reacher  $\Delta$=25",
        fontsize=20, fontweight="bold",
    )
    all_success_rates = [sr for _, sr in dense_pts + singlek_pts + k96_pts]
    y_min = max(0, min(all_success_rates) - 5)
    y_max = min(102, max(all_success_rates) + 5)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95)

    fig.tight_layout()

    out = HERE / "plot_flops.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
