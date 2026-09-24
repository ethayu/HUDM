"""Regenerate plot.png from the data/ in this folder.

Reacher (swm/ReacherDMControl-v0) goal_offset=50, 100 episodes, horizon=4
(20-step plan / 4-step execute blocks), sepopt K48-192 checkpoint
(checkpoints_mwm/mwm_paper10_reacher_k48_72_96_120_144_168_192_sepopt_actckpt_20260917
unless noted). Three series:

- Runs 02-26, K96-192 range (475 cells): the scheduled-fidelity results,
  every "coarsest" schedule endpoint floored at K=96 (level index 2) --
  purple diamonds + Pareto frontier. From
  release20260728_dense_reacher_goal50_plan50_execute20_k96to192slice_sepopt_all_fidelity_schedules.yaml,
  built into data/k96to192_slice_summary.json (which also contains a
  redundant re-run of runs 27-31, filtered out here).
- Runs 27-31 (5 individually-trained fixed-K checkpoints x 25 CEM combos =
  125 cells): each of K=192/168/144/120/96 trained separately as its own
  single-fidelity checkpoint -- orange triangles + Pareto frontier. Same
  source config as above, built into data/new_sweep_partial_singlek_summary.json.
- Dense checkpoint, individual levels, K96-192 only (5 of 7 levels x 25 CEM
  combos = 125 cells plotted): the SAME shared sepopt checkpoint held fixed
  (no scheduling) at each level -- teal squares + Pareto frontier. Levels
  K48/K72 (level0/level1) are excluded from the plot to match the K96-192
  range of the other two series; they're still present, unfiltered, in
  data/dense_checkpoint_individual_levels_summary.json (175 cells total,
  all 7 levels) -- see that file directly for the full K48-192 picture.
  Includes n_iter=5 cells (previously excluded in one revision of this
  script; put back on request, no data-quality issue was ever found with
  them). From
  release20260728_dense_reacher_goal50_plan50_execute20_dense_checkpoint_individual_levels.yaml,
  built into data/dense_checkpoint_individual_levels_summary.json.

Usage: python generate_plot.py  (writes plot.png in this directory)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

HERE = Path(__file__).resolve().parent

PURPLE = "#8e5fd1"
ORANGE = "#e07b1a"
TEAL = "#1f9e8e"
GRID = "#d9d9d6"

ADAPTIVE_PREFIXES = {"02", "05", "06", "07", "08", "11", "12", "13", "14", "17",
                     "18", "19", "20", "21", "22", "23", "24", "25", "26"}


def bits_per_ep(run: dict) -> float:
    episodes = run.get("episodes") or 100
    return run.get("bits_used_total", 0) / episodes / 1e6


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left frontier: maximize success rate, minimize cost."""
    pts_sorted = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier = []
    best_sr = -1.0
    for bits, sr in pts_sorted:
        if sr > best_sr:
            frontier.append((bits, sr))
            best_sr = sr
    return frontier


def load(name: str) -> dict:
    return json.load(open(HERE / "data" / name))


def main() -> None:
    singlek = load("new_sweep_partial_singlek_summary.json")
    k96to192 = load("k96to192_slice_summary.json")
    dense_levels = load("dense_checkpoint_individual_levels_summary.json")

    k96_runs = [r for r in k96to192["runs"] if r["base_name"].split("_", 1)[0] in ADAPTIVE_PREFIXES]
    k96_pts = [(bits_per_ep(r), r["success_rate"]) for r in k96_runs]
    k96_done, k96_target = len(k96_runs), 19 * 25

    singlek_pts = [(bits_per_ep(r), r["success_rate"]) for r in singlek["runs"]]
    s_done = s_target = len(singlek["runs"])

    # Dense-sliced: only K>=96 levels shown, matching the K96-192 range of
    # the other two series (level0_k48/level1_k72 excluded from the plot;
    # still present in data/dense_checkpoint_individual_levels_summary.json).
    dense_runs = [r for r in dense_levels["runs"]
                  if r["base_name"].split("_", 1)[0] not in {"level0", "level1"}]
    dense_pts = [(bits_per_ep(r), r["success_rate"]) for r in dense_runs]
    d_done = d_target = len(dense_pts)

    k96_frontier = pareto_frontier(k96_pts)
    singlek_frontier = pareto_frontier(singlek_pts)
    dense_frontier = pareto_frontier(dense_pts)

    def draw_series(target_ax, marker_scale=1.0):
        # Dimmed all-cells scatter (same visual language across all three series).
        target_ax.scatter(*zip(*dense_pts), s=14 * marker_scale, color=TEAL, alpha=0.25,
                           marker="s", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*singlek_pts), s=14 * marker_scale, color=ORANGE, alpha=0.25,
                           marker="^", zorder=3, linewidths=0)
        target_ax.scatter(*zip(*k96_pts), s=14 * marker_scale, color=PURPLE, alpha=0.25,
                           zorder=4, linewidths=0)
        # Frontier lines + solid markers.
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

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=18)
    ax.set_title(
        r"Reacher  $\Delta$=50",
        fontsize=20, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95)

    # Inset: zoom on the low-cost region where the Pareto frontiers' elbows
    # sit.
    frontier_x_max = max(p[0] for p in k96_frontier)
    x_hi = max(20.0, frontier_x_max * 2.5)
    axins = ax.inset_axes([0.62, 0.58, 0.36, 0.38])
    axins.set_facecolor("#fcfcfb")
    draw_series(axins, marker_scale=1.6)
    axins.set_xlim(0, x_hi)
    axins.set_ylim(0, 102)
    axins.grid(True, color=GRID, lw=0.6, zorder=0)
    axins.tick_params(labelsize=11)
    for spine in axins.spines.values():
        spine.set_color(GRID)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="#888888", lw=0.8)

    fig.tight_layout()

    out = HERE / "plot.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
