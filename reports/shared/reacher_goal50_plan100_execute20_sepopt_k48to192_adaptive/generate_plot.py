"""Regenerate plot.png from the data/ in this folder.

Same structure/style as reacher_goal25_sepopt_k48to192_all_configs, for
Reacher (swm/ReacherDMControl-v0) goal_offset=50, 100 episodes, horizon=4
(20-step plan / 4-step execute blocks), release20260728_dense_reacher_
goal50_plan50_execute20_all_fidelity_schedules.yaml.

Two series, both COMPLETE:
- Adaptive schedules: 19 schedules x 25 CEM combos = 475 cells, sepopt
  K48-192 checkpoint.
- Fixed K=192 (no scheduling) on the SAME sepopt K48-192 checkpoint: 25 CEM
  combos, from release20260728_dense_reacher_goal50_plan50_execute20_fixed_k192_sepopt.yaml
  (a separate config -- the fixed-K baselines were commented out of the
  adaptive config itself). This directly tests whether adaptive scheduling
  earns anything over just running the same checkpoint at full fidelity the
  whole time.

Usage: python generate_plot.py  (writes plot.png in this directory)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

BLUE = "#2a78d6"
BLACK = "#111111"
GRID = "#d9d9d6"


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


def main() -> None:
    adaptive = json.load(open(HERE / "data" / "sepopt_k48to192_adaptive_summary.json"))
    baseline = json.load(open(HERE / "data" / "fixed_k192_sepopt_summary.json"))

    n_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    n_target = adaptive.get("runs_target", len(adaptive["runs"]))
    b_done = baseline.get("runs_completed", len(baseline["runs"]))
    b_target = baseline.get("runs_target", len(baseline["runs"]))

    adaptive_pts = [(bits_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    frontier = pareto_frontier(adaptive_pts)
    baseline_pts = [(bits_per_ep(r), r["success_rate"]) for r in baseline["runs"]]

    def draw_series(target_ax, marker_scale=1.0):
        target_ax.scatter(*zip(*adaptive_pts), s=14 * marker_scale, color=BLUE, alpha=0.25,
                           zorder=2, linewidths=0)
        target_ax.plot(*zip(*frontier), color=BLUE, lw=2, zorder=3)
        target_ax.scatter(*zip(*frontier), s=45 * marker_scale, color=BLUE, zorder=4, linewidths=0)
        target_ax.scatter(*zip(*baseline_pts), s=32 * marker_scale, color=BLACK, alpha=0.65,
                           marker="s", zorder=3, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    ax.scatter([], [], s=45, color=BLUE, linewidths=0,
               label=f"Adaptive schedules (Pareto frontier, {n_done}/{n_target})")
    ax.scatter([], [], s=14, color=BLUE, alpha=0.25, linewidths=0,
               label="Adaptive schedules (all completed cells)")
    ax.scatter([], [], s=32, color=BLACK, alpha=0.65, marker="s", linewidths=0,
               label=f"Fixed K=192, no scheduling (same sepopt checkpoint, {b_done}/{b_target})")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_title(
        "Reacher  goal_offset=50  100 episodes  horizon 4 (plan20/execute4)\n"
        "adaptive fidelity scheduling vs. fixed K=192 -- sepopt K48-192 checkpoint",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Inset: zoom on the low-cost region where the Pareto frontier's elbow
    # sits (the frontier is flat/near-ceiling well before the bulk of the
    # scatter, which spreads out to much higher cost with no success gain).
    frontier_x_max = max(p[0] for p in frontier)
    x_hi = max(20.0, frontier_x_max * 2.5)
    axins = ax.inset_axes([0.62, 0.60, 0.36, 0.36])
    axins.set_facecolor("#fcfcfb")
    draw_series(axins, marker_scale=1.6)
    axins.set_xlim(0, x_hi)
    axins.set_ylim(30, 102)
    axins.grid(True, color=GRID, lw=0.6, zorder=0)
    axins.tick_params(labelsize=8)
    for spine in axins.spines.values():
        spine.set_color(GRID)
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="#888888", lw=0.8)

    fig.tight_layout()

    out = HERE / "plot.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
