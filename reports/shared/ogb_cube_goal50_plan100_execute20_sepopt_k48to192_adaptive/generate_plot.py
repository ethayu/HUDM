"""Regenerate plot.png from the data/ in this folder.

OGB-Cube (swm/OGBCube-v0) goal_offset=50, 100 episodes, horizon=4 (20-step
plan / 4-step execute blocks), sepopt K48-192 checkpoint
(checkpoints_dense_k48to192_sepopt_20260917/checkpoints_mwm/
mwm_paper10_ogb_cube_k48_72_96_120_144_168_192_sepopt_actckpt_20260917).
Structure mirrors the reacher goal50 report at
../reacher_goal50_plan100_execute20_sepopt_k48to192_adaptive/, minus the
K96-192-floor series (not yet run for OGB-Cube).

- Runs 02-26, K48-192 (19 adaptive schedules x 25 CEM combos = 475 cells,
  COMPLETE): blue dots + Pareto frontier. From
  release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules.yaml,
  data/adaptive_summary.json.
- Runs 27-31 (5 individually-trained fixed-K checkpoints x 25 CEM combos =
  125 cells, COMPLETE): orange triangles. Same config as above (unlike
  reacher, OGB-Cube's goal50 config ships these uncommented already), data/
  singlek_summary.json.
- Dense checkpoint, individual fixed levels (7 levels x 25 CEM combos = 175
  cells, COMPLETE): teal squares. From
  release20260728_dense_ogb_cube_goal50_plan50_execute20_dense_checkpoint_individual_levels.yaml,
  data/dense_checkpoint_individual_levels_summary.json.

All three series get their own Pareto frontier line + solid frontier
markers, with their non-frontier cells dimmed -- same visual language
throughout (matching ../ogb_cube_goal25_sepopt_k48to192_all_configs/).

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

BLUE = "#2a78d6"
ORANGE = "#e07b1a"
TEAL = "#1f9e8e"
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


def load(name: str) -> dict:
    return json.load(open(HERE / "data" / name))


def main() -> None:
    adaptive = load("adaptive_summary.json")
    singlek = load("singlek_summary.json")
    dense_levels = load("dense_checkpoint_individual_levels_summary.json")

    a_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    a_target = adaptive.get("runs_target", len(adaptive["runs"]))
    s_done = singlek.get("runs_completed", len(singlek["runs"]))
    s_target = singlek.get("runs_target", len(singlek["runs"]))
    d_target = len(dense_levels["runs"])
    dense_runs = [r for r in dense_levels["runs"] if r.get("n_iter") != 5]
    d_done = len(dense_runs)

    adaptive_pts = [(bits_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    singlek_pts = [(bits_per_ep(r), r["success_rate"]) for r in singlek["runs"]]
    dense_pts = [(bits_per_ep(r), r["success_rate"]) for r in dense_runs]

    adaptive_frontier = pareto_frontier(adaptive_pts)
    singlek_frontier = pareto_frontier(singlek_pts)
    dense_frontier = pareto_frontier(dense_pts)

    def draw_series(target_ax, marker_scale=1.0):
        # Dimmed all-cells scatter (same visual language across all three series).
        target_ax.scatter(*zip(*dense_pts), s=14 * marker_scale, color=TEAL, alpha=0.25,
                           marker="s", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*singlek_pts), s=14 * marker_scale, color=ORANGE, alpha=0.25,
                           marker="^", zorder=3, linewidths=0)
        target_ax.scatter(*zip(*adaptive_pts), s=14 * marker_scale, color=BLUE, alpha=0.25,
                           zorder=4, linewidths=0)
        # Frontier lines + solid markers.
        target_ax.plot(*zip(*dense_frontier), color=TEAL, lw=2, zorder=5)
        target_ax.scatter(*zip(*dense_frontier), s=40 * marker_scale, color=TEAL, marker="s",
                           zorder=6, linewidths=0)
        target_ax.plot(*zip(*singlek_frontier), color=ORANGE, lw=2, zorder=7)
        target_ax.scatter(*zip(*singlek_frontier), s=45 * marker_scale, color=ORANGE, marker="^",
                           zorder=8, linewidths=0)
        target_ax.plot(*zip(*adaptive_frontier), color=BLUE, lw=2, zorder=9)
        target_ax.scatter(*zip(*adaptive_frontier), s=45 * marker_scale, color=BLUE, zorder=10, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    ax.scatter([], [], s=45, color=BLUE, linewidths=0,
               label=f"Adaptive schedules (Pareto frontier, {a_done}/{a_target})")
    ax.scatter([], [], s=14, color=BLUE, alpha=0.25, linewidths=0,
               label="Adaptive schedules (all cells)")
    ax.scatter([], [], s=45, color=ORANGE, marker="^", linewidths=0,
               label=f"Individually-trained fixed-K (Pareto frontier, {s_done}/{s_target})")
    ax.scatter([], [], s=14, color=ORANGE, alpha=0.25, marker="^", linewidths=0,
               label="Individually-trained fixed-K (all cells)")
    ax.scatter([], [], s=40, color=TEAL, marker="s", linewidths=0,
               label=f"Dense-sliced fixed-level (Pareto frontier, {d_done}/{d_target})")
    ax.scatter([], [], s=14, color=TEAL, alpha=0.25, marker="s", linewidths=0,
               label="Dense-sliced fixed-level (all cells)")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_title(
        "OGB-Cube  goal_offset=50  100 episodes  horizon 4 (plan20/execute4)\n"
        "adaptive fidelity scheduling vs. individually-trained and dense-sliced fixed-K models\n"
        "sepopt K48-192 checkpoint",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # Inset: zoom on the low-cost region where the adaptive Pareto frontier's
    # elbow sits (the frontier is flat/near-ceiling well before the bulk of
    # the scatter, which spreads out to much higher cost with no success gain).
    frontier_x_max = max(p[0] for p in adaptive_frontier)
    x_hi = max(20.0, frontier_x_max * 2.5)
    axins = ax.inset_axes([0.62, 0.58, 0.36, 0.38])
    axins.set_facecolor("#fcfcfb")
    draw_series(axins, marker_scale=1.6)
    axins.set_xlim(0, x_hi)
    axins.set_ylim(0, 102)
    axins.grid(True, color=GRID, lw=0.6, zorder=0)
    axins.tick_params(labelsize=8)
    for spine in axins.spines.values():
        spine.set_color(GRID)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="#888888", lw=0.8)

    fig.tight_layout()

    out = HERE / "plot.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
