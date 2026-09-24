"""Regenerate plot.png from the data/ and config in this folder.

Same structure as tworoom_goal25_sepopt_k48to192_all_configs, for OGB-Cube
(swm/OGBCube-v0) goal_offset=25, 100 episodes, horizon=2.

IMPORTANT CAVEATS -- read before using this for the paper:

1. PARTIAL DATA: the sepopt K48-192 adaptive-schedule sweep (blue) was still
   running the last time it was refreshed (see the "note" field in
   data/sepopt_k48to192_adaptive_summary.json for the completed/target
   count). It ran as two concurrent --roles-filtered halves on separate
   GPUs, so data/sepopt_k48to192_adaptive_summary.json is built directly
   from individual per-cell summary.json sidecars rather than any top-level
   aggregate (see the reacher folder's README for why: the top-level
   aggregate gets overwritten by whichever half finishes last). No newer/
   complete source directory for this series was found on disk as of the
   last refresh -- rerun once one turns up.

2. Fixed-K series (orange) and dense-sliced-fixed-level series (teal) are
   BOTH now COMPLETE, replacing the old partial/mismatched-grid data:
   - data/individual_fixed_k_baselines_summary.json (110/110): 5
     individually-trained fixed-K checkpoints (K=192/168/144/120/96) on the
     same 22-combo grid as the adaptive sweep. Replaces the older
     data/fixed_k_baselines_summary.json (125 cells, 25-combo grid).
   - data/dense_checkpoint_individual_levels_summary.json (154/154): the
     SAME shared sepopt checkpoint, held fixed (no scheduling) at each of
     its 7 declared levels (K=48,72,96,120,144,168,192) in turn. Supersedes
     the old single-level data/sepopt_k96_fixed_cem_grid_summary.json and
     data/sepopt_k192_fixed_cem_grid_summary.json (the latter never
     completed) -- both are dropped from this plot in favor of the
     comprehensive all-levels series.
   Built by build_dense_levels_and_refreshed_fixedk_summary.py. All three
   series get their own Pareto frontier line + solid frontier markers, with
   their non-frontier cells dimmed -- same visual language throughout.

3. Dense-sliced fixed-level excludes n_iter=5 cells (119/154 plotted).

4. The K=48-96 restricted adaptive probe (data/sepopt_k48_96_probe_summary.json)
   is intentionally NOT drawn in this plot -- it remains in data/ and in the
   config/README for reference (it does extend the adaptive frontier's cheap
   end, see the README), but is excluded here to keep the figure to the
   three main series.

Checkpoint swap: adaptive-schedule runs and the dense-sliced-fixed-level
series use the sepopt K48-192 checkpoint
(checkpoints_dense_k48to192_sepopt_20260917/checkpoints_mwm/
mwm_paper10_ogb_cube_k48_72_96_120_144_168_192_sepopt_actckpt_20260917,
K=[48,72,96,120,144,168,192], epoch 9) instead of the original paper10 dense
checkpoint (K=[96,120,144,168,192]). The individually-trained fixed-K series
uses 5 separately-trained single-fidelity checkpoints instead.

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
EPISODES = 100  # all data files below use 100-episode evals

BLUE = "#2a78d6"
ORANGE = "#e07b1a"
TEAL = "#1f9e8e"
GRID = "#d9d9d6"


def bits_per_ep(run: dict) -> float:
    episodes = run.get("episodes") or EPISODES
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
    adaptive = load("sepopt_k48to192_adaptive_summary.json")
    fixedk = load("individual_fixed_k_baselines_summary.json")
    dense_levels = load("dense_checkpoint_individual_levels_summary.json")

    adaptive_runs = adaptive["runs"]
    fixedk_runs = fixedk["runs"]
    d_total = len(dense_levels["runs"])
    dense_runs = [r for r in dense_levels["runs"] if r.get("n_iter") != 5]
    a_shown = a_total = len(adaptive_runs)
    f_shown = f_total = len(fixedk_runs)
    d_shown = len(dense_runs)

    adaptive_pts = [(bits_per_ep(r), r["success_rate"]) for r in adaptive_runs]
    fixedk_pts = [(bits_per_ep(r), r["success_rate"]) for r in fixedk_runs]
    dense_pts = [(bits_per_ep(r), r["success_rate"]) for r in dense_runs]

    adaptive_frontier = pareto_frontier(adaptive_pts)
    fixedk_frontier = pareto_frontier(fixedk_pts)
    dense_frontier = pareto_frontier(dense_pts)

    def draw_series(target_ax, marker_scale=1.0):
        # Dimmed all-cells scatter (same visual language across all three series).
        target_ax.scatter(*zip(*dense_pts), s=14 * marker_scale, color=TEAL, alpha=0.25,
                           marker="s", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*fixedk_pts), s=14 * marker_scale, color=ORANGE, alpha=0.25,
                           marker="^", zorder=3, linewidths=0)
        target_ax.scatter(*zip(*adaptive_pts), s=14 * marker_scale, color=BLUE, alpha=0.25,
                           zorder=4, linewidths=0)
        # Frontier lines + solid markers.
        target_ax.plot(*zip(*dense_frontier), color=TEAL, lw=2, zorder=5)
        target_ax.scatter(*zip(*dense_frontier), s=40 * marker_scale, color=TEAL, marker="s",
                           zorder=6, linewidths=0)
        target_ax.plot(*zip(*fixedk_frontier), color=ORANGE, lw=2, zorder=7)
        target_ax.scatter(*zip(*fixedk_frontier), s=45 * marker_scale, color=ORANGE, marker="^",
                           zorder=8, linewidths=0)
        target_ax.plot(*zip(*adaptive_frontier), color=BLUE, lw=2, zorder=9)
        target_ax.scatter(*zip(*adaptive_frontier), s=45 * marker_scale, color=BLUE, zorder=10, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    ax.scatter([], [], s=45, color=BLUE, linewidths=0,
               label=f"Adaptive schedules (Pareto frontier, {a_shown}/{a_total})")
    ax.scatter([], [], s=14, color=BLUE, alpha=0.25, linewidths=0,
               label="Adaptive schedules (all cells)")
    ax.scatter([], [], s=45, color=ORANGE, marker="^", linewidths=0,
               label=f"Individually-trained fixed-K (Pareto frontier, {f_shown}/{f_total})")
    ax.scatter([], [], s=14, color=ORANGE, alpha=0.25, marker="^", linewidths=0,
               label="Individually-trained fixed-K (all cells)")
    ax.scatter([], [], s=40, color=TEAL, marker="s", linewidths=0,
               label=f"Dense-sliced fixed-level (Pareto frontier, {d_shown}/{d_total})")
    ax.scatter([], [], s=14, color=TEAL, alpha=0.25, marker="s", linewidths=0,
               label="Dense-sliced fixed-level (all cells)")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    title = (
        "OGB-Cube  goal_offset=25  100 episodes  horizon 2\n"
        "adaptive fidelity scheduling vs. individually-trained and dense-sliced fixed-K models\n"
        "sepopt K48-192 checkpoint"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # Inset: zoom on the 0-30M bits/episode region, where most points sit.
    axins = ax.inset_axes([0.60, 0.60, 0.38, 0.36])
    axins.set_facecolor("#fcfcfb")
    draw_series(axins, marker_scale=1.6)
    axins.set_xlim(0, 30)
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
