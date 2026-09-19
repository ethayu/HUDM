"""Regenerate plot.png from the data/ and config in this folder.

Same structure as tworoom_goal25_sepopt_k48to192_all_configs, for Reacher
(swm/ReacherDMControl-v0) goal_offset=25, 100 episodes, horizon=2.

IMPORTANT CAVEAT: the sepopt K48-192 adaptive-schedule sweep was still
running when this was generated (see the "note" field in
data/sepopt_k48to192_adaptive_summary.json for the completed/target count).
The Pareto frontier and any "best result" numbers here will likely shift as
more cells complete. Regenerate once the sweep finishes.

Checkpoint swap: adaptive-schedule runs (02-26) use the sepopt K48-192
checkpoint (checkpoints_dense_k48to192_sepopt_20260917_reacher/checkpoints_mwm/
mwm_paper10_reacher_k48_72_96_120_144_168_192_sepopt_actckpt_20260917,
K=[48,72,96,120,144,168,192], epoch 9) instead of the original paper10 dense
checkpoint (K=[96,120,144,168,192]). Fixed K=192 / Fixed K<192 use
individually-trained single-K checkpoints (untouched by the swap, reused from
the original sweep). Fixed K=96 (dense) uses the *original* paper10 dense
checkpoint held fixed at K=96 for the whole plan (also reused, not rerun).
All four series use episodes=100.

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
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLACK = "#111111"
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


def main() -> None:
    adaptive = json.load(open(HERE / "data" / "sepopt_k48to192_adaptive_summary.json"))
    baselines = json.load(open(HERE / "data" / "fixed_k_baselines_summary.json"))
    dense_k96 = json.load(open(HERE / "data" / "dense_k96_fixed_cem_grid_summary.json"))

    n_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    n_target = adaptive.get("runs_target", len(adaptive["runs"]))

    adaptive_pts = [(bits_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    frontier = pareto_frontier(adaptive_pts)

    k192_ckpt = "checkpoints_mwm/mwm_paper10_reacher_k192_release20260728"
    fixed192_pts = [
        (bits_per_ep(r), r["success_rate"])
        for r in baselines["runs"]
        if r.get("checkpoint_run_dir") == k192_ckpt
    ]
    fixedsub_pts = [
        (bits_per_ep(r), r["success_rate"])
        for r in baselines["runs"]
        if r.get("checkpoint_run_dir") != k192_ckpt
    ]
    dense_k96_pts = [(bits_per_ep(r), r["success_rate"]) for r in dense_k96["runs"]]

    def draw_series(target_ax, marker_scale=1.0):
        target_ax.scatter(*zip(*adaptive_pts), s=14 * marker_scale, color=BLUE, alpha=0.25,
                           zorder=2, linewidths=0)
        target_ax.plot(*zip(*frontier), color=BLUE, lw=2, zorder=3)
        target_ax.scatter(*zip(*frontier), s=45 * marker_scale, color=BLUE, zorder=4, linewidths=0)
        target_ax.scatter(*zip(*fixedsub_pts), s=14 * marker_scale, color=ORANGE, alpha=0.35,
                           marker="o", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*fixed192_pts), s=28 * marker_scale, color=BLACK, alpha=0.55,
                           marker="s", zorder=3, linewidths=0)
        target_ax.scatter(*zip(*dense_k96_pts), s=40 * marker_scale, color=AQUA, marker="^",
                           zorder=4, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    # Re-add labelled (invisible-duplicate-free) legend handles on the main axes.
    ax.scatter([], [], s=45, color=BLUE, linewidths=0,
               label=f"Winning adaptive schedules (Pareto frontier, PARTIAL {n_done}/{n_target})")
    ax.scatter([], [], s=14, color=ORANGE, alpha=0.35, marker="o", linewidths=0,
               label="Fixed K<192 (96/120/144/168, individually-trained)")
    ax.scatter([], [], s=28, color=BLACK, alpha=0.55, marker="s", linewidths=0,
               label="Fixed K=192 (baseline, individually-trained)")
    ax.scatter([], [], s=40, color=AQUA, marker="^", linewidths=0,
               label="Fixed K=96 (dense checkpoint, no scheduling)")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_title(
        "Reacher  goal_offset=25  100 episodes  horizon 2\n"
        "adaptive fidelity scheduling vs. fixed-K -- sepopt K48-192 checkpoint\n"
        f"(adaptive sweep PARTIAL: {n_done}/{n_target} cells complete)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # Inset: zoom on the 0-10M bits/episode region, where almost every point
    # sits, in the upper-right corner (kept clear of the lower-right legend).
    axins = ax.inset_axes([0.62, 0.60, 0.36, 0.36])
    axins.set_facecolor("#fcfcfb")
    draw_series(axins, marker_scale=1.6)
    axins.set_xlim(0, 10)
    axins.set_ylim(80, 102)
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
