"""Regenerate plot.png from the data/ and config in this folder.

Same structure as tworoom_goal25_sepopt_k48to192_all_configs, for OGB-Cube
(swm/OGBCube-v0) goal_offset=25, 100 episodes, horizon=2.

IMPORTANT CAVEATS -- read before using this for the paper:

1. PARTIAL DATA: the sepopt K48-192 adaptive-schedule sweep was still
   running when this was generated (see the "note" field in
   data/sepopt_k48to192_adaptive_summary.json for the completed/target
   count). It ran as two concurrent --roles-filtered halves on separate
   GPUs, so data/sepopt_k48to192_adaptive_summary.json is built directly
   from individual per-cell summary.json sidecars rather than any top-level
   aggregate (see the reacher folder's README for why: the top-level
   aggregate gets overwritten by whichever half finishes last).

2. The Fixed K=192 (sepopt checkpoint, no scheduling) series may not exist
   yet -- that eval was still queued/running when this was generated. This
   script skips that series gracefully if
   data/sepopt_k192_fixed_cem_grid_summary.json is absent; rerun once it
   lands for the full 4-series plot.

3. K=48-96 restricted adaptive probe (data/sepopt_k48_96_probe_summary.json,
   12 cells: 2 schedules x 6 CEM combos, same sepopt checkpoint restricted to
   levels 0-2 instead of the full 0-6 range) is folded into the SAME Pareto
   frontier as the full-range adaptive sweep -- it genuinely extends the
   frontier's cheap end (63% drops from 2.69M to 1.54M bits/ep; 79% drops
   from 20.16M to 11.52M bits/ep). Drawn as blue "+" markers, distinct from
   the round full-range adaptive points, so the two families stay
   visually traceable even though they share one frontier line.

Checkpoint swap: adaptive-schedule runs (02-26) use the sepopt K48-192
checkpoint (checkpoints_dense_k48to192_sepopt_20260917_ogb_cube/checkpoints_mwm/
mwm_paper10_ogb_cube_k48_72_96_120_144_168_192_sepopt_actckpt_20260917,
K=[48,72,96,120,144,168,192], epoch 9) instead of the original paper10 dense
checkpoint (K=[96,120,144,168,192]). Fixed K=192 / Fixed K<192 use
individually-trained single-K checkpoints (untouched by the swap, reused from
the original sweep). Fixed K=192 (sepopt) uses the SAME sepopt checkpoint as
the adaptive frontier, held fixed at K=192 (its own finest level) for the
whole plan -- a within-checkpoint ablation, not reused.

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
MAGENTA = "#e87ba4"
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

    k192_grid_path = HERE / "data" / "sepopt_k192_fixed_cem_grid_summary.json"
    sepopt_k192_pts = None
    if k192_grid_path.is_file():
        sepopt_k192 = json.load(open(k192_grid_path))
        sepopt_k192_pts = [(bits_per_ep(r), r["success_rate"]) for r in sepopt_k192["runs"]]

    k96_grid_path = HERE / "data" / "sepopt_k96_fixed_cem_grid_summary.json"
    sepopt_k96_pts = None
    if k96_grid_path.is_file():
        sepopt_k96 = json.load(open(k96_grid_path))
        sepopt_k96_pts = [(bits_per_ep(r), r["success_rate"]) for r in sepopt_k96["runs"]]

    k48_96_probe_path = HERE / "data" / "sepopt_k48_96_probe_summary.json"
    k48_96_probe_pts = None
    if k48_96_probe_path.is_file():
        k48_96_probe = json.load(open(k48_96_probe_path))
        k48_96_probe_pts = [(bits_per_ep(r), r["success_rate"]) for r in k48_96_probe["runs"]]

    n_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    n_target = adaptive.get("runs_target", len(adaptive["runs"]))
    status_tag = "COMPLETE" if n_done >= n_target else f"PARTIAL {n_done}/{n_target}"

    adaptive_pts = [(bits_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    # Frontier spans both the full K48-192 adaptive sweep and the K48-96
    # restricted probe -- the probe's cheap-end schedules extend it (see
    # README), so they belong in the same "adaptive scheduling" frontier.
    frontier_pts = adaptive_pts + (k48_96_probe_pts or [])
    frontier = pareto_frontier(frontier_pts)

    k192_ckpt = "checkpoints_mwm/mwm_paper10_ogb_cube_k192_release20260728"
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

    # zorder is deliberately layered so blue (adaptive) always draws on top of
    # magenta/green (sepopt fixed K=96/K=192), which draw on top of
    # black/orange (baselines) -- otherwise overlapping points hide the
    # adaptive-schedule result.
    def draw_series(target_ax, marker_scale=1.0):
        target_ax.scatter(*zip(*fixedsub_pts), s=14 * marker_scale, color=ORANGE, alpha=0.35,
                           marker="o", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*fixed192_pts), s=28 * marker_scale, color=BLACK, alpha=0.55,
                           marker="s", zorder=3, linewidths=0)
        if sepopt_k96_pts:
            target_ax.scatter(*zip(*sepopt_k96_pts), s=40 * marker_scale, color=MAGENTA,
                               marker="D", zorder=4, linewidths=0)
        if sepopt_k192_pts:
            target_ax.scatter(*zip(*sepopt_k192_pts), s=40 * marker_scale, color=AQUA,
                               marker="^", zorder=4, linewidths=0)
        target_ax.scatter(*zip(*adaptive_pts), s=14 * marker_scale, color=BLUE, alpha=0.25,
                           zorder=5, linewidths=0)
        if k48_96_probe_pts:
            target_ax.scatter(*zip(*k48_96_probe_pts), s=48 * marker_scale, color=BLUE,
                               marker="P", edgecolors="white", linewidths=0.6 * marker_scale,
                               zorder=6, alpha=0.9)
        target_ax.plot(*zip(*frontier), color=BLUE, lw=2, zorder=7)
        target_ax.scatter(*zip(*frontier), s=45 * marker_scale, color=BLUE, zorder=8, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    # Re-add labelled (invisible) legend handles on the main axes.
    ax.scatter([], [], s=14, color=ORANGE, alpha=0.35, marker="o", linewidths=0,
               label="Fixed K<192 (96/120/144/168, individually-trained)")
    ax.scatter([], [], s=28, color=BLACK, alpha=0.55, marker="s", linewidths=0,
               label="Fixed K=192 (baseline, individually-trained)")
    if sepopt_k96_pts:
        ax.scatter([], [], s=40, color=MAGENTA, marker="D", linewidths=0,
                   label="Fixed K=96 (sepopt checkpoint, no scheduling)")
    if sepopt_k192_pts:
        ax.scatter([], [], s=40, color=AQUA, marker="^", linewidths=0,
                   label="Fixed K=192 (sepopt checkpoint, no scheduling)")
    if k48_96_probe_pts:
        ax.scatter([], [], s=48, color=BLUE, marker="P", edgecolors="white", linewidths=0.6,
                   label="K=48-96 restricted adaptive schedules (probe, 12/12)")
    ax.scatter([], [], s=45, color=BLUE, linewidths=0,
               label=f"Winning adaptive schedules (Pareto frontier, {status_tag})")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    title = (
        "OGB-Cube  goal_offset=25  100 episodes  horizon 2\n"
        "adaptive fidelity scheduling vs. fixed-K -- sepopt K48-192 checkpoint"
    )
    if status_tag != "COMPLETE":
        title += f"\n(adaptive sweep {status_tag} cells complete)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Inset: zoom on the 0-30M bits/episode region, where most points sit, in
    # the upper-right corner (kept clear of the lower-right legend).
    axins = ax.inset_axes([0.60, 0.62, 0.38, 0.34])
    axins.set_facecolor("#fcfcfb")
    draw_series(axins, marker_scale=1.6)
    axins.set_xlim(0, 30)
    axins.set_ylim(40, 82)
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
