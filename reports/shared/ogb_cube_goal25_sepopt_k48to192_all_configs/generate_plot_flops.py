"""Regenerate plot_flops.png -- same structure as generate_plot.py, but using
REAL audited dynamics FLOPs (dynamics_flops_total) as the x-axis instead of
the bits/latent-work proxy (bits_used_total).

dynamics_flops_total in data/sepopt_k48to192_adaptive_summary.json,
data/fixed_k_baselines_summary.json, and data/sepopt_k96_fixed_cem_grid_summary.json
was reconstructed post-hoc: the FLOP audit (flop_accounting: dynamics_audit)
silently recorded 0 for every run in this repo due to a
torch.inference_mode()/FlopCounterMode interaction bug (see
mwm/diagnostics/flops.py, fixed after these runs completed). Since
FlopCounterMode counts are a pure function of tensor shapes (not data
values) and scale exactly linearly in batch*samples (verified empirically),
the true dynamics_flops_total was recovered from each eval.json's saved
per-CEM-iteration trace combined with a small shape-only calibration pass
against the real checkpoints -- no re-running of the actual benchmark was
needed. All 565 cells across adaptive (418) + fixed-K baselines (125) +
Fixed K=96 sepopt grid (22) reconstructed with zero shape-matching
mismatches (see "dynamics_flops_reconstructed_posthoc" on each run).

data/individual_fixed_k_baselines_summary.json (110 cells) and
data/dense_checkpoint_individual_levels_summary.json (154 cells) are a
separate, newer pair of sweeps (run after the FLOP-audit bug was fixed) --
their dynamics_flops_total is real audited data from the start, not
reconstructed. They replace fixed_k_baselines_summary.json /
sepopt_k96_fixed_cem_grid_summary.json as the plotted fixed-K and
dense-sliced series (see the README).

NOTE: the K=48-96 restricted adaptive-schedule probe
(data/sepopt_k48_96_probe_summary.json) is intentionally NOT plotted here
(excluded on request).

NOTE: dense-sliced fixed-level excludes n_iter=5 cells (119/154 plotted).

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
    adaptive = json.load(open(HERE / "data" / "sepopt_k48to192_adaptive_summary.json"))
    fixedk = json.load(open(HERE / "data" / "individual_fixed_k_baselines_summary.json"))
    dense = json.load(open(HERE / "data" / "dense_checkpoint_individual_levels_summary.json"))

    adaptive_runs = adaptive["runs"]
    fixedk_runs = fixedk["runs"]
    d_total = len(dense["runs"])
    dense_runs = [r for r in dense["runs"] if r.get("n_iter") != 5]
    a_shown = a_total = len(adaptive_runs)
    f_shown = f_total = len(fixedk_runs)
    d_shown = len(dense_runs)

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
               label=f"MWM (scheduled) (Pareto frontier, {a_shown}/{a_total})")
    ax.scatter([], [], s=14, color=PURPLE, alpha=0.25, linewidths=0,
               label="MWM (scheduled) (all cells)")
    ax.scatter([], [], s=45, color=ORANGE, marker="^", linewidths=0,
               label=f"Baseline (Pareto frontier, {f_shown}/{f_total})")
    ax.scatter([], [], s=14, color=ORANGE, alpha=0.25, marker="^", linewidths=0,
               label="Baseline (all cells)")
    ax.scatter([], [], s=40, color=TEAL, marker="s", linewidths=0,
               label=f"MWM (fixed) (Pareto frontier, {d_shown}/{d_total})")
    ax.scatter([], [], s=14, color=TEAL, alpha=0.25, marker="s", linewidths=0,
               label="MWM (fixed) (all cells)")

    ax.set_xlabel("Audited dynamics GFLOPs per episode", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    title = r"OGB-Cube  $\Delta$=25"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylim(0, 102)
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

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
