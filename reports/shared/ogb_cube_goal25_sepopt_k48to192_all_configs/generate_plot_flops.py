"""Regenerate plot_flops.png -- same structure as generate_plot.py, but using
REAL audited dynamics FLOPs (dynamics_flops_total) as the x-axis instead of
the bits/latent-work proxy (bits_used_total).

dynamics_flops_total in data/*.json was reconstructed post-hoc: the FLOP
audit (flop_accounting: dynamics_audit) silently recorded 0 for every run in
this repo due to a torch.inference_mode()/FlopCounterMode interaction bug
(see mwm/diagnostics/flops.py, fixed after these runs completed). Since
FlopCounterMode counts are a pure function of tensor shapes (not data
values) and scale exactly linearly in batch*samples (verified empirically),
the true dynamics_flops_total was recovered from each eval.json's saved
per-CEM-iteration trace combined with a small shape-only calibration pass
against the real checkpoints -- no re-running of the actual benchmark was
needed. All 556 cells across adaptive (409/418, still finishing) + fixed-K
baselines (125) + Fixed K=96 sepopt grid (22) reconstructed with zero
shape-matching mismatches (see "dynamics_flops_reconstructed_posthoc" on
each run).

NOTE: the K=48-96 restricted adaptive-schedule probe
(data/sepopt_k48_96_probe_summary.json) is intentionally NOT plotted here
(excluded on request) and was not reconstructed.

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

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
MAGENTA = "#e87ba4"
BLACK = "#111111"
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
    baselines = json.load(open(HERE / "data" / "fixed_k_baselines_summary.json"))
    sepopt_k96 = json.load(open(HERE / "data" / "sepopt_k96_fixed_cem_grid_summary.json"))

    n_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    n_target = adaptive.get("runs_target", len(adaptive["runs"]))
    status_tag = "COMPLETE" if n_done >= n_target else f"PARTIAL {n_done}/{n_target}"

    adaptive_pts = [(gflops_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    frontier = pareto_frontier(adaptive_pts)

    k192_ckpt = "checkpoints_mwm/mwm_paper10_ogb_cube_k192_release20260728"
    fixed192_pts = [
        (gflops_per_ep(r), r["success_rate"])
        for r in baselines["runs"]
        if r.get("checkpoint_run_dir") == k192_ckpt
    ]
    fixedsub_pts = [
        (gflops_per_ep(r), r["success_rate"])
        for r in baselines["runs"]
        if r.get("checkpoint_run_dir") != k192_ckpt
    ]
    sepopt_k96_pts = [(gflops_per_ep(r), r["success_rate"]) for r in sepopt_k96["runs"]]

    def draw_series(target_ax, marker_scale=1.0):
        target_ax.scatter(*zip(*fixedsub_pts), s=14 * marker_scale, color=ORANGE, alpha=0.35,
                           marker="o", zorder=2, linewidths=0)
        target_ax.scatter(*zip(*fixed192_pts), s=28 * marker_scale, color=BLACK, alpha=0.55,
                           marker="s", zorder=3, linewidths=0)
        target_ax.scatter(*zip(*sepopt_k96_pts), s=40 * marker_scale, color=MAGENTA,
                           marker="D", zorder=4, linewidths=0)
        target_ax.scatter(*zip(*adaptive_pts), s=14 * marker_scale, color=BLUE, alpha=0.25,
                           zorder=5, linewidths=0)
        target_ax.plot(*zip(*frontier), color=BLUE, lw=2, zorder=6)
        target_ax.scatter(*zip(*frontier), s=45 * marker_scale, color=BLUE, zorder=7, linewidths=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    draw_series(ax)
    ax.scatter([], [], s=14, color=ORANGE, alpha=0.35, marker="o", linewidths=0,
               label="Fixed K<192 (96/120/144/168, individually-trained)")
    ax.scatter([], [], s=28, color=BLACK, alpha=0.55, marker="s", linewidths=0,
               label="Fixed K=192 (baseline, individually-trained)")
    ax.scatter([], [], s=40, color=MAGENTA, marker="D", linewidths=0,
               label="Fixed K=96 (sepopt checkpoint, no scheduling)")
    ax.scatter([], [], s=45, color=BLUE, linewidths=0,
               label=f"Winning adaptive schedules (Pareto frontier, {status_tag})")

    ax.set_xlabel("Audited dynamics GFLOPs per episode", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    title = (
        "OGB-Cube  goal_offset=25  100 episodes  horizon 2\n"
        "adaptive fidelity scheduling vs. fixed-K -- sepopt K48-192 checkpoint\n"
        "(real audited FLOPs)"
    )
    if status_tag != "COMPLETE":
        title += f"\n(adaptive sweep {status_tag} cells complete)"
    ax.set_title(title, fontsize=13, fontweight="bold")
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
