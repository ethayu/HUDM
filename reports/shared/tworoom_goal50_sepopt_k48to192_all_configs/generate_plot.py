"""Regenerate plot.png from the data/ and config in this folder.

Same structure as the tworoom_goal25_sepopt_k48to192_all_configs figure, for
TwoRoom goal_offset=50 (horizon=4, budget=100) instead of goal25.

IMPORTANT CAVEAT -- read before using this for the paper:

PARTIAL DATA: the sepopt K48-192 adaptive-schedule sweep was still running
when this was generated (see the "note" field in
data/sepopt_k48to192_adaptive_summary.json for the completed/target count).
The Pareto frontier and any "best result" numbers here will likely shift as
more schedules complete. Regenerate once the sweep finishes for the final
numbers.

(An earlier version of this data was trimmed down from a 200-episode run to
match the other three series at episodes=100 -- see git history / the
K48-192 sepopt subset probe earlier in this project. The adaptive sweep was
since relaunched at a native episodes=100 using a purpose-built manifest
(configs/manifest/release20260728_tworoom_goal50_exact_seed42_n100.yaml), so
the current data/sepopt_k48to192_adaptive_summary.json is NOT trimmed -- all
four series are natively episodes=100.)

Checkpoint swap: adaptive-schedule runs (02-26) use the sepopt K48-192
checkpoint (checkpoints_dense_k12to96_sepopt_20260916_tworoom_k48to192/
checkpoints_mwm/mwm_paper10_tworoom_k48_72_96_120_144_168_192_sepopt_20260917,
K=[48,72,96,120,144,168,192], epoch 9). Fixed K=192 / Fixed K<192 use
individually-trained single-K checkpoints (untouched by the swap, reused from
the original sweep). Fixed K=192 (sepopt) uses the SAME sepopt checkpoint as
the adaptive frontier, held fixed at K=192 (its own finest level) for the
whole plan -- a NEW eval, not reused, see
data/sepopt_k192_fixed_cem_grid_summary.json.

(An earlier version of this plot had a "Fixed K=96, original dense
checkpoint" series here instead -- removed since it wasn't a fair comparison
against the sepopt checkpoint's own frontier.)

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
ORANGE = "#eb6834"
AQUA = "#1baf7a"
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
    baselines = json.load(open(HERE / "data" / "fixed_k_baselines_summary.json"))
    sepopt_k192 = json.load(open(HERE / "data" / "sepopt_k192_fixed_cem_grid_summary.json"))

    n_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    n_target = adaptive.get("runs_target", len(adaptive["runs"]))

    adaptive_pts = [(bits_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    frontier = pareto_frontier(adaptive_pts)

    k192_ckpt = "checkpoints_mwm/mwm_paper10_tworoom_k192_release20260728"
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
    sepopt_k192_pts = [(bits_per_ep(r), r["success_rate"]) for r in sepopt_k192["runs"]]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    # zorder is deliberately layered so blue (adaptive) always draws on top of
    # green (sepopt fixed K=192), which draws on top of black/orange
    # (baselines) -- otherwise overlapping points hide the adaptive-schedule
    # result.
    ax.scatter(*zip(*fixedsub_pts), s=14, color=ORANGE, alpha=0.35, marker="o", zorder=2,
               linewidths=0, label="Fixed K<192 (96/120/144/168, individually-trained, n=100)")

    ax.scatter(*zip(*fixed192_pts), s=28, color=BLACK, alpha=0.55, marker="s", zorder=3,
               linewidths=0, label="Fixed K=192 (baseline, individually-trained, n=100)")

    ax.scatter(*zip(*sepopt_k192_pts), s=40, color=AQUA, marker="^", zorder=4, linewidths=0,
               label="Fixed K=192 (sepopt checkpoint, no scheduling, n=100)")

    ax.scatter(*zip(*adaptive_pts), s=14, color=BLUE, alpha=0.25, zorder=5, linewidths=0)
    ax.plot(*zip(*frontier), color=BLUE, lw=2, zorder=6)
    ax.scatter(*zip(*frontier), s=45, color=BLUE, zorder=7, linewidths=0,
               label=f"Winning adaptive schedules (Pareto frontier, PARTIAL {n_done}/{n_target})")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_title(
        "TwoRoom  goal_offset=50  horizon 4\n"
        "adaptive fidelity scheduling vs. fixed-K -- sepopt K48-192 checkpoint\n"
        f"(adaptive sweep PARTIAL: {n_done}/{n_target} cells complete)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()

    out = HERE / "plot.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
