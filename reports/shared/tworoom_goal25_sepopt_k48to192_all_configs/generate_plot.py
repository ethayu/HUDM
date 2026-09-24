"""Regenerate plot.png from the data/ and config in this folder.

Reproduces the same plot style as the original release20260728 TwoRoom
goal25 all-configs figure (rollouts/tworoom_release20260728_goal25_100ep_all_configs.png
in the HUDM-mwm-ethan repo), but with the "winning adaptive schedules" series
computed from the sepopt K48-192 checkpoint
(checkpoints_dense_k12to96_sepopt_20260916_tworoom_k48to192/checkpoints_mwm/
mwm_paper10_tworoom_k48_72_96_120_144_168_192_sepopt_20260917, K=[48,72,96,
120,144,168,192], epoch 9) instead of the original paper10 dense checkpoint
(K=[96,120,144,168,192]).

The two individually-trained fixed-K baseline series (Fixed K=192, Fixed
K<192) are unchanged from the original sweep -- those runs use
individually-trained single-K checkpoints that were never touched by the
checkpoint swap, so their results were reused rather than recomputed (see
data/fixed_k_baselines_summary.json and the note field in that file).

A fourth series, Fixed K=192 (sepopt checkpoint, no scheduling), is a NEW
eval (not reused): the *sepopt* K48-192 checkpoint -- same checkpoint the
adaptive frontier uses -- with mpc/cem/rollout all fixed at K=192 (its own
finest/native level) for the whole plan, no fidelity transitions at all,
across the same pop_size x n_iter grid. This answers "does simply fixing the
new checkpoint at its finest level already dominate, or does adaptive
scheduling actually buy something on top of it?" -- see
data/sepopt_k192_fixed_cem_grid_summary.json and
config_sepopt_k192_fixed_cem_grid.yaml.

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
EPISODES = 100  # all data files below use 100-episode evals

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLACK = "#111111"
GRID = "#d9d9d6"


def bits_per_ep(run: dict) -> float:
    return run.get("bits_used_total", 0) / EPISODES / 1e6


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
               linewidths=0, label="Fixed K<192 (96/120/144/168, Baseline)")

    ax.scatter(*zip(*fixed192_pts), s=28, color=BLACK, alpha=0.55, marker="s", zorder=3,
               linewidths=0, label="Fixed K=192 (Baseline)")

    ax.scatter(*zip(*sepopt_k192_pts), s=40, color=AQUA, marker="^", zorder=4, linewidths=0,
               label="MWM (fixed) (K=192)")

    ax.scatter(*zip(*adaptive_pts), s=14, color=BLUE, alpha=0.25, zorder=5, linewidths=0)
    ax.plot(*zip(*frontier), color=BLUE, lw=2, zorder=6)
    ax.scatter(*zip(*frontier), s=45, color=BLUE, zorder=7, linewidths=0,
               label="MWM (scheduled) (Pareto frontier, changing fidelity)")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_title(
        r"TwoRoom  $\Delta$=25",
        fontsize=16, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()

    out = HERE / "plot.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
