"""Regenerate plot.png from the data/ in this folder.

Same structure/plotting conventions as the tworoom_goal50_sepopt_k48to192_all_configs
and reacher_goal50_sepopt_k48to192_all_configs figures, for PushT goal_offset=50
(horizon=5, budget=100) on the sepopt K48-192 checkpoint.

IMPORTANT CAVEATS -- read before using this for the paper:

PARTIAL DATA: the full-range (K=48-192) adaptive-schedule sweep was still
running when this was generated (see the "note"/"runs_completed"/
"runs_target" fields in data/sepopt_k48to192_adaptive_summary.json). Only
schedules 01-21 (of 26 adaptive schedules; 27-31 are the individually-
trained fixed-K baselines, not yet started) had landed cells at generation
time. The Pareto frontier and any "best result" numbers here will shift,
likely upward, as more schedules complete -- especially schedules in the
06/07/20/23 family, which were the strongest performers across every other
environment in this project. Regenerate once the sweep finishes.

NO FIXED-K BASELINES YET: schedules 27-31 (individually-trained fixed-K
checkpoints: K=192/168/144/120/96, run fresh in this same sweep at horizon=5
on the current pinned n100 manifest -- NOT reused/manifest-drifted, unlike
the old fixed_k_baselines_summary.json below) had not started at generation
time, so this plot has no fixed-K comparison series at all yet.

data/fixed_k_baselines_summary.json (125 cells) is NOT plotted and should
NOT be used for comparison: it was extracted from the OLD pre-checkpoint-
swap sweep, which used (a) the old individually-trained-per-K checkpoints
under the OLD pre-swap naming, (b) horizon=4 (not 5), and (c) the unsuffixed
non-pinned manifest -- a three-way mismatch against the current sepopt
adaptive frontier, not just a manifest drift like other shared folders'
reused baselines. Kept only for reference; superseded once schedules 27-31
land in this same sweep.

K48-192 full-range horizon=5 was itself chosen after a separate K120-192
restricted probe at horizon=5 (data/sepopt_k120_192_probe_summary.json, 12/12
complete, same checkpoint/manifest/horizon) showed a dramatic improvement
over the same K-range probe at horizon=4 (41% vs 34% best success) -- that
finding motivated relaunching the full K-range at horizon=5 rather than
horizon=4. The K120-192 probe points are merged into the plotted Pareto
frontier here (same treatment as the K96-192 probes in the TwoRoom/Reacher
shared folders) since they were found to extend it.

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
    k120_192_probe = json.load(open(HERE / "data" / "sepopt_k120_192_probe_summary.json"))

    n_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    n_target = adaptive.get("runs_target", len(adaptive["runs"]))

    adaptive_pts = [(bits_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    k120_192_probe_pts = [(bits_per_ep(r), r["success_rate"]) for r in k120_192_probe["runs"]]
    frontier = pareto_frontier(adaptive_pts + k120_192_probe_pts)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    ax.scatter(*zip(*adaptive_pts), s=14, color=BLUE, alpha=0.25, zorder=5, linewidths=0)
    ax.scatter(*zip(*k120_192_probe_pts), s=48, color=BLUE, marker="P",
               edgecolors="white", linewidths=0.6, zorder=6, alpha=0.9)
    ax.plot(*zip(*frontier), color=BLUE, lw=2, zorder=7)
    ax.scatter(*zip(*frontier), s=45, color=BLUE, zorder=8, linewidths=0)

    ax.scatter([], [], s=48, color=BLUE, marker="P", edgecolors="white", linewidths=0.6,
               label="MWM (scheduled) (K120-192 probe)")
    ax.scatter([], [], s=45, color=BLUE, linewidths=0,
               label="MWM (scheduled) (Pareto frontier)")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=18)
    title = r"PushT  $\Delta$=50"
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95)

    fig.tight_layout()

    out = HERE / "plot.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
