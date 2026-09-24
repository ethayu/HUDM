"""Regenerate plot_flops.png: TwoRoom goal50 (horizon 4, 100 episodes) on REAL
audited dynamics FLOPs (dynamics_flops_total).

Only the Slurm array 8634465 series have native FLOPs (run
build_k96to192_summary.py first):

- Joint K96-192 checkpoint, adaptive schedules (runs 02-26) -- purple,
  data/k96to192_adaptive_summary.json.
- Individually-trained fixed-K (runs 27-31, K=192/168/144/120/96) -- orange,
  data/k96to192_singlek_summary.json.

The sepopt series and the older fixed-K baselines in this folder all have
dynamics_flops_total=0 (never audited), so they are excluded here.

Usage: python generate_plot_flops.py  (writes plot_flops.png)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

PURPLE = "#8e5fd1"
ORANGE = "#e07b1a"
GRID = "#d9d9d6"


def gflops_per_ep(run: dict) -> float:
    return run.get("dynamics_flops_total", 0) / (run.get("episodes") or 100) / 1e9


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left frontier: maximize success rate, minimize cost."""
    frontier, best_sr = [], -1.0
    for cost, sr in sorted(points, key=lambda p: (p[0], -p[1])):
        if sr > best_sr:
            frontier.append((cost, sr))
            best_sr = sr
    return frontier


def load(name: str) -> dict:
    return json.load(open(HERE / "data" / name))


def main() -> None:
    k96 = load("k96to192_adaptive_summary.json")
    singlek = load("k96to192_singlek_summary.json")
    singlek_ks = Counter(r["base_name"].split("_")[2] for r in singlek["runs"])
    singlek_ks_str = ", ".join(f"{k}:{n}" for k, n in sorted(singlek_ks.items()))

    series = [
        ([(gflops_per_ep(r), r["success_rate"]) for r in singlek["runs"]], ORANGE, "^",
         f"Individually-trained fixed-K ({singlek['runs_completed']}/{singlek['runs_target']}; {singlek_ks_str})"),
        ([(gflops_per_ep(r), r["success_rate"]) for r in k96["runs"]], PURPLE, "D",
         f"Joint K96-192 ckpt, adaptive ({k96['runs_completed']}/{k96['runs_target']})"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    for z, (pts, color, marker, label) in enumerate(series):
        frontier = pareto_frontier(pts)
        ax.scatter(*zip(*pts), s=14, color=color, alpha=0.25, marker=marker, zorder=2 + z, linewidths=0)
        ax.plot(*zip(*frontier), color=color, lw=2, zorder=10 + z)
        ax.scatter(*zip(*frontier), s=42, color=color, marker=marker, zorder=20 + z, linewidths=0,
                   label=f"{label} -- frontier, max {max(p[1] for p in pts):.0f}%")

    ax.set_xlabel("Audited dynamics GFLOPs per episode", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_title(
        "TwoRoom  goal_offset=50  100 episodes  horizon 4\n"
        "adaptive scheduling (joint K96-192 checkpoint) vs. individually-trained fixed-K\n"
        "(real audited FLOPs; sweep PARTIAL -- counts in legend)",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()

    out = HERE / "plot_flops.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
