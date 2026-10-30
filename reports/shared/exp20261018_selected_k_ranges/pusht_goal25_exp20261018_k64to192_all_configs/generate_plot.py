"""Regenerate plot.png (bits axis) and plot_flops.png (audited dynamics FLOPs)
from the data/ summaries written by build_summary.py.

Same plot style as the earlier sepopt K48/K120-192 pusht goal25 folders,
on release hudm-mwm-exponential-k1to192-20261018 restricted to K=64..192
(100 episodes):

- MWM (scheduled): the 19 adaptive schedules on the new dense checkpoint -- blue.
- MWM (fixed): the same dense checkpoint held at one level K=64..192 -- teal.
- Single-d: the release's singleton checkpoints K=64..192 -- orange.

Every cell carries native dynamics_flops_total, so all three series appear on
both plots.

Usage: python generate_plot.py  (writes plot.png and plot_flops.png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TITLE = r"PushT  $\Delta$=25"

ORANGE = "#e07b1a"
TEAL = "#1f9e8e"
BLUE = "#2a78d6"
GRID = "#d9d9d6"


def bits_per_ep(run: dict) -> float:
    return run.get("bits_used_total", 0) / (run.get("episodes") or 100) / 1e6


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


def plot(series: list[tuple], cost_fn, xlabel: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    for z, (runs, color, marker, label) in enumerate(series):
        pts = [(cost_fn(r), r["success_rate"]) for r in runs]
        frontier = pareto_frontier(pts)
        # MWM series get an "(all cells)" / "(Pareto frontier)" legend split;
        # Single-$d$ keeps its single bare label.
        is_mwm = label.startswith("MWM")
        ax.scatter(*zip(*pts), s=14, color=color, alpha=0.25, marker=marker, zorder=2 + z, linewidths=0,
                   label=f"{label} (all cells)" if is_mwm else None)
        ax.plot(*zip(*frontier), color=color, lw=2, zorder=10 + z)
        ax.scatter(*zip(*frontier), s=42, color=color, marker=marker, zorder=20 + z, linewidths=0,
                   label=f"{label} (Pareto frontier)" if is_mwm else label)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Success rate (%)", fontsize=18)
    ax.set_title(TITLE, fontsize=20, fontweight="bold")
    ax.set_ylim(0, 102)
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95)
    fig.tight_layout()

    out = HERE / out_name
    fig.savefig(out)
    print(f"saved {out}")


def main() -> None:
    series = [
        (load("dense_fixed_levels_summary.json")["runs"], TEAL, "s", "MWM (fixed)"),
        (load("singlek_summary.json")["runs"], ORANGE, "^", "Single-$d$"),
        (load("adaptive_summary.json")["runs"], BLUE, "o", "MWM (scheduled)"),
    ]
    plot(series, bits_per_ep, "Bits per episode (×10$^6$)", "plot.png")
    plot(series, gflops_per_ep, "Audited dynamics GFLOPs per episode", "plot_flops.png")


if __name__ == "__main__":
    main()
