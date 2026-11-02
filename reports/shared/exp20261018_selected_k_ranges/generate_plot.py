"""Combined 2x4 figure of the selected K range per env / goal on release
hudm-mwm-exponential-k1to192-20261018 (100 episodes each).

Rows: Delta=25, Delta=50. Columns: OGB Cube, PushT, Reacher, TwoRoom. Each panel
reads the data/ summaries of its <folder>/ in this directory (built by that
folder's build_summary.py) and draws the same three series as the per-folder
plots:

- MWM (scheduled): the adaptive schedules on the dense checkpoint -- blue.
- MWM (fixed): the dense checkpoint held at one level in the range -- teal.
- Single-d: the release's singleton checkpoints in the range -- orange.

PushT uses the original K=64..192 all-configs sweep, which still carries all
19 schedules; the 6 duplicates (05, 11, 12, 13, 14, 17 -- same level decisions
as 02, 06, 07, 08, 02, see scripts/research/gen_exp20261018_k4to16_configs.py)
are dropped so every panel shows the same 13 schedules.

Usage: python generate_plot.py  (writes plot.png and plot_flops.png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SHARED = HERE

# (row, col) -> (label, folder, K range)
PANELS = {
    (0, 0): ("OGB Cube", "ogb_cube_goal25_exp20261018_k8to16_schedules", "8–16"),
    (1, 0): ("OGB Cube", "ogb_cube_goal50_exp20261018_k16to32_schedules", "16–32"),
    (0, 1): ("PushT", "pusht_goal25_exp20261018_k64to192_all_configs", "64–192"),
    (1, 1): ("PushT", "pusht_goal50_exp20261018_k64to192_all_configs", "64–192"),
    (0, 2): ("Reacher", "reacher_goal25_exp20261018_k8to32_schedules", "8–32"),
    (1, 2): ("Reacher", "reacher_goal50_exp20261018_k16to32_schedules", "16–32"),
    (0, 3): ("TwoRoom", "tworoom_goal25_exp20261018_k4to16_schedules", "4–16"),
    (1, 3): ("TwoRoom", "tworoom_goal50_exp20261018_k4to16_schedules", "4–16"),
}
DUPLICATE_SCHEDULES = ("05_", "11_", "12_", "13_", "14_", "17_")

ORANGE = "#e07b1a"
TEAL = "#1f9e8e"
BLUE = "#2a78d6"
GRID = "#d9d9d6"
SERIES = (
    ("dense_fixed_levels_summary.json", TEAL, "s", "MWM (fixed)"),
    ("singlek_summary.json", ORANGE, "^", "Single-$d$"),
    ("adaptive_summary.json", BLUE, "o", "MWM (scheduled)"),
)


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


def load(folder: str, file_name: str) -> list[dict]:
    runs = json.load(open(SHARED / folder / "data" / file_name))["runs"]
    if file_name == "adaptive_summary.json":
        runs = [r for r in runs if not r["base_name"].startswith(DUPLICATE_SCHEDULES)]
    return runs


def plot(cost_fn, xlabel: str, out_name: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for (row, col), (label, folder, k_range) in PANELS.items():
        ax = axes[row, col]
        ax.set_facecolor("#fcfcfb")
        for z, (file_name, color, marker, _) in enumerate(SERIES):
            pts = [(cost_fn(r), r["success_rate"]) for r in load(folder, file_name)]
            frontier = pareto_frontier(pts)
            ax.scatter(*zip(*pts), s=10, color=color, alpha=0.25, marker=marker, zorder=2 + z, linewidths=0)
            ax.plot(*zip(*frontier), color=color, lw=2, zorder=10 + z)
            ax.scatter(*zip(*frontier), s=30, color=color, marker=marker, zorder=20 + z, linewidths=0)
        goal = 25 if row == 0 else 50
        ax.set_title(rf"{label}  $\Delta$={goal}  (K={k_range})", fontsize=15, fontweight="bold")
        ax.set_ylim(0, 102)
        ax.set_xscale("log")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(axis="both", labelsize=12)
        if col == 0:
            ax.set_ylabel("Success rate (%)", fontsize=14)
        if row == 1:
            ax.set_xlabel(xlabel, fontsize=14)

    handles = [Line2D([], [], color=c, marker=m, lw=2, markersize=8, label=f"{lbl} (Pareto frontier)")
               for _, c, m, lbl in reversed(SERIES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = HERE / out_name
    fig.savefig(out)
    print(f"saved {out}")


def main() -> None:
    plot(bits_per_ep, "Bits per episode (×10$^6$)", "plot.png")
    plot(gflops_per_ep, "Audited dynamics GFLOPs per episode", "plot_flops.png")


if __name__ == "__main__":
    main()
