"""Regenerate plot_new.png (bits axis) and plot_flops.png (real audited FLOPs)
from the data/ summaries written by build_new_results_summary.py.

Series (100 episodes):
- Joint K96-192 checkpoint, adaptive schedules (runs 02-26) -- purple.
- Individually-trained fixed-K (runs 27-31) -- orange.
- Dense-sliced fixed-level: the sepopt K48-192 checkpoint held at one level
  (no scheduling) -- teal.
  Both fixed-K series are plotted at K=120-192 only (FIXED_K_MIN); the K48/
  K72/K96 cells are still in data/k96to192_singlek_summary.json and
  data/dense_checkpoint_individual_levels_summary.json.
- Bits plot only: sepopt K48-192 checkpoint, adaptive schedules -- blue, from
  data/sepopt_k48to192_adaptive_summary.json. It has no audited FLOPs
  (dynamics_flops_total=0), so it is left off the FLOPs plot.

Usage: python generate_plot_new.py  (writes plot_new.png and plot_flops.png)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIXED_K_MIN = 120  # drop fixed-K cells below this K from both fixed-K series
TITLE = r"TwoRoom  $\Delta$=50"

PURPLE = "#8e5fd1"
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


def fixed_k(run: dict) -> int:
    # Fixed-K cells use a single K throughout; read it from the recorded
    # usage rather than base_name ("27_all_fixed_finest" carries no K).
    counts = run["schedule_k_counts"]
    if isinstance(counts, str):
        counts = json.loads(counts)
    (k,) = counts
    return int(k)


def load(name: str) -> dict:
    return json.load(open(HERE / "data" / name))


def count(summary: dict, runs: list[dict] | None = None) -> str:
    if runs is not None:
        return f"{len(runs)} cells"
    return f"{summary['runs_completed']}/{summary['runs_target']}"


def plot(series: list[tuple], cost_fn, xlabel: str, subtitle: str, out_name: str) -> None:
    # plot_flops.png only: zoom the y-axis to the actual data range. The
    # bits-axis plot_new.png output (out_name != "plot_flops.png") is left
    # exactly as before -- same hardcoded ylim.
    is_flops = out_name == "plot_flops.png"
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    all_success_rates: list[float] = []
    for z, (runs, color, marker, label) in enumerate(series):
        pts = [(cost_fn(r), r["success_rate"]) for r in runs]
        all_success_rates.extend(sr for _, sr in pts)
        frontier = pareto_frontier(pts)
        ax.scatter(*zip(*pts), s=14, color=color, alpha=0.25, marker=marker, zorder=2 + z, linewidths=0)
        ax.plot(*zip(*frontier), color=color, lw=2, zorder=10 + z)
        ax.scatter(*zip(*frontier), s=42, color=color, marker=marker, zorder=20 + z, linewidths=0,
                   label=label)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Success rate (%)", fontsize=18)
    ax.set_title(TITLE, fontsize=20, fontweight="bold")
    if is_flops and all_success_rates:
        y_min = max(0, min(all_success_rates) - 5)
        y_max = min(102, max(all_success_rates) + 5)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0, 102)
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95)
    fig.tight_layout()

    out = HERE / out_name
    fig.savefig(out)
    print(f"saved {out}")


def main() -> None:
    k96 = load("k96to192_adaptive_summary.json")
    singlek = load("k96to192_singlek_summary.json")
    dense = load("dense_checkpoint_individual_levels_summary.json")
    sepopt = load("sepopt_k48to192_adaptive_summary.json")

    dense_runs = [r for r in dense["runs"] if fixed_k(r) >= FIXED_K_MIN]
    singlek_runs = [r for r in singlek["runs"] if fixed_k(r) >= FIXED_K_MIN]

    flops_series = [
        (dense_runs, TEAL, "s", "MWM (fixed)"),
        (singlek_runs, ORANGE, "^", "Single-$d$"),
        (k96["runs"], PURPLE, "D", "MWM (scheduled) (K96-192)"),
    ]
    sepopt_series = (sepopt["runs"], BLUE, "o", "MWM (scheduled) (K48-192)")

    plot([flops_series[0], flops_series[1], sepopt_series, flops_series[2]], bits_per_ep,
         "Bits per episode (×10$^6$)",
         "adaptive scheduling vs. individually-trained and dense-sliced fixed-K",
         "plot_new.png")
    plot(flops_series, gflops_per_ep, "Audited dynamics GFLOPs per episode",
         "adaptive scheduling vs. fixed-K (real audited FLOPs)",
         "plot_flops.png")


if __name__ == "__main__":
    main()
