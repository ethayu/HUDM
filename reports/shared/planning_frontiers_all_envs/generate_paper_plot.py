"""Paper figure: empirical Pareto frontiers (success vs. dynamics GFLOPs per
episode) for Baseline, MWM (fixed level), and MWM (scheduled); 2x4 panels
(rows: goal offset 25/50, columns: environment). Also prints the per-panel
numbers quoted in the text and writes them to frontier_stats.json.

All inputs are existing summaries under reports/shared (see SOURCES); the
three scheduled sweeps without audited FLOPs use data/*_flops_reconstructed.json
from reconstruct_flops.py. Within each panel every series shares the pinned
n100 manifest and horizon, and only the CEM (pop_size, n_iter) cells common to
all three series are used.

Usage: python generate_paper_plot.py [OUTPUT_PDF]
  (default: planning_frontiers.pdf in this directory)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent

ORANGE = "#eb6834"
BLUE = "#2a78d6"
GREEN = "#1f9e5a"
GRID = "#d9d9d6"

ENVS = [("tworoom", "TwoRoom"), ("pusht", "PushT"), ("ogb", "OGBench-Cube"), ("reacher", "Reacher")]
GOALS = [25, 50]
SERIES = [("baseline", "Baseline", ORANGE, "o"),
          ("fixed", "MWM (fixed level)", BLUE, "s"),
          ("scheduled", "MWM (scheduled)", GREEN, "D")]

T25, T50 = "tworoom_goal25_sepopt_k48to192_all_configs", "tworoom_goal50_sepopt_k48to192_all_configs"
P25, P50 = "pusht_goal25_sepopt_k48to192_all_configs", "pusht_goal50_plan100_execute20_sepopt_k120to192_adaptive"
O25, O50 = "ogb_cube_goal25_sepopt_k48to192_all_configs", "ogb_cube_goal50_plan100_execute20_sepopt_k48to192_adaptive"
R25, R50 = "reacher_goal25_sepopt_k48to192_all_configs", "reacher_goal50_plan100_execute20_sepopt_k48to192_adaptive"
LOCAL = "planning_frontiers_all_envs"

SOURCES = {
    ("tworoom", 25): {
        "baseline": [f"{T25}/data/k96to192_singlek_summary.json"],
        "fixed": [f"{T25}/data/dense_checkpoint_individual_levels_summary.json"],
        "scheduled": [f"{LOCAL}/data/tworoom_goal25_scheduled_flops_reconstructed.json"],
    },
    ("tworoom", 50): {
        "baseline": [f"{T50}/data/k96to192_singlek_summary.json"],
        "fixed": [f"{T50}/data/dense_checkpoint_individual_levels_summary.json"],
        "scheduled": [f"{LOCAL}/data/tworoom_goal50_scheduled_flops_reconstructed.json"],
    },
    ("pusht", 25): {
        "baseline": [f"{P25}/data/fixed_k192_baseline_summary.json",
                     f"{P25}/data/fixed_k_lt192_baseline_summary.json"],
        "fixed": [f"{P25}/data/sepopt_k{k}_fixed_cem_grid_summary.json" for k in (120, 144, 168, 192)],
        "scheduled": [f"{P25}/data/sepopt_k48to192_adaptive_summary.json"],
    },
    ("pusht", 50): {
        "baseline": [f"{P50}/data/singlek_summary.json"],
        "fixed": [f"{P50}/data/dense_checkpoint_individual_levels_summary.json"],
        "scheduled": [f"{P50}/data/adaptive_summary.json"],
    },
    ("ogb", 25): {
        "baseline": [f"{O25}/data/individual_fixed_k_baselines_summary.json"],
        "fixed": [f"{O25}/data/dense_checkpoint_individual_levels_summary.json"],
        "scheduled": [f"{O25}/data/sepopt_k48to192_adaptive_summary.json"],
    },
    ("ogb", 50): {
        "baseline": [f"{O50}/data/singlek_summary.json"],
        "fixed": [f"{O50}/data/dense_checkpoint_individual_levels_summary.json"],
        "scheduled": [f"{O50}/data/adaptive_summary.json"],
    },
    ("reacher", 25): {
        "baseline": [f"{R25}/data/individual_fixed_k_baselines_summary.json"],
        "fixed": [f"{R25}/data/dense_checkpoint_individual_levels_summary.json"],
        "scheduled": [f"{LOCAL}/data/reacher_goal25_scheduled_flops_reconstructed.json"],
    },
    ("reacher", 50): {
        "baseline": [f"{R50}/data/new_sweep_partial_singlek_summary.json"],
        "fixed": [f"{R50}/data/dense_checkpoint_individual_levels_summary.json"],
        "scheduled": [f"{R50}/data/sepopt_k48to192_adaptive_summary.json"],
    },
}


def cem_cell(run: dict) -> tuple[int, int]:
    params = run["sweep_params"]
    if isinstance(params, str):
        params = json.loads(params)
    return int(params["planner.pop_size"]), int(params["planner.n_iter"])


def load_panel(env: str, goal: int) -> dict[str, list[dict]]:
    runs = {}
    for name, files in SOURCES[(env, goal)].items():
        runs[name] = [r for f in files for r in json.load(open(SHARED / f))["runs"]]
        assert all(r["dynamics_flops_total"] > 0 for r in runs[name]), (env, goal, name)
    common = set.intersection(*({cem_cell(r) for r in rs} for rs in runs.values()))
    return {name: [r for r in rs if cem_cell(r) in common] for name, rs in runs.items()}


def points(runs: list[dict]) -> list[tuple[float, float]]:
    return [(r["dynamics_flops_total"] / r["episodes"] / 1e9, float(r["success_rate"])) for r in runs]


def pareto_frontier(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    frontier, best = [], -1.0
    for cost, sr in sorted(pts, key=lambda p: (p[0], -p[1])):
        if sr > best:
            frontier.append((cost, sr))
            best = sr
    return frontier


def cost_to_reach(frontier: list[tuple[float, float]], target: float) -> float | None:
    return next((c for c, s in frontier if s >= target), None)


def success_at(frontier: list[tuple[float, float]], budget: float) -> float | None:
    feasible = [s for c, s in frontier if c <= budget]
    return max(feasible) if feasible else None


def panel_stats(frontiers: dict[str, list[tuple[float, float]]], n_cells: dict[str, int]) -> dict:
    stats = {"cells": n_cells}
    for name, fr in frontiers.items():
        stats[name] = {"max_success": fr[-1][1], "cost_at_max": fr[-1][0], "min_cost": fr[0][0]}
    for ref in ("baseline", "fixed"):
        # Matched success: the best success both frontiers reach.
        target = min(frontiers[ref][-1][1], frontiers["scheduled"][-1][1])
        c_ref, c_sch = cost_to_reach(frontiers[ref], target), cost_to_reach(frontiers["scheduled"], target)
        # Matched compute: the cost at which the reference frontier peaks.
        budget = frontiers[ref][-1][0]
        stats[f"scheduled_vs_{ref}"] = {
            "matched_success_target": target,
            "cost_ratio_ref_over_scheduled": c_ref / c_sch,
            "matched_compute_gflops": budget,
            "success_gain_at_ref_peak_cost": success_at(frontiers["scheduled"], budget) - frontiers[ref][-1][1],
        }
    return stats


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "planning_frontiers.pdf"
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})

    fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.1), sharey=True)
    all_stats = {}
    for row, goal in enumerate(GOALS):
        for col, (env, label) in enumerate(ENVS):
            ax = axes[row, col]
            panel = load_panel(env, goal)
            frontiers = {}
            for name, _, color, marker in SERIES:
                pts = points(panel[name])
                fr = pareto_frontier(pts)
                frontiers[name] = fr
                ax.scatter(*zip(*pts), s=2.5, color=color, alpha=0.18, linewidths=0, zorder=2)
                xs, ys = zip(*fr)
                ax.step(xs, ys, where="post", color=color, lw=1.1, zorder=4)
                ax.plot(xs, ys, ls="none", marker=marker, ms=2.4, color=color, zorder=5)
            all_stats[f"{env}_goal{goal}"] = panel_stats(frontiers, {k: len(v) for k, v in panel.items()})

            ax.set_xscale("log")
            ax.set_ylim(-5, 105)
            ax.grid(True, color=GRID, lw=0.5)
            ax.tick_params(length=2, pad=1.5)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if row == 0:
                ax.set_title(label, fontsize=8)
        axes[row, 0].set_ylabel(f"Success (%)\n" + rf"$\Delta={goal}$")

    handles = [Line2D([], [], color=c, marker=m, ms=3, lw=1.1, label=lab) for _, lab, c, m in SERIES]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.5)
    fig.supxlabel("Dynamics GFLOPs per episode (log scale)", fontsize=7, y=0.01)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94), w_pad=0.5, h_pad=0.6)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01, dpi=200)
    (HERE / "frontier_stats.json").write_text(json.dumps(all_stats, indent=1))
    print(f"saved {out}")
    for key, s in all_stats.items():
        b, f, sc = s["baseline"], s["fixed"], s["scheduled"]
        vb, vf = s["scheduled_vs_baseline"], s["scheduled_vs_fixed"]
        print(f"{key:16s} cells={s['cells']}  max: B {b['max_success']:.0f}@{b['cost_at_max']:.1f}  "
              f"F {f['max_success']:.0f}@{f['cost_at_max']:.1f}  S {sc['max_success']:.0f}@{sc['cost_at_max']:.1f}  | "
              f"costratio@{vb['matched_success_target']:.0f}% B/S={vb['cost_ratio_ref_over_scheduled']:.2f} "
              f"@{vf['matched_success_target']:.0f}% F/S={vf['cost_ratio_ref_over_scheduled']:.2f}  | "
              f"gain@Bpeak {vb['success_gain_at_ref_peak_cost']:+.0f}  gain@Fpeak {vf['success_gain_at_ref_peak_cost']:+.0f}")


if __name__ == "__main__":
    main()
