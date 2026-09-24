"""Numbers quoted in the paper's compute-budget section, computed from exactly
the series drawn in each folder's plot_flops.png (same files and filters as
the per-folder generate_plot_flops.py scripts; TwoRoom's plot has no script
in its folder, so its filters are taken from the legend: K120-192 for both
fixed-K series, joint K96-192 checkpoint for the adaptive series).

Usage: python stats_from_plot_flops.py  (prints; writes plot_flops_stats.json)
"""
from __future__ import annotations

import json
from pathlib import Path

from generate_paper_plot import cost_to_reach, pareto_frontier, success_at

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
ADAPTIVE_PREFIXES = {"02", "05", "06", "07", "08", "11", "12", "13", "14", "17",
                     "18", "19", "20", "21", "22", "23", "24", "25", "26"}


def runs(folder: str, name: str) -> list[dict]:
    return json.load(open(SHARED / folder / "data" / name))["runs"]


def level_ge(k: int):
    return lambda r: int(r["base_name"].split("_k", 1)[1].split("_", 1)[0]) >= k


def panels() -> dict[str, dict[str, list[dict]]]:
    t25, t50 = "tworoom_goal25_sepopt_k48to192_all_configs", "tworoom_goal50_sepopt_k48to192_all_configs"
    p25, p50 = "pusht_goal25_sepopt_k48to192_all_configs", "pusht_goal50_plan100_execute20_sepopt_k120to192_adaptive"
    o25, o50 = "ogb_cube_goal25_sepopt_k48to192_all_configs", "ogb_cube_goal50_plan100_execute20_sepopt_k48to192_adaptive"
    r25, r50 = "reacher_goal25_sepopt_k48to192_all_configs", "reacher_goal50_plan100_execute20_sepopt_k48to192_adaptive"
    adaptive_slice = lambda r: r["base_name"].split("_", 1)[0] in ADAPTIVE_PREFIXES
    out = {}
    for tag, f in (("tworoom_goal25", t25), ("tworoom_goal50", t50)):
        out[tag] = {
            "baseline": [r for r in runs(f, "k96to192_singlek_summary.json") if "k96" not in r["base_name"]],
            "fixed": [r for r in runs(f, "dense_checkpoint_individual_levels_summary.json") if level_ge(120)(r)],
            "scheduled": runs(f, "k96to192_adaptive_summary.json"),
        }
    out["pusht_goal25"] = {
        "baseline": [r for n in ("fixed_k192_baseline_summary.json", "fixed_k_lt192_baseline_summary.json")
                     for r in runs(p25, n) if "k96" not in r["base_name"]],
        "fixed": [r for k in (120, 144, 168, 192) for r in runs(p25, f"sepopt_k{k}_fixed_cem_grid_summary.json")],
        "scheduled": runs(p25, "sepopt_k48to192_adaptive_summary.json"),
    }
    out["pusht_goal50"] = {
        "baseline": [r for r in runs(p50, "singlek_summary.json") if not r["base_name"].startswith("31")],
        "fixed": runs(p50, "dense_checkpoint_individual_levels_summary.json"),
        "scheduled": runs(p50, "adaptive_summary.json"),
    }
    out["ogb_goal25"] = {
        "baseline": runs(o25, "individual_fixed_k_baselines_summary.json"),
        "fixed": [r for r in runs(o25, "dense_checkpoint_individual_levels_summary.json") if r["n_iter"] != 5],
        "scheduled": runs(o25, "sepopt_k48to192_adaptive_summary.json"),
    }
    out["ogb_goal50"] = {
        "baseline": runs(o50, "singlek_summary.json"),
        "fixed": [r for r in runs(o50, "dense_checkpoint_individual_levels_summary.json") if r["n_iter"] != 5],
        "scheduled": runs(o50, "adaptive_summary.json"),
    }
    out["reacher_goal25"] = {
        "baseline": runs(r25, "individual_fixed_k_baselines_summary.json"),
        "fixed": [r for r in runs(r25, "dense_checkpoint_individual_levels_summary.json") if level_ge(96)(r)],
        "scheduled": [r for r in runs(r25, "k96to192_slice_summary.json") if adaptive_slice(r)],
    }
    out["reacher_goal50"] = {
        "baseline": runs(r50, "new_sweep_partial_singlek_summary.json"),
        "fixed": [r for r in runs(r50, "dense_checkpoint_individual_levels_summary.json") if level_ge(96)(r)],
        "scheduled": [r for r in runs(r50, "k96to192_slice_summary.json") if adaptive_slice(r)],
    }
    return out


def main() -> None:
    stats = {}
    for tag, series in panels().items():
        fr = {k: pareto_frontier([(r["dynamics_flops_total"] / r["episodes"] / 1e9, float(r["success_rate"]))
                                  for r in v]) for k, v in series.items()}
        s = {"cells": {k: len(v) for k, v in series.items()},
             "frontiers": {k: [(round(c, 2), sr) for c, sr in v] for k, v in fr.items()}}
        for a, b in (("baseline", "fixed"), ("baseline", "scheduled"), ("fixed", "scheduled")):
            target = min(fr[a][-1][1], fr[b][-1][1])
            budget = fr[a][-1][0]
            gain = success_at(fr[b], budget)
            s[f"{b}_vs_{a}"] = {"matched_success": target,
                                 "cost_ratio": cost_to_reach(fr[a], target) / cost_to_reach(fr[b], target),
                                 "success_gain_at_ref_peak_cost": None if gain is None else gain - fr[a][-1][1]}
        stats[tag] = s
        pk = "  ".join(f"{k[0].upper()} {v[-1][1]:.0f}@{v[-1][0]:.1f}" for k, v in fr.items())
        rs = "  ".join(f"{k}: @{v['matched_success']:.0f}% x{v['cost_ratio']:.2f} gain {v['success_gain_at_ref_peak_cost']}"
                       for k, v in s.items() if "_vs_" in k)
        print(f"{tag:15s} {s['cells']}\n   peaks {pk}\n   {rs}")
    (HERE / "plot_flops_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
