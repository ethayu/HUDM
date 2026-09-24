"""Reconstruct dynamics_flops_total for the three scheduled sweeps whose FLOP
audit recorded 0 (runs predating the inference_mode/FlopCounterMode fix):

- TwoRoom goal25 (horizon 2) and goal50 (horizon 4, partial 93/418),
  tworoom_goal*_sepopt_k48to192_all_configs/data/sepopt_k48to192_adaptive_summary.json
- Reacher goal25 (horizon 2),
  reacher_goal25_sepopt_k48to192_all_configs/data/sepopt_k48to192_adaptive_summary.json

FlopCounterMode counts depend only on tensor shapes and are linear in the
number of candidates, so each run's total is
    sum_L  count(CEM calls at base level L) * pop_size * f(rollout_levels(L)),
where rollout_levels(L) comes from the run's own FidelityScheduler config and
f is measured once per level sequence on the real checkpoint (batch=1,
samples=1). A per-checkpoint scale factor (median audited/reconstructed over
the same checkpoint's natively audited fixed-level sweep at the same horizon)
absorbs the small (<0.5%) residual. Validated on the natively audited Reacher
goal50 scheduled sweep (475 runs): max |rel err| 0.5% before scaling.

Usage: python reconstruct_flops.py   (writes data/*_flops_reconstructed.json)
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import yaml

from flop_calib import Calib

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
FIXED = {"enabled": True, "mpc": {"mode": "fixed", "level": "finest"},
         "cem": {"mode": "fixed", "level": "base"}, "rollout": {"mode": "fixed", "level": "base"}}

JOBS = [
    ("tworoom_goal25", "tworoom_goal25_sepopt_k48to192_all_configs"),
    ("tworoom_goal50", "tworoom_goal50_sepopt_k48to192_all_configs"),
    ("reacher_goal25", "reacher_goal25_sepopt_k48to192_all_configs"),
]


def main() -> None:
    for tag, folder in JOBS:
        src = SHARED / folder
        cfg = yaml.safe_load(open(src / "config_sepopt_k48to192.yaml"))
        horizon = int(cfg["run_defaults"]["planner"]["horizon"])
        sched = {r["name"]: r["planner"]["scheduler"] for r in cfg["runs"]}
        calib = Calib(cfg["runs"][0]["checkpoint"], horizon=horizon)

        dense = json.load(open(src / "data" / "dense_checkpoint_individual_levels_summary.json"))["runs"]
        scale = median(r["dynamics_flops_total"] / calib.run_flops(r, FIXED) for r in dense)

        summary = json.load(open(src / "data" / "sepopt_k48to192_adaptive_summary.json"))
        for run in summary["runs"]:
            run["dynamics_flops_total"] = int(round(scale * calib.run_flops(run, sched[run["base_name"]])))
            run["dynamics_flops_reconstructed_posthoc"] = True
        summary["flops_reconstruction"] = {"script": "reconstruct_flops.py", "horizon": horizon,
                                           "scale_from_fixed_level_audit": scale}
        out = HERE / "data" / f"{tag}_scheduled_flops_reconstructed.json"
        out.write_text(json.dumps(summary, indent=1))
        print(f"{tag}: {len(summary['runs'])} runs, H={horizon}, scale={scale:.5f} -> {out.name}")


if __name__ == "__main__":
    main()
