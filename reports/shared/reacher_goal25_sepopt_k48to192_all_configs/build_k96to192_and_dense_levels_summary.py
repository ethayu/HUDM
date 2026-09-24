"""Build summaries for two more completed reacher goal25 sweeps on the sepopt
K48-192 checkpoint, both now COMPLETE. Mirrors the goal50 counterpart in
../reacher_goal50_plan100_execute20_sepopt_k48to192_adaptive/.

- k96to192_slice_summary.json (528/528): the same 19 adaptive schedules + 5
  individually-trained fixed-K baselines (runs 02-31) as
  sepopt_k48to192_adaptive_summary.json / fixed_k_baselines_summary.json,
  but with every "coarsest" schedule endpoint floored at K=96 (level index
  2) instead of K=48 -- see
  release20260728_dense_reacher_goal25_k96to192slice_sepopt_all_fidelity_schedules.yaml.
- dense_checkpoint_individual_levels_summary.json (154/154): the SAME shared
  sepopt checkpoint, held fixed (no scheduling) at each of its 7 declared
  levels in turn (K=48,72,96,120,144,168,192) -- see
  release20260728_dense_reacher_goal25_dense_checkpoint_individual_levels.yaml.
  Distinct from the individually-trained fixed-K baselines: same one
  checkpoint, just sliced to each level, vs. 5 separately trained
  single-K checkpoints.
- individual_fixed_k_baselines_summary.json (110/110): a freshly re-run
  version of runs 27-31 (individually-trained fixed-K, K=192/168/144/120/96)
  on the goal25 22-combo sweep grid (matching series #1/#3's grid exactly),
  from
  release20260728_dense_reacher_goal25_individual_fixed_k_baselines.yaml.
  Replaces the older data/fixed_k_baselines_summary.json (125 cells, 25
  combos/run, no exclusions -- a different, wider grid) as the plotted
  series #2.

Usage: python build_k96to192_and_dense_levels_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mwm.benchmark.summary import eval_summary_row  # noqa: E402

HERE = Path(__file__).resolve().parent

K96TO192_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_sepopt_k96to192slice_reacher_goal25_all_fidelity_schedules"
)
DENSE_LEVELS_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_reacher_goal25_dense_checkpoint_individual_levels"
)
INDIVIDUAL_FIXED_K_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_reacher_goal25_individual_fixed_k_baselines"
)

K96TO192_TARGET = 528
DENSE_LEVELS_TARGET = 154
INDIVIDUAL_FIXED_K_TARGET = 110


def build_summary(source_dir: Path, target: int, note: str, out_name: str) -> None:
    rows = []
    for eval_path in sorted(source_dir.glob("*/eval.json")):
        payload = json.loads(eval_path.read_text())
        rows.append(eval_summary_row(eval_path.parent.name, eval_path, payload))
    out = {
        "source_output_dir": str(source_dir),
        "note": note,
        "runs_completed": len(rows),
        "runs_target": target,
        "runs": rows,
    }
    (HERE / "data" / out_name).write_text(json.dumps(out, indent=2))
    print(f"{out_name}: {len(rows)}/{target}")


def main() -> None:
    build_summary(
        K96TO192_SOURCE_DIR,
        K96TO192_TARGET,
        (
            "COMPLETE -- runs 02-31 (19 adaptive schedules + 5 individually-trained "
            "fixed-K baselines) on the sepopt K48-192 checkpoint, but every 'coarsest' "
            "schedule endpoint floored at K=96 (level index 2) instead of K=48. From "
            "release20260728_dense_reacher_goal25_k96to192slice_sepopt_all_fidelity_schedules.yaml."
        ),
        "k96to192_slice_summary.json",
    )
    build_summary(
        DENSE_LEVELS_SOURCE_DIR,
        DENSE_LEVELS_TARGET,
        (
            "COMPLETE -- the shared sepopt K48-192 checkpoint held fixed (no "
            "scheduling) at each of its 7 declared levels (K=48,72,96,120,144,168,192) "
            "in turn. Distinct from the individually-trained fixed-K baselines: one "
            "checkpoint sliced per level, not 5 separately trained checkpoints. From "
            "release20260728_dense_reacher_goal25_dense_checkpoint_individual_levels.yaml."
        ),
        "dense_checkpoint_individual_levels_summary.json",
    )
    build_summary(
        INDIVIDUAL_FIXED_K_SOURCE_DIR,
        INDIVIDUAL_FIXED_K_TARGET,
        (
            "COMPLETE -- runs 27-31 (5 individually-trained fixed-K checkpoints, "
            "K=192/168/144/120/96), re-run on the goal25 22-combo sweep grid "
            "(matching series #1/#3 exactly) instead of the older 25-combo grid. "
            "From release20260728_dense_reacher_goal25_individual_fixed_k_baselines.yaml."
        ),
        "individual_fixed_k_baselines_summary.json",
    )


if __name__ == "__main__":
    main()
