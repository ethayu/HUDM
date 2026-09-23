"""Build summaries for two newly-completed OGB-Cube goal25 sweeps on the
sepopt K48-192 checkpoint. Mirrors the reacher goal25 counterpart,
../reacher_goal25_sepopt_k48to192_all_configs/build_k96to192_and_dense_levels_summary.py
(minus the K96-192-floor adaptive-schedule slice, which has not been run for
OGB-Cube on any goal offset -- see the goal50 OGB-Cube report's README).

- dense_checkpoint_individual_levels_summary.json (154/154): the shared
  sepopt K48-192 checkpoint held fixed (no scheduling) at each of its 7
  declared levels in turn (K=48,72,96,120,144,168,192), on the same
  22-combo sweep grid as the adaptive schedules. From
  release20260728_ogb_cube_goal25_dense_checkpoint_individual_levels.
  This series did not exist at all in the original (PARTIAL) version of
  this report.
- individual_fixed_k_baselines_summary.json (110/110): a freshly re-run
  version of the individually-trained fixed-K baselines (K=192/168/144/
  120/96), on the goal25 22-combo sweep grid (matching series #1's grid
  exactly), from release20260728_ogb_cube_goal25_individual_fixed_k_baselines.
  Replaces the older data/fixed_k_baselines_summary.json (125 cells, 25
  combos/run, a different/wider grid) as the plotted fixed-K series.

The adaptive K48-192 schedule sweep itself (data/sepopt_k48to192_adaptive_
summary.json, 262/418) is NOT refreshed by this script -- no newer/complete
source directory for it was found on disk; it is carried over unchanged
from the original PARTIAL version of this report.

Usage: python build_dense_levels_and_refreshed_fixedk_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mwm.benchmark.summary import eval_summary_row  # noqa: E402

HERE = Path(__file__).resolve().parent

DENSE_LEVELS_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_ogb_cube_goal25_dense_checkpoint_individual_levels"
)
INDIVIDUAL_FIXED_K_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_ogb_cube_goal25_individual_fixed_k_baselines"
)

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
        DENSE_LEVELS_SOURCE_DIR,
        DENSE_LEVELS_TARGET,
        (
            "COMPLETE -- the shared sepopt K48-192 checkpoint held fixed (no "
            "scheduling) at each of its 7 declared levels (K=48,72,96,120,144,168,192) "
            "in turn. Distinct from the individually-trained fixed-K baselines: one "
            "checkpoint sliced per level, not 5 separately trained checkpoints. From "
            "release20260728_ogb_cube_goal25_dense_checkpoint_individual_levels."
        ),
        "dense_checkpoint_individual_levels_summary.json",
    )
    build_summary(
        INDIVIDUAL_FIXED_K_SOURCE_DIR,
        INDIVIDUAL_FIXED_K_TARGET,
        (
            "COMPLETE -- 5 individually-trained fixed-K checkpoints "
            "(K=192/168/144/120/96), re-run on the goal25 22-combo sweep grid "
            "(matching the adaptive-schedule series' grid exactly) instead of the "
            "older 25-combo grid. From "
            "release20260728_ogb_cube_goal25_individual_fixed_k_baselines."
        ),
        "individual_fixed_k_baselines_summary.json",
    )


if __name__ == "__main__":
    main()
