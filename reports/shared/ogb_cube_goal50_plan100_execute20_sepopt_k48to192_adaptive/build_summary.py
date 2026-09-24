"""Build summaries for OGB-Cube goal50 sweeps on the sepopt K48-192
checkpoint. Mirrors the reacher goal50 report at
../reacher_goal50_plan100_execute20_sepopt_k48to192_adaptive/.

- adaptive_summary.json / singlek_summary.json: split from the ONE unified
  release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules.yaml
  sweep (600/600 cells, COMPLETE) into runs 02-26 (19 adaptive schedules x 25
  CEM combos = 475 cells) and runs 27-31 (5 individually-trained fixed-K
  checkpoints x 25 CEM combos = 125 cells). Unlike reacher, OGB-Cube's
  goal50 config ships runs 27-31 uncommented in the same file already -- no
  separate standalone config was needed.
- dense_checkpoint_individual_levels_summary.json: the shared sepopt
  checkpoint held fixed (no scheduling) at each of its 7 declared levels in
  turn (K=48,72,96,120,144,168,192) -- from
  release20260728_dense_ogb_cube_goal50_plan50_execute20_dense_checkpoint_individual_levels.yaml.
  May be PARTIAL (job 8628664 still running as of this writing) -- rerun
  this script to refresh once complete.

Usage: python build_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mwm.benchmark.summary import eval_summary_row  # noqa: E402

HERE = Path(__file__).resolve().parent

ALL_FIDELITY_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_dense_ogb_cube_goal50_plan100_execute20_all_fidelity_schedules"
)
DENSE_LEVELS_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_ogb_cube_goal50_dense_checkpoint_individual_levels"
)

ADAPTIVE_PREFIXES = {"02", "05", "06", "07", "08", "11", "12", "13", "14", "17",
                     "18", "19", "20", "21", "22", "23", "24", "25", "26"}
SINGLEK_PREFIXES = {"27", "28", "29", "30", "31"}

ADAPTIVE_TARGET = 19 * 25
SINGLEK_TARGET = 5 * 25
DENSE_LEVELS_TARGET = 7 * 25


def base_prefix(base_name: str) -> str:
    return base_name.split("_", 1)[0]


def main() -> None:
    adaptive_rows, singlek_rows = [], []
    for eval_path in sorted(ALL_FIDELITY_SOURCE_DIR.glob("*/eval.json")):
        payload = json.loads(eval_path.read_text())
        prefix = base_prefix(str(payload.get("base_name", "")))
        row = eval_summary_row(eval_path.parent.name, eval_path, payload)
        if prefix in ADAPTIVE_PREFIXES:
            adaptive_rows.append(row)
        elif prefix in SINGLEK_PREFIXES:
            singlek_rows.append(row)
        else:
            raise ValueError(f"unrecognized base_name prefix {prefix!r} in {eval_path}")

    dense_rows = []
    for eval_path in sorted(DENSE_LEVELS_SOURCE_DIR.glob("*/eval.json")):
        payload = json.loads(eval_path.read_text())
        dense_rows.append(eval_summary_row(eval_path.parent.name, eval_path, payload))

    def write(rows, target, note, out_name):
        out = {
            "source_output_dir": str(ALL_FIDELITY_SOURCE_DIR),
            "note": note,
            "runs_completed": len(rows),
            "runs_target": target,
            "runs": rows,
        }
        (HERE / "data" / out_name).write_text(json.dumps(out, indent=2))
        print(f"{out_name}: {len(rows)}/{target}")

    write(
        adaptive_rows, ADAPTIVE_TARGET,
        "COMPLETE -- runs 02-26 (19 adaptive schedules) from "
        "release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules.yaml, "
        "sepopt K48-192 checkpoint.",
        "adaptive_summary.json",
    )
    write(
        singlek_rows, SINGLEK_TARGET,
        "COMPLETE -- runs 27-31 (5 individually-trained fixed-K checkpoints, "
        "K=192/168/144/120/96) from the same config as the adaptive schedules.",
        "singlek_summary.json",
    )
    (HERE / "data" / "dense_checkpoint_individual_levels_summary.json").write_text(json.dumps({
        "source_output_dir": str(DENSE_LEVELS_SOURCE_DIR),
        "note": (
            "The shared sepopt K48-192 checkpoint held fixed (no scheduling) at "
            "each of its 7 declared levels in turn. From "
            "release20260728_dense_ogb_cube_goal50_plan50_execute20_dense_checkpoint_individual_levels.yaml. "
            "May be PARTIAL -- check runs_completed/runs_target."
        ),
        "runs_completed": len(dense_rows),
        "runs_target": DENSE_LEVELS_TARGET,
        "runs": dense_rows,
    }, indent=2))
    print(f"dense_checkpoint_individual_levels_summary.json: {len(dense_rows)}/{DENSE_LEVELS_TARGET}")


if __name__ == "__main__":
    main()
