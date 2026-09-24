"""Build data/ summaries for the completed TwoRoom goal50 sweeps on /vast/projects:

- k96to192_adaptive_summary.json (418/418): runs 02-26, adaptive schedules on
  the original joint K96-192 checkpoint
  (checkpoints_mwm/mwm_paper10_tworoom_k96_120_144_168_192_release20260728) --
  NOT the sepopt checkpoint. Slurm array 8634465, config
  configs/research/release20260728_dense_tworoom_goal50_plan50_execute20_k96to192_all_fidelity_schedules.yaml.
- k96to192_singlek_summary.json (110/110): runs 27-31 of the same sweep, the
  individually-trained fixed-K checkpoints (K=192/168/144/120/96), re-run
  with real audited FLOPs.
- dense_checkpoint_individual_levels_summary.json (154/154): the sepopt
  K48-192 checkpoint held fixed (no scheduling) at each of its 7 levels.
  Slurm array 8635617, config
  configs/research/release20260728_dense_tworoom_goal50_plan50_execute20_dense_checkpoint_individual_levels.yaml.

All three carry native dynamics_flops_total. Every sweep here: goal_offset=50,
horizon 4, budget 100, 100 episodes, 22 CEM cells per run (25-cell
pop_size x n_iter grid minus 3 sweep_exclude cells).

Usage: python build_new_results_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mwm.benchmark.summary import eval_summary_row  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = Path("/vast/projects/dineshj/lab/aurora/reports/research")

K96TO192_SOURCE_DIR = RESEARCH / "release20260728_k96to192_tworoom_goal50_plan100_execute20_all_fidelity_schedules"
DENSE_LEVELS_SOURCE_DIR = RESEARCH / "release20260728_tworoom_goal50_dense_checkpoint_individual_levels"

ADAPTIVE_PREFIXES = {"02", "05", "06", "07", "08", "11", "12", "13", "14", "17",
                     "18", "19", "20", "21", "22", "23", "24", "25", "26"}
CELLS_PER_RUN = 22


def load_rows(source_dir: Path) -> list[dict]:
    rows = []
    # summary.json is written last, so its presence marks a finished cell.
    for summary_path in sorted(source_dir.glob("*/summary.json")):
        eval_path = summary_path.parent / "eval.json"
        rows.append(eval_summary_row(summary_path.parent.name, eval_path, json.loads(eval_path.read_text())))
    return rows


def write(name: str, source_dir: Path, rows: list[dict], target: int, note: str) -> None:
    out = {
        "source_output_dir": str(source_dir),
        "note": note,
        "runs_completed": len(rows),
        "runs_target": target,
        "runs": rows,
    }
    (HERE / "data" / name).write_text(json.dumps(out, indent=2))
    print(f"{name}: {len(rows)}/{target}")


def main() -> None:
    sweep = load_rows(K96TO192_SOURCE_DIR)
    adaptive = [r for r in sweep if r["base_name"].split("_", 1)[0] in ADAPTIVE_PREFIXES]
    singlek = [r for r in sweep if r["base_name"].split("_", 1)[0] not in ADAPTIVE_PREFIXES]
    write("k96to192_adaptive_summary.json", K96TO192_SOURCE_DIR, adaptive,
          len(ADAPTIVE_PREFIXES) * CELLS_PER_RUN,
          "Runs 02-26, joint K96-192 checkpoint (NOT sepopt), adaptive schedules.")
    write("k96to192_singlek_summary.json", K96TO192_SOURCE_DIR, singlek, 5 * CELLS_PER_RUN,
          "Runs 27-31, individually-trained fixed-K checkpoints (K=192/168/144/120/96).")
    write("dense_checkpoint_individual_levels_summary.json", DENSE_LEVELS_SOURCE_DIR,
          load_rows(DENSE_LEVELS_SOURCE_DIR), 7 * CELLS_PER_RUN,
          "Sepopt K48-192 checkpoint held fixed at each of its 7 levels (no scheduling).")


if __name__ == "__main__":
    main()
