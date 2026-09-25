"""Build summaries for PushT goal25 sweeps on the sepopt K48-192 checkpoint,
restricted to the K120-192 range. goal25 counterpart of
../pusht_goal50_plan100_execute20_sepopt_k120to192_adaptive/. The full
K48-192 goal25 adaptive series lives in ../pusht_goal25_sepopt_k48to192_all_configs/.

- adaptive_summary.json / singlek_summary.json: split from the K120-192
  slice sweep
  (release20260728_dense_pusht_goal25_k120to192slice_sepopt_all_fidelity_schedules.yaml)
  into runs 02-26 (19 adaptive schedules, "coarsest" floored at K=120,
  level index 3) and runs 27-31 (5 individually-trained fixed-K
  checkpoints, K=192/168/144/120/96 -- unaffected by the K-floor change
  since each uses its own single-level checkpoint).
- dense_checkpoint_individual_levels_summary.json: the shared sepopt
  checkpoint held fixed (no scheduling) at each of its K>=120 levels
  (K=120,144,168,192 -- K48/72/96 were dropped from this sweep on
  request) -- from
  release20260728_dense_pusht_goal25_dense_checkpoint_k120to192_levels.yaml.
- singlek_retrain_summary.json: the Single-d series re-evaluated with the
  separate-optimizer single-K retrains (GitHub release
  hudm-mwm-single-k48to192-sepopt-retrain-20260922, K=120/144/168/192, same
  eval/planner/sweep settings) -- from
  release20260728_dense_pusht_goal25_singlek_sepopt_retrain20260922_k120to192.yaml.
  This is the Single-d series the plots draw; singlek_summary.json (old
  release20260728 single-K checkpoints) is kept for reference only.

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

K120SLICE_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_pusht_goal25_k120to192slice_sepopt_all_fidelity_schedules"
)
DENSE_LEVELS_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_pusht_goal25_dense_checkpoint_k120to192_levels"
)
SINGLEK_RETRAIN_SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_pusht_goal25_singlek_sepopt_retrain20260922_k120to192"
)

ADAPTIVE_PREFIXES = {"02", "05", "06", "07", "08", "11", "12", "13", "14", "17",
                     "18", "19", "20", "21", "22", "23", "24", "25", "26"}
SINGLEK_PREFIXES = {"27", "28", "29", "30", "31"}

ADAPTIVE_TARGET = 19 * 25
SINGLEK_TARGET = 5 * 25
DENSE_LEVELS_TARGET = 4 * 25
SINGLEK_RETRAIN_TARGET = 4 * 25


def base_prefix(base_name: str) -> str:
    return base_name.split("_", 1)[0]


def main() -> None:
    adaptive_rows, singlek_rows = [], []
    for eval_path in sorted(K120SLICE_SOURCE_DIR.glob("*/eval.json")):
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
            "source_output_dir": str(K120SLICE_SOURCE_DIR),
            "note": note,
            "runs_completed": len(rows),
            "runs_target": target,
            "runs": rows,
        }
        (HERE / "data" / out_name).write_text(json.dumps(out, indent=2))
        print(f"{out_name}: {len(rows)}/{target}")

    write(
        adaptive_rows, ADAPTIVE_TARGET,
        "Runs 02-26 (19 adaptive schedules) from the K120-192 slice sweep -- "
        "every 'coarsest' schedule endpoint floored at K=120 (level index 3) "
        "instead of K=48. From "
        "release20260728_dense_pusht_goal25_k120to192slice_sepopt_all_fidelity_schedules.yaml.",
        "adaptive_summary.json",
    )
    write(
        singlek_rows, SINGLEK_TARGET,
        "Runs 27-31 (5 individually-trained fixed-K checkpoints, "
        "K=192/168/144/120/96) from the same K120-192 slice sweep config.",
        "singlek_summary.json",
    )
    (HERE / "data" / "dense_checkpoint_individual_levels_summary.json").write_text(json.dumps({
        "source_output_dir": str(DENSE_LEVELS_SOURCE_DIR),
        "note": (
            "The shared sepopt K48-192 checkpoint held fixed (no scheduling) at "
            "each of its K>=120 levels (K48/72/96 dropped from this sweep on "
            "request). From "
            "release20260728_dense_pusht_goal25_dense_checkpoint_k120to192_levels.yaml."
        ),
        "runs_completed": len(dense_rows),
        "runs_target": DENSE_LEVELS_TARGET,
        "runs": dense_rows,
    }, indent=2))
    print(f"dense_checkpoint_individual_levels_summary.json: {len(dense_rows)}/{DENSE_LEVELS_TARGET}")

    retrain_rows = []
    for eval_path in sorted(SINGLEK_RETRAIN_SOURCE_DIR.glob("*/eval.json")):
        payload = json.loads(eval_path.read_text())
        retrain_rows.append(eval_summary_row(eval_path.parent.name, eval_path, payload))
    (HERE / "data" / "singlek_retrain_summary.json").write_text(json.dumps({
        "source_output_dir": str(SINGLEK_RETRAIN_SOURCE_DIR),
        "note": (
            "Runs 27-30 re-evaluated with the separate-optimizer single-K "
            "retrains (release hudm-mwm-single-k48to192-sepopt-retrain-20260922, "
            "K=192/168/144/120, canonical epoch 9). From "
            "release20260728_dense_pusht_goal25_singlek_sepopt_retrain20260922_k120to192.yaml."
        ),
        "runs_completed": len(retrain_rows),
        "runs_target": SINGLEK_RETRAIN_TARGET,
        "runs": retrain_rows,
    }, indent=2))
    print(f"singlek_retrain_summary.json: {len(retrain_rows)}/{SINGLEK_RETRAIN_TARGET}")


if __name__ == "__main__":
    main()
