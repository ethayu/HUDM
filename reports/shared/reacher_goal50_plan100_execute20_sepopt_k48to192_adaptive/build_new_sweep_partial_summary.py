"""Build partial-progress summaries from the in-flight release20260728 reacher
goal50 all_fidelity_schedules sweep (relaunched to /vast/projects/dineshj/lab/
aurora/... after the local reports/research/ copy became unavailable).

Splits completed cells into:
- new_sweep_partial_adaptive_summary.json: the 19 adaptive-schedule base runs
  (02,05,06,07,08,11,12,13,14,17,18,19,20,21,22,23,24,25,26) -- same set as
  the already-COMPLETE sepopt_k48to192_adaptive_summary.json in this folder,
  just from the freshly re-run sweep.
- new_sweep_partial_singlek_summary.json: the 5 new single-K fixed-finest
  baselines (27=K192, 28=K168, 29=K144, 30=K120, 31=K96) that were NOT part
  of the original adaptive config and have no prior data in this folder.

Usage: python build_new_sweep_partial_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mwm.benchmark.summary import eval_summary_row  # noqa: E402

SOURCE_DIR = Path(
    "/vast/projects/dineshj/lab/aurora/reports/research/"
    "release20260728_dense_reacher_goal50_plan100_execute20_all_fidelity_schedules"
)

ADAPTIVE_PREFIXES = {"02", "05", "06", "07", "08", "11", "12", "13", "14", "17",
                      "18", "19", "20", "21", "22", "23", "24", "25", "26"}
SINGLEK_PREFIXES = {"27", "28", "29", "30", "31"}

ADAPTIVE_TARGET = 19 * 25
SINGLEK_TARGET = 5 * 25

HERE = Path(__file__).resolve().parent


def base_prefix(base_name: str) -> str:
    return base_name.split("_", 1)[0]


def main() -> None:
    adaptive_rows = []
    singlek_rows = []
    for eval_path in sorted(SOURCE_DIR.glob("*/eval.json")):
        payload = json.loads(eval_path.read_text())
        base_name = str(payload.get("base_name", ""))
        prefix = base_prefix(base_name)
        row = eval_summary_row(eval_path.parent.name, eval_path, payload)
        if prefix in ADAPTIVE_PREFIXES:
            adaptive_rows.append(row)
        elif prefix in SINGLEK_PREFIXES:
            singlek_rows.append(row)
        else:
            raise ValueError(f"unrecognized base_name prefix {prefix!r} in {eval_path}")

    adaptive_out = {
        "source_output_dir": str(SOURCE_DIR),
        "note": (
            "PARTIAL -- in-flight re-run of the 19 adaptive schedule runs "
            "(release20260728_dense_reacher_goal50_plan50_execute20_all_fidelity_schedules.yaml), "
            "relaunched to /vast/projects/dineshj/lab/aurora/ after node failures "
            "hit the original job and the local reports/research/ copy became unavailable."
        ),
        "runs_completed": len(adaptive_rows),
        "runs_target": ADAPTIVE_TARGET,
        "runs": adaptive_rows,
    }
    singlek_out = {
        "source_output_dir": str(SOURCE_DIR),
        "note": (
            "PARTIAL -- new single-K fixed-finest baselines (K=192/168/144/120/96) "
            "added to the sweep config for the first time, same sepopt K48-192 "
            "checkpoint as the adaptive schedules. In flight at "
            "/vast/projects/dineshj/lab/aurora/."
        ),
        "runs_completed": len(singlek_rows),
        "runs_target": SINGLEK_TARGET,
        "runs": singlek_rows,
    }

    (HERE / "data" / "new_sweep_partial_adaptive_summary.json").write_text(json.dumps(adaptive_out, indent=2))
    (HERE / "data" / "new_sweep_partial_singlek_summary.json").write_text(json.dumps(singlek_out, indent=2))
    print(f"adaptive: {len(adaptive_rows)}/{ADAPTIVE_TARGET}")
    print(f"single-K: {len(singlek_rows)}/{SINGLEK_TARGET}")


if __name__ == "__main__":
    main()
