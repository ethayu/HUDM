"""Build data/ summaries of every existing single-d result on release
hudm-mwm-exponential-k1to192-20261018 (the release's singleton checkpoints
checkpoints_mwm/hudm_exp20261018_<env>_single_k<K>), all envs and goals.

Two files per env / goal, because the planner settings differ:

- <env>_goal<g>_sweep_summary.json: every single-d cell of the CEM
  pop_size x n_iter sweeps (same planner, manifest and grid as the panels'
  schedule sweeps), for every K that has one:
    - K=<floor>..192 from reports/research/<env>_goal<g>_exp20261018_k<floor>to192_all_configs
    - OGB Cube K=4 from the extra baselines of
      ogb_cube_goal25_exp20261018_k4to16_schedules / ogb_cube_goal50_exp20261018_k4to32_schedules
  K below the floor (K=1,2 everywhere; also K=4 for Reacher and K=4..32 for
  PushT) have no sweep.
- <env>_goal<g>_horizon5_iter30_summary.json: one cell per K for all
  K=1..192, from reports/research/<env>_goal<g>_exp20261018_single_k1to192_horizon5_iter30
  (fixed protocol: horizon=5, receding_horizon=5, pop_size=300, n_iter=30).

Each row carries "source" (the reports/research dir it came from) and "k".

Usage: python build_summary.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from mwm.benchmark.summary import eval_summary_row  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = ROOT / "reports" / "research"
TAG = "exp20261018"
ENVS = ("ogb_cube", "pusht", "reacher", "tworoom")
GOALS = (25, 50)


def load_single_rows(source_dir: Path) -> list[dict]:
    rows = []
    # summary.json is written last, so its presence marks a finished cell.
    for summary_path in sorted(source_dir.glob("*single_k*/summary.json")):
        eval_path = summary_path.parent / "eval.json"
        row = eval_summary_row(summary_path.parent.name, eval_path, json.loads(eval_path.read_text()))
        row["source"] = source_dir.name
        row["k"] = int(re.search(r"single_k(\d+)", summary_path.parent.name).group(1))
        rows.append(row)
    return rows


def write(name: str, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["k"], r.get("pop_size") or 0, r.get("n_iter") or 0))
    out = {
        "sources": sorted({r["source"] for r in rows}),
        "k_values": sorted({r["k"] for r in rows}),
        "runs": rows,
    }
    (HERE / "data" / name).write_text(json.dumps(out, indent=2))
    print(f"{name}: {len(rows)} cells, K={out['k_values']}")


def main() -> None:
    (HERE / "data").mkdir(exist_ok=True)
    for env in ENVS:
        for goal in GOALS:
            prefix = f"{env}_goal{goal}_{TAG}"
            fixed = RESEARCH / f"{prefix}_single_k1to192_horizon5_iter30"
            sweeps = [d for d in sorted(RESEARCH.glob(f"{prefix}_k*"))
                      if d.is_dir() and d != fixed and any(d.glob("*single_k*/summary.json"))]
            write(f"{env}_goal{goal}_sweep_summary.json", [r for d in sweeps for r in load_single_rows(d)])
            write(f"{env}_goal{goal}_horizon5_iter30_summary.json", load_single_rows(fixed))


if __name__ == "__main__":
    main()
