from __future__ import annotations

from hudm.experiment import (
    aggregate_summary,
    load_experiment_spec,
    run_experiment,
)
from hudm.task_sampling import enumerate_rollout_candidates, rollout_id, select_rollouts


def run_eval(cfg_path: str) -> str:
    return run_experiment(cfg_path)


def main(cfg_path: str) -> None:
    run_eval(cfg_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python planner_eval.py <path/to/config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
