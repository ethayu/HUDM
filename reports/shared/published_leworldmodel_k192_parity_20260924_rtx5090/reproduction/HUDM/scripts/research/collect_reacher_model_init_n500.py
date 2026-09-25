from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scripts.research.collect_reacher_checkpoint_n500 import HDF5_SHA256, _eval_provenance


INIT_SEEDS = (0, 1, 2, 42, 100)
EVAL_SEEDS = (0, 1, 2, 42, 100)
ROOT = Path(__file__).resolve().parents[2]
T_CRIT_95_DF4 = 2.7764451051977987
PAPER_PLANNER = {"elite_frac": 0.1, "n_iter": 10, "pop_size": 300, "topk": 30}
EVALUATION_DATA = {
    "dataset_path": "data/upstream/reacher.h5",
    "dataset_format": "hdf5",
    "dataset_sha256": HDF5_SHA256,
    "goal_offset": 25,
    "goal_indexing": "upstream_lewm_end_exclusive",
    "effective_goal_offset": 24,
    "manifest_pairs": 500,
    "eval_budget_per_batch": 50,
    "action_preprocessing": "standard_scaler",
    "sampling": "upstream_lewm",
    "rollout_semantics": "upstream_lewm_historical",
    "policy_semantics": "upstream_lewm_historical",
}


def _row(path: Path, role: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == role]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {role!r} row in {path}, found {len(rows)}")
    return rows[0]


def _paper_provenance(
    row: dict[str, str], root: Path = ROOT
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    runtime, planner, evaluation_data = _eval_provenance(row, root)
    if planner != PAPER_PLANNER:
        raise RuntimeError(
            f"Paper evaluator planner parameters mismatch: {planner} != {PAPER_PLANNER}"
        )
    if evaluation_data != EVALUATION_DATA:
        raise RuntimeError(
            f"Paper evaluator data/goal provenance mismatch: "
            f"{evaluation_data} != {EVALUATION_DATA}"
        )
    return runtime, planner, evaluation_data


def _validate_pair(upstream: dict[str, str], candidate: dict[str, str], eval_seed: int) -> None:
    for row, label in ((upstream, "upstream"), (candidate, "candidate")):
        if int(row["episodes"]) != 500 or int(row["seed"]) != eval_seed:
            raise RuntimeError(f"{label} seed/episode mismatch for evaluation seed {eval_seed}")
    for field in ("manifest_sha256", "manifest_file_sha256"):
        if upstream[field] != candidate[field]:
            raise RuntimeError(f"{field} mismatch for evaluation seed {eval_seed}")


def _ci95(values: list[float]) -> list[float]:
    center = mean(values)
    radius = T_CRIT_95_DF4 * stdev(values) / math.sqrt(len(values))
    return [center - radius, center + radius]


def collect(
    root: Path,
    *,
    report_prefix: str,
    baseline_report_tag: str,
    baseline_candidate_role: str,
) -> dict[str, object]:
    by_init: list[dict[str, object]] = []
    for init_seed in (*INIT_SEEDS, 3072):
        paired: list[dict[str, object]] = []
        for eval_seed in EVAL_SEEDS:
            baseline_csv = (
                root / "rollouts" / f"{baseline_report_tag}_seed{eval_seed}_n500" / "summary.csv"
            )
            upstream = _row(baseline_csv, "upstream_lewm_converted")
            if init_seed == 3072:
                candidate_csv = baseline_csv
                candidate_role = baseline_candidate_role
            else:
                candidate_csv = (
                    root
                    / "rollouts"
                    / f"{report_prefix}_init{init_seed}_seed{eval_seed}_n500"
                    / "summary.csv"
                )
                candidate_role = f"mwm_model_init_{init_seed}"
            candidate = _row(candidate_csv, candidate_role)
            _validate_pair(upstream, candidate, eval_seed)
            upstream_runtime, upstream_planner, upstream_data = _paper_provenance(upstream, root)
            candidate_runtime, candidate_planner, candidate_data = _paper_provenance(candidate, root)
            if upstream_runtime != candidate_runtime:
                raise RuntimeError(
                    f"Environment runtime mismatch for model-init {init_seed}, eval seed {eval_seed}"
                )
            if upstream_planner != candidate_planner or upstream_data != candidate_data:
                raise RuntimeError(
                    f"Evaluator provenance mismatch for model-init {init_seed}, "
                    f"eval seed {eval_seed}"
                )
            upstream_rate = float(upstream["success_rate"])
            candidate_rate = float(candidate["success_rate"])
            paired.append(
                {
                    "eval_seed": eval_seed,
                    "episodes": 500,
                    "upstream_success_rate": upstream_rate,
                    "candidate_success_rate": candidate_rate,
                    "candidate_minus_upstream": candidate_rate - upstream_rate,
                    "manifest_sha256": candidate["manifest_sha256"],
                    "manifest_file_sha256": candidate["manifest_file_sha256"],
                    "upstream_summary_csv": str(baseline_csv.relative_to(root)),
                    "candidate_summary_csv": str(candidate_csv.relative_to(root)),
                    "env_runtime": candidate_runtime,
                    "planner_params": candidate_planner,
                    "evaluation_data": candidate_data,
                }
            )
        candidate_rates = [float(row["candidate_success_rate"]) for row in paired]
        deltas = [float(row["candidate_minus_upstream"]) for row in paired]
        delta_ci95 = _ci95(deltas)
        by_init.append(
            {
                "model_init_seed": init_seed,
                "paired_results": paired,
                "aggregate": {
                    "candidate_mean_success_rate": mean(candidate_rates),
                    "candidate_sample_stdev": stdev(candidate_rates),
                    "mean_paired_delta": mean(deltas),
                    "paired_delta_sample_stdev": stdev(deltas),
                    "mean_paired_delta_ci95": delta_ci95,
                    "zero_in_mean_paired_delta_ci95": delta_ci95[0] <= 0.0 <= delta_ci95[1],
                },
            }
        )

    upstream_rates = [
        float(row["upstream_success_rate"])
        for row in by_init[0]["paired_results"]  # type: ignore[index]
    ]
    runtime_values = {
        json.dumps(row["env_runtime"], sort_keys=True)
        for branch in by_init
        for row in branch["paired_results"]  # type: ignore[index]
    }
    if len(runtime_values) != 1:
        raise RuntimeError(f"Environment runtime changed across the init sweep: {sorted(runtime_values)}")
    closest = min(
        by_init,
        key=lambda branch: abs(float(branch["aggregate"]["mean_paired_delta"])),  # type: ignore[index]
    )
    candidate_means = [
        float(branch["aggregate"]["candidate_mean_success_rate"])  # type: ignore[index]
        for branch in by_init
    ]
    payload: dict[str, object] = {
        "status": "pass",
        "environment": "swm/ReacherDMControl-v0",
        "episodes_per_eval_seed": 500,
        "evaluation_seeds": list(EVAL_SEEDS),
        "model_init_seeds": [*INIT_SEEDS, 3072],
        "baseline_report_tag": baseline_report_tag,
        "environment_runtime": by_init[0]["paired_results"][0]["env_runtime"],  # type: ignore[index]
        "planner_params": PAPER_PLANNER,
        "evaluation_data": EVALUATION_DATA,
        "upstream": {
            "mean_success_rate": mean(upstream_rates),
            "sample_stdev": stdev(upstream_rates),
        },
        "by_model_init_seed": by_init,
        "construction_seed_sensitivity": {
            "candidate_mean_success_rate_across_initializations": mean(candidate_means),
            "candidate_mean_success_rate_sample_stdev_across_initializations": stdev(candidate_means),
            "candidate_mean_success_rate_min": min(candidate_means),
            "candidate_mean_success_rate_max": max(candidate_means),
            "candidate_mean_success_rate_range": max(candidate_means) - min(candidate_means),
            "descriptive_closest_model_init_seed": int(closest["model_init_seed"]),
            "descriptive_closest_mean_paired_delta": float(
                closest["aggregate"]["mean_paired_delta"]  # type: ignore[index]
            ),
            "descriptive_closest_mean_paired_delta_ci95": closest["aggregate"][  # type: ignore[index]
                "mean_paired_delta_ci95"
            ],
            "interpretation": (
                "The closest seed is descriptive after selection across six tested constructions; "
                "its interval must not be treated as an unadjusted confirmatory parity test."
            ),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the controlled Reacher construction-seed sweep.")
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--baseline-report-tag", required=True)
    parser.add_argument("--baseline-candidate-role", required=True)
    args = parser.parse_args()
    payload = collect(
        ROOT,
        report_prefix=args.report_prefix,
        baseline_report_tag=args.baseline_report_tag,
        baseline_candidate_role=args.baseline_candidate_role,
    )
    destination = ROOT / "reports" / "research" / args.report_prefix / "model_init_n500_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
