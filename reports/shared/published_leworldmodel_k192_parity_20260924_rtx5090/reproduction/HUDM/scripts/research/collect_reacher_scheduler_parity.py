from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2, 42, 100)
T_CRIT_95_DF4 = 2.7764451051977987
SELECTION_BASIS = (
    "minimum absolute mean candidate-minus-upstream delta under the paper's "
    "10-iteration Reacher CEM recipe"
)
SELECTION_INTERPRETATION = (
    "Operational branch choice for the follow-on construction-seed sensitivity sweep; "
    "it is not proof of the released checkpoint's historical LR horizon because the "
    "released construction RNG and training runtime are unrecoverable."
)
BRANCHES = {
    "paper10": {
        "candidate_role": "paper10_nativeh5_nodecoder_init3072_fit0",
        "paper_cem": "single_level_reacher_paper10_nativeh5_parity_20260806",
        "public_cem": "single_level_reacher_paper10_nativeh5_publiccem30_parity_20260806",
    },
    "epoch10_horizon100": {
        "candidate_role": "epoch10_horizon100_nativeh5_nodecoder_init3072_fit0",
        "paper_cem": "single_level_reacher_epoch10_horizon100_nativeh5_parity_20260806",
        "public_cem": "single_level_reacher_epoch10_horizon100_nativeh5_publiccem30_parity_20260806",
    },
}


def _load_summary(root: Path, report_tag: str, candidate_role: str) -> dict[str, Any]:
    path = root / "reports" / "research" / report_tag / "n500_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"Summary did not pass: {path}")
    if payload.get("candidate_role") != candidate_role:
        raise RuntimeError(
            f"Candidate role mismatch in {path}: {payload.get('candidate_role')!r} != {candidate_role!r}"
        )
    if payload.get("seeds") != list(SEEDS) or payload.get("episodes_per_seed") != 500:
        raise RuntimeError(f"Seed/episode contract mismatch in {path}")
    rows = payload.get("paired_results")
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        raise RuntimeError(f"Paired-row contract mismatch in {path}")
    for expected_seed, row in zip(SEEDS, rows, strict=True):
        if int(row["seed"]) != expected_seed or int(row["episodes"]) != 500:
            raise RuntimeError(f"Paired-row seed/episode mismatch in {path}")
    return payload


def _ci95(values: list[float]) -> list[float]:
    center = mean(values)
    radius = T_CRIT_95_DF4 * stdev(values) / math.sqrt(len(values))
    return [center - radius, center + radius]


def _checked_common_upstream(left: dict[str, Any], right: dict[str, Any], planner: str) -> None:
    for left_row, right_row in zip(left["paired_results"], right["paired_results"], strict=True):
        seed = int(left_row["seed"])
        if int(right_row["seed"]) != seed:
            raise RuntimeError(f"{planner}: branch seed mismatch")
        for field in (
            "upstream_success_rate",
            "manifest_sha256",
            "manifest_file_sha256",
        ):
            if left_row[field] != right_row[field]:
                raise RuntimeError(f"{planner}: upstream {field} mismatch for seed {seed}")


def _planner_comparison(
    paper10: dict[str, Any], horizon100: dict[str, Any], planner: str
) -> dict[str, Any]:
    _checked_common_upstream(paper10, horizon100, planner)
    candidate_differences = [
        float(hundred["candidate_success_rate"]) - float(ten["candidate_success_rate"])
        for ten, hundred in zip(
            paper10["paired_results"], horizon100["paired_results"], strict=True
        )
    ]
    return {
        "upstream_mean_success_rate": float(
            paper10["aggregate"]["upstream_mean_success_rate"]
        ),
        "paper10_candidate_mean_success_rate": float(
            paper10["aggregate"]["candidate_mean_success_rate"]
        ),
        "paper10_mean_paired_delta": float(paper10["aggregate"]["mean_paired_delta"]),
        "paper10_mean_paired_delta_ci95": _ci95(
            [float(row["candidate_minus_upstream"]) for row in paper10["paired_results"]]
        ),
        "horizon100_candidate_mean_success_rate": float(
            horizon100["aggregate"]["candidate_mean_success_rate"]
        ),
        "horizon100_mean_paired_delta": float(
            horizon100["aggregate"]["mean_paired_delta"]
        ),
        "horizon100_mean_paired_delta_ci95": _ci95(
            [float(row["candidate_minus_upstream"]) for row in horizon100["paired_results"]]
        ),
        "horizon100_minus_paper10_candidate_by_seed": candidate_differences,
        "horizon100_minus_paper10_candidate_mean": mean(candidate_differences),
        "horizon100_minus_paper10_candidate_ci95": _ci95(candidate_differences),
    }


def collect(root: Path) -> dict[str, Any]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for branch, spec in BRANCHES.items():
        candidate_role = str(spec["candidate_role"])
        loaded[branch] = {
            planner: _load_summary(root, str(spec[planner]), candidate_role)
            for planner in ("paper_cem", "public_cem")
        }

    expected_recipes = {
        "paper_cem": {
            "threshold": 0.1,
            "planner": {"elite_frac": 0.1, "n_iter": 10, "pop_size": 300, "topk": 30},
        },
        "public_cem": {
            "threshold": 0.05,
            "planner": {"elite_frac": 0.1, "n_iter": 30, "pop_size": 300, "topk": 30},
        },
    }
    expected_data = {
        "dataset_path": "data/upstream/reacher.h5",
        "dataset_format": "hdf5",
        "dataset_sha256": "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06",
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
    for branch in loaded.values():
        for planner, expected in expected_recipes.items():
            runtime = branch[planner].get("environment_runtime")
            expected_threshold = float(expected["threshold"])
            if not isinstance(runtime, dict) or float(runtime.get("reacher_qpos_threshold", -1)) != expected_threshold:
                raise RuntimeError(
                    f"{planner}: expected explicit Reacher qpos threshold {expected_threshold}, got {runtime}"
                )
            if runtime.get("post_reset_validation") != "passed":
                raise RuntimeError(f"{planner}: Reacher threshold was not validated after reset: {runtime}")
            if branch[planner].get("planner_params") != expected["planner"]:
                raise RuntimeError(
                    f"{planner}: effective planner parameters mismatch: "
                    f"{branch[planner].get('planner_params')} != {expected['planner']}"
                )
            if branch[planner].get("evaluation_data") != expected_data:
                raise RuntimeError(
                    f"{planner}: native-HDF5/goal provenance mismatch: "
                    f"{branch[planner].get('evaluation_data')} != {expected_data}"
                )

    comparisons = {
        planner: _planner_comparison(
            loaded["paper10"][planner], loaded["epoch10_horizon100"][planner], planner
        )
        for planner in ("paper_cem", "public_cem")
    }
    paper = comparisons["paper_cem"]
    selected = min(
        ("paper10", "epoch10_horizon100"),
        key=lambda branch: abs(
            float(
                paper[
                    "paper10_mean_paired_delta"
                    if branch == "paper10"
                    else "horizon100_mean_paired_delta"
                ]
            )
        ),
    )
    return {
        "status": "pass",
        "environment": "swm/ReacherDMControl-v0",
        "episodes_per_seed": 500,
        "seeds": list(SEEDS),
        "selection_basis": SELECTION_BASIS,
        "selection_interpretation": SELECTION_INTERPRETATION,
        "historical_scheduler_identification": "unrecoverable_from_released_artifact",
        "evaluated_recipe_axes": expected_recipes,
        "evaluation_data": expected_data,
        "selected_scheduler_branch": selected,
        "comparisons": comparisons,
        "source_reports": {
            branch: {planner: str(spec[planner]) for planner in ("paper_cem", "public_cem")}
            for branch, spec in BRANCHES.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the two controlled Reacher LR horizons.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / "research"
        / "single_level_reacher_parity_20260806"
        / "scheduler_branch_comparison.json",
    )
    args = parser.parse_args()
    payload = collect(args.root.resolve())
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
