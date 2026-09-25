from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLD_TAG = "reacher_identity_joint10_n500_upstream_exact_20260806"
NEW_TAG = "reacher_identity_joint10_n500_historical_threshold010_20260806"
SEEDS = (0, 1, 2, 42, 100)
ROLES = ("000_upstream", "001_retrained_joint10")
DESTINATION = (
    ROOT
    / "reports"
    / "research"
    / "single_level_reacher_parity_20260806"
    / "reacher_eval_semantics_causal.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    raise RuntimeError(
        "Superseded collector: its threshold-0.1 runs applied the override before Reacher reset, "
        "then reset-time task recompilation restored 0.05. Use "
        "collect_upstream_reacher_evaluator_parity.py and outputs with "
        "env_runtime.post_reset_validation=passed."
    )
    threshold_rows = []
    for seed in SEEDS:
        for role in ROLES:
            old_path = ROOT / "rollouts" / f"{OLD_TAG}_seed{seed}_n500" / role / "eval.json"
            new_path = ROOT / "rollouts" / f"{NEW_TAG}_seed{seed}_n500" / role / "eval.json"
            old = _load(old_path)
            new = _load(new_path)
            old_success = list(old["swm_results"]["episode_successes"])
            new_success = list(new["swm_results"]["episode_successes"])
            if len(old_success) != 500 or len(new_success) != 500:
                raise RuntimeError(f"Unexpected episode count for seed={seed}, role={role}")
            runtime = new.get("env_runtime", {})
            if float(runtime.get("reacher_qpos_threshold", -1)) != 0.1:
                raise RuntimeError(f"Historical threshold was not applied in {new_path}")
            defaults = list(runtime.get("dependency_default_qpos_thresholds", []))
            if len(defaults) != 50 or set(map(float, defaults)) != {0.05}:
                raise RuntimeError(f"Unexpected dependency threshold provenance in {new_path}")
            changed = sum(a != b for a, b in zip(old_success, new_success, strict=True))
            threshold_rows.append(
                {
                    "seed": seed,
                    "role": role,
                    "episodes": 500,
                    "success_labels_changed": changed,
                    "threshold_0_05_success_rate": float(old["swm_results"]["success_rate"]),
                    "threshold_0_10_success_rate": float(new["swm_results"]["success_rate"]),
                }
            )
    if any(row["success_labels_changed"] for row in threshold_rows):
        raise RuntimeError("Threshold changed at least one paired episode label")

    ten_path = (
        ROOT
        / "rollouts"
        / "reacher_paper_metric_seed42_n50_threshold010_20260806"
        / "eval.json"
    )
    thirty_path = (
        ROOT
        / "rollouts"
        / "reacher_historical_code_seed42_n50_iter30_threshold010_20260806"
        / "eval.json"
    )
    hdf5_path = (
        ROOT
        / "rollouts"
        / "reacher_historical_code_native_h5_seed42_n50_iter30_threshold010_20260806"
        / "eval.json"
    )
    historical_goal_path = (
        ROOT
        / "rollouts"
        / "reacher_historical_goal24_native_h5_seed42_n50_iter30_threshold010_20260806"
        / "eval.json"
    )
    historical_goal_threshold005_path = (
        ROOT
        / "rollouts"
        / "reacher_historical_goal24_native_h5_seed42_n50_iter30_threshold005_20260806"
        / "eval.json"
    )
    ten = _load(ten_path)
    thirty = _load(thirty_path)
    hdf5 = _load(hdf5_path)
    historical_goal = _load(historical_goal_path)
    historical_goal_threshold005 = _load(historical_goal_threshold005_path)
    if ten["manifest"]["manifest_sha256"] != thirty["manifest"]["manifest_sha256"]:
        raise RuntimeError("N=50 CEM comparisons did not use the same manifest")
    if float(ten["env_runtime"]["reacher_qpos_threshold"]) != 0.1:
        raise RuntimeError("10-iteration run did not use threshold 0.1")
    if float(thirty["env_runtime"]["reacher_qpos_threshold"]) != 0.1:
        raise RuntimeError("30-iteration run did not use threshold 0.1")
    if float(hdf5["env_runtime"]["reacher_qpos_threshold"]) != 0.1:
        raise RuntimeError("Native-HDF5 run did not use threshold 0.1")
    lance_pairs = [pair for batch in thirty["batches"] for pair in batch["pairs"]]
    hdf5_pairs = [pair for batch in hdf5["batches"] for pair in batch["pairs"]]
    historical_goal_pairs = [
        pair for batch in historical_goal["batches"] for pair in batch["pairs"]
    ]
    if lance_pairs != hdf5_pairs:
        raise RuntimeError("Lossy-Lance and lossless-HDF5 runs did not sample the same pairs")
    if [pair["start_row"] for pair in hdf5_pairs] != [
        pair["start_row"] for pair in historical_goal_pairs
    ]:
        raise RuntimeError("Goal-indexing comparison did not preserve start rows")
    if any(
        historical["goal_row"] != corrected["goal_row"] - 1
        for corrected, historical in zip(hdf5_pairs, historical_goal_pairs, strict=True)
    ):
        raise RuntimeError("Historical goal rows are not exactly one row before corrected goal rows")
    if historical_goal.get("goal_indexing") != {
        "mode": "upstream_lewm_end_exclusive",
        "requested_offset": 25,
        "effective_offset": 24,
    }:
        raise RuntimeError("Historical goal-indexing provenance is missing or incorrect")
    if float(historical_goal_threshold005["env_runtime"]["reacher_qpos_threshold"]) != 0.05:
        raise RuntimeError("Historical-goal threshold-0.05 provenance is incorrect")
    if [pair for batch in historical_goal_threshold005["batches"] for pair in batch["pairs"]] != historical_goal_pairs:
        raise RuntimeError("Historical-goal threshold replay did not preserve pairs")

    ten_rate = float(ten["swm_results"]["success_rate"])
    thirty_rate = float(thirty["swm_results"]["success_rate"])
    hdf5_rate = float(hdf5["swm_results"]["success_rate"])
    historical_goal_rate = float(historical_goal["swm_results"]["success_rate"])
    figure_rate = 86.0
    payload = {
        "status": "pass",
        "threshold_causal_replay": {
            "threshold_0_05": 0.05,
            "threshold_0_10": 0.1,
            "paired_episode_labels": 5000,
            "changed_episode_labels": sum(
                int(row["success_labels_changed"]) for row in threshold_rows
            ),
            "rows": threshold_rows,
            "conclusion": (
                "The threshold provenance changed, but it had no observed effect on either "
                "checkpoint over these 5,000 paired episode labels."
            ),
        },
        "cem_iteration_causal_replay": {
            "episodes": 50,
            "seed": 42,
            "threshold": 0.1,
            "manifest_sha256": ten["manifest"]["manifest_sha256"],
            "ten_iterations": {
                "success_rate": ten_rate,
                "cem_cost_calls": int(ten["planning_diagnostics"]["cem_cost_calls"]),
            },
            "thirty_iterations": {
                "success_rate": thirty_rate,
                "cem_cost_calls": int(thirty["planning_diagnostics"]["cem_cost_calls"]),
            },
            "thirty_minus_ten_percentage_points": thirty_rate - ten_rate,
            "paper_figure_6_visible_rate": figure_rate,
            "ten_minus_figure_percentage_points": ten_rate - figure_rate,
            "thirty_minus_figure_percentage_points": thirty_rate - figure_rate,
            "paper_rate_binomial_standard_error_percentage_points": 100.0
            * math.sqrt((figure_rate / 100.0) * (1.0 - figure_rate / 100.0) / 50.0),
            "conclusion": (
                "Using the initial executable 30-iteration CEM recipe raises this exact N=50 "
                "sample by 2 points but does not reproduce Figure 6's visible 86% value."
            ),
        },
        "lossless_pixel_causal_replay": {
            "episodes": 50,
            "seed": 42,
            "threshold": 0.1,
            "cem_iterations": 30,
            "sample_pairs_equal": True,
            "jpeg_lance_success_rate": thirty_rate,
            "lossless_hdf5_success_rate": hdf5_rate,
            "hdf5_minus_lance_percentage_points": hdf5_rate - thirty_rate,
            "changed_episode_labels": sum(
                a != b
                for a, b in zip(
                    thirty["swm_results"]["episode_successes"],
                    hdf5["swm_results"]["episode_successes"],
                    strict=True,
                )
            ),
            "lance_only_successes": sum(
                a and not b
                for a, b in zip(
                    thirty["swm_results"]["episode_successes"],
                    hdf5["swm_results"]["episode_successes"],
                    strict=True,
                )
            ),
            "hdf5_only_successes": sum(
                b and not a
                for a, b in zip(
                    thirty["swm_results"]["episode_successes"],
                    hdf5["swm_results"]["episode_successes"],
                    strict=True,
                )
            ),
            "hdf5_cem_cost_calls": int(hdf5["planning_diagnostics"]["cem_cost_calls"]),
            "hdf5_minus_figure_percentage_points": hdf5_rate - figure_rate,
            "conclusion": (
                "Lossless pixels materially alter individual CEM outcomes (13 of 50 labels), "
                "but reduce the net score by 2 points on this sample. JPEG conversion therefore "
                "does not explain the reported 86% value; strict HDF5 evaluation widens its gap."
            ),
        },
        "goal_indexing_causal_replay": {
            "episodes": 50,
            "seed": 42,
            "threshold": 0.1,
            "cem_iterations": 30,
            "requested_goal_offset": 25,
            "corrected_effective_goal_offset": 25,
            "historical_effective_goal_offset": 24,
            "start_rows_equal": True,
            "corrected_success_rate": hdf5_rate,
            "historical_success_rate": historical_goal_rate,
            "historical_minus_corrected_percentage_points": historical_goal_rate - hdf5_rate,
            "historical_minus_figure_percentage_points": historical_goal_rate - figure_rate,
            "changed_episode_labels": sum(
                a != b
                for a, b in zip(
                    hdf5["swm_results"]["episode_successes"],
                    historical_goal["swm_results"]["episode_successes"],
                    strict=True,
                )
            ),
            "corrected_only_successes": sum(
                a and not b
                for a, b in zip(
                    hdf5["swm_results"]["episode_successes"],
                    historical_goal["swm_results"]["episode_successes"],
                    strict=True,
                )
            ),
            "historical_only_successes": sum(
                b and not a
                for a, b in zip(
                    hdf5["swm_results"]["episode_successes"],
                    historical_goal["swm_results"]["episode_successes"],
                    strict=True,
                )
            ),
            "historical_goal_threshold_0_05_success_rate": float(
                historical_goal_threshold005["swm_results"]["success_rate"]
            ),
            "historical_goal_threshold_changed_labels": sum(
                a != b
                for a, b in zip(
                    historical_goal_threshold005["swm_results"]["episode_successes"],
                    historical_goal["swm_results"]["episode_successes"],
                    strict=True,
                )
            ),
            "conclusion": (
                "The paper-era end-exclusive slice targets start+24 despite requesting 25. "
                "That one-row shift changes 12/50 outcomes and recovers 4 percentage points, "
                "but the exact released checkpoint remains 4 points below Figure 6."
            ),
        },
    }
    DESTINATION.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
