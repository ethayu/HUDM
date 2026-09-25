from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import yaml

from mwm.data.manifest import load_manifest, manifest_file_sha256


SEEDS = (0, 1, 2, 42, 100)
ROOT = Path(__file__).resolve().parents[2]
HDF5_SHA256 = "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06"


def _row(path: Path, role: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == role]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {role!r} row in {path}, found {len(rows)}")
    return rows[0]


def _eval_provenance(
    row: dict[str, str], root: Path = ROOT
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    path = Path(row["output_json"])
    if not path.is_absolute():
        path = root / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = payload.get("env_runtime")
    if not isinstance(runtime, dict) or "reacher_qpos_threshold" not in runtime:
        raise RuntimeError(f"Missing explicit Reacher environment runtime provenance in {path}")
    if runtime.get("post_reset_validation") != "passed":
        raise RuntimeError(f"Reacher threshold was not validated after reset in {path}")
    diagnostics = payload.get("planning_diagnostics", {})
    calls = int(diagnostics.get("cem_cost_calls", 0))
    if calls <= 0:
        raise RuntimeError(f"Missing CEM-call provenance in {path}")
    expected = {"upstream_lewm_historical": calls}
    for field in ("rollout_semantics_counts", "policy_semantics_counts"):
        if diagnostics.get(field) != expected:
            raise RuntimeError(f"Historical evaluator semantics mismatch for {field} in {path}")
    planner = payload.get("planner_params")
    if not isinstance(planner, dict):
        raise RuntimeError(f"Missing effective planner parameters in {path}")
    normalized = {
        "elite_frac": float(planner.get("elite_frac", -1)),
        "n_iter": int(planner.get("n_iter", -1)),
        "pop_size": int(planner.get("pop_size", -1)),
        "topk": int(planner.get("topk", -1)),
    }
    if normalized["n_iter"] <= 0 or normalized["pop_size"] <= 0 or normalized["topk"] <= 0:
        raise RuntimeError(f"Invalid effective planner parameters in {path}: {normalized}")
    config_ref = payload.get("config")
    if not isinstance(config_ref, dict) or not config_ref.get("resolved_path"):
        raise RuntimeError(f"Missing resolved evaluation config provenance in {path}")
    config_path = Path(str(config_ref["resolved_path"]))
    if not config_path.is_absolute():
        config_path = root / config_path
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if config_ref.get("sha256") != config_sha:
        raise RuntimeError(f"Resolved evaluation config hash mismatch in {path}")
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = resolved.get("data", {})
    eval_cfg = resolved.get("eval", {})
    env_cfg = resolved.get("env", {})
    planner_cfg = resolved.get("planner", {})
    if (
        resolved.get("env_id") != "swm/ReacherDMControl-v0"
        or data_cfg.get("format") != "hdf5"
        or data_cfg.get("identity_path") != "data/upstream/reacher.h5"
        or data_cfg.get("action_preprocessing") != "standard_scaler"
        or data_cfg.get("keys_to_load") != ["pixels", "action", "qpos", "qvel", "observation"]
        or int(eval_cfg.get("episodes", -1)) != 500
        or int(eval_cfg.get("goal_offset", -1)) != 25
        or eval_cfg.get("goal_indexing") != "upstream_lewm_end_exclusive"
        or int(eval_cfg.get("seed", -1)) != int(row.get("seed", -2))
        or int(eval_cfg.get("budget", -1)) != 50
        or int(eval_cfg.get("num_envs", -1)) != 50
        or eval_cfg.get("sampling") != "upstream_lewm"
        or int(env_cfg.get("max_episode_steps", -1)) != 100
        or env_cfg.get("goal_conditioned") is not True
        or env_cfg.get("kwargs", {}).get("task") != "qpos_match"
        or float(env_cfg.get("runtime", {}).get("reacher_qpos_threshold", -1))
        != float(runtime["reacher_qpos_threshold"])
        or resolved.get("restore", {}).get("import_path")
        != "mwm.swm.restore.reacher_qpos_match_restore_spec"
        or planner_cfg.get("rollout_semantics") != "upstream_lewm_historical"
        or planner_cfg.get("policy_semantics") != "upstream_lewm_historical"
    ):
        raise RuntimeError(f"Resolved historical evaluator contract mismatch in {config_path}")
    manifest_ref = payload.get("manifest")
    if not isinstance(manifest_ref, dict) or not manifest_ref.get("path"):
        raise RuntimeError(f"Missing evaluation manifest provenance in {path}")
    manifest_path = Path(str(manifest_ref["path"]))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_manifest(manifest_path)
    actual_file_sha = manifest_file_sha256(manifest_path)
    if manifest_ref.get("sha256") != actual_file_sha:
        raise RuntimeError(f"Evaluation manifest file hash mismatch in {path}")
    if manifest_ref.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise RuntimeError(f"Evaluation manifest content hash mismatch in {path}")
    if row.get("manifest_file_sha256") != actual_file_sha:
        raise RuntimeError(f"Summary CSV manifest file hash is detached from {path}")
    if row.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise RuntimeError(f"Summary CSV manifest content hash is detached from {path}")
    if (
        int(manifest.get("seed", -1)) != int(row.get("seed", -2))
        or int(manifest.get("eval_budget", -1)) != 50
        or manifest.get("env_id") != "swm/ReacherDMControl-v0"
        or manifest.get("restore_spec") != "reacher_qpos_match_qpos_qvel"
    ):
        raise RuntimeError(f"Evaluation manifest task/seed/budget mismatch in {manifest_path}")
    metadata = manifest.get("dataset_metadata")
    expected_metadata = {
        "path": "data/upstream/reacher.h5",
        "format": "hdf5",
        "member_sha256": HDF5_SHA256,
    }
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise RuntimeError(f"Official native-HDF5 provenance mismatch in {manifest_path}")
    if (
        manifest.get("dataset_path") != "data/upstream/reacher.h5"
        or int(manifest.get("goal_offset", -1)) != 25
        or manifest.get("goal_indexing") != "upstream_lewm_end_exclusive"
        or int(manifest.get("effective_goal_offset", -1)) != 24
    ):
        raise RuntimeError(f"Historical goal-indexing provenance mismatch in {manifest_path}")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 500:
        raise RuntimeError(f"Expected 500 manifest pairs in {manifest_path}")
    if any(
        int(pair["goal_step"]) - int(pair["start_step"]) != 24
        or int(pair["goal_row"]) - int(pair["start_row"]) != 24
        for pair in pairs
    ):
        raise RuntimeError(f"Manifest does not use effective start+24 goals in {manifest_path}")
    evaluation_data = {
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
    return runtime, normalized, evaluation_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a candidate-only Reacher N=500 parity sweep.")
    parser.add_argument("--report-tag", required=True)
    parser.add_argument("--candidate-role", required=True)
    args = parser.parse_args()

    paired = []
    for seed in SEEDS:
        candidate_summary = ROOT / "rollouts" / f"{args.report_tag}_seed{seed}_n500" / "summary.csv"
        upstream_summary = candidate_summary
        upstream = _row(candidate_summary, "upstream_lewm_converted")
        candidate = _row(candidate_summary, args.candidate_role)
        upstream_runtime, upstream_planner, upstream_data = _eval_provenance(upstream)
        candidate_runtime, candidate_planner, candidate_data = _eval_provenance(candidate)
        if upstream_runtime != candidate_runtime:
            raise RuntimeError(f"Environment runtime mismatch for seed {seed}")
        if upstream_planner != candidate_planner:
            raise RuntimeError(f"Planner parameter mismatch for seed {seed}")
        if upstream_data != candidate_data:
            raise RuntimeError(f"Evaluation data/goal provenance mismatch for seed {seed}")
        for row, label in ((upstream, "upstream"), (candidate, "candidate")):
            if int(row["episodes"]) != 500 or int(row["seed"]) != seed:
                raise RuntimeError(f"{label} seed/episode mismatch for seed {seed}")
        if upstream["manifest_sha256"] != candidate["manifest_sha256"]:
            raise RuntimeError(f"Manifest content mismatch for seed {seed}")
        if upstream["manifest_file_sha256"] != candidate["manifest_file_sha256"]:
            raise RuntimeError(f"Manifest file mismatch for seed {seed}")
        upstream_rate = float(upstream["success_rate"])
        candidate_rate = float(candidate["success_rate"])
        paired.append(
            {
                "seed": seed,
                "episodes": 500,
                "upstream_success_rate": upstream_rate,
                "candidate_success_rate": candidate_rate,
                "candidate_minus_upstream": candidate_rate - upstream_rate,
                "manifest_sha256": candidate["manifest_sha256"],
                "manifest_file_sha256": candidate["manifest_file_sha256"],
                "upstream_summary_csv": str(upstream_summary.relative_to(ROOT)),
                "candidate_summary_csv": str(candidate_summary.relative_to(ROOT)),
                "env_runtime": candidate_runtime,
                "planner_params": candidate_planner,
                "evaluation_data": candidate_data,
            }
        )

    upstream_rates = [row["upstream_success_rate"] for row in paired]
    candidate_rates = [row["candidate_success_rate"] for row in paired]
    deltas = [row["candidate_minus_upstream"] for row in paired]
    runtime_values = {json.dumps(row["env_runtime"], sort_keys=True) for row in paired}
    if len(runtime_values) != 1:
        raise RuntimeError(f"Environment runtime changed across seeds: {sorted(runtime_values)}")
    planner_values = {json.dumps(row["planner_params"], sort_keys=True) for row in paired}
    if len(planner_values) != 1:
        raise RuntimeError(f"Planner parameters changed across seeds: {sorted(planner_values)}")
    data_values = {json.dumps(row["evaluation_data"], sort_keys=True) for row in paired}
    if len(data_values) != 1:
        raise RuntimeError(f"Evaluation data/goal provenance changed across seeds: {sorted(data_values)}")
    payload = {
        "status": "pass",
        "environment": "swm/ReacherDMControl-v0",
        "episodes_per_seed": 500,
        "seeds": list(SEEDS),
        "candidate_role": args.candidate_role,
        "environment_runtime": paired[0]["env_runtime"],
        "planner_params": paired[0]["planner_params"],
        "evaluation_data": paired[0]["evaluation_data"],
        "paired_results": paired,
        "aggregate": {
            "upstream_mean_success_rate": mean(upstream_rates),
            "upstream_sample_stdev": stdev(upstream_rates),
            "candidate_mean_success_rate": mean(candidate_rates),
            "candidate_sample_stdev": stdev(candidate_rates),
            "mean_paired_delta": mean(deltas),
            "paired_delta_sample_stdev": stdev(deltas),
        },
    }
    report_root = ROOT / "reports" / "research" / args.report_tag
    report_root.mkdir(parents=True, exist_ok=True)
    destination = report_root / "n500_summary.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
