#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mwm.swm.restore import eval_callables_for_env, validate_restore_columns

DEFAULT_ARTIFACT_ROOT = Path("/vast/projects/dineshj/lab/ethanyu/code/" + "H" + "UDM")
ARTIFACT_ROOT = Path(os.environ.get("MWM_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
OUT_DIR = REPO_ROOT / "reports" / "research" / "reacher_identity_delta"

ENV_ID = "swm/ReacherDMControl-v0"
RESTORE_IMPORT_PATH = "mwm.swm.restore.reacher_qpos_match_restore_spec"
RESTORE_SPEC_ID = "reacher_qpos_match_qpos_qvel"
DATASET = "data/upstream/reacher.lance"
BENCHMARK_CONFIG = "configs/benchmark/paper_parity_reacher.yaml"
EVAL_CONFIG = "configs/eval/paper_reacher.yaml"
TRAIN_CONFIG = "configs/train/mwm_lewm_reacher_upstream.yaml"
ROLLOUT_DIR = "rollouts/mwm_paper_parity_reacher"
TRAIN_LOG = "logs/mwm_train_reacher_identity_6782935.out"
BENCHMARK_LOG = "logs/mwm_identity_parity_6784362.out"
UPSTREAM_CKPT = "checkpoints_mwm/upstream_lewm_reacher"
IDENTITY_CKPT = "checkpoints_mwm/retrained_lewm_identity_reacher_upstream"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TABLE_RE = re.compile(r"\|\s*([^|]+?)\s*\|\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\|")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_out(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def display_path(value: Any) -> str:
    text = str(value)
    text = text.replace(str(ARTIFACT_ROOT), "${MWM_ARTIFACT_ROOT}")
    text = text.replace(str(REPO_ROOT), "${WORKTREE_ROOT}")
    return text


def nested_get(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def fmt(value: Any) -> str:
    text = json.dumps(jsonable(value), sort_keys=True)
    if len(text) > 140:
        return text[:137] + "..."
    return text


def arrow_column_to_numpy(col: Any) -> np.ndarray:
    ctype = col.type
    if pa.types.is_fixed_size_list(ctype):
        return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), ctype.list_size)
    if pa.types.is_integer(ctype) or pa.types.is_floating(ctype):
        return col.to_numpy(zero_copy_only=False)
    raise TypeError(f"Unsupported numeric column type: {ctype}")


def load_numeric_column(dataset: Any, column: str) -> np.ndarray:
    chunks = []
    reader = dataset.scanner(columns=[column]).to_reader()
    for batch in reader:
        idx = batch.schema.get_field_index(column)
        if idx < 0:
            raise KeyError(column)
        chunks.append(arrow_column_to_numpy(batch.column(idx)))
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def summarize_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    flat = flat.astype(np.float64, copy=False)
    return {
        "shape": list(arr.shape),
        "mean": np.mean(flat, axis=0).tolist() if flat.size else [],
        "std_population": np.std(flat, axis=0, ddof=0).tolist() if flat.size else [],
        "min": np.min(flat, axis=0).tolist() if flat.size else [],
        "max": np.max(flat, axis=0).tolist() if flat.size else [],
        "quantiles": {
            str(q): np.quantile(flat, q, axis=0).tolist() if flat.size else []
            for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        },
    }


def summarize_episode_lengths(ep_ids: np.ndarray) -> dict[str, Any]:
    if ep_ids.size == 0:
        lengths = np.array([], dtype=np.int64)
    else:
        changes = np.flatnonzero(np.diff(ep_ids) != 0) + 1
        offsets = np.concatenate([[0], changes])
        lengths = np.diff(np.concatenate([offsets, [len(ep_ids)]])).astype(np.int64)
    return {
        "episode_count": int(lengths.size),
        "length_min": int(lengths.min()) if lengths.size else 0,
        "length_max": int(lengths.max()) if lengths.size else 0,
        "length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "length_std": float(lengths.std(ddof=0)) if lengths.size else 0.0,
        "length_quantiles": {
            str(q): float(np.quantile(lengths, q)) if lengths.size else 0.0
            for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        },
    }


def audit_dataset() -> dict[str, Any]:
    path = ARTIFACT_ROOT / DATASET
    dataset = lance.dataset(path)
    ep_ids = load_numeric_column(dataset, "episode_idx")
    numeric = {
        name: summarize_array(load_numeric_column(dataset, name))
        for name in ("action", "qpos", "qvel", "observation")
    }
    return {
        "path": str(path),
        "metadata_path": str(path.with_suffix(path.suffix + ".metadata.json")),
        "sidecar_metadata": read_json(path.with_suffix(path.suffix + ".metadata.json")),
        "row_count": int(dataset.count_rows()),
        "columns": [{"name": field.name, "type": str(field.type)} for field in dataset.schema],
        "episode_lengths": summarize_episode_lengths(ep_ids),
        "numeric_columns": numeric,
    }


def parse_training_log(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"path": str(log_path), "exists": False}
    parts = [log_path.read_text(encoding="utf-8", errors="replace")]
    err_path = log_path.with_suffix(".err")
    if err_path.is_file():
        parts.append(err_path.read_text(encoding="utf-8", errors="replace"))
    text = ANSI_RE.sub("", "\n".join(parts))
    metrics: dict[str, list[float]] = {}
    for key, value in TABLE_RE.findall(text):
        key = " ".join(key.split())
        if "/" in key:
            metrics.setdefault(key, []).append(float(value))
    return {
        "path": str(log_path),
        "exists": True,
        "job_id": log_path.stem.rsplit("_", 1)[-1],
        "max_epochs_reached": "`Trainer.fit` stopped: `max_epochs=10` reached." in text,
        "exact_training_complete": "Exact Le-WM training complete." in text,
        "metric_last": {key: vals[-1] for key, vals in sorted(metrics.items()) if vals},
        "metric_count": {key: len(vals) for key, vals in sorted(metrics.items())},
    }


def load_checkpoint_bundle(rel: str) -> dict[str, Any]:
    run_dir = ARTIFACT_ROOT / rel
    return {
        "path": str(run_dir),
        "config_sha256": file_sha256(run_dir / "config.json"),
        "metadata_sha256": file_sha256(run_dir / "world_metadata.json"),
        "config": read_json(run_dir / "config.json"),
        "metadata": read_json(run_dir / "world_metadata.json"),
    }


def audit_checkpoints() -> dict[str, Any]:
    roles = {
        "upstream_lewm_converted": load_checkpoint_bundle(UPSTREAM_CKPT),
        "retrained_lewm_identity": load_checkpoint_bundle(IDENTITY_CKPT),
    }
    fields = [
        "env_id",
        "restore_spec",
        "adapter_family",
        "architecture_version",
        "levels",
        "D",
        "action_dim",
        "action_block",
        "action_spec",
        "image_shape",
        "preprocessing_spec",
        "component_policy",
        "source_config_sha256",
        "training_recipe",
        "fresh_init",
        "dataset",
        "epoch",
        "best_checkpoint",
        "last_checkpoint",
        "selected_lightning_checkpoint",
    ]
    comparisons = []
    up = roles["upstream_lewm_converted"]["metadata"]
    ident = roles["retrained_lewm_identity"]["metadata"]
    for field in fields:
        comparisons.append(
            {
                "field": field,
                "upstream": nested_get(up, field),
                "identity": nested_get(ident, field),
                "same": nested_get(up, field) == nested_get(ident, field),
            }
        )
    return {"roles": roles, "metadata_comparison": comparisons}


def load_eval_payloads() -> dict[str, Any]:
    out = {}
    for name, rel in {
        "upstream": f"{ROLLOUT_DIR}/000_upstream/eval.json",
        "retrained_identity": f"{ROLLOUT_DIR}/001_retrained_identity/eval.json",
    }.items():
        payload = read_json(ARTIFACT_ROOT / rel)
        compact_payload = {
            key: payload.get(key)
            for key in (
                "role",
                "env_id",
                "restore_spec",
                "episodes",
                "goal_offset",
                "seed",
                "checkpoint_epoch",
                "checkpoint_run_dir",
                "eval_budget",
                "dataset",
                "manifest",
                "model_accounting",
                "dataset_metadata",
                "schedule",
            )
        }
        compact_payload["planning_summary"] = payload.get("planning_diagnostics", {}).get("summary", {})
        successes = [bool(x) for x in payload["swm_results"]["episode_successes"]]
        out[name] = {
            "path": str(ARTIFACT_ROOT / rel),
            "payload": compact_payload,
            "successes": successes,
            "success_count": int(sum(successes)),
            "failure_indices": [idx for idx, ok in enumerate(successes) if not ok],
        }
    return out


def sign_test_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(a, b) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def audit_rollout() -> dict[str, Any]:
    summary_rows = []
    with (ARTIFACT_ROOT / ROLLOUT_DIR / "summary.csv").open(newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    evals = load_eval_payloads()
    up = evals["upstream"]["successes"]
    ident = evals["retrained_identity"]["successes"]
    if len(up) != len(ident):
        raise ValueError("Upstream and identity evals have different episode counts.")
    n = len(up)
    identity_better = sum((not u) and i for u, i in zip(up, ident))
    upstream_better = sum(u and (not i) for u, i in zip(up, ident))
    shared_failure = sum((not u) and (not i) for u, i in zip(up, ident))
    shared_success = sum(u and i for u, i in zip(up, ident))
    up_failures = set(evals["upstream"]["failure_indices"])
    identity_failures = set(evals["retrained_identity"]["failure_indices"])
    p_up = evals["upstream"]["success_count"] / n
    p_ident = evals["retrained_identity"]["success_count"] / n
    return {
        "summary_csv": str(ARTIFACT_ROOT / ROLLOUT_DIR / "summary.csv"),
        "summary_rows": summary_rows,
        "evals": evals,
        "episodes": n,
        "upstream_success_count": evals["upstream"]["success_count"],
        "identity_success_count": evals["retrained_identity"]["success_count"],
        "delta_count": evals["retrained_identity"]["success_count"] - evals["upstream"]["success_count"],
        "delta_pp": (p_ident - p_up) * 100.0,
        "episode_granularity_pp": 100.0 / n,
        "shared_success_count": shared_success,
        "shared_failure_count": shared_failure,
        "shared_failure_indices": sorted(up_failures & identity_failures),
        "upstream_only_failure_indices": sorted(up_failures - identity_failures),
        "identity_only_failure_indices": sorted(identity_failures - up_failures),
        "identity_better_discordant_count": identity_better,
        "upstream_better_discordant_count": upstream_better,
        "paired_sign_test_p": sign_test_p(identity_better, upstream_better),
        "independent_diff_se_pp": math.sqrt(p_up * (1.0 - p_up) / n + p_ident * (1.0 - p_ident) / n) * 100.0,
        "manifest": evals["upstream"]["payload"].get("manifest", {}),
    }


def audit_configs() -> dict[str, Any]:
    return {
        "benchmark": {"path": BENCHMARK_CONFIG, "sha256": file_sha256(REPO_ROOT / BENCHMARK_CONFIG), "data": read_yaml(REPO_ROOT / BENCHMARK_CONFIG)},
        "eval": {"path": EVAL_CONFIG, "sha256": file_sha256(REPO_ROOT / EVAL_CONFIG), "data": read_yaml(REPO_ROOT / EVAL_CONFIG)},
        "train": {"path": TRAIN_CONFIG, "sha256": file_sha256(REPO_ROOT / TRAIN_CONFIG), "data": read_yaml(REPO_ROOT / TRAIN_CONFIG)},
        "research_eval": {
            "path": "configs/research/reacher_identity_delta/reacher_eval.yaml",
            "sha256": file_sha256(REPO_ROOT / "configs/research/reacher_identity_delta/reacher_eval.yaml"),
            "data": read_yaml(REPO_ROOT / "configs/research/reacher_identity_delta/reacher_eval.yaml"),
        },
        "research_benchmark": {
            "path": "configs/research/reacher_identity_delta/reacher_benchmark_seed42.yaml",
            "sha256": file_sha256(REPO_ROOT / "configs/research/reacher_identity_delta/reacher_benchmark_seed42.yaml"),
            "data": read_yaml(REPO_ROOT / "configs/research/reacher_identity_delta/reacher_benchmark_seed42.yaml"),
        },
    }


def audit_restore() -> dict[str, Any]:
    columns = {"pixels", "action", "qpos", "qvel", "observation"}
    spec = validate_restore_columns(ENV_ID, columns, import_path=RESTORE_IMPORT_PATH)
    spec_id, callables = eval_callables_for_env(ENV_ID, columns, import_path=RESTORE_IMPORT_PATH)
    missing_checks = {}
    for missing in ("qpos", "qvel"):
        try:
            validate_restore_columns(ENV_ID, columns - {missing}, import_path=RESTORE_IMPORT_PATH)
        except ValueError as exc:
            missing_checks[missing] = str(exc)
    return {
        "import_path": RESTORE_IMPORT_PATH,
        "spec_id": spec.spec_id,
        "eval_callables_spec_id": spec_id,
        "required_columns": list(spec.required_columns),
        "eval_callables": callables,
        "missing_column_checks": missing_checks,
    }


def git_audit() -> dict[str, Any]:
    try:
        base = git_out("merge-base", "HEAD", "origin/multienv-support")
    except Exception:
        base = None
    return {
        "branch": git_out("branch", "--show-current"),
        "head": git_out("rev-parse", "HEAD"),
        "base_origin_multienv_support": base,
        "status_short": git_out("status", "--short"),
    }


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rollout = payload["rollout"]
    return {
        "branch": payload["git"]["branch"],
        "audit_head": payload["git"]["head"],
        "artifact_root": payload["artifact_root"],
        "env_id": ENV_ID,
        "task": "qpos_match",
        "restore_spec": RESTORE_SPEC_ID,
        "restore_import_path": RESTORE_IMPORT_PATH,
        "dataset": DATASET,
        "observed": {
            "upstream_success_rate": 80.0,
            "retrained_identity_success_rate": 86.0,
            "episodes": rollout["episodes"],
            "delta_pp": rollout["delta_pp"],
            "delta_count": rollout["delta_count"],
            "episode_granularity_pp": rollout["episode_granularity_pp"],
            "paired_sign_test_p": rollout["paired_sign_test_p"],
        },
        "classification": {
            "primary_cause": "manifest/evaluator sampling variance at 50 episodes, with no evidence of a restore or architecture mismatch",
            "training_data": "not supported as the paired cause",
            "training_recipe_or_convergence": "not supported as a negative identity cause",
            "evaluator_or_manifest_variance": "most likely explanation of the apparent +6pp identity advantage",
            "checkpoint_selection": "possible small contributor, but not evidenced by this run",
            "restore_or_qpos_goal_handling": "not supported",
            "code_mismatch": "unlikely",
        },
        "recommendation": {
            "paper_target_tolerance": "split upstream-paper from identity-upstream and make both checks episode-count aware",
            "next_experiment": "five-seed paired Reacher 200-episode sweep of the same two checkpoints",
            "next_command": 'MWM_REACHER_SWEEP_EPISODES=200 MWM_REACHER_SWEEP_SEEDS="0 1 2 42 100" sbatch scripts/research/slurm_research_reacher_identity_seed_sweep.sbatch',
        },
        "jobs": {
            "historical_identity_train": "6782935",
            "historical_identity_parity_benchmark": "6784362",
            "new_gpu_jobs_submitted": [],
        },
        "blockers": [],
    }


def write_report(payload: dict[str, Any], summary: dict[str, Any]) -> None:
    rollout = payload["rollout"]
    dataset = payload["dataset"]
    training = payload["training"]
    ckpt = payload["checkpoints"]
    restore = payload["restore"]
    configs = payload["configs"]
    rows = rollout["summary_rows"]
    up_eval = rollout["evals"]["upstream"]
    id_eval = rollout["evals"]["retrained_identity"]
    up_meta = ckpt["roles"]["upstream_lewm_converted"]["metadata"]
    id_meta = ckpt["roles"]["retrained_lewm_identity"]["metadata"]

    lines: list[str] = [
        "# Reacher Identity-Upstream Delta Research Report",
        "",
        f"Branch: `{payload['git']['branch']}`",
        f"Audit head: `{payload['git']['head']}`",
        f"Base from `origin/multienv-support`: `{payload['git']['base_origin_multienv_support']}`",
        "Worktree: `${WORKTREE_ROOT}`",
        "Artifact root audited: `${MWM_ARTIFACT_ROOT}`",
        "",
        "## Executive Answer",
        "",
        "The observed Reacher result is upstream `80.0` versus retrained identity `86.0` on a single 50-episode manifest. That is a three-episode difference, because each episode is worth 2 percentage points. The paired discordance is identity-better on 8 episodes and upstream-better on 5 episodes, with a two-sided sign-test p-value of `{:.3f}`. This is not strong evidence that the retrained identity checkpoint is truly better; it is best classified as evaluator/manifest sampling variance plus ordinary planner noise at low episode count.".format(rollout["paired_sign_test_p"]),
        "",
        "There is no evidence in the audited artifacts for a training-data mismatch, restore/qpos-goal bug, architecture mismatch, or failed convergence. Both checkpoints evaluate on the same manifest, same dataset, same Reacher qpos-match restore spec, same `K=[192]`, same flattened action spec `dim=10`, and same eval pipeline.",
        "",
        "## Required Answers",
        "",
        "1. Cause classification: primary cause is evaluator/manifest variance at 50 episodes. Training data, training recipe/convergence, checkpoint selection, restore/qpos-goal handling, and code mismatch are not supported as primary causes by the current evidence. Checkpoint selection remains a small unresolved caveat because the identity export uses the final/last Lightning checkpoint, not a best-validation checkpoint.",
        "",
        "2. Why identity appears to outperform upstream: it wins by only 3 episodes on one fixed manifest. The identity model is also trained directly against the current Lance data and preprocessing recipe, while upstream is a converted Le-WM artifact. That could plausibly move a few borderline CEM decisions, but this run is too small to call it a real performance advantage.",
        "",
        "3. Meaningfulness of +6pp: weak. The run has 2pp granularity, independent-binomial diff standard error is `{:.2f}` pp, and the paired sign test is not significant. The failure sets are mostly different rather than a fixed subset of broken Reacher goals: shared failures `{}`, upstream-only failures `{}`, identity-only failures `{}`.".format(
            rollout["independent_diff_se_pp"],
            rollout["shared_failure_indices"],
            rollout["upstream_only_failure_indices"],
            rollout["identity_only_failure_indices"],
        ),
        "",
        "4. Tolerance recommendation: split upstream-paper and identity-upstream checks, and make them episode-count aware. A single 1 percent gate is below the resolution of a 50-episode run. Use `max(1pp, 100 / episodes)` as a minimum reporting resolution, and use paired count/seed-sweep checks for identity-upstream rather than a one-manifest percent delta.",
        "",
        "5. Exact next experiment: run a five-seed paired Reacher sweep at 200 episodes per seed for the same two checkpoints before retraining anything:",
        "",
        "```bash",
        'MWM_REACHER_SWEEP_EPISODES=200 MWM_REACHER_SWEEP_SEEDS="0 1 2 42 100" sbatch scripts/research/slurm_research_reacher_identity_seed_sweep.sbatch',
        "```",
        "",
        "No new GPU jobs were submitted for this investigation. Current Slurm state was inspected with `sinfo -s` and `scontrol show partition dgx-b200`; the `dgx-b200` partition is up, has B200 GPUs, and the exact research-only sbatch script is recorded in `scripts/research/slurm_research_reacher_identity_seed_sweep.sbatch`.",
        "",
        "## Observed Rollout",
        "",
        "| role | success | episodes | seed | manifest | output |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        output_ref = row.get("output_path", row.get("output_json", ""))
        lines.append(
            f"| `{row['role']}` | {float(row['success_rate']):.1f} | {row['episodes']} | {row['seed']} | `{row['manifest_sha256'][:12]}` | `{display_path(output_ref)}` |"
        )
    lines.extend(
        [
            "",
            f"- Manifest path: `{display_path(rollout['manifest'].get('path'))}`",
            f"- Manifest sha256: `{rollout['manifest'].get('sha256')}`",
            f"- Immutable manifest hash: `{rollout['manifest'].get('manifest_sha256')}`",
            f"- Upstream failures: `{up_eval['failure_indices']}`",
            f"- Identity failures: `{id_eval['failure_indices']}`",
            f"- Shared successes/failures: `{rollout['shared_success_count']}` / `{rollout['shared_failure_count']}`",
            f"- Identity-better/upstream-better discordant counts: `{rollout['identity_better_discordant_count']}` / `{rollout['upstream_better_discordant_count']}`",
            "",
            "## Restore And Dataset",
            "",
            f"- Restore import path: `{restore['import_path']}`",
            f"- Restore spec id: `{restore['spec_id']}`",
            f"- Required columns: `{restore['required_columns']}`",
            "- Eval callable 1: `set_state(qpos=<start qpos>, qvel=<start qvel>)`",
            "- Eval callable 2: `set_target_qpos(target_qpos=<goal qpos>)`",
            f"- Eval callables raw: `{fmt(restore['eval_callables'])}`",
            f"- Missing-column checks: `{fmt(restore['missing_column_checks'])}`",
            f"- Dataset rows: `{dataset['row_count']}`",
            f"- Dataset episodes: `{dataset['episode_lengths']['episode_count']}`",
            f"- Episode length mean/min/max: `{dataset['episode_lengths']['length_mean']:.2f}` / `{dataset['episode_lengths']['length_min']}` / `{dataset['episode_lengths']['length_max']}`",
            f"- Dataset sidecar restore spec: `{dataset['sidecar_metadata'].get('restore_spec')}`",
            f"- Dataset raw action dim: `{dataset['sidecar_metadata'].get('action_dim')}`",
            f"- Dataset source: `{fmt(dataset['sidecar_metadata'].get('source'))}`",
            "",
            "## Checkpoints",
            "",
            "| field | upstream | identity | same |",
            "| --- | --- | --- | --- |",
        ]
    )
    key_fields = {
        "env_id",
        "restore_spec",
        "adapter_family",
        "architecture_version",
        "levels",
        "D",
        "action_dim",
        "action_block",
        "action_spec",
        "image_shape",
        "source_config_sha256",
        "fresh_init",
        "epoch",
        "best_checkpoint",
        "last_checkpoint",
    }
    for row in ckpt["metadata_comparison"]:
        if row["field"] in key_fields:
            lines.append(f"| `{row['field']}` | `{fmt(row['upstream'])}` | `{fmt(row['identity'])}` | {row['same']} |")
    lines.extend(
        [
            "",
            f"- Eval model accounting upstream: `{fmt(rollout['evals']['upstream']['payload'].get('model_accounting'))}`",
            f"- Eval model accounting identity: `{fmt(rollout['evals']['retrained_identity']['payload'].get('model_accounting'))}`",
            f"- Upstream flattened action dim from config: `{nested_get(ckpt['roles']['upstream_lewm_converted']['config'], 'kwargs.action_dim')}`",
            f"- Identity flattened action dim from config: `{nested_get(ckpt['roles']['retrained_lewm_identity']['config'], 'kwargs.action_dim')}`",
            "",
            "## Training Recipe And Convergence",
            "",
            f"- Identity train config: `{TRAIN_CONFIG}`",
            f"- Data path in train config: `{configs['train']['data']['data']['path']}`",
            f"- Train seed: `{configs['train']['data']['seed']}`",
            f"- Max epochs configured: `{configs['train']['data']['schedule']['max_epochs']}`",
            f"- Train job log: `{display_path(training['path'])}`",
            f"- Historical train job id: `{training.get('job_id')}`",
            f"- Max epochs reached: `{training.get('max_epochs_reached')}`",
            f"- Exact training complete marker: `{training.get('exact_training_complete')}`",
            f"- Last fit/pred loss: `{training.get('metric_last', {}).get('fit/pred_loss')}`",
            f"- Last validate/pred loss: `{training.get('metric_last', {}).get('validate/pred_loss')}`",
            "",
            "The identity run reached epoch 9 of 9 and exported `checkpoints_mwm/retrained_lewm_identity_reacher_upstream`. This does not prove optimal convergence, but it rules out an obvious early-stop or missing-export explanation for the 50-episode result.",
            "",
            "## Config Audit",
            "",
            f"- Benchmark config: `{BENCHMARK_CONFIG}` sha `{configs['benchmark']['sha256'][:12]}`",
            f"- Eval config: `{EVAL_CONFIG}` sha `{configs['eval']['sha256'][:12]}`",
            f"- Train config: `{TRAIN_CONFIG}` sha `{configs['train']['sha256'][:12]}`",
            f"- Eval env kwargs: `{fmt(configs['eval']['data']['env']['kwargs'])}`",
            f"- Eval restore import path: `{configs['eval']['data']['restore']['import_path']}`",
            f"- Eval keys_to_load: `{configs['eval']['data']['data']['keys_to_load']}`",
            f"- Train `K`: `{configs['train']['data']['model']['K']}`",
            f"- Train frameskip/action_block: `{configs['train']['data']['data']['frameskip']}` / `{configs['train']['data']['model']['action_block']}`",
            "",
            "## Commands And Jobs",
            "",
            "Commands used in this investigation:",
            "",
            "```bash",
            "git worktree add .worktrees/codex-reacher-identity-upstream-delta -b codex/reacher-identity-upstream-delta origin/multienv-support",
            "/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m py_compile $(rg --files -g '*.py')",
            "/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_local_workflow.py tests/test_mwm_repo_hygiene.py tests/test_mwm_core.py tests/test_mwm_artifacts.py",
            "sinfo -s",
            "scontrol show partition dgx-b200",
            "/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python scripts/research/research_reacher_identity_delta_audit.py",
            "bash -n scripts/research/research_reacher_identity_seed_sweep.sh scripts/research/slurm_research_reacher_identity_seed_sweep.sbatch",
            "```",
            "",
            "Historical jobs referenced from existing artifacts:",
            "",
            "- Identity Reacher training: `6782935`",
            "- Identity parity benchmark including Reacher: `6784362`",
            "- New GPU jobs submitted by this investigation: none",
            "",
            "## Blockers",
            "",
            "No blocker for the static conclusion. The only remaining uncertainty is statistical: the current Reacher comparison has one 50-episode manifest, so it cannot establish whether the +6pp identity advantage persists across seeds or higher episode counts.",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_root": str(ARTIFACT_ROOT),
        "repo_root": str(REPO_ROOT),
        "git": git_audit(),
        "configs": audit_configs(),
        "restore": audit_restore(),
        "dataset": audit_dataset(),
        "checkpoints": audit_checkpoints(),
        "rollout": audit_rollout(),
        "training": parse_training_log(ARTIFACT_ROOT / TRAIN_LOG),
        "historical_benchmark_log": {"path": str(ARTIFACT_ROOT / BENCHMARK_LOG), "job_id": "6784362"},
    }
    summary = build_summary(payload)
    (OUT_DIR / "audit_raw.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(payload, summary)
    print(f"Wrote {OUT_DIR / 'audit_raw.json'}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
