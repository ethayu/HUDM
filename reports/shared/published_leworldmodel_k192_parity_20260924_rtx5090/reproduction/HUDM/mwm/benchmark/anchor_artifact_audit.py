"""Fail-closed provenance and accounting audit for five-anchor results.

The five-anchor result table is intentionally small, but some rows may point
at a canonical 250-episode artifact while others point at a dedicated
50-episode artifact.  This module validates the *referenced artifacts* rather
than trusting the derived table.  It supports both legacy JSON and the
capsule-plus-archive representation and materializes each unique eval archive
at most once per audit invocation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

from mwm.benchmark.anchor_diagnostic import ANCHORS, SCHEMA_VERSION as RESULTS_SCHEMA_VERSION
from mwm.benchmark.config import DEFAULTS, checkpoint_mapping, safe_name
from mwm.benchmark.eval_artifacts import (
    PLANNING_REF_SCHEMA,
    PLANNING_REF_VERSION,
    load_eval_artifact,
)
from mwm.benchmark.summary import eval_summary_row
from mwm.benchmark.sweep import expand_benchmark_runs
from mwm.config_cli import load_config
from mwm.data.manifest import load_manifest, manifest_file_sha256
from mwm.fidelity import FidelityScheduler
from mwm.io import file_sha256, load_json


SCHEMA_VERSION = "mwm.release20260728.anchor_artifact_audit/v2"
DEFAULT_INPUT = Path(
    "reports/research/release20260728_schedule_screening/five_anchor_results/results.json"
)
PAIR_FIELDS = ("episode", "start_step", "goal_step", "start_row", "goal_row")
SOURCE_EPISODES = {"canonical_250_prefix": 250, "screen50": 50}


class AnchorArtifactAuditError(RuntimeError):
    """Raised after a strict audit finds one or more invariant violations."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class _MatrixExpectation:
    canonical_config: Path
    manifest_path: Path
    manifest: dict[str, Any]
    runs: dict[int, Any]


@dataclass(frozen=True)
class _SourceArtifact:
    run_dir: Path
    payload: dict[str, Any]
    config: dict[str, Any]
    manifest_path: Path
    manifest: dict[str, Any]
    metric: dict[str, Any]
    episode_traces: list[dict[str, Any]]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=True,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(dict(value))
    return rows


def _plain_config(path: Path) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a configuration mapping")
    return dict(value)


def _pair_identity(value: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple(int(value[field]) for field in PAIR_FIELDS)  # type: ignore[return-value]


def _trace_identity(value: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(value["dataset_episode"]),
        int(value["start_step"]),
        int(value["goal_step"]),
    )


def _close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _benchmark_manifest_path(root: Path, cfg: Any) -> Path:
    raw = OmegaConf.to_container(cfg.get("manifest", {}), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("benchmark manifest configuration is not a mapping")
    merged = dict(raw)
    config_ref = merged.pop("config", None)
    if config_ref is not None:
        manifest_cfg_path = _resolve(root, str(config_ref))
        manifest_cfg = OmegaConf.to_container(OmegaConf.load(manifest_cfg_path), resolve=True)
        if not isinstance(manifest_cfg, dict):
            raise ValueError(f"{manifest_cfg_path} is not a mapping")
        merged = {**manifest_cfg, **merged}
    if "path" not in merged:
        group = str(merged.get("group", "")).strip()
        if not group:
            raise ValueError("benchmark manifest config has neither path nor group")
        directory = Path(str(merged.get("dir", "rollouts/manifests")))
        merged["path"] = directory / f"{safe_name(group)}_seed{int(cfg.seed)}.json"
    return _resolve(root, str(merged["path"]))


def _row_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('canonical_config', '?')}[{row.get('matrix_index', '?')}]"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_planning_sidecar(run_dir: Path, diagnostics: dict[str, Any]) -> None:
    sidecar_path = run_dir / "planning_diagnostics.json"
    sidecar = load_json(sidecar_path)
    marker = sidecar.get("_artifact")
    if isinstance(marker, dict) and marker.get("schema") == PLANNING_REF_SCHEMA:
        _check(
            int(marker.get("version", -1)) == PLANNING_REF_VERSION,
            f"{sidecar_path}: unsupported planning ref version",
        )
        _check(marker.get("eval_path") == "eval.json", f"{sidecar_path}: unsafe eval reference")
        _check(
            marker.get("json_pointer") == "/planning_diagnostics",
            f"{sidecar_path}: wrong JSON pointer",
        )
        _check(
            str(marker.get("section_sha256", "")) == _sha256_json(diagnostics),
            f"{sidecar_path}: section SHA-256 differs from the materialized diagnostics",
        )
        aggregate = {key: value for key, value in diagnostics.items() if key != "trace"}
        observed_aggregate = {
            key: value for key, value in sidecar.items() if key not in {"_artifact", "trace"}
        }
        _check(
            observed_aggregate == aggregate,
            f"{sidecar_path}: aggregate values differ from the materialized diagnostics",
        )
        trace_ref = sidecar.get("trace")
        _check(isinstance(trace_ref, dict), f"{sidecar_path}: trace reference is missing")
        _check(
            int(trace_ref.get("length", -1)) == len(diagnostics.get("trace", [])),
            f"{sidecar_path}: trace reference length mismatch",
        )
        return
    _check(sidecar == diagnostics, f"{sidecar_path}: legacy sidecar differs from eval.json")


def _load_matrix_expectation(
    root: Path,
    canonical_config: str,
) -> _MatrixExpectation:
    config_path = _resolve(root, canonical_config)
    cfg = load_config(DEFAULTS, config_path)
    runs = {int(run.matrix_index): run for run in expand_benchmark_runs(cfg)}
    manifest_path = _benchmark_manifest_path(root, cfg)
    manifest = load_manifest(manifest_path)
    return _MatrixExpectation(config_path, manifest_path, manifest, runs)


def _load_source(
    root: Path,
    run_dir: Path,
) -> _SourceArtifact:
    required = (
        "eval.json",
        "resolved_config.yaml",
        "metrics.jsonl",
        "summary.json",
        "planning_diagnostics.json",
        "episode_traces.jsonl",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    _check(not missing, f"{run_dir}: missing completion sidecars {missing}")

    # This is deliberately the only full eval load in this function.  For a
    # compact artifact it verifies compressed and uncompressed hashes, archive
    # JSON, section hashes, and the reconstructed scientific-payload hash.
    payload = load_eval_artifact(run_dir / "eval.json", verify="full")
    _check(isinstance(payload, dict), f"{run_dir}/eval.json is not an object")
    diagnostics = payload.get("planning_diagnostics")
    _check(isinstance(diagnostics, dict), f"{run_dir}: missing planning diagnostics")
    _verify_planning_sidecar(run_dir, diagnostics)

    config_path = run_dir / "resolved_config.yaml"
    config = _plain_config(config_path)
    config_info = payload.get("config")
    _check(isinstance(config_info, dict), f"{run_dir}: missing config provenance")
    config_sha = file_sha256(config_path)
    _check(config_sha == str(config_info.get("sha256", "")), f"{run_dir}: config SHA-256 mismatch")
    referenced_config = _resolve(root, str(config_info.get("resolved_path", "")))
    _check(referenced_config == config_path.resolve(), f"{run_dir}: config path does not resolve to sidecar")

    manifest_info = payload.get("manifest")
    _check(isinstance(manifest_info, dict), f"{run_dir}: missing manifest provenance")
    manifest_path = _resolve(root, str(manifest_info.get("path", "")))
    _check(manifest_path.is_file(), f"{run_dir}: referenced manifest is missing")
    _check(
        manifest_file_sha256(manifest_path) == str(manifest_info.get("sha256", "")),
        f"{run_dir}: manifest file SHA-256 mismatch",
    )
    manifest = load_manifest(manifest_path)
    _check(
        str(manifest.get("manifest_sha256", ""))
        == str(manifest_info.get("manifest_sha256", "")),
        f"{run_dir}: semantic manifest SHA-256 mismatch",
    )

    metrics = _jsonl(run_dir / "metrics.jsonl")
    _check(len(metrics) == 1, f"{run_dir}: expected exactly one metrics row")
    metric = metrics[0]
    summary = load_json(run_dir / "summary.json")
    _check(summary.get("run") == metric, f"{run_dir}: summary and metrics sidecars differ")
    metric_output = str(metric.get("output_json", ""))
    _check(
        _resolve(root, metric_output) == (run_dir / "eval.json").resolve(),
        f"{run_dir}: metrics output_json points at another artifact",
    )
    regenerated = eval_summary_row(
        str(payload.get("benchmark_name", payload.get("cell_id", ""))),
        metric_output,
        payload,
    )
    _check(regenerated == metric, f"{run_dir}: metrics row does not regenerate from eval payload")

    episode_traces = _jsonl(run_dir / "episode_traces.jsonl")
    return _SourceArtifact(
        run_dir=run_dir,
        payload=payload,
        config=config,
        manifest_path=manifest_path,
        manifest=manifest,
        metric=metric,
        episode_traces=episode_traces,
    )


def _audit_matrix_identity(
    row: Mapping[str, Any],
    source: _SourceArtifact,
    expected: _MatrixExpectation,
) -> None:
    key = _row_key(row)
    index = int(row["matrix_index"])
    _check(index in expected.runs, f"{key}: matrix index does not exist")
    run = expected.runs[index]
    anchor_by_values = {
        (anchor.pop_size, anchor.elite_frac, anchor.n_iter): anchor for anchor in ANCHORS
    }
    values = (int(row["pop_size"]), float(row["elite_frac"]), int(row["n_iter"]))
    _check(values in anchor_by_values, f"{key}: row planner values are not one of the five anchors")
    anchor = anchor_by_values[values]
    _check(str(row["anchor_id"]) == anchor.anchor_id, f"{key}: anchor id/value mismatch")
    _check(str(row["anchor_label"]) == anchor.label, f"{key}: anchor label/value mismatch")
    _check(
        int(row["nominal_candidate_iterations"]) == anchor.nominal_candidate_iterations,
        f"{key}: nominal candidate-iteration mismatch",
    )
    expected_values = (
        int(run.planner.pop_size),
        float(run.planner.elite_frac),
        int(run.planner.n_iter),
    )
    _check(values == expected_values, f"{key}: row planner values differ from canonical matrix cell")
    _check(str(row["cell_id"]) == str(run.cell_id), f"{key}: cell id differs from canonical matrix")
    _check(str(row["schedule"]) == str(run.base_name), f"{key}: schedule id differs from canonical matrix")
    _check(
        str(row["schedule_label"]) == str(run.get("schedule", run.base_name)),
        f"{key}: schedule label differs from canonical matrix",
    )
    expected_dir_name = f"{index:03d}_{run.cell_id}"
    _check(source.run_dir.name == expected_dir_name, f"{key}: source directory/cell identity mismatch")

    payload = source.payload
    metric = source.metric
    _check(str(payload.get("cell_id", "")) == str(run.cell_id), f"{key}: payload cell id mismatch")
    _check(str(payload.get("benchmark_name", "")) == str(run.cell_id), f"{key}: payload benchmark name mismatch")
    _check(str(payload.get("base_name", "")) == str(run.base_name), f"{key}: payload base name mismatch")
    _check(str(payload.get("schedule", "")) == str(row["schedule_label"]), f"{key}: payload schedule mismatch")
    _check(str(metric.get("cell_id", "")) == str(run.cell_id), f"{key}: metrics cell id mismatch")
    _check(str(metric.get("base_name", "")) == str(run.base_name), f"{key}: metrics base name mismatch")
    _check(str(metric.get("schedule", "")) == str(row["schedule_label"]), f"{key}: metrics schedule mismatch")

    expected_role = str(run.get("role", run.base_name))
    _check(str(payload.get("role", "")) == expected_role, f"{key}: payload role differs from canonical matrix")
    _check(str(metric.get("role", "")) == expected_role, f"{key}: metrics role differs from canonical matrix")

    expected_checkpoint = checkpoint_mapping(run)
    resolved_checkpoint = source.config.get("checkpoint")
    _check(isinstance(resolved_checkpoint, dict), f"{key}: resolved checkpoint config is missing")
    expected_checkpoint_dir = str(expected_checkpoint["run_dir"])
    _check(
        str(resolved_checkpoint.get("run_dir", "")) == expected_checkpoint_dir,
        f"{key}: resolved checkpoint run_dir differs from canonical matrix",
    )
    _check(
        str(payload.get("checkpoint_run_dir", "")) == expected_checkpoint_dir,
        f"{key}: payload checkpoint run_dir differs from canonical matrix",
    )
    _check(
        str(metric.get("checkpoint_run_dir", "")) == expected_checkpoint_dir,
        f"{key}: metrics checkpoint run_dir differs from canonical matrix",
    )
    expected_epoch = expected_checkpoint.get("epoch")
    _check(
        resolved_checkpoint.get("epoch") == expected_epoch,
        f"{key}: resolved checkpoint epoch differs from canonical matrix",
    )
    payload_epoch = int(payload.get("checkpoint_epoch", -1))
    _check(payload_epoch >= 0, f"{key}: payload checkpoint epoch is missing")
    _check(int(metric.get("checkpoint_epoch", -1)) == payload_epoch, f"{key}: metrics checkpoint epoch mismatch")
    if expected_epoch is not None:
        _check(payload_epoch == int(expected_epoch), f"{key}: loaded checkpoint epoch differs from canonical matrix")

    planner = source.config.get("planner")
    _check(isinstance(planner, dict), f"{key}: resolved planner config is missing")
    expected_planner = run.get("planner")
    _check(expected_planner is not None, f"{key}: canonical run planner config is missing")
    expected_scheduler = OmegaConf.to_container(expected_planner.get("scheduler"), resolve=True)
    _check(isinstance(expected_scheduler, dict), f"{key}: canonical scheduler config is missing")
    resolved_scheduler = planner.get("scheduler")
    _check(isinstance(resolved_scheduler, dict), f"{key}: resolved scheduler config is missing")
    _check(
        resolved_scheduler == expected_scheduler,
        f"{key}: resolved scheduler differs from canonical matrix",
    )
    planner_params = payload.get("planner_params")
    _check(isinstance(planner_params, dict), f"{key}: payload planner params are missing")
    for field, observed in zip(("pop_size", "elite_frac", "n_iter"), values, strict=True):
        _check(_close(planner.get(field), observed), f"{key}: resolved planner {field} mismatch")
        _check(_close(planner_params.get(field), observed), f"{key}: payload planner {field} mismatch")
        _check(_close(metric.get(field), observed), f"{key}: metrics planner {field} mismatch")
    topk = max(1, min(values[0], int(round(values[0] * values[1]))))
    _check(int(planner_params.get("topk", -1)) == topk, f"{key}: payload topk mismatch")
    _check(int(metric.get("topk", -1)) == topk, f"{key}: metrics topk mismatch")

    eval_cfg = source.config.get("eval")
    _check(isinstance(eval_cfg, dict), f"{key}: resolved eval config is missing")
    _check(str(source.config.get("env_id", "")) == str(row["env_id"]), f"{key}: resolved env mismatch")
    _check(str(payload.get("env_id", "")) == str(row["env_id"]), f"{key}: payload env mismatch")
    _check(str(metric.get("env_id", "")) == str(row["env_id"]), f"{key}: metrics env mismatch")
    _check(int(eval_cfg.get("goal_offset", -1)) == int(row["goal_offset"]), f"{key}: resolved goal mismatch")
    _check(int(payload.get("goal_offset", -1)) == int(row["goal_offset"]), f"{key}: payload goal mismatch")
    _check(int(metric.get("goal_offset", -1)) == int(row["goal_offset"]), f"{key}: metrics goal mismatch")


def _audit_episode_prefix(
    row: Mapping[str, Any],
    source: _SourceArtifact,
    expected: _MatrixExpectation,
) -> None:
    key = _row_key(row)
    source_kind = str(row.get("source_kind", ""))
    _check(source_kind in SOURCE_EPISODES, f"{key}: unsupported source kind {source_kind!r}")
    episodes = SOURCE_EPISODES[source_kind]
    payload = source.payload
    metric = source.metric
    eval_cfg = source.config["eval"]
    _check(int(payload.get("episodes", -1)) == episodes, f"{key}: payload episode count mismatch")
    _check(int(metric.get("episodes", -1)) == episodes, f"{key}: metrics episode count mismatch")
    _check(int(eval_cfg.get("episodes", -1)) == episodes, f"{key}: resolved episode count mismatch")
    _check(len(source.manifest.get("pairs", [])) == episodes, f"{key}: source manifest pair count mismatch")
    _check(len(source.episode_traces) == episodes, f"{key}: episode trace count mismatch")

    batches = payload.get("batches")
    _check(isinstance(batches, list), f"{key}: payload batches are missing")
    pairs = [pair for batch in batches for pair in batch.get("pairs", [])]
    batch_outcomes = [
        bool(value)
        for batch in batches
        for value in batch.get("swm_results", {}).get("episode_successes", [])
    ]
    source_pairs = list(source.manifest.get("pairs", []))
    _check(
        [_pair_identity(pair) for pair in pairs]
        == [_pair_identity(pair) for pair in source_pairs],
        f"{key}: eval batch pair identities differ from source manifest",
    )
    successes = payload.get("swm_results", {}).get("episode_successes", [])
    _check(isinstance(successes, list) and len(successes) == episodes, f"{key}: payload outcomes are incomplete")
    _check(batch_outcomes == [bool(value) for value in successes], f"{key}: batch/top-level outcomes differ")

    canonical_pairs = list(expected.manifest.get("pairs", []))
    _check(len(canonical_pairs) >= 50, f"{key}: canonical manifest has fewer than 50 pairs")
    prefix_pairs = [_pair_identity(pair) for pair in pairs[:50]]
    _check(
        prefix_pairs == [_pair_identity(pair) for pair in canonical_pairs[:50]],
        f"{key}: source does not use the exact canonical 50-pair prefix",
    )
    outcomes = [bool(value) for value in successes[:50]]
    _check(int(row.get("discovery_episodes", -1)) == 50, f"{key}: discovery episode count is not 50")
    _check(list(row.get("discovery_outcomes", [])) == outcomes, f"{key}: result/source outcomes differ")
    _check(int(row.get("successes", -1)) == sum(outcomes), f"{key}: discovery success count mismatch")
    _check(_close(row.get("success_rate_fraction"), sum(outcomes) / 50), f"{key}: fractional success mismatch")
    _check(_close(row.get("success_rate_percent"), sum(outcomes) * 2), f"{key}: percent success mismatch")

    traces = source.episode_traces[:50]
    _check(
        [int(trace.get("episode_index", -1)) for trace in traces] == list(range(50)),
        f"{key}: episode trace prefix indices are not canonical",
    )
    _check([bool(trace.get("success")) for trace in traces] == outcomes, f"{key}: episode trace outcomes differ")
    _check(
        [_trace_identity(trace) for trace in traces]
        == [(pair[0], pair[1], pair[2]) for pair in prefix_pairs],
        f"{key}: episode trace dataset identities differ",
    )

    review = payload.get("review_rollouts")
    _check(isinstance(review, list) and len(review) == episodes, f"{key}: review rollouts are incomplete")
    _check(
        [int(item.get("episode_index", -1)) for item in review[:50]] == list(range(50)),
        f"{key}: review rollout indices differ",
    )
    _check([bool(item.get("success")) for item in review[:50]] == outcomes, f"{key}: review outcomes differ")
    _check(
        [
            (
                int(item["dataset_episode"]),
                int(item["start_step"]),
                int(item["goal_step"]),
                int(item["start_row"]),
                int(item["goal_row"]),
            )
            for item in review[:50]
        ]
        == prefix_pairs,
        f"{key}: review rollout dataset identities differ",
    )


def _audit_planning_and_cost(row: Mapping[str, Any], source: _SourceArtifact) -> None:
    key = _row_key(row)
    payload = source.payload
    metric = source.metric
    planner = source.config["planner"]
    eval_cfg = source.config["eval"]
    diagnostics = payload["planning_diagnostics"]
    summary = diagnostics.get("summary")
    trace = diagnostics.get("trace")
    batches = payload["batches"]
    _check(isinstance(summary, dict), f"{key}: planning summary is missing")
    _check(isinstance(trace, list), f"{key}: planning trace is missing")

    episodes = int(payload["episodes"])
    budget = int(eval_cfg["budget"])
    action_block = int(planner["action_block"])
    receding_horizon = int(planner["receding_horizon"])
    replan_interval = action_block * receding_horizon
    _check(replan_interval > 0, f"{key}: nonpositive replanning interval")
    replans_per_episode = math.ceil(budget / replan_interval)
    _check(replans_per_episode == 5, f"{key}: expected five replans per episode, got {replans_per_episode}")

    outer_batch_sizes = [len(batch.get("pairs", [])) for batch in batches]
    _check(sum(outer_batch_sizes) == episodes, f"{key}: outer batch sizes do not cover episodes")
    num_envs = int(eval_cfg["num_envs"])
    _check(all(0 < size <= num_envs for size in outer_batch_sizes), f"{key}: invalid outer batch size")
    planner_batch_size = planner.get("batch_size", num_envs)
    if planner_batch_size in (None, "auto"):
        planner_batch_size = num_envs
    planner_batch_size = int(planner_batch_size)
    _check(planner_batch_size > 0, f"{key}: nonpositive planner batch size")
    n_iter = int(row["n_iter"])
    pop_size = int(row["pop_size"])
    horizon = int(planner["horizon"])

    model_accounting = payload.get("model_accounting")
    _check(isinstance(model_accounting, dict), f"{key}: model accounting is missing")
    num_levels = int(model_accounting.get("num_levels", -1))
    levels = model_accounting.get("K")
    _check(num_levels > 0, f"{key}: model level count is missing")
    _check(
        isinstance(levels, list) and len(levels) == num_levels,
        f"{key}: model K ladder is missing or inconsistent",
    )
    k_levels = [int(value) for value in levels]
    scheduler_cfg = planner.get("scheduler")
    _check(isinstance(scheduler_cfg, dict), f"{key}: resolved scheduler config is missing")
    scheduler = FidelityScheduler.from_config(
        scheduler_cfg,
        num_levels=num_levels,
        horizon=horizon,
        levels=k_levels,
        min_k=1,
        max_k=max(k_levels),
        supports_arbitrary_k=bool(model_accounting.get("supports_arbitrary_k", False)),
        selectable_ks=(
            [int(value) for value in model_accounting.get("supported_k", {}).get("values", [])]
            if scheduler_cfg.get("fidelity_unit", "level") == "k"
            else None
        ),
    )

    expected_plans = len(batches) * replans_per_episode
    _check(int(diagnostics.get("plans", -1)) == expected_plans, f"{key}: batched solver plan count mismatch")
    _check(int(summary.get("replans", -1)) == expected_plans, f"{key}: planning summary replan mismatch")
    weighted_replans = sum(size * replans_per_episode for size in outer_batch_sizes)
    _check(weighted_replans == episodes * 5, f"{key}: batch-weighted episode replans mismatch")

    expected_calls = sum(
        math.ceil(size / planner_batch_size) * replans_per_episode * n_iter
        for size in outer_batch_sizes
    )
    _check(len(trace) == expected_calls, f"{key}: planning trace/CEM-call count mismatch")
    for field in ("cem_cost_calls",):
        _check(int(diagnostics.get(field, -1)) == expected_calls, f"{key}: diagnostics {field} mismatch")
        _check(int(summary.get(field, -1)) == expected_calls, f"{key}: summary {field} mismatch")
        _check(int(metric.get(field, -1)) == expected_calls, f"{key}: metrics {field} mismatch")

    concatenated: list[dict[str, Any]] = []
    for batch_index, (batch, batch_size) in enumerate(zip(batches, outer_batch_sizes, strict=True)):
        batch_diag = batch.get("planning_diagnostics")
        _check(isinstance(batch_diag, dict), f"{key}: batch {batch_index} diagnostics missing")
        batch_trace = batch_diag.get("trace")
        batch_summary = batch_diag.get("summary")
        _check(isinstance(batch_trace, list), f"{key}: batch {batch_index} trace missing")
        _check(isinstance(batch_summary, dict), f"{key}: batch {batch_index} summary missing")
        expected_batch_calls = math.ceil(batch_size / planner_batch_size) * replans_per_episode * n_iter
        _check(len(batch_trace) == expected_batch_calls, f"{key}: batch {batch_index} CEM-call mismatch")
        _check(int(batch_summary.get("replans", -1)) == replans_per_episode, f"{key}: batch replan mismatch")
        _check(int(batch_summary.get("cem_cost_calls", -1)) == expected_batch_calls, f"{key}: batch CEM summary mismatch")
        _check(
            int(batch_summary.get("flop_audit_error_count", -1)) == 0,
            f"{key}: batch {batch_index} has FLOP audit errors",
        )
        groups: dict[tuple[int, int, int], list[int]] = {}
        for diag in batch_trace:
            group = (
                int(diag.get("mpc_iter", -1)),
                int(diag.get("batch_start", -1)),
                int(diag.get("batch_end", -1)),
            )
            groups.setdefault(group, []).append(int(diag.get("cem_iter", -1)))
        _check(len(groups) == replans_per_episode * math.ceil(batch_size / planner_batch_size), f"{key}: CEM grouping mismatch")
        for (mpc_iter, start, stop), iterations in groups.items():
            _check(0 <= mpc_iter < replans_per_episode, f"{key}: invalid MPC iteration")
            _check(0 <= start < stop <= batch_size, f"{key}: invalid planner microbatch span")
            _check(stop - start <= planner_batch_size, f"{key}: planner microbatch exceeds batch_size")
            _check(sorted(iterations) == list(range(n_iter)), f"{key}: incomplete CEM iteration sequence")
        batch_sums = {
            "candidate_action_values": sum(
                int(diag.get("candidate_action_values", -1)) for diag in batch_trace
            ),
            "dynamics_flops_total": sum(
                int(diag.get("model_dynamics_flops", -1)) for diag in batch_trace
            ),
            "latent_work_total": sum(
                int(diag.get("model_latent_work", -1)) for diag in batch_trace
            ),
        }
        for field, expected_value in batch_sums.items():
            _check(
                int(batch_summary.get(field, -1)) == expected_value,
                f"{key}: batch {batch_index} {field} trace-sum mismatch",
            )
        _check(
            int(batch_summary.get("total_bits_used_estimate", -1))
            == batch_sums["latent_work_total"] * 32,
            f"{key}: batch {batch_index} bit/latent accounting mismatch",
        )
        concatenated.extend(batch_trace)
    _check(concatenated == trace, f"{key}: top-level planning trace differs from batch traces")

    action_dims: set[int] = set()
    for diag in trace:
        width = int(diag.get("batch_end", -1)) - int(diag.get("batch_start", -1))
        action_dim = int(diag.get("action_dim", -1))
        action_dims.add(action_dim)
        _check(int(diag.get("cem_cost_calls", -1)) == 1, f"{key}: trace CEM call is not atomic")
        _check(int(diag.get("num_samples", -1)) == pop_size, f"{key}: trace population mismatch")
        _check(int(diag.get("topk", -1)) == int(metric["topk"]), f"{key}: trace topk mismatch")
        _check(int(diag.get("horizon", -1)) == horizon, f"{key}: trace horizon mismatch")
        mpc_iter = int(diag.get("mpc_iter", -1))
        cem_iter = int(diag.get("cem_iter", -1))
        mpc_progress = 0.0 if replans_per_episode <= 1 else mpc_iter / (replans_per_episode - 1)
        expected_decision = scheduler.decision(
            cem_iter=cem_iter,
            n_iter=n_iter,
            mpc_progress=mpc_progress,
        )
        expected_metadata = expected_decision.metadata
        expected_levels = [
            int(value) if value is not None else None
            for value in expected_decision.rollout_level_indices
        ]
        _check(_close(diag.get("mpc_progress"), mpc_progress), f"{key}: trace MPC progress mismatch")
        _check(
            _close(diag.get("cem_progress"), expected_decision.cem_progress),
            f"{key}: trace CEM progress mismatch",
        )
        for field, expected_value in (
            ("mpc_level_idx", expected_metadata.get("mpc_level_idx")),
            ("base_level_idx", expected_decision.base_level_idx),
            ("terminal_level_idx", expected_metadata.get("terminal_level_idx")),
            ("mpc_k", expected_metadata.get("mpc_k")),
            ("base_k", expected_decision.base_k),
            ("terminal_k", expected_metadata.get("terminal_k")),
        ):
            _check(diag.get(field) == expected_value, f"{key}: trace realized {field} mismatch")
        _check(
            diag.get("rollout_level_indices") == expected_levels,
            f"{key}: trace realized rollout level ladder mismatch",
        )
        expected_ks = (
            [int(value) for value in expected_decision.rollout_ks]
            if expected_decision.rollout_ks is not None
            else None
        )
        _check(diag.get("rollout_ks") == expected_ks, f"{key}: trace realized rollout K ladder mismatch")
        for field, expected_value in (
            ("model_base_level_idx", expected_decision.base_level_idx),
            ("model_terminal_level_idx", expected_metadata.get("terminal_level_idx")),
            ("model_base_k", expected_decision.base_k),
            ("model_terminal_k", expected_metadata.get("terminal_k")),
        ):
            _check(diag.get(field) == expected_value, f"{key}: model-observed {field} mismatch")
        _check(
            diag.get("model_rollout_level_indices") == expected_levels,
            f"{key}: model-observed rollout level ladder mismatch",
        )
        _check(
            diag.get("model_rollout_ks") == expected_ks,
            f"{key}: model-observed rollout K ladder mismatch",
        )
        expected_values = width * pop_size * horizon * action_dim
        _check(
            int(diag.get("candidate_action_values", -1)) == expected_values,
            f"{key}: trace candidate-action cost mismatch",
        )
        _check(diag.get("model_flop_accounting") == "dynamics_audit", f"{key}: wrong FLOP accounting mode")
        _check(not diag.get("model_flop_audit_error"), f"{key}: trace contains a FLOP audit error")
        _check(int(diag.get("model_dynamics_flops", 0)) > 0, f"{key}: missing trace dynamics FLOPs")
        _check(int(diag.get("model_latent_work", 0)) > 0, f"{key}: missing trace latent work")
    _check(len(action_dims) == 1 and next(iter(action_dims)) > 0, f"{key}: inconsistent action dimension")

    sums = {
        "cem_cost_calls": sum(int(diag["cem_cost_calls"]) for diag in trace),
        "candidate_action_values": sum(int(diag["candidate_action_values"]) for diag in trace),
        "dynamics_flops_total": sum(int(diag["model_dynamics_flops"]) for diag in trace),
        "latent_work_total": sum(int(diag["model_latent_work"]) for diag in trace),
    }
    for field, expected_value in sums.items():
        _check(int(diagnostics.get(field, -1)) == expected_value, f"{key}: diagnostics {field} trace-sum mismatch")
        _check(int(summary.get(field, -1)) == expected_value, f"{key}: summary {field} trace-sum mismatch")
        _check(int(metric.get(field, -1)) == expected_value, f"{key}: metrics {field} trace-sum mismatch")
    _check(int(diagnostics.get("flop_audit_error_count", -1)) == 0, f"{key}: nonzero FLOP audit errors")
    _check(int(summary.get("flop_audit_error_count", -1)) == 0, f"{key}: summary FLOP audit errors")
    _check(int(diagnostics.get("bits_used_total", -1)) == sums["latent_work_total"] * 32, f"{key}: bit/latent accounting mismatch")
    _check(int(summary.get("total_bits_used_estimate", -1)) == sums["latent_work_total"] * 32, f"{key}: summary bit/latent accounting mismatch")

    per_episode_fields = {
        "dynamics_flops_per_episode": "dynamics_flops_total",
        "latent_work_per_episode": "latent_work_total",
        "plan_time_sec_per_episode": "plan_time_total_sec",
        "wall_time_sec_per_episode": "wall_time_sec",
    }
    for result_field, metric_field in per_episode_fields.items():
        _check(
            _close(row.get(result_field), float(metric[metric_field]) / episodes),
            f"{key}: {result_field} normalization mismatch",
        )


def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrices = payload.get("matrices")
    if not isinstance(matrices, Sequence) or isinstance(matrices, (str, bytes)):
        raise ValueError("anchor results matrices are not a list")
    for matrix in matrices:
        if not isinstance(matrix, Mapping):
            raise ValueError("anchor results contain a non-mapping matrix")
        matrix_rows = matrix.get("rows")
        if not isinstance(matrix_rows, Sequence) or isinstance(matrix_rows, (str, bytes)):
            raise ValueError("anchor result matrix rows are not a list")
        _check(
            int(matrix.get("cells", len(matrix_rows))) == len(matrix_rows),
            "matrix cell count differs from its rows",
        )
        observed_frontier = sum(
            bool(row.get("pareto_frontier"))
            for row in matrix_rows
            if isinstance(row, Mapping)
        )
        _check(
            int(matrix.get("pareto_cells", observed_frontier)) == observed_frontier,
            "matrix Pareto count differs from its rows",
        )
        for raw_row in matrix_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("anchor results contain a non-mapping row")
            row = dict(raw_row)
            _check(
                str(row.get("canonical_config", "")) == str(matrix.get("canonical_config", "")),
                "row canonical_config differs from containing matrix",
            )
            _check(
                str(row.get("screen_config", "")) == str(matrix.get("screen_config", "")),
                "row screen_config differs from containing matrix",
            )
            rows.append(row)
    return rows


def audit_anchor_artifacts(
    results: Mapping[str, Any] | str | Path = DEFAULT_INPUT,
    *,
    repo_root: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Audit every unique source referenced by a five-anchor results payload.

    ``load_eval_artifact(..., verify="full")`` is called once per unique
    ``source_run_dir``. Rows are grouped by source and each materialized
    payload is released after its group is checked, so the strict audit is
    bounded by one source artifact rather than the full diagnostic corpus.
    """

    root = Path(repo_root).resolve()
    if isinstance(results, Mapping):
        payload = dict(results)
        input_path: str | None = None
    else:
        path = _resolve(root, results)
        payload = load_json(path)
        input_path = str(path)

    issues: list[str] = []
    try:
        _check(
            payload.get("schema_version") == RESULTS_SCHEMA_VERSION,
            "unsupported five-anchor results schema",
        )
        rows = _rows(payload)
        _check(int(payload.get("cells", len(rows))) == len(rows), "results cell count mismatch")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        rows = []
        issues.append(str(exc))

    matrix_cache: dict[str, _MatrixExpectation] = {}
    failed_matrices: set[str] = set()
    rows_by_source: dict[Path, list[dict[str, Any]]] = {}
    source_references: Counter[Path] = Counter()
    source_kind_counts: Counter[str] = Counter()
    for row in rows:
        run_dir = _resolve(root, str(row.get("source_run_dir", "")))
        source_references[run_dir] += 1
        rows_by_source.setdefault(run_dir, []).append(row)
        source_kind_counts[str(row.get("source_kind", ""))] += 1

    verified_sources = 0
    full_load_attempts = 0
    for run_dir, source_rows in rows_by_source.items():
        full_load_attempts += 1
        try:
            source = _load_source(root, run_dir)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                f"{_row_key(source_rows[0])}: cannot verify source artifact: {exc}"
            )
            continue
        verified_sources += 1
        for row in source_rows:
            key = _row_key(row)
            canonical_key = str(row.get("canonical_config", ""))
            if canonical_key not in matrix_cache and canonical_key not in failed_matrices:
                try:
                    matrix_cache[canonical_key] = _load_matrix_expectation(root, canonical_key)
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(f"{key}: cannot load canonical matrix/manifest: {exc}")
                    failed_matrices.add(canonical_key)
            if canonical_key in failed_matrices:
                continue
            try:
                _check(
                    file_sha256(source.run_dir / "resolved_config.yaml")
                    == str(row.get("config_sha256", "")),
                    f"{key}: row config SHA-256 mismatch",
                )
                _check(
                    str(source.manifest.get("manifest_sha256", ""))
                    == str(row.get("manifest_sha256", "")),
                    f"{key}: row semantic manifest SHA-256 mismatch",
                )
                _audit_matrix_identity(row, source, matrix_cache[canonical_key])
                _audit_episode_prefix(row, source, matrix_cache[canonical_key])
                _audit_planning_and_cost(row, source)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                issues.append(str(exc))

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not issues else "failed",
        "input_path": input_path,
        "rows_referenced": len(rows),
        "unique_sources_referenced": len(source_references),
        "unique_sources_verified": verified_sources,
        "matrices_verified": len(matrix_cache),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "source_reference_counts": {
            str(path): count for path, count in sorted(source_references.items(), key=lambda item: str(item[0]))
        },
        "full_artifact_load_attempts": full_load_attempts,
        "full_archive_loads": verified_sources,
        "peak_materialized_sources": 1 if verified_sources else 0,
        "replans_per_episode": 5,
        "checkpoint_identity_scope": (
            "canonical/resolved/payload/metric run_dir agreement plus loaded-epoch "
            "consistency; checkpoint file bytes were not recorded by evaluation"
        ),
        "checkpoint_content_hash_verified": False,
        "configured_scheduler_and_realized_trace_verified": not issues,
        "issues": issues,
    }
    if issues and strict:
        preview = "; ".join(issues[:5])
        raise AnchorArtifactAuditError(
            f"Five-anchor artifact audit found {len(issues)} violation(s): {preview}",
            report,
        )
    return report


__all__ = [
    "AnchorArtifactAuditError",
    "DEFAULT_INPUT",
    "SCHEMA_VERSION",
    "audit_anchor_artifacts",
]
