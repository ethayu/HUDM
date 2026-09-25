"""Statistical analysis for the release20260728 five-anchor diagnostic.

The diagnostic is a finite, paired benchmark: within one environment/goal
matrix every schedule sees the same 50 episode initial conditions.  This
module therefore keeps episode pairing intact in every comparison and avoids
pretending that the 1,040 aggregate cell rates are independent observations.

The analysis is deliberately separate from :mod:`anchor_diagnostic` so it can
evolve without changing code imported by active benchmark workers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from mwm.benchmark.screening import _atomic_write_text


SCHEMA_VERSION = "mwm.release20260728.anchor_analysis/v4"
DEFAULT_INPUT = Path(
    "reports/research/release20260728_schedule_screening/five_anchor_results/results.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "reports/research/release20260728_schedule_screening/five_anchor_analysis"
)
DEFAULT_COVARIATE_INPUT = Path(
    "reports/research/release20260728_schedule_screening/"
    "five_anchor_episode_covariates.json"
)
DEFAULT_BOOTSTRAP_REPEATS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_728

_LABEL_RE = re.compile(r"^MPC=(.*?)\s*\|\s*CEM=(.*?)\s*\|\s*Rollout=(.*?)$")
_SCHEDULE_ID_RE = re.compile(r"^(\d+)_")

_EXPECTED_ANCHORS = {
    "a1": {"pop_size": 20, "elite_frac": 0.2, "n_iter": 5, "matrix_offset": 5},
    "a2": {"pop_size": 50, "elite_frac": 0.2, "n_iter": 10, "matrix_offset": 31},
    "a3": {"pop_size": 100, "elite_frac": 0.2, "n_iter": 15, "matrix_offset": 57},
    "a4": {"pop_size": 150, "elite_frac": 0.2, "n_iter": 20, "matrix_offset": 83},
    "a5": {"pop_size": 200, "elite_frac": 0.2, "n_iter": 30, "matrix_offset": 109},
}
_EXPECTED_SCHEDULE_NUMBERS = set(range(1, 27))


class AnchorAnalysisError(ValueError):
    """Raised when final diagnostic data violate the paired-design contract."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schedule_number(schedule: str) -> int:
    match = _SCHEDULE_ID_RE.match(str(schedule))
    if match is None:
        raise AnchorAnalysisError(f"Cannot parse schedule number from {schedule!r}.")
    return int(match.group(1))


@lru_cache(maxsize=None)
def source_axis_signature(schedule_label: str) -> tuple[str, str, str]:
    """Normalize the three labels while retaining their configured stages."""

    match = _LABEL_RE.match(str(schedule_label).strip())
    if match is None:
        raise AnchorAnalysisError(f"Malformed schedule label {schedule_label!r}.")
    raw_mpc, raw_cem, raw_rollout = (part.strip().lower() for part in match.groups())
    mpc = {
        "-": "fixed_finest",
        "fixed": "fixed_finest",
        "coarse -> fine": "coarse_to_fine",
    }.get(raw_mpc)
    cem = {
        "-": "fixed_base",
        "fixed": "fixed_base",
        "base": "fixed_base",
        "coarse -> base": "coarse_to_base",
        "base -> fine": "base_to_fine",
        "coarse -> fine": "coarse_to_fine",
    }.get(raw_cem)
    rollout = {
        "fixed": "fixed_base",
        "base": "fixed_base",
        "fine -> base": "fine_to_base",
        "base -> coarse": "base_to_coarse",
        "fine -> coarse": "fine_to_coarse",
    }.get(raw_rollout)
    if mpc is None or cem is None or rollout is None:
        raise AnchorAnalysisError(
            f"Unsupported operational schedule tokens in {schedule_label!r}: "
            f"mpc={raw_mpc!r}, cem={raw_cem!r}, rollout={raw_rollout!r}."
        )
    return mpc, cem, rollout


def _interp_level(start: int, end: int, progress: float) -> int:
    # This intentionally mirrors mwm.fidelity._interp_level, including
    # Python's round-to-even behavior at exact half levels.
    return int(round(float(start) + (float(end) - float(start)) * float(progress)))


def _realized_decisions(
    schedule_label: str,
    *,
    n_iter: int,
    horizon: int,
    replans: int = 5,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Resolve a source label to the exact level decisions used at runtime."""

    mpc_mode, cem_mode, rollout_mode = source_axis_signature(schedule_label)
    decisions = []
    for replan in range(replans):
        mpc_progress = 0.0 if replans <= 1 else replan / (replans - 1)
        mpc_level = 4 if mpc_mode == "fixed_finest" else _interp_level(0, 4, mpc_progress)
        for iteration in range(n_iter):
            cem_progress = 1.0 if n_iter <= 1 else iteration / (n_iter - 1)
            if cem_mode == "fixed_base":
                base_level = mpc_level
            elif cem_mode == "coarse_to_base":
                base_level = _interp_level(0, mpc_level, cem_progress)
            elif cem_mode == "base_to_fine":
                base_level = _interp_level(mpc_level, 4, cem_progress)
            elif cem_mode == "coarse_to_fine":
                base_level = _interp_level(0, 4, cem_progress)
            else:  # pragma: no cover - source_axis_signature is exhaustive
                raise AnchorAnalysisError(f"Unsupported CEM mode {cem_mode!r}.")
            if rollout_mode == "fixed_base":
                rollout = (base_level,) * horizon
            else:
                if rollout_mode == "fine_to_base":
                    start, end = 4, base_level
                elif rollout_mode == "base_to_coarse":
                    start, end = base_level, 0
                elif rollout_mode == "fine_to_coarse":
                    start, end = 4, 0
                else:  # pragma: no cover - source_axis_signature is exhaustive
                    raise AnchorAnalysisError(f"Unsupported rollout mode {rollout_mode!r}.")
                rollout = tuple(
                    start
                    if horizon <= 1
                    else _interp_level(start, end, step / (horizon - 1))
                    for step in range(horizon)
                )
            decisions.append((mpc_level, base_level, rollout))
    return tuple(decisions)


@lru_cache(maxsize=None)
def realized_policy_signature(schedule_label: str) -> str:
    """Hash the exact decisions across both benchmark horizons/all anchors.

    Both goal designs make five replans.  Including every iteration ladder and
    both horizons makes equality stronger than comparing the human-readable
    labels: equal hashes imply the same MPC, CEM-base, and rollout levels at
    every decision point exercised by this diagnostic.
    """

    decisions = {
        f"iter{n_iter}_h{horizon}": _realized_decisions(
            schedule_label,
            n_iter=n_iter,
            horizon=horizon,
        )
        for n_iter in (5, 10, 15, 20, 30)
        for horizon in (5, 10)
    }
    return _canonical_sha256(decisions)


def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrices = payload.get("matrices")
    if not isinstance(matrices, list):
        raise AnchorAnalysisError("Result payload has no matrix list.")
    rows: list[dict[str, Any]] = []
    for matrix in matrices:
        if not isinstance(matrix, Mapping) or not isinstance(matrix.get("rows"), list):
            raise AnchorAnalysisError("Malformed matrix result entry.")
        rows.extend(dict(row) for row in matrix["rows"])
    return rows


def validate_results(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = _rows(payload)
    if len(rows) != 8 * 26 * 5:
        raise AnchorAnalysisError(f"Expected 1,040 result rows, found {len(rows)}.")
    if int(payload.get("cells", -1)) != len(rows):
        raise AnchorAnalysisError(
            f"Top-level cell count is {payload.get('cells')!r}, expected {len(rows)}."
        )
    matrix_keys = sorted({str(row["canonical_config"]) for row in rows})
    if len(matrix_keys) != 8:
        raise AnchorAnalysisError(f"Expected eight matrices, found {len(matrix_keys)}.")
    seen: set[tuple[str, int, str]] = set()
    schedule_names: dict[int, str] = {}
    schedule_labels: dict[int, str] = {}
    env_goals: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        schedule = str(row["schedule"])
        schedule_number = _schedule_number(schedule)
        anchor_id = str(row["anchor_id"])
        key = (str(row["canonical_config"]), schedule_number, anchor_id)
        if key in seen:
            raise AnchorAnalysisError(f"Duplicate matrix/schedule/anchor row: {key}.")
        seen.add(key)
        if schedule_number not in _EXPECTED_SCHEDULE_NUMBERS:
            raise AnchorAnalysisError(f"{key} has schedule number outside 1..26.")
        anchor = _EXPECTED_ANCHORS.get(anchor_id)
        if anchor is None:
            raise AnchorAnalysisError(f"{key} has unexpected anchor {anchor_id!r}.")
        expected_index = (schedule_number - 1) * 120 + int(anchor["matrix_offset"])
        if int(row.get("matrix_index", -1)) != expected_index:
            raise AnchorAnalysisError(
                f"{key} has matrix_index={row.get('matrix_index')!r}, expected {expected_index}."
            )
        for field in ("pop_size", "n_iter"):
            if int(row.get(field, -1)) != int(anchor[field]):
                raise AnchorAnalysisError(
                    f"{key} has {field}={row.get(field)!r}, expected {anchor[field]}."
                )
        if not math.isclose(
            float(row.get("elite_frac", math.nan)),
            float(anchor["elite_frac"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise AnchorAnalysisError(
                f"{key} has elite_frac={row.get('elite_frac')!r}, expected {anchor['elite_frac']}."
            )
        expected_work = int(anchor["pop_size"]) * int(anchor["n_iter"])
        if int(row.get("nominal_candidate_iterations", -1)) != expected_work:
            raise AnchorAnalysisError(
                f"{key} has nominal_candidate_iterations="
                f"{row.get('nominal_candidate_iterations')!r}, expected {expected_work}."
            )
        if int(row.get("discovery_episodes", -1)) != 50:
            raise AnchorAnalysisError(f"{key} does not declare 50 discovery episodes.")
        outcomes = row.get("discovery_outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != 50:
            raise AnchorAnalysisError(f"{key} does not contain exactly 50 paired outcomes.")
        if any(not isinstance(value, bool) for value in outcomes):
            raise AnchorAnalysisError(f"{key} contains a non-boolean paired outcome.")
        successes = sum(outcomes)
        if int(row.get("successes", -1)) != successes:
            raise AnchorAnalysisError(f"{key} success count disagrees with its outcomes.")
        expected_rate = successes / 50.0
        if not math.isclose(
            float(row.get("success_rate_fraction", math.nan)),
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AnchorAnalysisError(f"{key} success-rate fraction disagrees with outcomes.")
        if not math.isclose(
            float(row.get("success_rate_percent", math.nan)),
            100.0 * expected_rate,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise AnchorAnalysisError(f"{key} success-rate percent disagrees with outcomes.")
        dynamics_cost = float(row["dynamics_flops_per_episode"])
        if not math.isfinite(dynamics_cost) or dynamics_cost <= 0:
            raise AnchorAnalysisError(f"{key} has non-positive or non-finite dynamics cost.")
        schedule_label = str(row["schedule_label"])
        source_axis_signature(schedule_label)
        realized_policy_signature(schedule_label)
        previous_name = schedule_names.setdefault(schedule_number, schedule)
        previous_label = schedule_labels.setdefault(schedule_number, schedule_label)
        if previous_name != schedule or previous_label != schedule_label:
            raise AnchorAnalysisError(
                f"Schedule {schedule_number:02d} changes name or label across matrices."
            )
        goal_offset = int(row["goal_offset"])
        if goal_offset not in {25, 50}:
            raise AnchorAnalysisError(f"{key} has unsupported goal offset {goal_offset}.")
        env_goals[str(row["env_id"])].add(goal_offset)
    per_matrix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_matrix[str(row["canonical_config"])].append(row)
    malformed: dict[str, Any] = {}
    expected_grid = {
        (schedule_number, anchor_id)
        for schedule_number in _EXPECTED_SCHEDULE_NUMBERS
        for anchor_id in _EXPECTED_ANCHORS
    }
    for matrix_key, values in per_matrix.items():
        observed_grid = {
            (_schedule_number(str(row["schedule"])), str(row["anchor_id"])) for row in values
        }
        env_goal = {(str(row["env_id"]), int(row["goal_offset"])) for row in values}
        if len(values) != 130 or observed_grid != expected_grid or len(env_goal) != 1:
            malformed[matrix_key] = {
                "rows": len(values),
                "missing_grid_cells": len(expected_grid - observed_grid),
                "extra_grid_cells": len(observed_grid - expected_grid),
                "env_goal_values": sorted(env_goal),
            }
    if malformed:
        raise AnchorAnalysisError(f"Malformed matrix grids: {malformed}.")
    if len(env_goals) != 4 or any(goals != {25, 50} for goals in env_goals.values()):
        raise AnchorAnalysisError(
            "Expected four environments with exactly one goal-25 and one goal-50 matrix; "
            f"found {dict(sorted(env_goals.items()))}."
        )
    return {
        "rows": len(rows),
        "matrices": len(matrix_keys),
        "schedules": len(schedule_names),
        "anchors": len(_EXPECTED_ANCHORS),
        "paired_episodes": 50,
        "matrix_keys": matrix_keys,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise AnchorAnalysisError(f"{path} contains a non-object JSONL row.")
                    values.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorAnalysisError(f"Cannot read {path}: {exc}") from exc
    return values


def audit_source_artifacts(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[tuple[str, int], list[int]]]:
    """Validate cost normalization and recover trajectory-cluster identities."""

    root = Path(repo_root).resolve()
    rows = _rows(payload)
    errors: list[str] = []
    clusters: dict[tuple[str, int], list[int]] = {}
    pair_identities: dict[str, list[tuple[int, int, int]]] = {}
    source_counts: dict[str, int] = defaultdict(int)
    episode_counts: dict[int, int] = defaultdict(int)
    for row in rows:
        key = (str(row["canonical_config"]), int(row["matrix_index"]))
        run_dir = Path(str(row["source_run_dir"]))
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        metrics_rows = _jsonl(run_dir / "metrics.jsonl")
        traces = _jsonl(run_dir / "episode_traces.jsonl")
        try:
            planning = json.loads((run_dir / "planning_diagnostics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnchorAnalysisError(f"Cannot read {run_dir / 'planning_diagnostics.json'}: {exc}") from exc
        if len(metrics_rows) != 1:
            errors.append(f"{key}: expected one metrics row, found {len(metrics_rows)}")
            continue
        metric = metrics_rows[0]
        episodes = int(metric.get("episodes", -1))
        expected_episodes = 250 if row["source_kind"] == "canonical_250_prefix" else 50
        source_counts[str(row["source_kind"])] += 1
        episode_counts[episodes] += 1
        if episodes != expected_episodes or len(traces) != episodes:
            errors.append(
                f"{key}: source episodes expected={expected_episodes}, metrics={episodes}, traces={len(traces)}"
            )
            continue
        prefix = traces[:50]
        if [int(trace.get("episode_index", -1)) for trace in prefix] != list(range(50)):
            errors.append(f"{key}: noncanonical episode_index prefix")
        outcomes = [trace.get("success") for trace in prefix]
        if outcomes != row["discovery_outcomes"]:
            errors.append(f"{key}: result outcomes differ from source trace")
        identities = [
            (
                int(trace.get("dataset_episode", -1)),
                int(trace.get("start_step", -1)),
                int(trace.get("goal_step", -1)),
            )
            for trace in prefix
        ]
        matrix_key = str(row["canonical_config"])
        if matrix_key in pair_identities and pair_identities[matrix_key] != identities:
            errors.append(f"{key}: paired manifest identities differ within matrix")
        else:
            pair_identities[matrix_key] = identities
        clusters[key] = [identity[0] for identity in identities]
        if str(metric.get("config_sha256", "")) != str(row["config_sha256"]):
            errors.append(f"{key}: config hash mismatch")
        if str(metric.get("manifest_sha256", "")) != str(row["manifest_sha256"]):
            errors.append(f"{key}: manifest hash mismatch")
        expected_cost = float(metric.get("dynamics_flops_total", math.nan)) / episodes
        if not math.isclose(
            expected_cost,
            float(row["dynamics_flops_per_episode"]),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            errors.append(f"{key}: dynamics FLOP normalization mismatch")
        plans = int(metric.get("plans", -1))
        n_iter = int(metric.get("n_iter", -1))
        cem_calls = int(metric.get("cem_cost_calls", -1))
        expected_batched_solver_calls = (episodes // 50) * 5 if episodes % 50 == 0 else -1
        if plans != expected_batched_solver_calls:
            errors.append(
                f"{key}: expected five vectorized solver calls per 50-episode batch, "
                f"got plans={plans}, episodes={episodes}"
            )
        if cem_calls != episodes * n_iter * 5:
            errors.append(
                f"{key}: expected cem_cost_calls=episodes*n_iter*5, got "
                f"{cem_calls}/{episodes}/{n_iter}"
            )
        for field in (
            "plans",
            "cem_cost_calls",
            "dynamics_flops_total",
            "latent_work_total",
        ):
            if int(planning.get(field, -1)) != int(metric.get(field, -2)):
                errors.append(f"{key}: planning/metrics {field} mismatch")
        if int(planning.get("flop_audit_error_count", -1)) != 0:
            errors.append(f"{key}: nonzero or missing flop_audit_error_count")
        trace_ref = planning.get("trace")
        trace_length = (
            int(trace_ref.get("length", -1))
            if isinstance(trace_ref, Mapping)
            else len(trace_ref) if isinstance(trace_ref, list) else -1
        )
        if trace_length != cem_calls:
            errors.append(f"{key}: planning trace length {trace_length} != CEM calls {cem_calls}")
    if errors:
        preview = "; ".join(errors[:10])
        raise AnchorAnalysisError(
            f"Source artifact audit found {len(errors)} invariant violation(s): {preview}"
        )
    matrix_cluster_summary = {
        matrix_key: {
            "windows": len(identities),
            "unique_dataset_episodes": len({identity[0] for identity in identities}),
            "unique_start_goal_windows": len(set(identities)),
        }
        for matrix_key, identities in sorted(pair_identities.items())
    }
    by_env_goal: dict[tuple[str, int], set[int]] = {}
    matrix_row = {str(row["canonical_config"]): row for row in rows}
    for matrix_key, identities in pair_identities.items():
        row = matrix_row[matrix_key]
        by_env_goal[(str(row["env_id"]), int(row["goal_offset"]))] = {
            identity[0] for identity in identities
        }
    cross_goal_overlap = {}
    for env_id in sorted({key[0] for key in by_env_goal}):
        short = by_env_goal.get((env_id, 25))
        long = by_env_goal.get((env_id, 50))
        if short is not None and long is not None:
            cross_goal_overlap[env_id] = len(short & long)
    return (
        {
            "status": "passed",
            "cells_audited": len(rows),
            "source_kind_counts": dict(sorted(source_counts.items())),
            "artifact_episode_counts": {str(key): value for key, value in sorted(episode_counts.items())},
            "flop_audit_error_count": 0,
            "episodes_per_vectorized_solver_batch": 50,
            "batched_solver_calls_per_batch": 5,
            "replanning_rounds_per_episode": 5,
            "cem_cost_calls_per_episode_iteration": 5.0,
            "matrix_episode_clusters": matrix_cluster_summary,
            "cross_goal_dataset_episode_overlap": cross_goal_overlap,
            "wall_time_scientific_cost_axis": False,
            "primary_cost_axis": "dynamics_flops_per_episode",
        },
        clusters,
    )


def load_episode_covariate_summary(
    path: str | Path,
    *,
    result_payload: Mapping[str, Any],
    episode_clusters: Mapping[tuple[str, int], Sequence[int]],
) -> dict[str, Any]:
    """Validate and summarize the independent pre-outcome structure artifact."""

    from mwm.benchmark.episode_covariates import validate_covariate_payload

    covariate_path = Path(path).resolve()
    try:
        raw = covariate_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorAnalysisError(
            f"Cannot read episode covariates {covariate_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AnchorAnalysisError(f"Episode covariates {covariate_path} are not an object.")
    try:
        validation = validate_covariate_payload(payload)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise AnchorAnalysisError(f"Invalid episode covariates {covariate_path}: {exc}") from exc

    result_rows = _rows(result_payload)
    matrix_reference: dict[str, Mapping[str, Any]] = {}
    for row in result_rows:
        matrix_reference.setdefault(str(row["canonical_config"]), row)
    records = payload.get("records")
    if not isinstance(records, Mapping):  # defended again for type narrowing
        raise AnchorAnalysisError("Validated covariate payload has no record mapping.")
    by_matrix: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records.values():
        if not isinstance(record, Mapping):
            raise AnchorAnalysisError("Validated covariate record is not a mapping.")
        by_matrix[str(record["canonical_config"])].append(record)
    if set(by_matrix) != set(matrix_reference):
        raise AnchorAnalysisError(
            "Covariate/result canonical matrices differ: "
            f"covariates={sorted(by_matrix)}, results={sorted(matrix_reference)}."
        )

    matrix_summaries = []
    for matrix_key in sorted(by_matrix):
        values = sorted(by_matrix[matrix_key], key=lambda row: int(row["episode_index"]))
        reference = matrix_reference[matrix_key]
        if {
            (str(row["env_id"]), int(row["goal_offset"])) for row in values
        } != {(str(reference["env_id"]), int(reference["goal_offset"]))}:
            raise AnchorAnalysisError(f"Covariate env/goal identity differs for {matrix_key}.")
        cluster_key = (matrix_key, int(reference["matrix_index"]))
        expected_clusters = list(episode_clusters.get(cluster_key, ()))
        observed_clusters = [int(row["dataset_episode"]) for row in values]
        if observed_clusters != expected_clusters:
            raise AnchorAnalysisError(
                f"Covariate dataset-episode identities differ from audited outcomes for {matrix_key}."
            )
        covariate_names = sorted(
            {
                str(name)
                for row in values
                for name in dict(row.get("covariates", {}))
            }
        )
        summaries: dict[str, dict[str, Any]] = {}
        for name in covariate_names:
            observed = [dict(row["covariates"])[name] for row in values]
            nonnull = [value for value in observed if value is not None]
            if nonnull and all(isinstance(value, bool) for value in nonnull):
                summaries[name] = {
                    "kind": "boolean",
                    "episodes": len(observed),
                    "nonnull": len(nonnull),
                    "true_count": int(sum(bool(value) for value in nonnull)),
                    "true_fraction": float(np.mean(nonnull)),
                }
            else:
                numeric = np.asarray([float(value) for value in nonnull], dtype=float)
                summaries[name] = {
                    "kind": "numeric",
                    "episodes": len(observed),
                    "nonnull": len(nonnull),
                    "null": len(observed) - len(nonnull),
                    "min": float(np.min(numeric)) if len(numeric) else None,
                    "median": float(np.median(numeric)) if len(numeric) else None,
                    "mean": float(np.mean(numeric)) if len(numeric) else None,
                    "max": float(np.max(numeric)) if len(numeric) else None,
                }
        matrix_summaries.append(
            {
                "canonical_config": matrix_key,
                "env_id": str(reference["env_id"]),
                "goal_offset": int(reference["goal_offset"]),
                "episodes": len(values),
                "covariates": summaries,
            }
        )
    return {
        "status": "passed",
        "schema_version": str(payload.get("schema_version", "")),
        "path": str(covariate_path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "validation": validation,
        "matrix_structure": matrix_summaries,
        "episode_records": [
            {
                "canonical_config": str(record["canonical_config"]),
                "env_id": str(record["env_id"]),
                "goal_offset": int(record["goal_offset"]),
                "episode_index": int(record["episode_index"]),
                "dataset_episode": int(record["dataset_episode"]),
                "covariates": dict(record["covariates"]),
            }
            for matrix_key in sorted(by_matrix)
            for record in sorted(
                by_matrix[matrix_key],
                key=lambda value: int(value["episode_index"]),
            )
        ],
        "raw_dataset_rows_content_hashed": False,
        "interpretation": "pre-outcome descriptive covariates; no schedule outcome was used",
    }


def _pareto_mask(costs: np.ndarray, successes: np.ndarray) -> np.ndarray:
    """Weak Pareto mask, retaining exact ties, in O(n log n)."""

    costs = np.asarray(costs, dtype=float)
    successes = np.asarray(successes, dtype=float)
    order = np.argsort(costs, kind="stable")
    mask = np.zeros(len(costs), dtype=bool)
    best_cheaper = -np.inf
    position = 0
    while position < len(order):
        stop = position + 1
        cost = costs[order[position]]
        while stop < len(order) and costs[order[stop]] == cost:
            stop += 1
        group = order[position:stop]
        group_best = float(np.max(successes[group]))
        if group_best > best_cheaper:
            mask[group[successes[group] == group_best]] = True
        best_cheaper = max(best_cheaper, group_best)
        position = stop
    return mask


def _epsilon_pareto_mask(costs: np.ndarray, successes: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Keep cells lacking a no-more-expensive practically better competitor."""

    costs = np.asarray(costs, dtype=float)
    successes = np.asarray(successes, dtype=float)
    mask = np.ones(len(costs), dtype=bool)
    for candidate in range(len(costs)):
        eligible = (costs <= costs[candidate]) & (
            successes >= successes[candidate] + float(epsilon)
        )
        eligible[candidate] = False
        if np.any(eligible):
            mask[candidate] = False
    return mask


def _cluster_sums(
    episode_differences: Sequence[float],
    cluster_ids: Sequence[Any],
) -> np.ndarray:
    """Aggregate paired differences at the independent trajectory unit."""

    if len(episode_differences) != len(cluster_ids):
        raise AnchorAnalysisError(
            "Paired differences and trajectory-cluster identities have different lengths."
        )
    totals: dict[Any, float] = {}
    for difference, raw_cluster in zip(episode_differences, cluster_ids, strict=True):
        key = (
            json.dumps(raw_cluster, sort_keys=True)
            if isinstance(raw_cluster, (dict, list))
            else raw_cluster
        )
        totals[key] = totals.get(key, 0.0) + float(difference)
    return np.asarray(list(totals.values()), dtype=float)


def _exact_cluster_sign_flip_p(
    episode_differences: Sequence[float],
    cluster_ids: Sequence[Any],
) -> float:
    """Exact-enumeration two-sided cluster sign-flip p-value.

    Validity requires independent cluster totals whose joint null distribution
    is invariant under independent sign flips (as with independent symmetric
    cluster totals); policy labels were not randomized by the benchmark design.
    Dynamic programming evaluates the complete sign-flip distribution without
    assuming that the 50 windows are independent and without explicitly
    enumerating ``2**n_clusters`` sign vectors.
    """

    cluster_totals = _cluster_sums(episode_differences, cluster_ids)
    nonzero = [float(value) for value in cluster_totals if float(value) != 0.0]
    if not nonzero:
        return 1.0
    # Operational-policy rows can average several runtime-equivalent source
    # labels. Their paired differences are therefore small rational numbers
    # rather than only {-1, 0, 1}. Integerizing those rationals avoids a
    # floating-key state explosion in the exact dynamic program.
    fractions = [Fraction(value).limit_denominator(1_000_000) for value in nonzero]
    rational = all(
        math.isclose(float(fraction), value, rel_tol=0.0, abs_tol=1e-12)
        for fraction, value in zip(fractions, nonzero, strict=True)
    )
    if rational:
        scale = math.lcm(*(fraction.denominator for fraction in fractions))
        values: list[float | int] = [int(fraction * scale) for fraction in fractions]
    else:  # pragma: no cover - defensive fallback for future non-rational scores
        values = nonzero
    observed = abs(sum(values))
    distribution: dict[float | int, int] = {0: 1}
    for value in values:
        updated: dict[float | int, int] = defaultdict(int)
        for total, multiplicity in distribution.items():
            updated[total + value] += multiplicity
            updated[total - value] += multiplicity
        distribution = dict(updated)
    tolerance = 1e-12
    tail = sum(
        multiplicity
        for total, multiplicity in distribution.items()
        if abs(float(total)) + tolerance >= observed
    )
    return float(tail / (2 ** len(nonzero)))


def _cluster_bootstrap_weights(
    cluster_ids: Sequence[Any],
    *,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return row weights from a trajectory-cluster bootstrap.

    A selected dataset episode contributes all of its nonoverlapping windows.
    When every row has a unique trajectory this reduces to the ordinary paired
    episode bootstrap.
    """

    if len(cluster_ids) != 50:
        raise AnchorAnalysisError(f"Expected 50 episode cluster IDs, found {len(cluster_ids)}.")
    unique: list[Any] = []
    cluster_index: dict[Any, int] = {}
    row_clusters = []
    for raw in cluster_ids:
        # JSON-like IDs become stable dictionary keys.
        key = json.dumps(raw, sort_keys=True) if isinstance(raw, (dict, list)) else raw
        if key not in cluster_index:
            cluster_index[key] = len(unique)
            unique.append(key)
        row_clusters.append(cluster_index[key])
    if repeats <= 0:
        raise AnchorAnalysisError("bootstrap repeats must be positive.")
    cluster_count = len(unique)
    # Multinomial counts are exactly the occupancy counts from drawing
    # ``cluster_count`` clusters with replacement, but avoid the previous
    # O(repeats * cluster_count**2) equality scan.
    counts = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count, dtype=float),
        size=repeats,
    )
    row_weights = counts[:, np.asarray(row_clusters, dtype=np.int64)].astype(float)
    row_weights /= row_weights.sum(axis=1, keepdims=True)
    return row_weights


def _row_clusters(row: Mapping[str, Any]) -> list[Any]:
    raw = row.get("episode_cluster_ids")
    return list(raw) if isinstance(raw, list) else list(range(50))


def frontier_stability(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Paired trajectory-cluster bootstrap of sparse-frontier membership.

    Cost is held at its audited aggregate value.  One bootstrap draw resamples
    the same 50 episode indices for every cell in a matrix, preserving the
    experiment's common-random-numbers design.
    """

    if repeats <= 0:
        raise AnchorAnalysisError("bootstrap repeats must be positive.")
    # Alias executions are technical replicates, not additional policies or
    # independent episode samples.
    rows = _operational_representative_rows(rows)
    rng = np.random.default_rng(int(seed))
    by_matrix: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_matrix[str(row["canonical_config"])].append(row)
    output: list[dict[str, Any]] = []
    for matrix_key in sorted(by_matrix):
        matrix_rows = sorted(
            by_matrix[matrix_key],
            key=lambda row: (_schedule_number(str(row["schedule"])), str(row["anchor_id"])),
        )
        outcomes = np.asarray([row["discovery_outcomes"] for row in matrix_rows], dtype=float)
        costs = np.asarray([row["dynamics_flops_per_episode"] for row in matrix_rows], dtype=float)
        empirical = _pareto_mask(costs, outcomes.mean(axis=1))
        epsilon_frontier = _epsilon_pareto_mask(costs, outcomes.mean(axis=1), epsilon=0.05)
        counts = np.zeros(len(matrix_rows), dtype=np.int64)
        clusters = _row_clusters(matrix_rows[0])
        if any(_row_clusters(row) != clusters for row in matrix_rows[1:]):
            raise AnchorAnalysisError(f"Episode cluster identities differ within {matrix_key}.")
        weights = _cluster_bootstrap_weights(clusters, repeats=repeats, rng=rng)
        boot_success = weights @ outcomes.T
        for replicate in range(repeats):
            counts += _pareto_mask(costs, boot_success[replicate])
        best_counts = np.zeros(len(matrix_rows), dtype=np.int64)
        rank_sums = np.zeros(len(matrix_rows), dtype=float)
        for anchor_id in ("a1", "a2", "a3", "a4", "a5"):
            positions = np.asarray(
                [pos for pos, row in enumerate(matrix_rows) if row["anchor_id"] == anchor_id],
                dtype=np.int64,
            )
            scores = boot_success[:, positions]
            best_counts[positions] = np.sum(scores == scores.max(axis=1, keepdims=True), axis=0)
            competition_ranks = 1 + np.sum(
                scores[:, :, None] < scores[:, None, :], axis=2
            )
            rank_sums[positions] = competition_ranks.sum(axis=0)
        # Cluster weights can sum to 1 +/- a few ulps, so an all-success cell
        # may otherwise serialize an impossible upper endpoint such as
        # 1.0000000000000004.
        rate_intervals = np.clip(
            np.quantile(boot_success, [0.025, 0.975], axis=0),
            0.0,
            1.0,
        )
        for position, row in enumerate(matrix_rows):
            output.append(
                {
                    "canonical_config": matrix_key,
                    "env_id": str(row["env_id"]),
                    "goal_offset": int(row["goal_offset"]),
                    "matrix_index": int(row["matrix_index"]),
                    "schedule": str(row["schedule"]),
                    "policy_id": f"P{_schedule_number(str(row['schedule'])):02d}",
                    "anchor_id": str(row["anchor_id"]),
                    "success_rate_fraction": float(outcomes[position].mean()),
                    "dynamics_flops_per_episode": float(costs[position]),
                    "empirical_frontier": bool(empirical[position]),
                    "epsilon_5pp_frontier": bool(epsilon_frontier[position]),
                    "paired_trajectory_cluster_bootstrap_frontier_selection_frequency": float(
                        counts[position] / repeats
                    ),
                    "paired_trajectory_cluster_bootstrap_best_at_anchor_selection_frequency": float(
                        best_counts[position] / repeats
                    ),
                    "paired_trajectory_cluster_bootstrap_mean_competition_rank_at_anchor": float(
                        rank_sums[position] / repeats
                    ),
                    "paired_trajectory_cluster_bootstrap_success_95_interval": [
                        float(rate_intervals[0, position]),
                        float(rate_intervals[1, position]),
                    ],
                }
            )
    return output


def _percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return [float(low), float(high)]


def _fixed_design_paired_bootstrap(
    pair_records: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    seed: int,
    weight_cache: dict[tuple[Any, ...], np.ndarray] | None = None,
) -> list[float]:
    """Bootstrap episodes within fixed matrices, preserving known pairing.

    Matrices from the same environment that have an identical ordered cluster
    panel reuse the same bootstrap weights.  This preserves cross-goal
    dependence where the benchmark reuses the exact dataset episodes, while
    matrices with different panels remain separately stratified.
    """

    by_matrix: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in pair_records:
        by_matrix[str(record["canonical_config"])].append(
            np.asarray(record["episode_differences"], dtype=float)
        )
    total_pairs = sum(len(values) for values in by_matrix.values())
    if total_pairs == 0:
        return [math.nan, math.nan]
    estimates = np.zeros(repeats, dtype=float)
    for matrix_key in sorted(by_matrix):
        # Average over all matched schedule/anchor contrasts in the matrix at
        # each episode, then use a shared resample for those correlated cells.
        episode_sum = np.stack(by_matrix[matrix_key], axis=0).sum(axis=0)
        records = [record for record in pair_records if str(record["canonical_config"]) == matrix_key]
        clusters = list(records[0]["episode_cluster_ids"])
        if any(list(record["episode_cluster_ids"]) != clusters for record in records[1:]):
            raise AnchorAnalysisError(f"Contrast cluster identities differ within {matrix_key}.")
        normalized_clusters = tuple(
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
            for value in clusters
        )
        env_values = {str(record["env_id"]) for record in records}
        if len(env_values) != 1:
            raise AnchorAnalysisError(f"Contrast environment identities differ within {matrix_key}.")
        env_id = next(iter(env_values))
        # The goal-25 and goal-50 OGB Cube/Reacher matrices reuse an identical
        # ordered dataset-episode panel. Keying by panel rather than config is
        # what preserves that known cross-goal dependence.
        cache_key = (env_id, normalized_clusters, int(repeats), int(seed))
        weights = weight_cache.get(cache_key) if weight_cache is not None else None
        if weights is None:
            # Panel-specific deterministic streams make a summary invariant to
            # which other strata happen to be requested and synchronize any
            # identical episode panel reused across goals.
            matrix_seed = int(
                _canonical_sha256([int(seed), env_id, normalized_clusters])[:16],
                16,
            )
            weights = _cluster_bootstrap_weights(
                clusters,
                repeats=repeats,
                rng=np.random.default_rng(matrix_seed),
            )
            if weight_cache is not None:
                weight_cache[cache_key] = weights
        estimates += (weights @ episode_sum) / total_pairs
    return _percentile_interval(estimates)


def _lowest_realized_policy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select one fixed lowest-numbered label for each realized policy."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["canonical_config"]),
            str(row["anchor_id"]),
            realized_policy_signature(str(row["schedule_label"])),
        )
        grouped[key].append(row)
    representatives = []
    for values in grouped.values():
        representatives.append(dict(min(values, key=lambda row: _schedule_number(str(row["schedule"])))))
    return representatives


def _operational_representative_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse source aliases into one episode-clustered operational row.

    Alias labels exercise the same realized fidelity decisions but were run as
    separate GPU processes. Execution or mutable-provenance differences can
    make equal configured seeds yield different action trajectories. Treating
    those runs as extra independent episodes would be pseudoreplication, while
    choosing one arbitrary label would discard useful negative-control data.
    We therefore average aliases *within each of the same 50 episode
    identities*. The downstream bootstrap and sign-flip tests continue to
    resample or flip only at the dataset-episode cluster level.
    """

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["canonical_config"]),
            str(row["anchor_id"]),
            realized_policy_signature(str(row["schedule_label"])),
        )
        grouped[key].append(row)
    representatives: list[dict[str, Any]] = []
    for (matrix_key, anchor_id, signature), values in grouped.items():
        ordered = sorted(values, key=lambda row: _schedule_number(str(row["schedule"])))
        reference = ordered[0]
        reference_clusters = _row_clusters(reference)
        for candidate in ordered[1:]:
            for field in ("dynamics_flops_per_episode", "latent_work_per_episode"):
                if not math.isclose(
                    float(candidate.get(field, math.nan)),
                    float(reference.get(field, math.nan)),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise AnchorAnalysisError(
                        "Runtime-equivalent aliases have unequal audited work; refusing "
                        f"scientific collapse: {matrix_key} {anchor_id} signature={signature}, "
                        f"field={field}, reference={reference.get('schedule')}, "
                        f"candidate={candidate.get('schedule')}."
                    )
            if _row_clusters(candidate) != reference_clusters:
                raise AnchorAnalysisError(
                    "Runtime-equivalent aliases do not share episode clusters; refusing "
                    f"scientific collapse: {matrix_key} {anchor_id} signature={signature}."
                )
        outcomes = np.asarray(
            [row["discovery_outcomes"] for row in ordered],
            dtype=float,
        )
        if outcomes.shape != (len(ordered), 50):
            raise AnchorAnalysisError(
                f"Malformed alias outcome block at {matrix_key} {anchor_id} signature={signature}."
            )
        episode_means = outcomes.mean(axis=0)
        collapsed = dict(reference)
        collapsed["discovery_outcomes"] = episode_means.tolist()
        collapsed["successes"] = float(episode_means.sum())
        collapsed["success_rate_fraction"] = float(episode_means.mean())
        collapsed["success_rate_percent"] = float(100.0 * episode_means.mean())
        collapsed["alias_replicates"] = len(ordered)
        collapsed["alias_schedules"] = [str(row["schedule"]) for row in ordered]
        collapsed["alias_matrix_indices"] = [int(row["matrix_index"]) for row in ordered]
        collapsed["alias_source_kinds"] = [str(row["source_kind"]) for row in ordered]
        collapsed["alias_discordant_episode_count"] = int(
            np.sum(np.ptp(outcomes, axis=0) > 0)
        )
        rates = outcomes.mean(axis=1)
        collapsed["alias_success_rate_range_fraction"] = [
            float(np.min(rates)),
            float(np.max(rates)),
        ]
        representatives.append(collapsed)
    return representatives


def equivalence_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_signature: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_signature[realized_policy_signature(str(row["schedule_label"]))].add(str(row["schedule"]))
    classes = [sorted(values, key=_schedule_number) for values in by_signature.values() if len(values) > 1]
    classes.sort(key=lambda values: _schedule_number(values[0]))
    replicate_disagreements: list[dict[str, Any]] = []
    invariant_violations: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["canonical_config"]),
                str(row["anchor_id"]),
                realized_policy_signature(str(row["schedule_label"])),
            )
        ].append(row)
    comparisons = 0
    for (matrix_key, anchor_id, signature), values in grouped.items():
        if len(values) <= 1:
            continue
        reference = min(values, key=lambda row: _schedule_number(str(row["schedule"])))
        for candidate in values:
            if candidate is reference:
                continue
            comparisons += 1
            outcomes_equal = candidate["discovery_outcomes"] == reference["discovery_outcomes"]
            cost_equal = math.isclose(
                float(candidate["dynamics_flops_per_episode"]),
                float(reference["dynamics_flops_per_episode"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            left = np.asarray(reference["discovery_outcomes"], dtype=float)
            right = np.asarray(candidate["discovery_outcomes"], dtype=float)
            difference = right - left
            record = {
                "canonical_config": matrix_key,
                "env_id": str(reference["env_id"]),
                "goal_offset": int(reference["goal_offset"]),
                "anchor_id": anchor_id,
                "realized_policy_signature": signature,
                "reference": str(reference["schedule"]),
                "candidate": str(candidate["schedule"]),
                "reference_source_kind": str(reference["source_kind"]),
                "candidate_source_kind": str(candidate["source_kind"]),
                "outcomes_equal": outcomes_equal,
                "audited_cost_equal": cost_equal,
                "discordant_episodes": int(np.sum(difference != 0)),
                "success_delta_fraction": float(difference.mean()),
            }
            if not outcomes_equal:
                replicate_disagreements.append(record)
            if not cost_equal or _row_clusters(candidate) != _row_clusters(reference):
                invariant_violations.append(
                    {
                        **record,
                        "episode_clusters_equal": (
                            _row_clusters(candidate) == _row_clusters(reference)
                        ),
                    }
                )
    discordances = np.asarray(
        [row["discordant_episodes"] for row in replicate_disagreements], dtype=float
    )
    absolute_rate_deltas = np.asarray(
        [abs(float(row["success_delta_fraction"])) for row in replicate_disagreements],
        dtype=float,
    )
    return {
        "operational_schedule_count": len(by_signature),
        "policy_classes": [
            {
                "policy_id": f"P{_schedule_number(values[0]):02d}",
                "representative": values[0],
                "aliases": values,
                "realized_policy_signature": realized_policy_signature(
                    str(
                        next(
                            row["schedule_label"]
                            for row in rows
                            if str(row["schedule"]) == values[0]
                        )
                    )
                ),
            }
            for values in sorted(
                (sorted(values, key=_schedule_number) for values in by_signature.values()),
                key=lambda values: _schedule_number(values[0]),
            )
        ],
        "alias_classes": classes,
        "paired_alias_comparisons": comparisons,
        "exact_identity_comparisons": comparisons - len(replicate_disagreements),
        "outcome_nonidentity_comparisons": len(replicate_disagreements),
        "replicate_disagreements": replicate_disagreements,
        # Compatibility alias for readers of the v2 report.
        "mismatches": replicate_disagreements,
        "scientific_invariant_violations": invariant_violations,
        "replicate_discordant_episode_total": int(discordances.sum()),
        "replicate_discordant_episode_mean_per_comparison": (
            float(discordances.sum() / comparisons) if comparisons else 0.0
        ),
        "replicate_discordant_episode_max": (
            int(discordances.max()) if len(discordances) else 0
        ),
        "mean_absolute_alias_success_delta_fraction": (
            float(absolute_rate_deltas.sum() / comparisons) if comparisons else 0.0
        ),
        "max_absolute_alias_success_delta_fraction": (
            float(absolute_rate_deltas.max()) if len(absolute_rate_deltas) else 0.0
        ),
        "collapse_method": (
            "mean source-label outcome within each fixed dataset-episode cluster; "
            "aliases do not increase the episode sample size"
        ),
    }


_PLANNED_CONTRASTS: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {
    # Within a realized MPC/CEM policy, change only rollout fidelity.
    "rollout_fine_to_base_minus_fixed": (
        "rollout",
        ((6, 7), (18, 19), (21, 22), (24, 25)),
    ),
    "rollout_base_to_coarse_minus_fixed": (
        "rollout",
        ((1, 2), (6, 8), (18, 20), (21, 23), (24, 26)),
    ),
    "rollout_base_to_coarse_minus_fine_to_base": (
        "rollout",
        ((7, 8), (19, 20), (22, 23), (25, 26)),
    ),
    # CEM contrasts are cleanly identified with dynamic MPC.  With fixed-fine
    # MPC, `base -> fine` degenerates to fixed-fine and `coarse -> base`
    # degenerates to `coarse -> fine`.
    "cem_coarse_to_base_minus_fixed_dynamic_mpc": (
        "cem",
        ((18, 21), (19, 22), (20, 23)),
    ),
    "cem_base_to_fine_minus_fixed_dynamic_mpc": (
        "cem",
        ((18, 24), (19, 25), (20, 26)),
    ),
    "cem_base_to_fine_minus_coarse_to_base_dynamic_mpc": (
        "cem",
        ((21, 24), (22, 25), (23, 26)),
    ),
    "cem_coarse_to_fine_minus_fixed_fixed_mpc": (
        "cem",
        ((1, 6), (1, 7), (2, 8)),
    ),
    # This changes only the *configured* MPC stage.  Because downstream
    # `base` tokens reference MPC, it is a total outer-policy intervention,
    # not a direct isolated MPC-compute effect.
    "mpc_coarse_to_fine_minus_fixed_total_effect": (
        "mpc_total",
        ((1, 18), (1, 19), (2, 20), (6, 21), (7, 22), (8, 23), (1, 24), (1, 25), (2, 26)),
    ),
    # Nearly matched fidelity mass placed on different clocks: within-plan
    # optimizer progress (P06-08) versus across real replans (P18-20).
    "mpc_clock_minus_cem_clock": (
        "schedule_clock",
        ((6, 18), (7, 19), (8, 20)),
    ),
}


def _planned_contrast_records(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    representatives = _operational_representative_rows(rows)
    cells: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in representatives:
        cells[
            (
                str(row["canonical_config"]),
                str(row["anchor_id"]),
                _schedule_number(str(row["schedule"])),
            )
        ] = row
    matrix_anchors = sorted({(key[0], key[1]) for key in cells})
    result: dict[str, list[dict[str, Any]]] = {}
    for name, (_, policy_pairs) in _PLANNED_CONTRASTS.items():
        records: list[dict[str, Any]] = []
        for matrix_key, anchor_id in matrix_anchors:
            for reference_number, treatment_number in policy_pairs:
                reference = cells.get((matrix_key, anchor_id, reference_number))
                candidate = cells.get((matrix_key, anchor_id, treatment_number))
                if reference is None or candidate is None:
                    raise AnchorAnalysisError(
                        f"Planned contrast {name} is missing P{reference_number:02d}/"
                        f"P{treatment_number:02d} at {matrix_key} {anchor_id}."
                    )
                reference_outcomes = np.asarray(reference["discovery_outcomes"], dtype=float)
                candidate_outcomes = np.asarray(candidate["discovery_outcomes"], dtype=float)
                reference_clusters = _row_clusters(reference)
                if _row_clusters(candidate) != reference_clusters:
                    raise AnchorAnalysisError(
                        f"Planned contrast {name} has unequal trajectory clusters at "
                        f"{matrix_key} {anchor_id}."
                    )
                records.append(
                    {
                        "canonical_config": matrix_key,
                        "env_id": str(reference["env_id"]),
                        "goal_offset": int(reference["goal_offset"]),
                        "anchor_id": anchor_id,
                        "reference_schedule": str(reference["schedule"]),
                        "treatment_schedule": str(candidate["schedule"]),
                        "episode_differences": (candidate_outcomes - reference_outcomes).tolist(),
                        "episode_cluster_ids": reference_clusters,
                        "success_delta": float(candidate_outcomes.mean() - reference_outcomes.mean()),
                        "cost_delta": float(candidate["dynamics_flops_per_episode"])
                        - float(reference["dynamics_flops_per_episode"]),
                        "cost_ratio": float(candidate["dynamics_flops_per_episode"])
                        / float(reference["dynamics_flops_per_episode"]),
                    }
                )
        result[name] = records
    return result


def alias_estimator_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare alias-mean point estimates with one fixed representative label.

    The primary estimator averages technically repeated executions within each
    episode.  Because alias classes have unequal replicate counts and execution
    variance is not resampled, this sensitivity analysis repeats every
    structural contrast after selecting the lowest-numbered source label in
    each realized-policy alias class.
    """

    primary = _planned_contrast_records(rows)
    representative = _planned_contrast_records(_lowest_realized_policy_rows(rows))
    comparisons: list[dict[str, Any]] = []
    for name in _PLANNED_CONTRASTS:
        primary_values = np.asarray(
            [record["success_delta"] for record in primary[name]],
            dtype=float,
        )
        representative_values = np.asarray(
            [record["success_delta"] for record in representative[name]],
            dtype=float,
        )
        primary_mean = float(primary_values.mean())
        representative_mean = float(representative_values.mean())
        comparisons.append(
            {
                "contrast": name,
                "alias_mean_success_delta_fraction": primary_mean,
                "lowest_source_label_success_delta_fraction": representative_mean,
                "lowest_minus_alias_mean_fraction": representative_mean - primary_mean,
            }
        )
    largest = max(
        comparisons,
        key=lambda row: abs(float(row["lowest_minus_alias_mean_fraction"])),
    )
    return {
        "primary_estimator": "episode-wise mean across realized-policy source-label aliases",
        "sensitivity_estimator": "lowest-numbered source label per realized-policy alias class",
        "execution_variance_modeled": False,
        "contrasts": comparisons,
        "maximum_absolute_shift_fraction": abs(
            float(largest["lowest_minus_alias_mean_fraction"])
        ),
        "maximum_shift_contrast": str(largest["contrast"]),
    }


_TASK_STRUCTURE_HYPOTHESES = (
    {
        "env_id": "swm/ReacherDMControl-v0",
        "covariate": "expert_joint_path_tortuosity",
        "split": "within_matrix_median",
        "contrast": "rollout_fine_to_base_minus_fixed",
        "mechanism": "curved expert paths may amplify compounding rollout error",
    },
    {
        "env_id": "swm/PushT-v1",
        "covariate": "block_rotation_rad",
        "split": "within_matrix_median",
        "contrast": "rollout_fine_to_base_minus_fixed",
        "mechanism": "larger orientation changes may increase sensitivity to early rollout fidelity",
    },
    {
        "env_id": "swm/OGBCube-v0",
        "covariate": "cube_translation_m",
        "split": "within_matrix_median",
        "contrast": "cem_base_to_fine_minus_fixed_dynamic_mpc",
        "mechanism": "larger transport displacements may sharpen late CEM ranking demands",
    },
    {
        "env_id": "swm/TwoRoom-v1",
        "covariate": "cross_room",
        "split": "boolean",
        "contrast": "mpc_clock_minus_cem_clock",
        "mechanism": "cross-room windows add a doorway bottleneck and global-route decision",
    },
)


def _standardized_subgroup_summary(
    record_groups: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> dict[str, float | int]:
    """Equal-weight within-record subgroup means and effect modifiers."""

    complete = [
        (np.asarray(low, dtype=float), np.asarray(high, dtype=float))
        for low, high in record_groups
        if len(low) and len(high)
    ]
    if not complete:
        raise AnchorAnalysisError("No matched record contains both task-structure subgroups.")
    low_means = np.asarray([float(low.mean()) for low, _ in complete], dtype=float)
    high_means = np.asarray([float(high.mean()) for _, high in complete], dtype=float)
    modifiers = high_means - low_means
    return {
        "low_or_false_mean": float(low_means.mean()),
        "high_or_true_mean": float(high_means.mean()),
        "high_or_true_minus_low_or_false": float(modifiers.mean()),
        "matched_records": len(complete),
    }


def task_structure_heterogeneity(
    rows: Sequence[Mapping[str, Any]],
    covariate_summary: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Descriptive, hypothesis-led schedule effects split by task structure.

    These are deliberately point-estimate diagnostics. Each episode difference
    is paired and low/high means are first computed within a matched
    matrix-anchor-policy-pair record before records receive equal weight. This
    prevents unequal subgroup prevalence across goal matrices from inducing a
    pooled Simpson effect. Repeated records and reused dataset episodes remain
    correlated, so raw cell-episode counts are not inferential sample sizes.
    """

    if not covariate_summary or covariate_summary.get("status") != "passed":
        return []
    raw_records = covariate_summary.get("episode_records")
    if not isinstance(raw_records, list):
        return []
    covariates = {
        (str(record["canonical_config"]), int(record["episode_index"])): dict(
            record["covariates"]
        )
        for record in raw_records
    }
    dataset_episode = {
        (str(record["canonical_config"]), int(record["episode_index"])): int(
            record["dataset_episode"]
        )
        for record in raw_records
    }
    medians = {
        (str(matrix["canonical_config"]), str(name)): summary.get("median")
        for matrix in covariate_summary.get("matrix_structure", [])
        for name, summary in dict(matrix.get("covariates", {})).items()
        if summary.get("kind") == "numeric"
    }
    contrast_records = _planned_contrast_records(rows)
    output: list[dict[str, Any]] = []
    for hypothesis in _TASK_STRUCTURE_HYPOTHESES:
        record_groups: list[tuple[list[float], list[float]]] = []
        observation_counts = {"low_or_false": 0, "high_or_true": 0}
        cluster_ids: dict[str, set[tuple[str, int]]] = {
            "low_or_false": set(),
            "high_or_true": set(),
        }
        for record in contrast_records[str(hypothesis["contrast"])]:
            if str(record["env_id"]) != str(hypothesis["env_id"]):
                continue
            matrix_key = str(record["canonical_config"])
            within_record: dict[str, list[float]] = {
                "low_or_false": [],
                "high_or_true": [],
            }
            for episode_index, difference in enumerate(record["episode_differences"]):
                value = covariates[(matrix_key, episode_index)].get(hypothesis["covariate"])
                if value is None:
                    continue
                if hypothesis["split"] == "boolean":
                    group = "high_or_true" if bool(value) else "low_or_false"
                else:
                    median = medians[(matrix_key, str(hypothesis["covariate"]))]
                    if median is None:
                        continue
                    group = "high_or_true" if float(value) >= float(median) else "low_or_false"
                within_record[group].append(float(difference))
                observation_counts[group] += 1
                cluster_ids[group].add(
                    (str(record["env_id"]), dataset_episode[(matrix_key, episode_index)])
                )
            record_groups.append(
                (within_record["low_or_false"], within_record["high_or_true"])
            )
        try:
            standardized = _standardized_subgroup_summary(record_groups)
        except AnchorAnalysisError as exc:
            raise AnchorAnalysisError(f"Empty task-structure subgroup for {hypothesis}.") from exc
        output.append(
            {
                **hypothesis,
                "low_or_false_mean_success_delta_fraction": standardized["low_or_false_mean"],
                "high_or_true_mean_success_delta_fraction": standardized["high_or_true_mean"],
                "high_or_true_minus_low_or_false_fraction": standardized[
                    "high_or_true_minus_low_or_false"
                ],
                "standardization": (
                    "compute subgroup means within each matrix-anchor-policy-pair record, "
                    "then weight records equally"
                ),
                "matched_records": standardized["matched_records"],
                "cell_episode_observations": observation_counts,
                "distinct_environment_dataset_episode_clusters": {
                    "low_or_false": len(cluster_ids["low_or_false"]),
                    "high_or_true": len(cluster_ids["high_or_true"]),
                },
                "inferential_interval": None,
            }
        )
    return output


def _summarize_contrast_records(
    name: str,
    records: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    seed: int,
    weight_cache: dict[tuple[Any, ...], np.ndarray] | None = None,
) -> dict[str, Any]:
    axis, policy_pairs = _PLANNED_CONTRASTS[name]
    deltas = np.asarray([record["success_delta"] for record in records], dtype=float)
    cost_ratios = np.asarray([record["cost_ratio"] for record in records], dtype=float)
    cost_deltas = np.asarray([record["cost_delta"] for record in records], dtype=float)
    return {
        "contrast": name,
        "axis": axis,
        "policy_pairs": [f"P{left:02d}->P{right:02d}" for left, right in policy_pairs],
        "matched_cell_pairs": len(records),
        "matrices": len({str(record["canonical_config"]) for record in records}),
        "mean_paired_success_delta_fraction": float(deltas.mean()) if len(deltas) else None,
        "median_paired_success_delta_fraction": float(np.median(deltas)) if len(deltas) else None,
        "fixed_matrix_paired_episode_cluster_bootstrap_95_interval": (
            _fixed_design_paired_bootstrap(
                records,
                repeats=repeats,
                seed=seed,
                weight_cache=weight_cache,
            )
            if records
            else None
        ),
        "cell_pair_direction_counts": {
            "treatment_better": int(np.sum(deltas > 0)),
            "tied": int(np.sum(deltas == 0)),
            "treatment_worse": int(np.sum(deltas < 0)),
        },
        "mean_cost_ratio": float(cost_ratios.mean()) if len(cost_ratios) else None,
        "mean_cost_delta_per_episode": float(cost_deltas.mean()) if len(cost_deltas) else None,
    }


def stage_contrasts(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    records_by_contrast = _planned_contrast_records(rows)
    results: list[dict[str, Any]] = []
    for contrast_position, name in enumerate(_PLANNED_CONTRASTS):
        weight_cache: dict[tuple[Any, ...], np.ndarray] = {}
        results.append(
            _summarize_contrast_records(
                name,
                records_by_contrast[name],
                repeats=repeats,
                seed=seed + contrast_position,
                weight_cache=weight_cache,
            )
        )
    return results


def stage_contrast_strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Summarize each contrast by matrix, anchor, and matrix-anchor interaction."""

    records_by_contrast = _planned_contrast_records(rows)
    results: list[dict[str, Any]] = []
    groupers = {
        "matrix": lambda record: (str(record["canonical_config"]),),
        "anchor": lambda record: (str(record["anchor_id"]),),
        "matrix_anchor": lambda record: (
            str(record["canonical_config"]),
            str(record["anchor_id"]),
        ),
    }
    for contrast_position, name in enumerate(_PLANNED_CONTRASTS):
        records = records_by_contrast[name]
        contrast_seed = seed + contrast_position
        weight_cache: dict[tuple[Any, ...], np.ndarray] = {}
        for scope, grouper in groupers.items():
            grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                grouped[grouper(record)].append(record)
            for group_key in sorted(grouped):
                values = grouped[group_key]
                summary = _summarize_contrast_records(
                    name,
                    values,
                    repeats=repeats,
                    seed=contrast_seed,
                    weight_cache=weight_cache,
                )
                matrix_values = {str(record["canonical_config"]) for record in values}
                env_values = {str(record["env_id"]) for record in values}
                goal_values = {int(record["goal_offset"]) for record in values}
                anchor_values = {str(record["anchor_id"]) for record in values}
                summary.update(
                    {
                        "scope": scope,
                        "canonical_config": next(iter(matrix_values)) if len(matrix_values) == 1 else None,
                        "env_id": next(iter(env_values)) if len(env_values) == 1 else None,
                        "goal_offset": next(iter(goal_values)) if len(goal_values) == 1 else None,
                        "anchor_id": next(iter(anchor_values)) if len(anchor_values) == 1 else None,
                    }
                )
                results.append(summary)
    return results


def paired_cell_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Trajectory-cluster-aware tests for each explicit exploratory contrast."""

    representatives = _operational_representative_rows(rows)
    cells = {
        (
            str(row["canonical_config"]),
            str(row["anchor_id"]),
            _schedule_number(str(row["schedule"])),
        ): row
        for row in representatives
    }
    matrix_anchors = sorted({(key[0], key[1]) for key in cells})
    output: list[dict[str, Any]] = []
    for contrast, (axis, policy_pairs) in _PLANNED_CONTRASTS.items():
        family_rows = []
        for matrix_key, anchor_id in matrix_anchors:
            for reference_number, treatment_number in policy_pairs:
                reference = cells[(matrix_key, anchor_id, reference_number)]
                treatment = cells[(matrix_key, anchor_id, treatment_number)]
                left = np.asarray(reference["discovery_outcomes"], dtype=float)
                right = np.asarray(treatment["discovery_outcomes"], dtype=float)
                clusters = _row_clusters(reference)
                if _row_clusters(treatment) != clusters:
                    raise AnchorAnalysisError(
                        f"Cell contrast {contrast} has unequal trajectory clusters at "
                        f"{matrix_key} {anchor_id}."
                    )
                differences = right - left
                treatment_mass = float(np.maximum(differences, 0).sum())
                reference_mass = float(np.maximum(-differences, 0).sum())
                discordant = int(np.sum(differences != 0))
                raw_p = _exact_cluster_sign_flip_p(differences.tolist(), clusters)
                family_rows.append(
                    {
                        "contrast_family": contrast,
                        "axis": axis,
                        "canonical_config": matrix_key,
                        "env_id": str(reference["env_id"]),
                        "goal_offset": int(reference["goal_offset"]),
                        "anchor_id": anchor_id,
                        "reference_policy": f"P{reference_number:02d}",
                        "treatment_policy": f"P{treatment_number:02d}",
                        "treatment_only_successes": treatment_mass,
                        "reference_only_successes": reference_mass,
                        "treatment_only_success_mass": treatment_mass,
                        "reference_only_success_mass": reference_mass,
                        "discordant_episodes": discordant,
                        "trajectory_clusters": len(
                            {
                                json.dumps(value, sort_keys=True)
                                if isinstance(value, (dict, list))
                                else value
                                for value in clusters
                            }
                        ),
                        "paired_risk_difference_fraction": float(differences.mean()),
                        "raw_two_sided_exact_cluster_sign_flip_p": raw_p,
                        "observed_absolute_difference_within_5pp": bool(
                            abs(float(differences.mean())) <= 0.05
                        ),
                    }
                )
        # Holm controls family-wise error within the explicitly named contrast
        # family; it is not used to turn the exploratory study confirmatory.
        order = sorted(
            range(len(family_rows)),
            key=lambda pos: family_rows[pos]["raw_two_sided_exact_cluster_sign_flip_p"],
        )
        running = 0.0
        adjusted = [1.0] * len(family_rows)
        for rank, position in enumerate(order):
            candidate = min(
                1.0,
                (len(family_rows) - rank)
                * float(family_rows[position]["raw_two_sided_exact_cluster_sign_flip_p"]),
            )
            running = max(running, candidate)
            adjusted[position] = running
        for row, value in zip(family_rows, adjusted, strict=True):
            row["holm_familywise_adjusted_cluster_sign_flip_p"] = value
        output.extend(family_rows)
    return output


def schedule_summaries(
    rows: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _operational_representative_rows(rows)
    probability = {
        (str(row["canonical_config"]), int(row["matrix_index"])): float(
            row["paired_trajectory_cluster_bootstrap_frontier_selection_frequency"]
        )
        for row in stability
    }
    best_probability = {
        (str(row["canonical_config"]), int(row["matrix_index"])): float(
            row["paired_trajectory_cluster_bootstrap_best_at_anchor_selection_frequency"]
        )
        for row in stability
    }
    bootstrap_rank = {
        (str(row["canonical_config"]), int(row["matrix_index"])): float(
            row["paired_trajectory_cluster_bootstrap_mean_competition_rank_at_anchor"]
        )
        for row in stability
    }
    by_stratum: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[(str(row["canonical_config"]), str(row["anchor_id"]))].append(row)
    regrets: dict[tuple[str, str, str], float] = {}
    ranks: dict[tuple[str, str, str], float] = {}
    for (matrix_key, anchor_id), values in by_stratum.items():
        rates = np.asarray([float(row["success_rate_fraction"]) for row in values])
        best = float(np.max(rates))
        # Average ranks make exact ties explicit.
        unique = sorted(set(rates), reverse=True)
        rank_for_rate: dict[float, float] = {}
        position = 1
        for rate in unique:
            count = int(np.sum(rates == rate))
            rank_for_rate[float(rate)] = position + (count - 1) / 2
            position += count
        for row in values:
            key = (matrix_key, anchor_id, str(row["schedule"]))
            rate = float(row["success_rate_fraction"])
            regrets[key] = best - rate
            ranks[key] = rank_for_rate[rate]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["schedule"])].append(row)
    output = []
    for schedule in sorted(grouped, key=_schedule_number):
        values = grouped[schedule]
        rates = np.asarray([float(row["success_rate_fraction"]) for row in values])
        costs = np.asarray([float(row["dynamics_flops_per_episode"]) for row in values])
        schedule_regrets = np.asarray(
            [regrets[(str(row["canonical_config"]), str(row["anchor_id"]), schedule)] for row in values]
        )
        schedule_ranks = np.asarray(
            [ranks[(str(row["canonical_config"]), str(row["anchor_id"]), schedule)] for row in values]
        )
        probs = np.asarray(
            [probability[(str(row["canonical_config"]), int(row["matrix_index"]))] for row in values]
        )
        best_probs = np.asarray(
            [best_probability[(str(row["canonical_config"]), int(row["matrix_index"]))] for row in values]
        )
        boot_ranks = np.asarray(
            [bootstrap_rank[(str(row["canonical_config"]), int(row["matrix_index"]))] for row in values]
        )
        anchor_rates = {
            anchor: float(
                np.mean(
                    [
                        float(row["success_rate_fraction"])
                        for row in values
                        if row["anchor_id"] == anchor
                    ]
                )
            )
            for anchor in ("a1", "a2", "a3", "a4", "a5")
        }
        output.append(
            {
                "policy_id": f"P{_schedule_number(schedule):02d}",
                "schedule": schedule,
                "schedule_label": str(values[0]["schedule_label"]),
                "source_axis_signature": list(source_axis_signature(str(values[0]["schedule_label"]))),
                "realized_policy_signature": realized_policy_signature(
                    str(values[0]["schedule_label"])
                ),
                "cells": len(values),
                "mean_success_rate_fraction": float(rates.mean()),
                "mean_success_regret_to_stratum_best": float(schedule_regrets.mean()),
                "mean_success_rank": float(schedule_ranks.mean()),
                "empirical_frontier_cells": int(
                    sum(
                        bool(
                            next(
                                item["empirical_frontier"]
                                for item in stability
                                if str(item["canonical_config"]) == str(row["canonical_config"])
                                and int(item["matrix_index"]) == int(row["matrix_index"])
                            )
                        )
                        for row in values
                    )
                ),
                "epsilon_5pp_frontier_cells": int(
                    sum(
                        bool(
                            next(
                                item["epsilon_5pp_frontier"]
                                for item in stability
                                if str(item["canonical_config"]) == str(row["canonical_config"])
                                and int(item["matrix_index"]) == int(row["matrix_index"])
                            )
                        )
                        for row in values
                    )
                ),
                "mean_bootstrap_frontier_selection_frequency": float(probs.mean()),
                "frontier_selection_frequency_mass": float(probs.sum()),
                "mean_bootstrap_best_at_anchor_selection_frequency": float(best_probs.mean()),
                "mean_bootstrap_competition_rank_at_anchor": float(boot_ranks.mean()),
                "mean_dynamics_flops_per_episode": float(costs.mean()),
                "success_rate_by_anchor": anchor_rates,
                "a5_minus_a1_mean_success_fraction": anchor_rates["a5"] - anchor_rates["a1"],
            }
        )
    return output


def goal_rank_stability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = _operational_representative_rows(rows)
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["env_id"]), int(row["goal_offset"]), str(row["schedule"]))].append(
            float(row["success_rate_fraction"])
        )
    envs = sorted({key[0] for key in grouped})
    output = []
    for env_id in envs:
        schedules = sorted(
            {key[2] for key in grouped if key[0] == env_id and key[1] == 25},
            key=_schedule_number,
        )
        left = [float(np.mean(grouped[(env_id, 25, schedule)])) for schedule in schedules]
        right = [float(np.mean(grouped[(env_id, 50, schedule)])) for schedule in schedules]
        raw_rho = float(spearmanr(left, right).statistic)
        rho = raw_rho if math.isfinite(raw_rho) else None
        output.append(
            {
                "env_id": env_id,
                "schedules": len(schedules),
                "goal25_vs_goal50_spearman_schedule_success": rho,
            }
        )
    return output


def budget_response(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _operational_representative_rows(rows)
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["canonical_config"]), str(row["schedule"]))][str(row["anchor_id"])] = float(
            row["success_rate_fraction"]
        )
    endpoint_deltas = []
    monotone = 0
    violations = 0
    for rates in grouped.values():
        sequence = [rates[anchor] for anchor in ("a1", "a2", "a3", "a4", "a5")]
        endpoint_deltas.append(sequence[-1] - sequence[0])
        if all(right >= left for left, right in zip(sequence, sequence[1:])):
            monotone += 1
        else:
            violations += 1
    values = np.asarray(endpoint_deltas, dtype=float)
    return {
        "matrix_schedule_curves": len(grouped),
        "nondecreasing_curves": monotone,
        "curves_with_at_least_one_downward_step": violations,
        "mean_a5_minus_a1_success_fraction": float(values.mean()),
        "median_a5_minus_a1_success_fraction": float(np.median(values)),
        "a5_better_tied_worse_counts": {
            "better": int(np.sum(values > 0)),
            "tied": int(np.sum(values == 0)),
            "worse": int(np.sum(values < 0)),
        },
    }


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f} pp"


def _matrix_name(row: Mapping[str, Any]) -> str:
    env = str(row["env_id"]).split("/")[-1].replace("-v0", "").replace("-v1", "")
    return f"{env} goal-{int(row['goal_offset'])}"


def render_report(analysis: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    summaries = list(analysis["schedule_summaries"])
    robust = sorted(
        summaries,
        key=lambda row: (
            float(row["mean_success_regret_to_stratum_best"]),
            -float(row["mean_bootstrap_frontier_selection_frequency"]),
        ),
    )
    efficient = sorted(
        summaries,
        key=lambda row: (
            -float(row["frontier_selection_frequency_mass"]),
            float(row["mean_dynamics_flops_per_episode"]),
        ),
    )
    by_matrix: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _operational_representative_rows(rows):
        by_matrix[str(row["canonical_config"])].append(row)
    matrix_lines = []
    for matrix_key in sorted(by_matrix):
        values = by_matrix[matrix_key]
        by_schedule: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_schedule[str(row["schedule"])].append(row)
        ranked = sorted(
            by_schedule.items(),
            key=lambda item: -float(np.mean([float(row["success_rate_fraction"]) for row in item[1]])),
        )
        leaders = ", ".join(
            f"P{_schedule_number(schedule):02d} / {schedule} "
            f"({100*np.mean([float(row['success_rate_fraction']) for row in cells]):.1f}%)"
            for schedule, cells in ranked[:3]
        )
        matrix_lines.append(f"- {_matrix_name(values[0])}: {leaders}")
    frontier_lines = []
    for matrix_key in sorted(by_matrix):
        values = [
            row
            for row in analysis["frontier_stability"]
            if str(row["canonical_config"]) == matrix_key
        ]
        empirical = [row for row in values if bool(row["empirical_frontier"])]
        practical = [row for row in values if bool(row["epsilon_5pp_frontier"])]
        stable = [
            row
            for row in empirical
            if float(
                row["paired_trajectory_cluster_bootstrap_frontier_selection_frequency"]
            )
            >= 0.5
        ]
        leaders = sorted(
            empirical,
            key=lambda row: -float(
                row["paired_trajectory_cluster_bootstrap_frontier_selection_frequency"]
            ),
        )[:5]
        leader_text = ", ".join(
            f"{row['policy_id']}/{row['anchor_id']} "
            f"({100*float(row['success_rate_fraction']):.1f}%, "
            f"freq={float(row['paired_trajectory_cluster_bootstrap_frontier_selection_frequency']):.2f})"
            for row in leaders
        )
        frontier_lines.append(
            f"- {_matrix_name(by_matrix[matrix_key][0])}: {len(empirical)} empirical cells, "
            f"{len(practical)} five-point-practical cells, {len(stable)} empirical cells "
            f"with selection frequency ≥0.50; highest-frequency examples: {leader_text}."
        )
    contrast_lines = []
    for contrast in analysis["stage_contrasts"]:
        interval = contrast["fixed_matrix_paired_episode_cluster_bootstrap_95_interval"]
        interval_text = (
            "n/a"
            if interval is None
            else f"[{100*interval[0]:+.1f}, {100*interval[1]:+.1f}] pp"
        )
        contrast_lines.append(
            f"- `{contrast['contrast']}`: {_fmt_pct(contrast['mean_paired_success_delta_fraction'])} "
            f"(paired trajectory-cluster bootstrap 95% interval {interval_text}); mean cost ratio "
            f"{contrast['mean_cost_ratio']:.3f}; {contrast['matched_cell_pairs']} matched cell pairs."
        )
    contrast_strata = list(analysis["stage_contrast_strata"])
    heterogeneity_lines = []
    for contrast in analysis["stage_contrasts"]:
        name = str(contrast["contrast"])
        matrix_values = [
            row
            for row in contrast_strata
            if row["scope"] == "matrix" and row["contrast"] == name
        ]
        anchors = sorted(
            (
                row
                for row in contrast_strata
                if row["scope"] == "anchor" and row["contrast"] == name
            ),
            key=lambda row: str(row["anchor_id"]),
        )
        matrix_deltas = np.asarray(
            [float(row["mean_paired_success_delta_fraction"]) for row in matrix_values],
            dtype=float,
        )
        low = min(
            matrix_values,
            key=lambda row: float(row["mean_paired_success_delta_fraction"]),
        )
        high = max(
            matrix_values,
            key=lambda row: float(row["mean_paired_success_delta_fraction"]),
        )
        anchor_text = ", ".join(
            f"{row['anchor_id']}={_fmt_pct(row['mean_paired_success_delta_fraction'])}"
            for row in anchors
        )
        heterogeneity_lines.append(
            f"- `{name}`: matrix signs +/0/- = "
            f"{int(np.sum(matrix_deltas > 0))}/{int(np.sum(matrix_deltas == 0))}/"
            f"{int(np.sum(matrix_deltas < 0))}; range {_matrix_name(low)} "
            f"{_fmt_pct(low['mean_paired_success_delta_fraction'])} to "
            f"{_matrix_name(high)} {_fmt_pct(high['mean_paired_success_delta_fraction'])}; "
            f"anchor means {anchor_text}."
        )
    goal_lines = [
        f"- {row['env_id']}: Spearman rho="
        + (
            f"{row['goal25_vs_goal50_spearman_schedule_success']:.3f}"
            if row["goal25_vs_goal50_spearman_schedule_success"] is not None
            else "undefined (at least one condition is constant)"
        )
        for row in analysis["goal_rank_stability"]
    ]
    aliases = analysis["equivalence_audit"]
    alias_sensitivity = analysis["alias_estimator_sensitivity"]
    alias_text = "; ".join(" / ".join(group) for group in aliases["alias_classes"])
    artifact_audit = analysis.get("artifact_audit") or {}
    cluster_index_audit = analysis.get("episode_cluster_index_audit") or {}
    audit_text = (
        f"The strict artifact gate verified all {artifact_audit['rows_referenced']} result "
        f"rows against {artifact_audit['unique_sources_verified']} unique source artifacts. "
        "It fully materialized legacy or compressed eval payloads, checked archive and "
        "scientific-payload integrity, rehashed resolved configs and manifests, regenerated "
        "metrics, confirmed canonical checkpoint-path/reported-epoch consistency, recomputed "
        "every realized scheduler decision, and reconciled episode, review, planner-trace, "
        "and audited-cost evidence. Checkpoint weight bytes were not hashed into the run "
        "artifacts, so this is not a retrospective checkpoint-content proof. "
        f"A subordinate cluster-index pass recovered paired trajectory identities for "
        f"{cluster_index_audit.get('cells_audited', 0)} rows; it found zero FLOP-audit "
        "errors and five replanning rounds per episode. Each round is one vectorized "
        "solver call per 50-environment batch, while diagnostics record five CEM cost "
        "calls per episode-iteration."
        if artifact_audit.get("status") == "passed"
        else "The required strict artifact-level integrity/provenance audit was not supplied."
    )
    covariate_audit = analysis.get("episode_covariates") or {}
    covariate_lines = []
    for matrix in covariate_audit.get("matrix_structure", []):
        fields = []
        for name, summary in sorted(dict(matrix["covariates"]).items()):
            if summary["kind"] == "boolean":
                fields.append(
                    f"`{name}`={100*float(summary['true_fraction']):.1f}% true"
                )
            else:
                if summary["median"] is None:
                    fields.append(f"`{name}` undefined for all episodes")
                else:
                    fields.append(
                        f"`{name}` median={float(summary['median']):.3g} "
                        f"[min={float(summary['min']):.3g}, max={float(summary['max']):.3g}]"
                        + (f" ({int(summary['null'])} undefined)" if summary["null"] else "")
                    )
        covariate_lines.append(
            f"- {_matrix_name(matrix)}: " + "; ".join(fields)
        )
    covariate_text = (
        "\n".join(covariate_lines)
        if covariate_lines
        else "- No validated pre-outcome covariate artifact was supplied."
    )
    structure_lines = [
        f"- {str(row['env_id']).split('/')[-1]} `{row['contrast']}` split by "
        f"`{row['covariate']}`: "
        f"low/false {_fmt_pct(row['low_or_false_mean_success_delta_fraction'])}, "
        f"high/true {_fmt_pct(row['high_or_true_mean_success_delta_fraction'])}, "
        f"modifier {_fmt_pct(row['high_or_true_minus_low_or_false_fraction'])}."
        for row in analysis.get("task_structure_heterogeneity", [])
    ]
    structure_text = (
        "\n".join(structure_lines)
        if structure_lines
        else "- No joined task-structure effect diagnostic was available."
    )
    covariate_intro = (
        "The independent covariate artifact passed its own 400-row schema and "
        "config/manifest-reference validation and was joined to the audited episode "
        "identities before analysis. It does not fingerprint the raw dataset rows, "
        "so it does not prove dataset immutability."
        if covariate_audit.get("status") == "passed"
        else "No validated pre-outcome covariate artifact was attached to this analysis."
    )
    budget = analysis["budget_response"]
    top_robust = ", ".join(f"{row['policy_id']} ({row['schedule']})" for row in robust[:5])
    top_efficient = ", ".join(f"{row['policy_id']} ({row['schedule']})" for row in efficient[:5])
    return rf"""# Release 2026-07-28 five-anchor schedule diagnostic

## Scope and strongest observations

This is a balanced **diagnostic**, not exhaustive frontier certification: 26
source-table schedule labels, five planner anchors, eight environment/goal
matrices, and the same 50 episode initial conditions within every matrix. The
anchors fix elite fraction at 0.2 and span `(population, iterations)` from
`(20,5)` to `(200,30)`.

{audit_text} Audited dynamics FLOPs are the scientific cost axis; wall time is
not pooled because 50- and 250-episode sources amortize setup differently and
ran under heterogeneous cluster load.

The lowest average regret schedules are **{top_robust}**. The largest paired-
bootstrap frontier-selection mass belongs to **{top_efficient}**. These rankings answer
different questions: regret rewards success at the same planner anchor,
whereas bootstrap selection frequency summarizes how often a cell remains on
the frontier under resampled episode panels.

Two binary frontier notions are retained in the machine-readable outputs: the
raw empirical sparse frontier and a descriptive five-percentage-point practical
frontier. Paired trajectory-cluster bootstrap membership frequency is usually
more informative than either binary label when schedules differ by only one or
two successes. It is a conditional bootstrap selection frequency—not a
posterior or confidence probability—and neither binary frontier is presented
as a confidence set.

Top mean-success schedules within each matrix (averaged over five anchors):

{chr(10).join(matrix_lines)}

## Empirical Pareto inventory

The exact frontier cells, audited costs, rates, and bootstrap frequencies are
in `pareto_frontier.csv`; `frontier_stability.csv` retains all 560 operational
policy-anchor cells. A cell can be empirically nondominated yet have low
selection frequency when its advantage is only one or two episode outcomes.

{chr(10).join(frontier_lines)}

## Stage-isolating matched contrasts

Every contrast below uses explicitly structured, exploratory realized-policy pairs while holding
the matrix, planner anchor, and all 50 episode identities fixed. Rollout and
dynamic-MPC CEM families isolate one realized stage. The MPC family is a total
outer-policy intervention because downstream `base` references propagate its
change. Intervals resample dataset-episode clusters *within the eight fixed
matrices*; they quantify episode sensitivity, not population-level uncertainty
over unseen tasks or checkpoints.

{chr(10).join(contrast_lines)}

Environment/goal and planner-anchor heterogeneity is not averaged away below.
Signs and ranges are descriptive; the machine-readable stratified table also
contains paired trajectory-cluster bootstrap intervals for every matrix,
anchor, and matrix-anchor interaction.

{chr(10).join(heterogeneity_lines)}

Cell-level paired tests flip each dataset-episode cluster total once, not each
repeated window independently. Their complete-enumeration cluster sign-flip
p-values require independent cluster totals with a joint null distribution
invariant under independent sign flips; treatment labels were not randomized
to trajectories. P-values are
Holm-adjusted within each named exploratory contrast family. No diagnostic-wide
global confidence-frontier claim is made.

## Planner-budget response

Across {budget['matrix_schedule_curves']} matrix-schedule curves, only
{budget['nondecreasing_curves']} are empirically nondecreasing at all five
anchors; {budget['curves_with_at_least_one_downward_step']} have at least one
downward step. The mean anchor-5 minus anchor-1 change is
{_fmt_pct(budget['mean_a5_minus_a1_success_fraction'])}. Thus additional CEM
work is not automatically additional task success. The anchors change both
population and iteration count and are not nested Monte Carlo samples, so this
is a budget-response curve—not a causal separation of population from
iterations.

Schedule ordering across goal lengths is:

{chr(10).join(goal_lines)}

Low rho is consistent with the bundled goal-50 condition changing which
approximation errors matter; high rho is consistent with schedule quality being
largely task-family intrinsic. With only 14 realized policies and finite,
frequently tied 50-episode panels, and no rank-correlation interval, rho remains
descriptive. This is
**not** a pure distance intervention: goal offset 25→50 is
bundled with horizon 5→10, executed blocks 2→4, and budget 50→100. The goal
manifests are not uniformly episode-paired. Pooled bootstrap intervals share
weights only when the audit finds an identical ordered `(environment,
dataset_episode)` panel across goals; no direct goal-25-minus-goal-50 success
contrast is treated as a randomized intervention.

## Runtime-equivalent controls

The source table contains 26 labels but only
{aliases['operational_schedule_count']} operational scheduler signatures. The
alias classes are {alias_text}. They agree exactly in
{aliases['exact_identity_comparisons']} of
{aliases['paired_alias_comparisons']} matrix-anchor comparisons; the remaining
{aliases['outcome_nonidentity_comparisons']} comparisons contain
{aliases['replicate_discordant_episode_total']} episode-level outcome
discordances. Audited work and episode identities agree in every collapsed
class. These are separate GPU executions of the same realized policy; observed
differences can reflect execution and/or mutable-provenance differences even
with the same configured seed. The primary operational-policy
estimator averages alias outcomes *within each episode identity* before any
frontier or contrast calculation. The bootstrap still has 50 episode clusters:
aliases reduce sensitivity to an arbitrary source-label choice but do not
inflate the scientific sample size. Alias classes have unequal numbers of
executions, and episode resampling does not model execution-to-execution
variance. The uncollapsed source-label rows remain in the finalized diagnostic
for reproducibility and sensitivity inspection. Replacing alias averaging with
the fixed lowest-numbered representative changes the nine structural contrast
point estimates by at most
{_fmt_pct(alias_sensitivity['maximum_absolute_shift_fraction'])}, for
`{alias_sensitivity['maximum_shift_contrast']}`. This is a sensitivity check,
not an execution-variance confidence interval.

The collapse follows directly from runtime semantics. If `M_r` is the MPC
level, CEM `base` means `M_r` rather than a fixed middle level; if `C_ri` is
the CEM level, rollout `base` means `C_ri`. Therefore under fixed-fine MPC,
`base -> fine` is constant fine and `coarse -> base` equals `coarse -> fine`.
Likewise, when CEM is fine, rollout `fine -> base` is fixed fine and
`base -> coarse` equals `fine -> coarse`. A naive three-way factorial model
would consequently be rank-deficient; the reported contrasts use only the
14 realized policies and explicitly identified policy pairs.

## Mechanistic model

Let `r` index MPC replans, `i` CEM iterations, `p` candidates, and `t` model
rollout steps. A first-order dynamics-work model is

\[
C(s,b) \approx \sum_r\sum_{{i=1}}^{{I_b}} P_b
                 \sum_{{t=1}}^H c(K_{{s,r,i,t}}),
\]

where the schedule chooses fidelity `K` and transformer cost `c(K)` grows
superlinearly in latent/feature width. MPC scheduling changes fidelity across replans;
CEM scheduling changes it across optimizer iterations; rollout scheduling
changes it down the imagined trajectory. The measured audited FLOPs—not this
approximation—define the reported frontier.

Suppose a fidelity-`K` one-step model has error `epsilon(K)` and latent dynamics
are locally `L`-Lipschitz. A standard telescoping argument gives the rollout
bound

\[
\|e_H\| \le \sum_{{t=1}}^H L^{{H-t}}\epsilon(K_t).
\]

The proof is the one-line recurrence
`||e_t|| <= L ||e_(t-1)|| + epsilon(K_t)`, unrolled from zero initial error.
It also yields a schedule-allocation result rather than only an intuition.
Relax discrete fidelity to positive effort `q_t`, assume
`epsilon(q)=A q^(-alpha)` with `alpha>0`, and constrain total rollout effort to
`sum_t q_t=B`. Minimizing the bound has the unique interior solution

\[
q_t^* = B\frac{{w_t^{{1/(\alpha+1)}}}}
                 {{\sum_u w_u^{{1/(\alpha+1)}}}},
\qquad w_t=L^{{H-t}}.
\]

This follows directly from the Lagrange condition
`alpha A w_t q_t^(-alpha-1)=lambda`. Therefore, when `L>1`, optimal effort is
monotonically decreasing down the imagined trajectory: a formal
`fine -> coarse` principle. The conclusion is conditional—contact can make
`L` state-dependent and terminal fidelity also changes the objective—but it
explains why early rollout accuracy can be worth more while later coarse steps
save compute before MPC replans.

For CEM, write the approximate candidate cost as
`J_hat(a)=J(a)+eta_K(a)`. If `|eta_K(a)| <= epsilon`, any pair with true cost
gap greater than `2 epsilon` keeps its ordering under the coarse model, since
the estimated gap differs from the true gap by at most `2 epsilon`. Let
`Delta_i` be the smallest decision-relevant true gap at CEM iteration `i`.
The cheapest ranking-safe fidelity is then

\[
K_i^*=\min\{{K:2\epsilon(K)<\Delta_i\}}.
\]

If CEM contraction makes `Delta_i` decrease and fidelity makes `epsilon(K)`
decrease, `K_i^*` is nondecreasing: a formal `coarse -> fine` principle.
Conversely, selecting the best of `P` candidates amplifies model exploitation.
For conditionally mean-zero, `sigma_K`-sub-Gaussian candidate errors, the mgf
bound gives
`E[max_p eta_p] <= sigma_K sqrt(2 log P)` (and the same bound for the most
favorable negative error). This explains why more search can hurt and why
moving to higher fidelity late in CEM should matter most at high population.

The same ranking lemma exposes an asymmetry between CEM and MPC clocks. Within
one CEM solve, later high-fidelity iterations can revise the distribution
*before* any candidate action is executed. Across MPC replans, the first action
block has already been committed. If the action margin at replan `r` is `g_r`,
the sufficient robust-choice condition is `g_r>2 epsilon_r`; violating it at an
early replan cannot be repaired retroactively. MPC `coarse -> fine` is therefore
appropriate only when early behavior has broad robust margins (for example,
smooth global progress) and late behavior has narrow terminal/contact margins.
It can fail when the episode begins near a discontinuity. This gives a concrete
reason to compare near-cost CEM-clock and MPC-clock schedules rather than
treating their ramps as interchangeable.

Rollout schedules also change the terminal objective, not just transition
cost. If `Pi_K` selects the prefix visible at terminal fidelity, then

\[
J_K(a)=\|\Pi_K(\hat z_H(a)-z_g)\|^2.
\]

For the same terminal state, lowering terminal fidelity from `D` to `K`
removes the nonnegative tail energy: for nested orthogonal projections,
`J_D=J_K+||Pi_{{>K}}(z_hat_H-z_g)||^2`, hence `J_K<=J_D` pointwise. The
minimizing action can nevertheless change because the removed term is
action-dependent. A base-to-coarse rollout can therefore act as useful
nuisance-dimension regularization or discard task-critical information; it is
not merely a cheaper simulation. Fine-to-base is the cleaner transition-
accuracy contrast because it retains the fixed-base terminal criterion.

## Pre-outcome task structure

{covariate_intro} These summaries describe the manifest windows only; they
neither use outcomes nor by themselves explain a schedule effect.

{covariate_text}

The following four task-structure splits were selected for their mechanistic
relevance, but remain **exploratory point estimates**: they have no inferential
interval, and repeated anchors, policy pairs, and dataset episodes make the raw
cell-episode counts non-independent. Displayed subgroup means are computed
within each matched matrix-anchor-policy-pair record and then averaged with
equal record weight. A positive modifier means that the named treatment
schedule did better relative to its reference in the high/true subgroup than
in the low/false subgroup.

{structure_text}

## Environment interpretation guide

- **Reacher** is the smoothest, lowest-dimensional control geometry. Stable
  coarse rankings and coarse-to-fine CEM/MPC are most plausible here. Its
  evaluator applies the 0.1-radian joint tolerance in raw coordinates rather
  than modulo \(2\pi\), so the wrapped-displacement covariate is a geometry
  descriptor rather than an exact restatement of success difficulty.
- **PushT** is contact-rich and orientation-sensitive. Early rollout accuracy
  and late-CEM fidelity should matter if small state errors change contact
  mode.
- **OGBench Cube** adds a five-dimensional manipulation action and contact-rich
  object transport. Its evaluator restores the target quaternion but declares
  success from cube position alone (within 0.04 m), so quaternion displacement
  is a manipulation-structure covariate, not a success criterion. Coarse global
  motion may still be useful, while grasp/contact phases may punish schedules
  that become coarse too early.
- **TwoRoom** has a topological bottleneck: coarse long-horizon predictions may
  identify the correct room/doorway cheaply, while precision should become
  valuable near the passage and goal. Goal-50 should strengthen the global-
  planning side of that tradeoff.

There is also a non-task architectural difference. OGBench Cube has five
primitive action dimensions, versus two in the other three environments; with
action blocks of five, its CEM vector has 125 dimensions at goal-25 and 250 at
goal-50, versus 50 and 100 elsewhere. A fixed population therefore covers a
much sparser fraction of Cube's search space. PushT, Reacher, and Cube use
history length three, so model rollout cost starts at scheduled position two;
TwoRoom uses history length one and consumes the full scheduled rollout. Thus
the same nominal rollout ramp exposes different active fidelity weights even
before considering contact or geometry.

These are mechanisms and hypotheses, not empirical proof. The covariate
modifiers above remain descriptive and need independent replication plus an
inferential interaction analysis. Matched contrasts agreeing across anchors
and stable bootstrap frontier-selection frequencies can show a robust schedule
effect; they can only be *consistent with*, not prove, the proposed mechanism.

## Statistical limits

Each uncollapsed source-label rate changes in two-percentage-point increments,
but alias-averaged operational rates can have finer fractional increments and
many schedule gaps remain uncertain. Empirical Pareto membership is unstable
when rates are tied or separated by only one or two episodes. The paired
bootstrap retains common episode difficulty and is preferable to independent
binomial intervals for schedule differences. It does not cover checkpoint
variation, training-seed variation, or untested planner settings. Operational
alias schedules are collapsed for factor contrasts so duplicated source-table
rows do not receive extra weight.
"""


def analyze_results(
    payload: Mapping[str, Any],
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    episode_clusters: Mapping[tuple[str, int], Sequence[int]] | None = None,
    artifact_audit: Mapping[str, Any] | None = None,
    episode_cluster_index_audit: Mapping[str, Any] | None = None,
    episode_covariates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation = validate_results(payload)
    rows = _rows(payload)
    if episode_clusters is not None:
        for row in rows:
            key = (str(row["canonical_config"]), int(row["matrix_index"]))
            if key not in episode_clusters:
                raise AnchorAnalysisError(f"Missing audited episode clusters for {key}.")
            row["episode_cluster_ids"] = list(episode_clusters[key])
    equivalence = equivalence_audit(rows)
    if int(equivalence["operational_schedule_count"]) != 14:
        raise AnchorAnalysisError(
            "Expected the 26 source labels to collapse to 14 realized policies, "
            f"found {equivalence['operational_schedule_count']}."
        )
    if equivalence["scientific_invariant_violations"]:
        first = equivalence["scientific_invariant_violations"][0]
        raise AnchorAnalysisError(
            "Runtime-equivalent schedule controls violate audited work or pairing; "
            "refusing to collapse aliases "
            f"or compute scientific summaries. First mismatch: {first}."
        )
    stability = frontier_stability(rows, repeats=repeats, seed=seed)
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "input_schema_version": str(payload.get("schema_version", "")),
        "input_sha256": _canonical_sha256(payload),
        "design_validation": validation,
        "bootstrap": {
            "method": (
                "fixed-matrix dataset-episode cluster bootstrap; identical ordered "
                "environment/cluster panels share weights across goals"
            ),
            "repeats": int(repeats),
            "seed": int(seed),
            "cost_treatment": "fixed audited aggregate cost per episode",
        },
        "paired_cell_inference": {
            "method": (
                "complete-enumeration two-sided trajectory-cluster sign-flip test "
                "under independent-cluster joint sign-flip invariance"
            ),
            "cluster_unit": "dataset_episode within matrix",
            "treatment_labels_randomized_to_trajectories": False,
            "operational_policy_estimator": (
                "episode-wise mean across runtime-equivalent source-label executions"
            ),
            "alias_replicates_increase_episode_sample_size": False,
            "multiple_testing": "Holm family-wise adjustment within each named contrast family",
            "global_confidence_frontier": False,
        },
        "artifact_audit": dict(artifact_audit) if artifact_audit is not None else None,
        "episode_cluster_index_audit": (
            dict(episode_cluster_index_audit)
            if episode_cluster_index_audit is not None
            else None
        ),
        "episode_covariates": (
            dict(episode_covariates) if episode_covariates is not None else None
        ),
        "task_structure_heterogeneity": task_structure_heterogeneity(
            rows,
            episode_covariates,
        ),
        "equivalence_audit": equivalence,
        "alias_estimator_sensitivity": alias_estimator_sensitivity(rows),
        "frontier_stability": stability,
        "stage_contrasts": stage_contrasts(rows, repeats=repeats, seed=seed),
        "stage_contrast_strata": stage_contrast_strata(
            rows,
            repeats=repeats,
            seed=seed,
        ),
        "paired_cell_comparisons": paired_cell_comparisons(rows),
        "schedule_summaries": schedule_summaries(rows, stability),
        "goal_rank_stability": goal_rank_stability(rows),
        "budget_response": budget_response(rows),
    }
    return analysis


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise AnchorAnalysisError(f"Refusing to write empty CSV {path}.")
    fieldnames = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def write_analysis(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    covariate_path: str | Path = DEFAULT_COVARIATE_INPUT,
) -> dict[str, Any]:
    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorAnalysisError(f"Cannot read finalized diagnostic {source}: {exc}") from exc
    repo_root = Path(__file__).resolve().parents[2]
    # The strict gate owns artifact/provenance validity.  The older lightweight
    # pass remains only to recover dataset-episode cluster IDs for paired
    # inference; its summary is reported separately and subordinately.
    from mwm.benchmark.anchor_artifact_audit import audit_anchor_artifacts

    artifact_audit = audit_anchor_artifacts(
        payload,
        repo_root=repo_root,
        strict=True,
    )
    cluster_index_audit, episode_clusters = audit_source_artifacts(
        payload,
        repo_root=repo_root,
    )
    covariates = load_episode_covariate_summary(
        covariate_path,
        result_payload=payload,
        episode_clusters=episode_clusters,
    )
    analysis = analyze_results(
        payload,
        repeats=repeats,
        seed=seed,
        episode_clusters=episode_clusters,
        artifact_audit=artifact_audit,
        episode_cluster_index_audit=cluster_index_audit,
        episode_covariates=covariates,
    )
    critical_modules = (
        "mwm/benchmark/anchor_diagnostic.py",
        "mwm/benchmark/anchor_analysis.py",
        "mwm/benchmark/anchor_artifact_audit.py",
        "mwm/benchmark/episode_covariates.py",
        "mwm/benchmark/eval_artifacts.py",
        "mwm/benchmark/screening.py",
    )
    analysis["analysis_code_provenance"] = {
        "input_file_sha256": _file_sha256(source),
        "critical_module_sha256": {
            relative: _file_sha256(repo_root / relative)
            for relative in critical_modules
        },
    }
    rows = _rows(payload)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        destination / "analysis.json",
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(destination / "schedule_summary.csv", analysis["schedule_summaries"])
    _write_csv(destination / "stage_contrasts.csv", analysis["stage_contrasts"])
    _write_csv(destination / "stage_contrast_strata.csv", analysis["stage_contrast_strata"])
    _write_csv(destination / "paired_cell_comparisons.csv", analysis["paired_cell_comparisons"])
    _write_csv(destination / "frontier_stability.csv", analysis["frontier_stability"])
    _write_csv(
        destination / "pareto_frontier.csv",
        [row for row in analysis["frontier_stability"] if row["empirical_frontier"]],
    )
    _atomic_write_text(destination / "report.md", render_report(analysis, rows))
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--episode-covariates", default=str(DEFAULT_COVARIATE_INPUT))
    args = parser.parse_args(argv)
    analysis = write_analysis(
        args.input,
        args.output_dir,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
        covariate_path=args.episode_covariates,
    )
    print(
        json.dumps(
            {
                "rows": analysis["design_validation"]["rows"],
                "operational_schedules": analysis["equivalence_audit"]["operational_schedule_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnchorAnalysisError",
    "analyze_results",
    "budget_response",
    "equivalence_audit",
    "frontier_stability",
    "goal_rank_stability",
    "load_episode_covariate_summary",
    "paired_cell_comparisons",
    "realized_policy_signature",
    "source_axis_signature",
    "schedule_summaries",
    "stage_contrast_strata",
    "stage_contrasts",
    "validate_results",
    "write_analysis",
]
