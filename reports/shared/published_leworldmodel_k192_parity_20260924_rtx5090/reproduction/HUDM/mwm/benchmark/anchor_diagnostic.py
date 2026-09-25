"""Sparse five-anchor diagnostic for the release20260728 schedule matrices.

The exhaustive release matrix has 120 planner configurations per schedule.
This module selects a controlled five-point planning-budget ladder while
holding ``elite_frac`` fixed.  Every selected cell keeps its canonical matrix
index and uses the exact 50-episode discovery prefix produced by
``mwm.benchmark.screening``.

The plan is also a source ledger.  Once a canonical-250 or dedicated-screen50
artifact is selected, subsequent refreshes verify and retain that exact source
instead of silently switching to a later artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mwm.benchmark.adaptive_workflow import indices_to_slurm_array
from mwm.benchmark.adaptive_workflow import run_screen_cell
from mwm.benchmark.config import DEFAULTS, manifest_path, merged_run_config
from mwm.benchmark.screening import (
    DEFAULT_GENERATED_DIR,
    PreparedScreen,
    _atomic_write_text,
    _resolve_from_root,
    prepare_all,
)
from mwm.benchmark.screening_sources import _candidate_source
from mwm.benchmark.sweep import expand_benchmark_runs
from mwm.config_cli import load_config


SCHEMA_VERSION = "mwm.release20260728.anchor_diagnostic/v1"
DEFAULT_PLAN_PATH = Path(
    "reports/research/release20260728_schedule_screening/five_anchor_plan.json"
)
DEFAULT_RESULTS_DIR = Path(
    "reports/research/release20260728_schedule_screening/five_anchor_results"
)
DEFAULT_LAUNCH_MANIFEST = Path(
    "reports/research/release20260728_schedule_screening/five_anchor_launch.json"
)


@dataclass(frozen=True)
class PlannerAnchor:
    anchor_id: str
    label: str
    pop_size: int
    elite_frac: float
    n_iter: int

    @property
    def nominal_candidate_iterations(self) -> int:
        return self.pop_size * self.n_iter


# Fixing elite_frac isolates the schedule comparison from changes in CEM
# selection pressure.  The population/iteration ladder spans 60x nominal
# candidate-iterations and includes the endpoints and center of both axes.
ANCHORS = (
    PlannerAnchor("a1", "very_low", 20, 0.2, 5),
    PlannerAnchor("a2", "low", 50, 0.2, 10),
    PlannerAnchor("a3", "medium", 100, 0.2, 15),
    PlannerAnchor("a4", "high", 150, 0.2, 20),
    PlannerAnchor("a5", "very_high", 200, 0.2, 30),
)


class AnchorDiagnosticError(RuntimeError):
    pass


def _anchor_for_run(run: Any) -> PlannerAnchor | None:
    key = (
        int(run.planner.pop_size),
        float(run.planner.elite_frac),
        int(run.planner.n_iter),
    )
    for anchor in ANCHORS:
        if key == (anchor.pop_size, anchor.elite_frac, anchor.n_iter):
            return anchor
    return None


def selected_anchor_runs(cfg: Any) -> list[tuple[Any, PlannerAnchor]]:
    selected = [
        (run, anchor)
        for run in expand_benchmark_runs(cfg)
        if (anchor := _anchor_for_run(run)) is not None
    ]
    schedules: dict[str, list[PlannerAnchor]] = {}
    for run, anchor in selected:
        schedules.setdefault(str(run.base_name), []).append(anchor)
    if len(schedules) != 26:
        raise AnchorDiagnosticError(f"Expected 26 schedules, found {len(schedules)}.")
    expected_ids = {anchor.anchor_id for anchor in ANCHORS}
    malformed = {
        schedule: sorted(anchor.anchor_id for anchor in anchors)
        for schedule, anchors in schedules.items()
        if len(anchors) != len(ANCHORS)
        or {anchor.anchor_id for anchor in anchors} != expected_ids
    }
    if malformed:
        raise AnchorDiagnosticError(f"Schedules do not contain the five anchor ladder: {malformed}")
    if len(selected) != 26 * len(ANCHORS):
        raise AnchorDiagnosticError(f"Expected 130 anchor cells, found {len(selected)}.")
    return selected


def _previous_matrices(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorDiagnosticError(f"Cannot read existing anchor plan {path}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AnchorDiagnosticError(f"Unsupported existing anchor plan schema in {path}.")
    matrices = payload.get("matrices")
    if not isinstance(matrices, list):
        raise AnchorDiagnosticError(f"Malformed matrices in {path}.")
    return {str(item["canonical_config"]): dict(item) for item in matrices}


def _launched_screen_cells(path: Path) -> set[tuple[str, int]]:
    """Return immutable ``(screen_config, matrix_index)`` launch targets.

    A canonical 250-episode array may complete after the diagnostic launch.
    These keys preserve the source assignment made at launch: a cell that was
    missing then must be finalized from its dedicated screen-50 artifact,
    rather than silently switching to the later canonical completion.
    """

    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorDiagnosticError(f"Cannot read anchor launch manifest {path}: {exc}") from exc
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != "immutable_launch_manifest"
    ):
        raise AnchorDiagnosticError(f"Unsupported anchor launch manifest in {path}.")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or int(payload.get("task_count", -1)) != len(tasks):
        raise AnchorDiagnosticError(f"Malformed task list in {path}.")
    launched: set[tuple[str, int]] = set()
    for position, task in enumerate(tasks):
        if not isinstance(task, dict) or int(task.get("task_index", -1)) != position:
            raise AnchorDiagnosticError(f"Task identity mismatch at position {position} in {path}.")
        key = (str(task.get("screen_config", "")), int(task.get("matrix_index", -1)))
        if not key[0] or key[1] < 0 or key in launched:
            raise AnchorDiagnosticError(f"Invalid or duplicate launch target {key} in {path}.")
        launched.add(key)
    return launched


def _source_candidate(
    *,
    prepared: PreparedScreen,
    source_kind: str,
    run: Any,
    canonical_cfg: Any,
    screen_cfg: Any,
    repo_root: Path,
) -> dict[str, Any] | None:
    if source_kind == "canonical_250_prefix":
        cfg = canonical_cfg
        output_dir = Path(prepared.canonical_output_dir)
        expected_episodes = 250
        expected_hash = prepared.canonical_manifest_sha256
    elif source_kind == "screen50":
        cfg = screen_cfg
        output_dir = Path(prepared.screen_output_dir)
        expected_episodes = 50
        expected_hash = prepared.screen_manifest_sha256
    else:
        raise AnchorDiagnosticError(f"Unknown source kind {source_kind!r}.")
    _, run_cfg = merged_run_config(cfg, run)
    return _candidate_source(
        source_kind=source_kind,
        output_dir=output_dir,
        run=run,
        run_cfg=run_cfg,
        manifest_path=manifest_path(cfg),
        manifest_sha256=expected_hash,
        episodes=expected_episodes,
        repo_root=repo_root,
    )


def build_plan(
    *,
    repo_root: str | Path,
    generated_dir: str | Path = DEFAULT_GENERATED_DIR,
    output_path: str | Path = DEFAULT_PLAN_PATH,
    launch_manifest_path: str | Path = DEFAULT_LAUNCH_MANIFEST,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = _resolve_from_root(root, output_path)
    previous = _previous_matrices(output)
    launched_screen_cells = _launched_screen_cells(
        _resolve_from_root(root, launch_manifest_path)
    )
    prepared_all = prepare_all(repo_root=root, generated_dir=generated_dir)
    matrices: list[dict[str, Any]] = []
    totals = {"selected": 0, "canonical_250_prefix": 0, "screen50": 0, "pending": 0}

    for prepared in prepared_all:
        canonical_cfg = load_config(DEFAULTS, _resolve_from_root(root, prepared.canonical_config))
        screen_cfg = load_config(DEFAULTS, _resolve_from_root(root, prepared.screen_config))
        selected = selected_anchor_runs(canonical_cfg)
        old = previous.get(prepared.canonical_config, {})
        for field in (
            "canonical_config_sha256",
            "canonical_manifest_sha256",
            "screen_manifest_sha256",
        ):
            old_value = old.get(field)
            if old_value is not None and str(old_value) != str(getattr(prepared, field)):
                raise AnchorDiagnosticError(
                    f"Pinned matrix provenance changed for {prepared.canonical_config}: {field}."
                )
        old_sources = old.get("sources", {})
        if not isinstance(old_sources, dict):
            raise AnchorDiagnosticError(f"Malformed pinned sources for {prepared.canonical_config}.")

        sources: dict[str, dict[str, Any]] = {}
        todo: list[int] = []
        anchor_cells: list[dict[str, Any]] = []
        counts = {"selected": len(selected), "canonical_250_prefix": 0, "screen50": 0, "pending": 0}
        for run, anchor in selected:
            index = int(run.matrix_index)
            old_source = old_sources.get(str(index))
            launched_for_screen = (prepared.screen_config, index) in launched_screen_cells
            if launched_for_screen:
                # Launch assignment is authoritative.  In particular, ignore
                # a canonical-250 artifact that appeared only after freeze.
                source = _source_candidate(
                    prepared=prepared,
                    source_kind="screen50",
                    run=run,
                    canonical_cfg=canonical_cfg,
                    screen_cfg=screen_cfg,
                    repo_root=root,
                )
                if old_source is not None and str(old_source.get("source_kind", "")) == "screen50":
                    if source != old_source:
                        raise AnchorDiagnosticError(
                            f"Pinned screen50 source changed or disappeared at "
                            f"{prepared.canonical_config} index {index}."
                        )
            elif old_source is not None:
                source_kind = str(old_source.get("source_kind", ""))
                source = _source_candidate(
                    prepared=prepared,
                    source_kind=source_kind,
                    run=run,
                    canonical_cfg=canonical_cfg,
                    screen_cfg=screen_cfg,
                    repo_root=root,
                )
                if source != old_source:
                    raise AnchorDiagnosticError(
                        f"Pinned source changed or disappeared at {prepared.canonical_config} index {index}."
                    )
            else:
                source = _source_candidate(
                    prepared=prepared,
                    source_kind="canonical_250_prefix",
                    run=run,
                    canonical_cfg=canonical_cfg,
                    screen_cfg=screen_cfg,
                    repo_root=root,
                )
                if source is None:
                    source = _source_candidate(
                        prepared=prepared,
                        source_kind="screen50",
                        run=run,
                        canonical_cfg=canonical_cfg,
                        screen_cfg=screen_cfg,
                        repo_root=root,
                    )
            if source is None:
                todo.append(index)
                counts["pending"] += 1
            else:
                sources[str(index)] = source
                counts[str(source["source_kind"])] += 1
            anchor_cells.append(
                {
                    "matrix_index": index,
                    "cell_id": str(run.cell_id),
                    "schedule": str(run.base_name),
                    "schedule_label": str(run.get("schedule", run.base_name)),
                    "anchor_id": anchor.anchor_id,
                    "anchor_label": anchor.label,
                    "planner": {
                        "pop_size": anchor.pop_size,
                        "elite_frac": anchor.elite_frac,
                        "n_iter": anchor.n_iter,
                        "nominal_candidate_iterations": anchor.nominal_candidate_iterations,
                    },
                }
            )
        matrix = {
            "canonical_config": prepared.canonical_config,
            "canonical_config_sha256": prepared.canonical_config_sha256,
            "canonical_manifest_sha256": prepared.canonical_manifest_sha256,
            "canonical_output_dir": prepared.canonical_output_dir,
            "screen_config": prepared.screen_config,
            "screen_manifest_sha256": prepared.screen_manifest_sha256,
            "screen_output_dir": prepared.screen_output_dir,
            "counts": counts,
            "selected_indices": [int(run.matrix_index) for run, _ in selected],
            "selected_slurm_array": indices_to_slurm_array(
                [int(run.matrix_index) for run, _ in selected]
            ),
            "todo_indices": todo,
            "todo_slurm_array": indices_to_slurm_array(todo),
            "sources": sources,
            "anchor_cells": anchor_cells,
        }
        matrices.append(matrix)
        for key in totals:
            totals[key] += int(counts[key])

    plan = {
        "schema_version": SCHEMA_VERSION,
        "design": {
            "purpose": "schedule diagnostic, not exhaustive frontier certification",
            "episodes": 50,
            "paired_manifest": "exact canonical prefix [0, 50)",
            "elite_fraction_control": 0.2,
            "anchors": [asdict(anchor) for anchor in ANCHORS],
            "schedules_per_matrix": 26,
            "matrices": 8,
        },
        "totals": totals,
        "matrices": matrices,
    }
    _atomic_write_text(output, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AnchorDiagnosticError(f"{path}:{line_number} is not a JSON object.")
                values.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorDiagnosticError(f"Cannot read {path}: {exc}") from exc
    return values


def pareto_indices(rows: list[dict[str, Any]]) -> list[int]:
    """Return row positions not weakly dominated in cost/success space."""

    frontier: list[int] = []
    for i, row in enumerate(rows):
        cost_i = float(row["dynamics_flops_per_episode"])
        success_i = float(row["success_rate_fraction"])
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            cost_j = float(other["dynamics_flops_per_episode"])
            success_j = float(other["success_rate_fraction"])
            if (
                cost_j <= cost_i
                and success_j >= success_i
                and (cost_j < cost_i or success_j > success_i)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def finalize_results(
    *,
    repo_root: str | Path,
    generated_dir: str | Path = DEFAULT_GENERATED_DIR,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    launch_manifest_path: str | Path = DEFAULT_LAUNCH_MANIFEST,
) -> dict[str, Any]:
    """Refresh the pinned plan and emit auditable 50-episode diagnostic data."""

    root = Path(repo_root).resolve()
    plan = build_plan(
        repo_root=root,
        generated_dir=generated_dir,
        output_path=plan_path,
        launch_manifest_path=launch_manifest_path,
    )
    if int(plan["totals"]["pending"]):
        raise AnchorDiagnosticError(
            f"Cannot finalize: {plan['totals']['pending']} anchor cells remain incomplete."
        )
    results_root = _resolve_from_root(root, output_dir)
    all_rows: list[dict[str, Any]] = []
    matrix_results: list[dict[str, Any]] = []
    for matrix in plan["matrices"]:
        cell_metadata = {
            int(cell["matrix_index"]): cell for cell in matrix["anchor_cells"]
        }
        rows: list[dict[str, Any]] = []
        for raw_index in matrix["selected_indices"]:
            index = int(raw_index)
            source = matrix["sources"].get(str(index))
            if not isinstance(source, dict):
                raise AnchorDiagnosticError(
                    f"Missing frozen source for {matrix['canonical_config']} index {index}."
                )
            run_dir = _resolve_from_root(root, str(source["run_dir"]))
            metrics = _jsonl_objects(run_dir / "metrics.jsonl")
            traces = _jsonl_objects(run_dir / "episode_traces.jsonl")
            if len(metrics) != 1:
                raise AnchorDiagnosticError(f"Expected one metrics row in {run_dir}.")
            observed_episodes = int(source["artifact_episodes"])
            if len(traces) != observed_episodes:
                raise AnchorDiagnosticError(
                    f"Expected {observed_episodes} episode traces in {run_dir}, found {len(traces)}."
                )
            prefix = traces[:50]
            if [int(trace.get("episode_index", -1)) for trace in prefix] != list(range(50)):
                raise AnchorDiagnosticError(f"Noncanonical discovery prefix in {run_dir}.")
            if any(not isinstance(trace.get("success"), bool) for trace in prefix):
                raise AnchorDiagnosticError(f"Missing boolean discovery outcome in {run_dir}.")
            metric = metrics[0]
            metadata = cell_metadata[index]
            successes = sum(bool(trace["success"]) for trace in prefix)
            episodes = int(metric.get("episodes", observed_episodes))
            if episodes != observed_episodes:
                raise AnchorDiagnosticError(f"Artifact/metrics episode mismatch in {run_dir}.")
            row = {
                "canonical_config": str(matrix["canonical_config"]),
                "screen_config": str(matrix["screen_config"]),
                "matrix_index": index,
                "cell_id": str(metadata["cell_id"]),
                "env_id": str(metric.get("env_id", "")),
                "goal_offset": int(metric.get("goal_offset", 0)),
                "schedule": str(metadata["schedule"]),
                "schedule_label": str(metadata["schedule_label"]),
                "anchor_id": str(metadata["anchor_id"]),
                "anchor_label": str(metadata["anchor_label"]),
                "pop_size": int(metadata["planner"]["pop_size"]),
                "elite_frac": float(metadata["planner"]["elite_frac"]),
                "n_iter": int(metadata["planner"]["n_iter"]),
                "nominal_candidate_iterations": int(
                    metadata["planner"]["nominal_candidate_iterations"]
                ),
                "discovery_episodes": 50,
                "successes": successes,
                "success_rate_fraction": successes / 50.0,
                "success_rate_percent": successes * 2.0,
                "dynamics_flops_per_episode": float(metric["dynamics_flops_total"]) / episodes,
                "latent_work_per_episode": float(metric["latent_work_total"]) / episodes,
                "plan_time_sec_per_episode": float(metric["plan_time_total_sec"]) / episodes,
                "wall_time_sec_per_episode": float(metric["wall_time_sec"]) / episodes,
                "source_kind": str(source["source_kind"]),
                "source_run_dir": str(source["run_dir"]),
                "config_sha256": str(source["config_sha256"]),
                "manifest_sha256": str(source["manifest_sha256"]),
                "discovery_outcomes": [bool(trace["success"]) for trace in prefix],
            }
            rows.append(row)
        frontier = set(pareto_indices(rows))
        for position, row in enumerate(rows):
            row["pareto_frontier"] = position in frontier
        all_rows.extend(rows)
        matrix_results.append(
            {
                "canonical_config": matrix["canonical_config"],
                "screen_config": matrix["screen_config"],
                "cells": len(rows),
                "pareto_cells": sum(bool(row["pareto_frontier"]) for row in rows),
                "rows": rows,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": (
            "Balanced five-anchor, 50-paired-episode diagnostic. Pareto membership is "
            "descriptive for this sparse design and is not exhaustive frontier certification."
        ),
        "design": plan["design"],
        "source_totals": plan["totals"],
        "cells": len(all_rows),
        "matrices": matrix_results,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        results_root / "results.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    fieldnames = [key for key in all_rows[0] if key != "discovery_outcomes"]
    csv_path = results_root / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in all_rows)
    return payload


def build_launch_manifest(
    plan: dict[str, Any],
    *,
    all_selected: bool = False,
) -> dict[str, Any]:
    """Freeze work into one balanced, immutable global task array.

    The normal manifest contains only missing cells.  ``all_selected`` is a
    provenance-repair mode: it records every selected cell as assigned to the
    dedicated screen output while identifying only non-screen sources as the
    recommended execution subset.  This lets a backfill replace reused legacy
    prefixes without rerunning already homogeneous screen cells.
    """

    matrices = list(plan.get("matrices", []))
    indices_by_matrix = [
        list(matrix.get("selected_indices" if all_selected else "todo_indices", []))
        for matrix in matrices
    ]
    tasks: list[dict[str, Any]] = []
    recommended: list[int] = []
    max_indices = max((len(values) for values in indices_by_matrix), default=0)
    for position in range(max_indices):
        for matrix, indices in zip(matrices, indices_by_matrix, strict=True):
            if position >= len(indices):
                continue
            matrix_index = int(indices[position])
            metadata = next(
                cell
                for cell in matrix["anchor_cells"]
                if int(cell["matrix_index"]) == matrix_index
            )
            source = matrix.get("sources", {}).get(str(matrix_index))
            source_kind_before = (
                str(source.get("source_kind", "pending"))
                if isinstance(source, dict)
                else "pending"
            )
            tasks.append(
                {
                    "task_index": len(tasks),
                    "screen_config": str(matrix["screen_config"]),
                    "screen_config_sha256": str(matrix["canonical_config_sha256"]),
                    "screen_manifest_sha256": str(matrix["screen_manifest_sha256"]),
                    "matrix_index": matrix_index,
                    "cell_id": str(metadata["cell_id"]),
                    "schedule": str(metadata["schedule"]),
                    "anchor_id": str(metadata["anchor_id"]),
                    "source_kind_before": source_kind_before,
                }
            )
            if not all_selected or source_kind_before != "screen50":
                recommended.append(len(tasks) - 1)
    # Replace the canonical hash placeholder with the generated screen config
    # hash at write time; keeping it in every task makes standalone execution
    # fail closed if a generated asset changes after submission.
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "immutable_launch_manifest",
        "launch_scope": "all_selected" if all_selected else "pending_only",
        "task_count": len(tasks),
        "recommended_task_count": len(recommended),
        "recommended_task_indices": recommended,
        "recommended_slurm_array": indices_to_slurm_array(recommended),
        "tasks": tasks,
    }


def write_launch_manifest(
    *,
    repo_root: str | Path,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    output_path: str | Path = DEFAULT_LAUNCH_MANIFEST,
    all_selected: bool = False,
) -> dict[str, Any]:
    from mwm.io import file_sha256

    root = Path(repo_root).resolve()
    plan_file = _resolve_from_root(root, plan_path)
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorDiagnosticError(f"Cannot read anchor plan {plan_file}: {exc}") from exc
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise AnchorDiagnosticError(f"Unsupported anchor plan schema in {plan_file}.")
    launch = build_launch_manifest(plan, all_selected=all_selected)
    launch["source_plan"] = str(plan_path)
    launch["source_plan_sha256"] = file_sha256(plan_file)
    for task in launch["tasks"]:
        config_path = _resolve_from_root(root, task["screen_config"])
        task["screen_config_sha256"] = file_sha256(config_path)
    output = _resolve_from_root(root, output_path)
    serialized = json.dumps(launch, indent=2, sort_keys=True) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") != serialized:
        raise AnchorDiagnosticError(
            f"Refusing to replace immutable launch manifest {output}; choose a new path."
        )
    _atomic_write_text(output, serialized)
    return launch


def run_launch_task(
    manifest_path: str | Path,
    *,
    task_index: int,
    repo_root: str | Path,
) -> dict[str, Any]:
    from mwm.io import file_sha256

    root = Path(repo_root).resolve()
    path = _resolve_from_root(root, manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorDiagnosticError(f"Cannot read launch manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != "immutable_launch_manifest":
        raise AnchorDiagnosticError(f"Unsupported launch manifest in {path}.")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or int(manifest.get("task_count", -1)) != len(tasks):
        raise AnchorDiagnosticError(f"Malformed task list in {path}.")
    if task_index < 0 or task_index >= len(tasks):
        raise AnchorDiagnosticError(f"Task index {task_index} outside [0, {len(tasks)}).")
    task = tasks[task_index]
    if int(task.get("task_index", -1)) != task_index:
        raise AnchorDiagnosticError(f"Task identity mismatch at position {task_index}.")
    screen_config = _resolve_from_root(root, str(task["screen_config"]))
    if file_sha256(screen_config) != str(task["screen_config_sha256"]):
        raise AnchorDiagnosticError(f"Generated screen config changed after launch freeze: {screen_config}")
    result = run_screen_cell(
        screen_config,
        matrix_index=int(task["matrix_index"]),
        num_shards=3120,
    )
    return {"task": task, "result": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("plan", "freeze-launch", "run-task", "finalize"),
        default="plan",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument("--generated-dir", default=str(DEFAULT_GENERATED_DIR))
    parser.add_argument("--output", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--launch-manifest", default=str(DEFAULT_LAUNCH_MANIFEST))
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--all-selected",
        action="store_true",
        help=(
            "freeze every selected cell into the screen-source ledger and "
            "recommend execution only for cells not already sourced from screen50"
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "finalize":
        result = finalize_results(
            repo_root=args.repo_root,
            generated_dir=args.generated_dir,
            plan_path=args.output,
            output_dir=args.results_dir,
            launch_manifest_path=args.launch_manifest,
        )
        print(json.dumps({"cells": result["cells"]}, sort_keys=True))
        return 0
    if args.command == "freeze-launch":
        launch = write_launch_manifest(
            repo_root=args.repo_root,
            plan_path=args.output,
            output_path=args.launch_manifest,
            all_selected=args.all_selected,
        )
        print(
            json.dumps(
                {
                    "task_count": launch["task_count"],
                    "recommended_task_count": launch["recommended_task_count"],
                    "recommended_slurm_array": launch["recommended_slurm_array"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-task":
        if args.task_index is None:
            parser.error("run-task requires --task-index")
        result = run_launch_task(
            args.launch_manifest,
            task_index=args.task_index,
            repo_root=args.repo_root,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    plan = build_plan(
        repo_root=args.repo_root,
        generated_dir=args.generated_dir,
        output_path=args.output,
        launch_manifest_path=args.launch_manifest,
    )
    print(json.dumps({"totals": plan["totals"]}, sort_keys=True))
    for matrix in plan["matrices"]:
        print(
            f"{matrix['screen_config']}: {matrix['counts']} "
            f"array={matrix['todo_slurm_array']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANCHORS",
    "AnchorDiagnosticError",
    "PlannerAnchor",
    "build_plan",
    "build_launch_manifest",
    "finalize_results",
    "pareto_indices",
    "run_launch_task",
    "selected_anchor_runs",
    "write_launch_manifest",
]
