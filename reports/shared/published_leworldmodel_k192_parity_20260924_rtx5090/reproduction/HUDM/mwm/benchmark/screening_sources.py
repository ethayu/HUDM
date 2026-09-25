from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mwm.benchmark.config import (
    DEFAULTS,
    manifest_path as benchmark_manifest_path,
    merged_run_config,
)
from mwm.benchmark.matrix import _completed_row, _configure_run_paths, _run_dir
from mwm.benchmark.screening import (
    NUM_MATRIX_CELLS,
    SCREENING_SCHEMA_VERSION,
    SCREEN_EPISODES,
    PreparedScreen,
    ScreeningPreparationError,
    _atomic_write_text,
    _relative_or_absolute,
    _resolve_from_root,
)
from mwm.benchmark.sweep import expand_benchmark_runs
from mwm.config_cli import load_config
from mwm.io import file_sha256


SOURCE_LEDGER_SCHEMA_VERSION = "mwm_release20260728_screening_sources_v1"


def _jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ScreeningPreparationError(f"Expected JSON objects in {path}.")
            if int(value.get("episode_index", -1)) != count:
                raise ScreeningPreparationError(
                    f"Expected canonical episode_index={count} in {path}."
                )
            if not isinstance(value.get("success"), bool):
                raise ScreeningPreparationError(
                    f"Episode trace {count} in {path} lacks a boolean success value."
                )
            count += 1
    return count


def _source_entry(
    *,
    source_kind: str,
    run_dir: Path,
    row: Mapping[str, Any],
    expected_episodes: int,
    expected_manifest_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    if int(row.get("episodes", -1)) != expected_episodes:
        raise ScreeningPreparationError(
            f"{run_dir}: completed row has {row.get('episodes')} episodes, "
            f"expected {expected_episodes}."
        )
    if str(row.get("manifest_sha256", "")) != expected_manifest_sha256:
        raise ScreeningPreparationError(f"{run_dir}: completed row has the wrong manifest hash.")
    trace_path = run_dir / "episode_traces.jsonl"
    trace_rows = _jsonl_count(trace_path)
    if trace_rows != expected_episodes:
        raise ScreeningPreparationError(
            f"{trace_path}: has {trace_rows} rows, expected {expected_episodes}."
        )
    metrics_path = run_dir / "metrics.jsonl"
    return {
        "source_kind": source_kind,
        "run_dir": _relative_or_absolute(run_dir, repo_root),
        "artifact_episodes": expected_episodes,
        "discovery_episode_range": [0, SCREEN_EPISODES],
        "cell_id": str(row.get("cell_id", "")),
        "config_sha256": str(row.get("config_sha256", "")),
        "manifest_sha256": expected_manifest_sha256,
        # These are the exact small inputs used by selection. They remain
        # stable if the detailed eval payload is losslessly recompressed.
        "selection_input_sha256": {
            "metrics.jsonl": file_sha256(metrics_path),
            "episode_traces.jsonl": file_sha256(trace_path),
        },
    }


def _candidate_source(
    *,
    source_kind: str,
    output_dir: Path,
    run: Any,
    run_cfg: Any,
    manifest_path: Path,
    manifest_sha256: str,
    episodes: int,
    repo_root: Path,
) -> dict[str, Any] | None:
    # Preserve the relative path spelling written by matrix.py while using an
    # absolute path only for filesystem access. Config equality is intentionally
    # strict and treats those spellings as provenance.
    run_dir_ref = _run_dir(output_dir, run, int(run.matrix_index))
    run_dir = run_dir_ref if run_dir_ref.is_absolute() else repo_root / run_dir_ref
    _configure_run_paths(run_cfg, run_dir_ref, manifest_path)
    row = _completed_row(run_dir, run_cfg, manifest_path)
    if row is None:
        return None
    return _source_entry(
        source_kind=source_kind,
        run_dir=run_dir,
        row=row,
        expected_episodes=episodes,
        expected_manifest_sha256=manifest_sha256,
        repo_root=repo_root,
    )


def _load_pinned_sources(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SOURCE_LEDGER_SCHEMA_VERSION:
        raise ScreeningPreparationError(f"Unsupported source ledger schema in {path}.")
    sources = payload.get("sources", {})
    if not isinstance(sources, dict):
        raise ScreeningPreparationError(f"Malformed sources mapping in {path}.")
    return {str(key): dict(value) for key, value in sources.items()}


def collect_matrix_sources(
    prepared: PreparedScreen,
    *,
    repo_root: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    """Pin discovery sources and return deterministic not-yet-run indices.

    A canonical 250-episode completion is preferred only when a cell has no
    previously pinned source. Once a screen50 completion is recorded, a later
    canonical completion cannot silently replace it; its exact input hashes
    must continue to match on every collector rerun.
    """

    root = Path(repo_root).resolve()
    canonical_config_path = _resolve_from_root(root, prepared.canonical_config)
    screen_config_path = _resolve_from_root(root, prepared.screen_config)
    canonical_cfg = load_config(DEFAULTS, canonical_config_path)
    screen_cfg = load_config(DEFAULTS, screen_config_path)
    canonical_manifest_path = benchmark_manifest_path(canonical_cfg)
    screen_manifest_path = benchmark_manifest_path(screen_cfg)
    canonical_output = Path(prepared.canonical_output_dir)
    screen_output = Path(prepared.screen_output_dir)
    runs = expand_benchmark_runs(canonical_cfg)
    if len(runs) != NUM_MATRIX_CELLS:
        raise ScreeningPreparationError(
            f"Expected {NUM_MATRIX_CELLS} canonical cells, got {len(runs)}."
        )

    ledger = _resolve_from_root(root, ledger_path)
    pinned = _load_pinned_sources(ledger)
    sources: dict[str, dict[str, Any]] = {}
    todo: list[int] = []
    counts = {"canonical_250_prefix": 0, "screen50": 0, "pending": 0}

    for run in runs:
        index = int(run.matrix_index)
        key = str(index)
        _, canonical_run_cfg = merged_run_config(canonical_cfg, run)
        _, screen_run_cfg = merged_run_config(screen_cfg, run)
        previous = pinned.get(key)
        if previous is not None:
            previous_kind = str(previous.get("source_kind", ""))
            if previous_kind == "canonical_250_prefix":
                current = _candidate_source(
                    source_kind=previous_kind,
                    output_dir=canonical_output,
                    run=run,
                    run_cfg=canonical_run_cfg,
                    manifest_path=canonical_manifest_path,
                    manifest_sha256=prepared.canonical_manifest_sha256,
                    episodes=250,
                    repo_root=root,
                )
            elif previous_kind == "screen50":
                current = _candidate_source(
                    source_kind=previous_kind,
                    output_dir=screen_output,
                    run=run,
                    run_cfg=screen_run_cfg,
                    manifest_path=screen_manifest_path,
                    manifest_sha256=prepared.screen_manifest_sha256,
                    episodes=SCREEN_EPISODES,
                    repo_root=root,
                )
            else:
                raise ScreeningPreparationError(
                    f"Pinned source has an unknown kind for matrix index {index}."
                )
            if current != previous:
                raise ScreeningPreparationError(
                    f"Pinned discovery source changed or disappeared for matrix index {index}."
                )
            selected = previous
        else:
            selected = _candidate_source(
                source_kind="canonical_250_prefix",
                output_dir=canonical_output,
                run=run,
                run_cfg=canonical_run_cfg,
                manifest_path=canonical_manifest_path,
                manifest_sha256=prepared.canonical_manifest_sha256,
                episodes=250,
                repo_root=root,
            )
            if selected is None:
                selected = _candidate_source(
                    source_kind="screen50",
                    output_dir=screen_output,
                    run=run,
                    run_cfg=screen_run_cfg,
                    manifest_path=screen_manifest_path,
                    manifest_sha256=prepared.screen_manifest_sha256,
                    episodes=SCREEN_EPISODES,
                    repo_root=root,
                )
        if selected is None:
            todo.append(index)
            counts["pending"] += 1
            continue
        if selected.get("cell_id") != str(run.cell_id):
            raise ScreeningPreparationError(
                f"Source cell identity mismatch at matrix index {index}."
            )
        sources[key] = selected
        counts[str(selected["source_kind"])] += 1

    payload = {
        "schema_version": SOURCE_LEDGER_SCHEMA_VERSION,
        "screening_schema_version": SCREENING_SCHEMA_VERSION,
        "canonical_config": prepared.canonical_config,
        "canonical_config_sha256": prepared.canonical_config_sha256,
        "screen_config": prepared.screen_config,
        "canonical_manifest_sha256": prepared.canonical_manifest_sha256,
        "screen_manifest_sha256": prepared.screen_manifest_sha256,
        "discovery_episode_range": [0, SCREEN_EPISODES],
        "counts": counts,
        "sources": sources,
        "todo_indices": todo,
    }
    _atomic_write_text(ledger, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def collect_all_sources(
    prepared: list[PreparedScreen],
    *,
    repo_root: str | Path,
    generated_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    target = _resolve_from_root(root, generated_dir)
    matrices: list[dict[str, Any]] = []
    totals = {"canonical_250_prefix": 0, "screen50": 0, "pending": 0}
    for item in prepared:
        ledger_path = target / f"{Path(item.screen_config).stem}_todo.json"
        payload = collect_matrix_sources(item, repo_root=root, ledger_path=ledger_path)
        for key in totals:
            totals[key] += int(payload["counts"][key])
        matrices.append(
            {
                "canonical_config": item.canonical_config,
                "screen_config": item.screen_config,
                "source_ledger": _relative_or_absolute(ledger_path, root),
                "counts": payload["counts"],
            }
        )
    summary = {
        "schema_version": SOURCE_LEDGER_SCHEMA_VERSION,
        "discovery_episode_range": [0, SCREEN_EPISODES],
        "totals": totals,
        "matrices": matrices,
    }
    _atomic_write_text(
        target / "screening_todo.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


__all__ = [
    "SOURCE_LEDGER_SCHEMA_VERSION",
    "collect_all_sources",
    "collect_matrix_sources",
]
