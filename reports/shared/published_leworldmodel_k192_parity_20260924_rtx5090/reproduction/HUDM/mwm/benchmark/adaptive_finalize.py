from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mwm.benchmark.config import DEFAULTS, cell_id, manifest_path, merged_run_config, validate_benchmark_matrix
from mwm.benchmark.matrix import _completed_row, _configure_run_paths, _run_dir
from mwm.benchmark.pareto import pareto_frontier, write_pareto_html
from mwm.benchmark.sweep import expand_benchmark_runs
from mwm.config_cli import load_config
from mwm.data.manifest import load_manifest
from mwm.io import file_sha256, load_json, write_json, write_metrics_jsonl


ADAPTIVE_SELECTION_SCHEMA = "mwm.adaptive_selection/v1"
ADAPTIVE_SUMMARY_SCHEMA = "mwm.adaptive_benchmark/v1"
ADAPTIVE_PARETO_SCHEMA = "mwm.adaptive_pareto/v1"
HELD_OUT_OUTCOME_BASIS = "held_out_episode_slice"


class AdaptiveFinalizeError(ValueError):
    pass


@dataclass(frozen=True)
class _MatrixCell:
    matrix_index: int
    run: Any
    run_cfg: Any
    run_dir: Path
    manifest_path: Path

    @property
    def cell_id(self) -> str:
        return cell_id(self.run, self.run_cfg)


@dataclass(frozen=True)
class _AdaptiveInputs:
    screen_cfg: Any
    confirmation_cfg: Any
    selection: dict[str, Any]
    selection_path: Path
    screen_cells: list[_MatrixCell]
    confirmation_cells: list[_MatrixCell]
    selected_indices: tuple[int, ...]
    selection_records: dict[int, dict[str, Any]]
    all_records: dict[int, dict[str, Any]]
    selection_episodes: int
    confirmation_episodes: int


def _matrix_cells(cfg_path: str | Path) -> tuple[Any, list[_MatrixCell]]:
    cfg = load_config(DEFAULTS, str(cfg_path), [])
    output_dir = Path(str(cfg.output_dir))
    shared_manifest = manifest_path(cfg)
    resolved: list[tuple[Any, Any]] = []
    cells: list[_MatrixCell] = []
    for matrix_index, run in enumerate(expand_benchmark_runs(cfg)):
        if "matrix_index" not in run:
            run.matrix_index = matrix_index
        actual_index = int(run.matrix_index)
        if actual_index != matrix_index:
            raise AdaptiveFinalizeError(
                f"{cfg_path}: noncanonical matrix_index {actual_index} at expansion position {matrix_index}"
            )
        _, run_cfg = merged_run_config(cfg, run)
        resolved.append((run, run_cfg))
        run_dir = _run_dir(output_dir, run, matrix_index)
        _configure_run_paths(run_cfg, run_dir, shared_manifest)
        cells.append(
            _MatrixCell(
                matrix_index=matrix_index,
                run=run,
                run_cfg=run_cfg,
                run_dir=run_dir,
                manifest_path=shared_manifest,
            )
        )
    validate_benchmark_matrix(cfg, resolved)
    return cfg, cells


def _int_index(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise AdaptiveFinalizeError(f"{label} must be an integer matrix index")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise AdaptiveFinalizeError(f"{label} must be an integer matrix index") from exc
    if str(value).strip() != str(index) and not isinstance(value, int):
        raise AdaptiveFinalizeError(f"{label} must be an exact integer matrix index")
    return index


def _indexed_records(value: Any, *, label: str) -> dict[int, dict[str, Any]]:
    if not isinstance(value, list):
        raise AdaptiveFinalizeError(f"selection manifest {label} must be a list")
    indexed: dict[int, dict[str, Any]] = {}
    for position, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise AdaptiveFinalizeError(f"selection manifest {label}[{position}] must be a mapping")
        index = _int_index(raw.get("matrix_index"), label=f"{label}[{position}].matrix_index")
        if index in indexed:
            raise AdaptiveFinalizeError(f"selection manifest {label} duplicates matrix_index {index}")
        indexed[index] = dict(raw)
    return indexed


def _discovery_source(record: dict[str, Any], *, label: str) -> tuple[str, dict[str, Any]]:
    discovery = record.get("discovery_source")
    if not isinstance(discovery, dict):
        raise AdaptiveFinalizeError(f"selection {label}.discovery_source must be a mapping")
    kind = str(discovery.get("kind", ""))
    if kind not in {"canonical_confirmation", "screen"}:
        raise AdaptiveFinalizeError(
            f"selection {label}.discovery_source.kind must be 'canonical_confirmation' or 'screen'"
        )
    return kind, dict(discovery)


def _validate_selection(
    payload: dict[str, Any],
    *,
    expected_cells: list[_MatrixCell],
) -> tuple[tuple[int, ...], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    if payload.get("schema_version") != ADAPTIVE_SELECTION_SCHEMA:
        raise AdaptiveFinalizeError(
            f"selection schema must be {ADAPTIVE_SELECTION_SCHEMA!r}, got {payload.get('schema_version')!r}"
        )
    raw_indices = payload.get("selected_matrix_indices")
    if not isinstance(raw_indices, list):
        raise AdaptiveFinalizeError("selection manifest selected_matrix_indices must be a list")
    selected = tuple(
        _int_index(value, label=f"selected_matrix_indices[{position}]")
        for position, value in enumerate(raw_indices)
    )
    if tuple(sorted(set(selected))) != selected:
        raise AdaptiveFinalizeError("selected_matrix_indices must be sorted and unique")
    if not selected:
        raise AdaptiveFinalizeError("selection manifest selected no confirmation cells")
    if payload.get("selected_count") != len(selected):
        raise AdaptiveFinalizeError("selection selected_count does not match selected_matrix_indices")

    expected_indices = {cell.matrix_index for cell in expected_cells}
    if not set(selected) <= expected_indices:
        raise AdaptiveFinalizeError(
            f"selection contains unexpected matrix indices: {sorted(set(selected) - expected_indices)}"
        )
    all_records = _indexed_records(payload.get("all_cells"), label="all_cells")
    if set(all_records) != expected_indices:
        missing = sorted(expected_indices - set(all_records))
        extra = sorted(set(all_records) - expected_indices)
        raise AdaptiveFinalizeError(f"selection all_cells coverage mismatch; missing={missing}, extra={extra}")
    selected_records = _indexed_records(payload.get("selected_cells"), label="selected_cells")
    if tuple(sorted(selected_records)) != selected:
        raise AdaptiveFinalizeError("selected_cells indices do not exactly match selected_matrix_indices")

    for cell in expected_cells:
        record = all_records[cell.matrix_index]
        selected_flag = record.get("selected")
        if not isinstance(selected_flag, bool) or selected_flag != (cell.matrix_index in selected_records):
            raise AdaptiveFinalizeError(
                f"selection all_cells[{cell.matrix_index}].selected disagrees with selected_matrix_indices"
            )
        record_cell_id = record.get("cell_id")
        if record_cell_id is not None and str(record_cell_id) != cell.cell_id:
            raise AdaptiveFinalizeError(
                f"selection matrix_index {cell.matrix_index} cell_id mismatch: "
                f"{record_cell_id!r} vs {cell.cell_id!r}"
            )
        _discovery_source(record, label=f"all_cells[{cell.matrix_index}]")
    for index, record in selected_records.items():
        if str(record.get("cell_id", expected_cells[index].cell_id)) != expected_cells[index].cell_id:
            raise AdaptiveFinalizeError(f"selected cell_id does not match canonical matrix_index {index}")
        reasons = record.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            raise AdaptiveFinalizeError(f"selected cell {index} must record at least one selection reason")
        if record.get("discovery_source") != all_records[index].get("discovery_source"):
            raise AdaptiveFinalizeError(
                f"selected_cells discovery source differs from all_cells for matrix_index {index}"
            )
    return selected, selected_records, all_records


def _uniform_episodes(cells: list[_MatrixCell], *, label: str) -> int:
    episodes = {int(cell.run_cfg.eval.episodes) for cell in cells}
    if len(episodes) != 1:
        raise AdaptiveFinalizeError(f"{label} matrix does not have one episode count: {sorted(episodes)}")
    return episodes.pop()


def _load_inputs(
    screen_cfg_path: str | Path,
    confirmation_cfg_path: str | Path,
    selection_path: str | Path,
) -> _AdaptiveInputs:
    screen_cfg, screen_cells = _matrix_cells(screen_cfg_path)
    confirmation_cfg, confirmation_cells = _matrix_cells(confirmation_cfg_path)
    if len(screen_cells) != len(confirmation_cells):
        raise AdaptiveFinalizeError(
            f"screen and confirmation matrices differ in size: {len(screen_cells)} vs {len(confirmation_cells)}"
        )
    for screen, confirmation in zip(screen_cells, confirmation_cells, strict=True):
        if screen.matrix_index != confirmation.matrix_index or screen.cell_id != confirmation.cell_id:
            raise AdaptiveFinalizeError(
                "screen and confirmation matrices must preserve canonical matrix_index/cell_id identity; "
                f"got {screen.matrix_index}:{screen.cell_id!r} and "
                f"{confirmation.matrix_index}:{confirmation.cell_id!r}"
            )
    screen_root = Path(str(screen_cfg.output_dir)).resolve()
    confirmation_root = Path(str(confirmation_cfg.output_dir)).resolve()
    if screen_root == confirmation_root:
        raise AdaptiveFinalizeError("screen and confirmation output roots must be separate")

    path = Path(selection_path)
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise AdaptiveFinalizeError("selection manifest must contain a JSON mapping")
    selected, records, all_records = _validate_selection(payload, expected_cells=screen_cells)
    selection_episodes = _uniform_episodes(screen_cells, label="screen")
    confirmation_episodes = _uniform_episodes(confirmation_cells, label="confirmation")
    selection_cfg = payload.get("selection_config")
    if not isinstance(selection_cfg, dict) or selection_cfg.get("discovery_episodes") != selection_episodes:
        raise AdaptiveFinalizeError(
            "selection_config.discovery_episodes does not match the screen matrix episode count"
        )
    if not 0 < selection_episodes < confirmation_episodes:
        raise AdaptiveFinalizeError(
            "episode contract requires 0 < screen < confirmation, got "
            f"{selection_episodes} and {confirmation_episodes}"
        )
    return _AdaptiveInputs(
        screen_cfg=screen_cfg,
        confirmation_cfg=confirmation_cfg,
        selection=payload,
        selection_path=path,
        screen_cells=screen_cells,
        confirmation_cells=confirmation_cells,
        selected_indices=selected,
        selection_records=records,
        all_records=all_records,
        selection_episodes=selection_episodes,
        confirmation_episodes=confirmation_episodes,
    )


def _episode_traces(path: Path, *, expected: int) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AdaptiveFinalizeError(f"{path}:{line_number} is not a JSON mapping")
                traces.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveFinalizeError(f"cannot read episode traces {path}: {exc}") from exc
    if len(traces) != expected:
        raise AdaptiveFinalizeError(f"{path} has {len(traces)} traces, expected {expected}")
    for index, trace in enumerate(traces):
        if int(trace.get("episode_index", -1)) != index:
            raise AdaptiveFinalizeError(f"{path} has noncanonical episode_index at position {index}")
        if not isinstance(trace.get("success"), bool):
            raise AdaptiveFinalizeError(f"{path} trace {index} lacks a boolean success outcome")
    return traces


def _pair_identity(trace: dict[str, Any]) -> tuple[Any, Any, Any]:
    return trace.get("dataset_episode"), trace.get("start_step"), trace.get("goal_step")


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AdaptiveFinalizeError(f"cannot fingerprint adaptive discovery inputs: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _assert_nested_manifest(inputs: _AdaptiveInputs) -> None:
    if not any(
        _discovery_source(record, label=f"all_cells[{index}]")[0] == "screen"
        for index, record in inputs.all_records.items()
    ):
        return
    screen_manifest = load_manifest(inputs.screen_cells[0].manifest_path)
    confirmation_manifest = load_manifest(inputs.confirmation_cells[0].manifest_path)
    screen_pairs = list(screen_manifest.get("pairs", []))
    confirmation_pairs = list(confirmation_manifest.get("pairs", []))
    if len(screen_pairs) != inputs.selection_episodes:
        raise AdaptiveFinalizeError(
            f"screen manifest has {len(screen_pairs)} pairs, expected {inputs.selection_episodes}"
        )
    if len(confirmation_pairs) != inputs.confirmation_episodes:
        raise AdaptiveFinalizeError(
            f"confirmation manifest has {len(confirmation_pairs)} pairs, expected {inputs.confirmation_episodes}"
        )
    if screen_pairs != confirmation_pairs[: inputs.selection_episodes]:
        raise AdaptiveFinalizeError("screen manifest pairs are not the exact logical prefix of confirmation pairs")


_NORMALIZED_COST_KEYS = (
    "dynamics_flops_total",
    "candidate_action_values",
    "latent_work_total",
    "bits_used_total",
    "cem_cost_calls",
    "plan_time_total_sec",
    "wall_time_sec",
)


def _heldout_row(
    source_row: dict[str, Any],
    confirmation_traces: list[dict[str, Any]],
    *,
    matrix_index: int,
    selection_episodes: int,
    selection_record: dict[str, Any],
) -> dict[str, Any]:
    canonical_episodes = len(confirmation_traces)
    if int(source_row.get("episodes", -1)) != canonical_episodes:
        raise AdaptiveFinalizeError(
            f"confirmation row {matrix_index} reports {source_row.get('episodes')} episodes, "
            f"but has {canonical_episodes} traces"
        )
    heldout = confirmation_traces[selection_episodes:]
    if not heldout:
        raise AdaptiveFinalizeError(f"confirmation row {matrix_index} has an empty held-out slice")
    successes = sum(bool(trace["success"]) for trace in heldout)
    row = dict(source_row)
    row.update(
        {
            "matrix_index": int(matrix_index),
            "canonical_episodes": canonical_episodes,
            "canonical_success_rate": float(source_row.get("success_rate", float("nan"))),
            "episodes": len(heldout),
            "successes": successes,
            "success_rate": 100.0 * successes / len(heldout),
            "outcome_basis": HELD_OUT_OUTCOME_BASIS,
            "confirmation_episode_slice": [selection_episodes, canonical_episodes],
            "selection_episodes": selection_episodes,
            "selection_reasons": list(selection_record.get("reasons", [])),
        }
    )
    for key in _NORMALIZED_COST_KEYS:
        if key in source_row:
            row[f"canonical_{key.removesuffix('_total')}_per_episode"] = (
                float(source_row[key]) / canonical_episodes
            )
    return row


def _collect_rows(inputs: _AdaptiveInputs) -> tuple[list[dict[str, Any]], list[str]]:
    _assert_nested_manifest(inputs)
    discovery_rows: dict[int, dict[str, Any]] = {}
    discovery_traces: dict[int, list[dict[str, Any]]] = {}
    missing_discovery: list[str] = []
    for index, record in sorted(inputs.all_records.items()):
        source, discovery = _discovery_source(record, label=f"all_cells[{index}]")
        cell = (
            inputs.confirmation_cells[index]
            if source == "canonical_confirmation"
            else inputs.screen_cells[index]
        )
        expected_episodes = (
            inputs.confirmation_episodes
            if source == "canonical_confirmation"
            else inputs.selection_episodes
        )
        row = _completed_row(cell.run_dir, cell.run_cfg, cell.manifest_path)
        if row is None:
            missing_discovery.append(f"{index}:{cell.cell_id} ({source})")
        elif int(row.get("episodes", -1)) != expected_episodes:
            raise AdaptiveFinalizeError(
                f"{source} discovery cell {index} reports {row.get('episodes')} episodes, "
                f"expected {expected_episodes}"
            )
        else:
            expected_output = discovery.get("output_json")
            if (
                expected_output is not None
                and Path(str(expected_output)).resolve() != (cell.run_dir / "eval.json").resolve()
            ):
                raise AdaptiveFinalizeError(f"selection discovery output path mismatch for matrix_index {index}")
            for selection_key, row_key in (
                ("config_sha256", "config_sha256"),
                ("manifest_sha256", "manifest_sha256"),
                ("manifest_file_sha256", "manifest_file_sha256"),
            ):
                if (
                    discovery.get(selection_key) is not None
                    and str(discovery[selection_key]) != str(row.get(row_key, ""))
                ):
                    raise AdaptiveFinalizeError(
                        f"selection discovery {selection_key} mismatch for matrix_index {index}"
                    )
            traces = _episode_traces(cell.run_dir / "episode_traces.jsonl", expected=expected_episodes)
            discovery_rows[index] = row
            discovery_traces[index] = traces[: inputs.selection_episodes]
    if missing_discovery:
        preview = ", ".join(missing_discovery[:10])
        suffix = f" (+{len(missing_discovery) - 10} more)" if len(missing_discovery) > 10 else ""
        raise AdaptiveFinalizeError(f"discovery coverage is incomplete; missing/stale cells: {preview}{suffix}")

    cost_key = str(inputs.selection.get("selection_config", {}).get("cost_key", ""))
    discovery_fingerprint = []
    for index, record in sorted(inputs.all_records.items()):
        row = discovery_rows[index]
        if cost_key not in row:
            raise AdaptiveFinalizeError(f"discovery row {index} lacks selected cost key {cost_key!r}")
        outcomes = [bool(trace["success"]) for trace in discovery_traces[index]]
        recorded_summary = record.get("discovery")
        expected_summary = {
            "episode_slice": [0, inputs.selection_episodes],
            "episodes": inputs.selection_episodes,
            "successes": sum(outcomes),
            "success_rate_fraction": sum(outcomes) / inputs.selection_episodes,
        }
        if not isinstance(recorded_summary, dict) or any(
            recorded_summary.get(key) != value for key, value in expected_summary.items()
        ):
            raise AdaptiveFinalizeError(f"selection discovery summary mismatch for matrix_index {index}")
        source_kind, source = _discovery_source(record, label=f"all_cells[{index}]")
        del source_kind
        discovery_fingerprint.append(
            {
                "matrix_index": index,
                "cell_id": str(record.get("cell_id", "")),
                "schedule": str(record.get("schedule", "")),
                "episodes": int(row["episodes"]),
                "total_cost": float(row[cost_key]),
                "discovery_source": source,
                "discovery_outcomes": outcomes,
            }
        )
    expected_input_sha = str(inputs.selection.get("screen", {}).get("input_sha256", ""))
    if not expected_input_sha or _canonical_sha256(discovery_fingerprint) != expected_input_sha:
        raise AdaptiveFinalizeError("selection discovery input fingerprint does not match frozen source artifacts")

    rows: list[dict[str, Any]] = []
    missing_confirmation: list[str] = []
    for index in inputs.selected_indices:
        confirmation = inputs.confirmation_cells[index]
        source_row = _completed_row(confirmation.run_dir, confirmation.run_cfg, confirmation.manifest_path)
        if source_row is None:
            missing_confirmation.append(f"{index}:{confirmation.cell_id}")
            continue
        confirmation_traces = _episode_traces(
            confirmation.run_dir / "episode_traces.jsonl", expected=inputs.confirmation_episodes
        )
        screen_pairs = [_pair_identity(trace) for trace in discovery_traces[index]]
        confirmation_prefix = [
            _pair_identity(trace) for trace in confirmation_traces[: inputs.selection_episodes]
        ]
        if screen_pairs != confirmation_prefix:
            raise AdaptiveFinalizeError(
                f"screen/confirmation trace pair identity differs for selected matrix_index {index}"
            )
        row = _heldout_row(
            source_row,
            confirmation_traces,
            matrix_index=index,
            selection_episodes=inputs.selection_episodes,
            selection_record=inputs.selection_records[index],
        )
        discovery_kind, _ = _discovery_source(
            inputs.all_records[index], label=f"all_cells[{index}]"
        )
        discovery_cell = (
            inputs.confirmation_cells[index]
            if discovery_kind == "canonical_confirmation"
            else inputs.screen_cells[index]
        )
        row["discovery_source"] = discovery_kind
        row["discovery_output_json"] = str(discovery_cell.run_dir / "eval.json")
        row["discovery_config_sha256"] = str(discovery_rows[index].get("config_sha256", ""))
        if str(row.get("cell_id", "")) != confirmation.cell_id:
            raise AdaptiveFinalizeError(f"confirmation summary cell_id mismatch for matrix_index {index}")
        rows.append(row)
    if missing_confirmation:
        preview = ", ".join(missing_confirmation[:10])
        suffix = f" (+{len(missing_confirmation) - 10} more)" if len(missing_confirmation) > 10 else ""
        raise AdaptiveFinalizeError(
            f"selected confirmation cells are incomplete; missing/stale cells: {preview}{suffix}"
        )
    return rows, [
        str(discovery_rows[index].get("manifest_sha256", "")) for index in sorted(discovery_rows)
    ]


def _pareto_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cost_key = "canonical_dynamics_flops_per_episode"
    frontier = pareto_frontier(rows, cost_key=cost_key)
    return {
        "schema_version": ADAPTIVE_PARETO_SCHEMA,
        "cost_key": cost_key,
        "outcome_key": "success_rate",
        "outcome_basis": HELD_OUT_OUTCOME_BASIS,
        "cells": [
            {
                "matrix_index": int(row["matrix_index"]),
                "cell_id": str(row.get("cell_id", "")),
                "cost": float(row[cost_key]),
                "held_out_success_rate": float(row["success_rate"]),
            }
            for row in frontier
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "matrix_index",
        "cell_id",
        "name",
        "schedule",
        "successes",
        "episodes",
        "success_rate",
        "outcome_basis",
        "confirmation_episode_slice",
        "canonical_episodes",
        "canonical_success_rate",
        "canonical_dynamics_flops_per_episode",
        "dynamics_flops_total",
        "pop_size",
        "elite_frac",
        "topk",
        "n_iter",
        "selection_reasons",
        "output_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                    for key in preferred
                }
            )


def _summary_payload(inputs: _AdaptiveInputs, rows: list[dict[str, Any]], pareto_path: Path) -> dict[str, Any]:
    return {
        "schema_version": ADAPTIVE_SUMMARY_SCHEMA,
        "title": f"{inputs.confirmation_cfg.title} — held-out confirmation",
        "runs": rows,
        "adaptive": {
            "selection_manifest": str(inputs.selection_path),
            "selection_manifest_sha256": file_sha256(inputs.selection_path),
            "selection_schema": ADAPTIVE_SELECTION_SCHEMA,
            "screen_output_dir": str(inputs.screen_cfg.output_dir),
            "confirmation_output_dir": str(inputs.confirmation_cfg.output_dir),
            "matrix_cells": len(inputs.screen_cells),
            "selected_cells": len(inputs.selected_indices),
            "screened_out_cells": len(inputs.screen_cells) - len(inputs.selected_indices),
            "canonical_discovery_cells": sum(
                _discovery_source(record, label=f"all_cells[{index}]")[0]
                == "canonical_confirmation"
                for index, record in inputs.all_records.items()
            ),
            "screen_discovery_cells": sum(
                _discovery_source(record, label=f"all_cells[{index}]")[0] == "screen"
                for index, record in inputs.all_records.items()
            ),
            "selection_episodes": inputs.selection_episodes,
            "confirmation_episode_slice": [inputs.selection_episodes, inputs.confirmation_episodes],
            "outcome_basis": HELD_OUT_OUTCOME_BASIS,
        },
        "pareto_cost": "canonical_dynamics_flops_per_episode",
        "pareto_html": str(pareto_path),
    }


def finalize_adaptive_benchmark(
    screen_cfg_path: str | Path,
    confirmation_cfg_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    from mwm.benchmark.html import write_review_html

    inputs = _load_inputs(screen_cfg_path, confirmation_cfg_path, selection_path)
    out = Path(output_dir)
    source_roots = {
        Path(str(inputs.screen_cfg.output_dir)).resolve(),
        Path(str(inputs.confirmation_cfg.output_dir)).resolve(),
    }
    if out.resolve() in source_roots:
        raise AdaptiveFinalizeError("adaptive aggregate output must not overwrite a screen or confirmation root")
    rows, _ = _collect_rows(inputs)
    plots_dir = out / "plots"
    pareto_html = plots_dir / "pareto.html"
    write_pareto_html(
        pareto_html,
        rows,
        cost_key="canonical_dynamics_flops_per_episode",
        provisional=False,
    )
    pareto = _pareto_payload(rows)
    summary = _summary_payload(inputs, rows, pareto_html)
    write_json(out / "summary.json", summary)
    write_metrics_jsonl(out / "metrics.jsonl", rows)
    _write_csv(out / "summary.csv", rows)
    write_json(out / "pareto.json", pareto)
    write_review_html(
        out / "review.html",
        str(summary["title"]),
        rows,
        [],
        plots=[],
        expected_cells=len(rows),
        pareto_html=str(pareto_html),
        include_rollouts=False,
    )
    return summary


def verify_adaptive_benchmark_output(
    screen_cfg_path: str | Path,
    confirmation_cfg_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    inputs = _load_inputs(screen_cfg_path, confirmation_cfg_path, selection_path)
    expected_rows, _ = _collect_rows(inputs)
    out = Path(output_dir)
    required = (
        out / "summary.json",
        out / "summary.csv",
        out / "metrics.jsonl",
        out / "pareto.json",
        out / "plots" / "pareto.html",
        out / "review.html",
    )
    errors = [f"missing or empty file: {path}" for path in required if not path.is_file() or path.stat().st_size == 0]
    if errors:
        raise AdaptiveFinalizeError("adaptive output is incomplete:\n- " + "\n- ".join(errors))
    summary = load_json(out / "summary.json")
    if summary.get("schema_version") != ADAPTIVE_SUMMARY_SCHEMA:
        errors.append("summary schema_version mismatch")
    if summary.get("runs") != expected_rows:
        errors.append("summary runs do not exactly regenerate held-out confirmation rows")
    expected_summary = _summary_payload(inputs, expected_rows, out / "plots" / "pareto.html")
    if summary != expected_summary:
        errors.append("summary does not exactly regenerate from selection and source cells")
    with (out / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        metrics = [json.loads(line) for line in handle if line.strip()]
    if metrics != expected_rows:
        errors.append("metrics.jsonl does not exactly regenerate held-out confirmation rows")
    expected_pareto = _pareto_payload(expected_rows)
    if load_json(out / "pareto.json") != expected_pareto:
        errors.append("pareto.json does not exactly regenerate from held-out confirmation rows")
    with (out / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if [int(row["matrix_index"]) for row in csv_rows] != list(inputs.selected_indices):
        errors.append("summary.csv matrix indices do not exactly match selected confirmation cells")
    review_text = (out / "review.html").read_text(encoding="utf-8")
    if "held-out confirmation" not in review_text.lower():
        errors.append("review.html does not identify held-out confirmation outcomes")
    if errors:
        raise AdaptiveFinalizeError("adaptive output failed verification:\n- " + "\n- ".join(errors))
    return {
        "output_dir": str(out),
        "screen_cells": len(inputs.screen_cells),
        "confirmed_cells": len(expected_rows),
        "held_out_episodes_per_cell": inputs.confirmation_episodes - inputs.selection_episodes,
        "pareto_cells": len(expected_pareto["cells"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize or verify a discovery-screen plus held-out-confirmation benchmark."
    )
    parser.add_argument("command", choices=("finalize", "verify"))
    parser.add_argument("--screen-config", required=True)
    parser.add_argument("--confirmation-config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    function = finalize_adaptive_benchmark if args.command == "finalize" else verify_adaptive_benchmark_output
    result = function(args.screen_config, args.confirmation_config, args.selection, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ADAPTIVE_PARETO_SCHEMA",
    "ADAPTIVE_SELECTION_SCHEMA",
    "ADAPTIVE_SUMMARY_SCHEMA",
    "AdaptiveFinalizeError",
    "finalize_adaptive_benchmark",
    "verify_adaptive_benchmark_output",
]
