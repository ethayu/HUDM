from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mwm.benchmark.adaptive_finalize import _matrix_cells
from mwm.benchmark.adaptive_selection import (
    build_selection_manifest,
    load_selection_manifest,
    write_selection_manifest,
)
from mwm.benchmark.eval_artifacts import compress_completed_eval
from mwm.benchmark.matrix import _completed_row
from mwm.benchmark.screening import screen_storage_preflight
from mwm.benchmark.config import DEFAULTS
from mwm.config_cli import load_config
from mwm.data.manifest import load_manifest
from mwm.io import file_sha256


WORKFLOW_SCHEMA = "mwm.adaptive_workflow/v1"


class AdaptiveWorkflowError(ValueError):
    pass


def indices_to_slurm_array(indices: list[int] | tuple[int, ...]) -> str:
    values = sorted(set(int(value) for value in indices))
    if any(value < 0 for value in values):
        raise AdaptiveWorkflowError("Slurm array indices must be nonnegative.")
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _episode_traces(path: Path, *, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AdaptiveWorkflowError(f"{path}:{line_number} is not a JSON mapping.")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveWorkflowError(f"Cannot read {path}: {exc}") from exc
    if len(rows) != expected:
        raise AdaptiveWorkflowError(f"{path} has {len(rows)} episode traces; expected {expected}.")
    for index, row in enumerate(rows):
        if int(row.get("episode_index", -1)) != index:
            raise AdaptiveWorkflowError(f"{path} has a noncanonical episode_index at position {index}.")
        if not isinstance(row.get("success"), bool):
            raise AdaptiveWorkflowError(f"{path} trace {index} lacks a boolean success value.")
    return rows


def _pair_identity(value: dict[str, Any]) -> tuple[int, int, int]:
    return int(value["episode"]), int(value["start_step"]), int(value["goal_step"])


def _trace_pair_identity(value: dict[str, Any]) -> tuple[int, int, int]:
    return int(value["dataset_episode"]), int(value["start_step"]), int(value["goal_step"])


def _validate_matrix_pair(screen_cfg_path: str | Path, confirmation_cfg_path: str | Path) -> tuple[Any, Any, Any, Any]:
    screen_cfg, screen_cells = _matrix_cells(screen_cfg_path)
    confirmation_cfg, confirmation_cells = _matrix_cells(confirmation_cfg_path)
    if len(screen_cells) != len(confirmation_cells):
        raise AdaptiveWorkflowError(
            f"Screen and confirmation matrices differ in size: {len(screen_cells)} vs {len(confirmation_cells)}."
        )
    for screen, confirmation in zip(screen_cells, confirmation_cells, strict=True):
        if screen.matrix_index != confirmation.matrix_index or screen.cell_id != confirmation.cell_id:
            raise AdaptiveWorkflowError(
                "Screening changed canonical matrix_index/cell_id identity at "
                f"{screen.matrix_index}:{screen.cell_id!r}."
            )
    screen_episodes = {int(cell.run_cfg.eval.episodes) for cell in screen_cells}
    confirmation_episodes = {int(cell.run_cfg.eval.episodes) for cell in confirmation_cells}
    if len(screen_episodes) != 1 or len(confirmation_episodes) != 1:
        raise AdaptiveWorkflowError("Each adaptive matrix must have one uniform episode count.")
    discovery_n = screen_episodes.pop()
    confirmation_n = confirmation_episodes.pop()
    if not 0 < discovery_n < confirmation_n:
        raise AdaptiveWorkflowError(
            f"Expected 0 < screen episodes < confirmation episodes, got {discovery_n} and {confirmation_n}."
        )
    screen_manifest = load_manifest(screen_cells[0].manifest_path)
    confirmation_manifest = load_manifest(confirmation_cells[0].manifest_path)
    screen_pairs = list(screen_manifest.get("pairs", []))
    confirmation_pairs = list(confirmation_manifest.get("pairs", []))
    if len(screen_pairs) != discovery_n or len(confirmation_pairs) != confirmation_n:
        raise AdaptiveWorkflowError("Manifest pair counts do not match their matrix episode counts.")
    if screen_pairs != confirmation_pairs[:discovery_n]:
        raise AdaptiveWorkflowError("Screen manifest is not the exact prefix of the confirmation manifest.")
    return (screen_cfg, screen_cells, confirmation_cfg, confirmation_cells)


def collect_discovery_rows(
    screen_cfg_path: str | Path,
    confirmation_cfg_path: str | Path,
    *,
    source_ledger_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    screen_cfg, screen_cells, confirmation_cfg, confirmation_cells = _validate_matrix_pair(
        screen_cfg_path, confirmation_cfg_path
    )
    discovery_n = int(screen_cells[0].run_cfg.eval.episodes)
    confirmation_n = int(confirmation_cells[0].run_cfg.eval.episodes)
    canonical_manifest = load_manifest(confirmation_cells[0].manifest_path)
    expected_prefix = [_pair_identity(pair) for pair in canonical_manifest["pairs"][:discovery_n]]
    source_ledger: dict[str, Any] | None = None
    pinned_sources: dict[str, Any] = {}
    if source_ledger_path is not None:
        try:
            source_ledger = json.loads(Path(source_ledger_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdaptiveWorkflowError(f"Cannot read source ledger {source_ledger_path}: {exc}") from exc
        if source_ledger.get("schema_version") != "mwm_release20260728_screening_sources_v1":
            raise AdaptiveWorkflowError(f"Unsupported source ledger schema in {source_ledger_path}.")
        raw_sources = source_ledger.get("sources")
        if not isinstance(raw_sources, dict):
            raise AdaptiveWorkflowError("Source ledger sources must be a mapping.")
        pinned_sources = raw_sources

    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    source_counts = {"canonical_confirmation": 0, "screen": 0}
    for screen, confirmation in zip(screen_cells, confirmation_cells, strict=True):
        pinned = pinned_sources.get(str(screen.matrix_index)) if source_ledger is not None else None
        if source_ledger is not None and pinned is None:
            missing.append(screen.matrix_index)
            continue
        if pinned is not None:
            raw_kind = str(pinned.get("source_kind", ""))
            if raw_kind == "canonical_250_prefix":
                source_kind = "canonical_confirmation"
                source_cell = confirmation
                expected_episodes = confirmation_n
            elif raw_kind == "screen50":
                source_kind = "screen"
                source_cell = screen
                expected_episodes = discovery_n
            else:
                raise AdaptiveWorkflowError(
                    f"Source ledger has invalid source_kind={raw_kind!r} at matrix_index {screen.matrix_index}."
                )
            source_row = _completed_row(source_cell.run_dir, source_cell.run_cfg, source_cell.manifest_path)
            if source_row is None:
                raise AdaptiveWorkflowError(
                    f"Pinned {raw_kind} source is no longer complete at matrix_index {screen.matrix_index}."
                )
            expected_run_dir = Path(str(pinned.get("run_dir", ""))).resolve()
            if expected_run_dir != source_cell.run_dir.resolve():
                raise AdaptiveWorkflowError(
                    f"Pinned source path changed at matrix_index {screen.matrix_index}."
                )
            for field in ("config_sha256", "manifest_sha256"):
                if str(pinned.get(field, "")) != str(source_row.get(field, "")):
                    raise AdaptiveWorkflowError(
                        f"Pinned source {field} changed at matrix_index {screen.matrix_index}."
                    )
            input_hashes = pinned.get("selection_input_sha256", {})
            if not isinstance(input_hashes, dict):
                raise AdaptiveWorkflowError(
                    f"Pinned source hashes are malformed at matrix_index {screen.matrix_index}."
                )
            for name in ("metrics.jsonl", "episode_traces.jsonl"):
                if str(input_hashes.get(name, "")) != file_sha256(source_cell.run_dir / name):
                    raise AdaptiveWorkflowError(
                        f"Pinned source {name} changed at matrix_index {screen.matrix_index}."
                    )
        else:
            canonical_row = _completed_row(
                confirmation.run_dir, confirmation.run_cfg, confirmation.manifest_path
            )
            if canonical_row is not None:
                source_kind = "canonical_confirmation"
                source_cell = confirmation
                source_row = canonical_row
                expected_episodes = confirmation_n
            else:
                screen_row = _completed_row(screen.run_dir, screen.run_cfg, screen.manifest_path)
                if screen_row is None:
                    missing.append(screen.matrix_index)
                    continue
                source_kind = "screen"
                source_cell = screen
                source_row = screen_row
                expected_episodes = discovery_n

        if int(source_row.get("episodes", -1)) != expected_episodes:
            raise AdaptiveWorkflowError(
                f"{source_kind} matrix_index {screen.matrix_index} reports "
                f"episodes={source_row.get('episodes')}, expected {expected_episodes}."
            )
        traces = _episode_traces(
            source_cell.run_dir / "episode_traces.jsonl", expected=expected_episodes
        )
        observed_prefix = [_trace_pair_identity(trace) for trace in traces[:discovery_n]]
        if observed_prefix != expected_prefix:
            raise AdaptiveWorkflowError(
                f"{source_kind} matrix_index {screen.matrix_index} does not use the canonical discovery pairs."
            )
        row = dict(source_row)
        row["matrix_index"] = int(screen.matrix_index)
        row["episode_successes"] = [bool(trace["success"]) for trace in traces]
        row["discovery_source"] = {
            "kind": source_kind,
            "episodes": expected_episodes,
            "output_json": str(source_cell.run_dir / "eval.json"),
            "config_sha256": str(source_row.get("config_sha256", "")),
            "manifest_sha256": str(source_row.get("manifest_sha256", "")),
            "manifest_file_sha256": str(source_row.get("manifest_file_sha256", "")),
        }
        rows.append(row)
        source_counts[source_kind] += 1

    metadata = {
        "schema_version": WORKFLOW_SCHEMA,
        "screen_config": str(screen_cfg_path),
        "screen_config_sha256": file_sha256(screen_cfg_path),
        "screen_output_dir": str(screen_cfg.output_dir),
        "confirmation_config": str(confirmation_cfg_path),
        "confirmation_config_sha256": file_sha256(confirmation_cfg_path),
        "confirmation_output_dir": str(confirmation_cfg.output_dir),
        "matrix_cells": len(screen_cells),
        "discovery_episodes": discovery_n,
        "confirmation_episodes": confirmation_n,
        "source_counts": source_counts,
    }
    if source_ledger_path is not None:
        metadata["source_ledger"] = str(source_ledger_path)
        metadata["source_ledger_sha256"] = file_sha256(source_ledger_path)
    return rows, missing, metadata


def workflow_status(
    screen_cfg_path: str | Path,
    confirmation_cfg_path: str | Path,
    *,
    selection_path: str | Path | None = None,
    source_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    rows, missing, metadata = collect_discovery_rows(
        screen_cfg_path,
        confirmation_cfg_path,
        source_ledger_path=source_ledger_path,
    )
    result = {
        **metadata,
        "discovery_complete": len(rows),
        "discovery_missing": len(missing),
        "screen_todo_indices": missing,
        "screen_todo_slurm_array": indices_to_slurm_array(missing),
    }
    if selection_path is not None:
        selection = load_selection_manifest(selection_path)
        _, _, _, confirmation_cells = _validate_matrix_pair(screen_cfg_path, confirmation_cfg_path)
        promotion_todo = [
            index
            for index in selection["selected_matrix_indices"]
            if _completed_row(
                confirmation_cells[index].run_dir,
                confirmation_cells[index].run_cfg,
                confirmation_cells[index].manifest_path,
            )
            is None
        ]
        result.update(
            {
                "selected_cells": len(selection["selected_matrix_indices"]),
                "promotion_todo_indices": promotion_todo,
                "promotion_todo_slurm_array": indices_to_slurm_array(promotion_todo),
            }
        )
    return result


def select_candidates(
    screen_cfg_path: str | Path,
    confirmation_cfg_path: str | Path,
    output_path: str | Path,
    *,
    source_ledger_path: str | Path | None = None,
    near_frontier_margin: float = 0.10,
    cost_regions: int = 8,
    familywise_alpha: float = 0.05,
    retain_uncertainty_nondominated: bool = False,
) -> dict[str, Any]:
    rows, missing, metadata = collect_discovery_rows(
        screen_cfg_path,
        confirmation_cfg_path,
        source_ledger_path=source_ledger_path,
    )
    if missing:
        preview = ", ".join(str(value) for value in missing[:20])
        suffix = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        raise AdaptiveWorkflowError(
            f"Discovery coverage is incomplete; missing matrix indices: {preview}{suffix}."
        )
    manifest = build_selection_manifest(
        rows,
        discovery_episodes=int(metadata["discovery_episodes"]),
        near_frontier_margin=near_frontier_margin,
        cost_regions=cost_regions,
        familywise_alpha=familywise_alpha,
        retain_uncertainty_nondominated=retain_uncertainty_nondominated,
        provenance=metadata,
    )
    write_selection_manifest(output_path, manifest)
    return manifest


def run_screen_cell(
    screen_cfg_path: str | Path,
    *,
    matrix_index: int,
    num_shards: int = 3120,
) -> dict[str, Any]:
    from mwm.benchmark.matrix import main as run_matrix

    if not 0 <= int(matrix_index) < int(num_shards):
        raise AdaptiveWorkflowError("matrix_index must satisfy 0 <= index < num_shards.")
    screen_cfg = load_config(DEFAULTS, screen_cfg_path)
    screening = screen_cfg.get("screening", {})
    canonical_config = screening.get("canonical_config") if screening is not None else None
    if not canonical_config:
        raise AdaptiveWorkflowError(
            "Generated screen config is missing screening.canonical_config provenance."
        )
    canonical_cfg = load_config(DEFAULTS, str(canonical_config))
    repo_root = Path(__file__).resolve().parents[2]
    storage = screen_storage_preflight(
        str(canonical_cfg.output_dir),
        str(screen_cfg.output_dir),
        repo_root=repo_root,
    )
    if not storage["ready"]:
        raise AdaptiveWorkflowError(
            "Screen output storage preflight failed: " + "; ".join(storage["issues"])
        )
    run_matrix(
        str(screen_cfg_path),
        resume=True,
        shard_index=int(matrix_index),
        num_shards=int(num_shards),
    )
    _, cells = _matrix_cells(screen_cfg_path)
    cell = cells[int(matrix_index)]
    row = _completed_row(cell.run_dir, cell.run_cfg, cell.manifest_path)
    if row is None:
        raise AdaptiveWorkflowError(
            f"Screen matrix_index {matrix_index} did not satisfy the completion contract."
        )
    compression = compress_completed_eval(cell.run_dir)
    if compression.get("status") in {"partial", "would_compress", "would_repair"}:
        raise AdaptiveWorkflowError(
            f"Screen matrix_index {matrix_index} was not durably compressed: {compression}."
        )
    return {
        "matrix_index": int(matrix_index),
        "cell_id": cell.cell_id,
        "output_dir": str(cell.run_dir),
        "compression": compression,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the adaptive schedule-frontier workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "select"):
        child = subparsers.add_parser(command)
        child.add_argument("--screen-config", required=True)
        child.add_argument("--confirmation-config", required=True)
        child.add_argument(
            "--source-ledger",
            help="Pinned discovery-source ledger written by mwm.benchmark.screening --collect-todo.",
        )
        if command == "status":
            child.add_argument("--selection")
        else:
            child.add_argument("--output", required=True)
            child.add_argument("--near-frontier-margin", type=float, default=0.10)
            child.add_argument("--cost-regions", type=int, default=8)
            child.add_argument("--familywise-alpha", type=float, default=0.05)
            child.add_argument("--retain-uncertainty-nondominated", action="store_true")
    run = subparsers.add_parser("run-screen-cell")
    run.add_argument("--screen-config", required=True)
    run.add_argument("--matrix-index", type=int, required=True)
    run.add_argument("--num-shards", type=int, default=3120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        result = workflow_status(
            args.screen_config,
            args.confirmation_config,
            selection_path=args.selection,
            source_ledger_path=args.source_ledger,
        )
    elif args.command == "select":
        result = select_candidates(
            args.screen_config,
            args.confirmation_config,
            args.output,
            source_ledger_path=args.source_ledger,
            near_frontier_margin=args.near_frontier_margin,
            cost_regions=args.cost_regions,
            familywise_alpha=args.familywise_alpha,
            retain_uncertainty_nondominated=args.retain_uncertainty_nondominated,
        )
    else:
        result = run_screen_cell(
            args.screen_config,
            matrix_index=args.matrix_index,
            num_shards=args.num_shards,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdaptiveWorkflowError",
    "WORKFLOW_SCHEMA",
    "collect_discovery_rows",
    "indices_to_slurm_array",
    "run_screen_cell",
    "select_candidates",
    "workflow_status",
]
