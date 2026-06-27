from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.checkpoint_verify import (
    load_checkpoint_metadata_for_benchmark,
    validate_benchmark_role_checkpoint_contract,
)
from mwm.benchmark.config import DEFAULTS as BENCHMARK_DEFAULTS
from mwm.benchmark.matrix_identity import expected_cells_from_resolved, load_expected_resolved, metric_identity
from mwm.benchmark.paper_targets import append_paper_target_errors
from mwm.benchmark.plot_contract import required_plots_for_benchmark
from mwm.data.manifest import load_manifest, manifest_file_sha256
from mwm.io import file_sha256, load_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _require_file(path: Path, errors: list[str]) -> bool:
    if not path.is_file():
        errors.append(f"missing file: {path}")
        return False
    if path.stat().st_size <= 0:
        errors.append(f"empty file: {path}")
        return False
    return True


def _review_href(path_text: Any, output_dir: Path) -> str:
    path = Path(str(path_text or ""))
    if not str(path):
        return ""
    try:
        if path.is_absolute():
            return path.relative_to(output_dir.resolve()).as_posix()
        return (Path.cwd() / path).resolve().relative_to(output_dir.resolve()).as_posix()
    except (OSError, ValueError):
        parts = path.parts
        if len(parts) >= 2 and parts[0] == "rollouts" and parts[1] == output_dir.name:
            return Path(*parts[2:]).as_posix()
        return path.as_posix()


def _has_resolved_ref(ref: Any) -> bool:
    return isinstance(ref, dict) and bool(ref.get("commit_id") or ref.get("sha256"))


def _validate_dependency_refs(dependencies: Any, *, label: str, errors: list[str]) -> None:
    if not isinstance(dependencies, dict):
        errors.append(f"{label} dependency refs are not a mapping")
        return
    for dep in ("stable-worldmodel", "stable-pretraining", "torch", "local_repo"):
        if dep not in dependencies:
            errors.append(f"{label} missing dependency ref {dep!r}")
        elif not _has_resolved_ref(dependencies[dep]):
            errors.append(f"{label} dependency ref {dep!r} lacks commit_id/sha256")
    local = dependencies.get("local_repo", {})
    if isinstance(local, dict) and local.get("dirty") and not local.get("diff_sha256"):
        errors.append(f"{label} local_repo is dirty but missing diff_sha256")


def verify_benchmark_output(
    cfg_path: str | Path = "configs/benchmark/scheduled_pusht.yaml",
    *,
    roles: Any = None,
) -> dict[str, Any]:
    cfg = OmegaConf.merge(BENCHMARK_DEFAULTS, OmegaConf.load(str(cfg_path)))
    resolved = load_expected_resolved(cfg, roles=roles)
    output_dir = Path(str(cfg.output_dir))
    errors: list[str] = []

    summary_path = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    metrics_path = output_dir / "metrics.jsonl"
    per_env_path = output_dir / "per_env_summary.csv"
    review_path = output_dir / "review.html"
    for path in (summary_path, summary_csv, metrics_path, per_env_path, review_path):
        _require_file(path, errors)

    if errors:
        raise ValueError("Benchmark output is incomplete:\n- " + "\n- ".join(errors))

    summary = load_json(summary_path)
    rows = list(summary.get("runs", []))
    csv_rows = _read_csv(summary_csv)
    metrics_rows = _read_jsonl(metrics_path)
    if len(rows) != len(resolved):
        errors.append(f"summary has {len(rows)} runs, expected {len(resolved)}")
    if len(csv_rows) != len(rows):
        errors.append(f"summary.csv has {len(csv_rows)} rows, expected {len(rows)}")
    if len(metrics_rows) != len(rows):
        errors.append(f"metrics.jsonl has {len(metrics_rows)} rows, expected {len(rows)}")
    elif metrics_rows != rows:
        errors.append("aggregate metrics.jsonl does not exactly regenerate summary rows")

    expected = expected_cells_from_resolved(resolved)
    identities = [metric_identity(row) for row in rows]
    present = set(identities)
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        errors.append(f"duplicate benchmark cells: {duplicates}")
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        errors.append(f"missing benchmark cells: {missing}")
    if extra:
        errors.append(f"unexpected benchmark cells: {extra}")
    append_paper_target_errors(cfg, rows, errors, roles={cell[2] for cell in expected})

    manifest_by_cell: dict[tuple[str, int, str], str] = {}
    manifest_by_env_seed: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        cell = metric_identity(row)
        run_output = Path(str(row.get("output_json", "")))
        if not _require_file(run_output, errors):
            continue
        payload = load_json(run_output)
        run_dir = run_output.parent
        sidecar_paths = (
            run_dir / "resolved_config.yaml",
            run_dir / "metrics.jsonl",
            run_dir / "episode_traces.jsonl",
            run_dir / "summary.json",
            run_dir / "dependencies.json",
            run_dir / "planning_diagnostics.json",
        )
        for path in sidecar_paths:
            _require_file(path, errors)

        if row.get("config_sha256") and file_sha256(run_dir / "resolved_config.yaml") != row["config_sha256"]:
            errors.append(f"config hash mismatch for {run_dir}")

        run_metrics = _read_jsonl(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").is_file() else []
        if len(run_metrics) != 1 or run_metrics[0] != row:
            errors.append(f"run metrics do not exactly regenerate summary row for {run_dir}")

        run_summary = load_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else {}
        if run_summary.get("run") != row:
            errors.append(f"run summary sidecar does not exactly regenerate summary row for {run_dir}")
        dependencies_sidecar = load_json(run_dir / "dependencies.json") if (run_dir / "dependencies.json").is_file() else {}
        if dependencies_sidecar != payload.get("dependencies", {}):
            errors.append(f"dependency sidecar does not match eval payload for {run_dir}")
        diagnostics_sidecar = load_json(run_dir / "planning_diagnostics.json") if (run_dir / "planning_diagnostics.json").is_file() else {}
        if diagnostics_sidecar != payload.get("planning_diagnostics", {}):
            errors.append(f"planning diagnostics sidecar does not match eval payload for {run_dir}")

        traces = _read_jsonl(run_dir / "episode_traces.jsonl") if (run_dir / "episode_traces.jsonl").is_file() else []
        if len(traces) != int(payload.get("episodes", 0)):
            errors.append(f"episode trace count mismatch for {run_dir}: {len(traces)} vs {payload.get('episodes')}")

        _validate_dependency_refs(payload.get("dependencies", {}), label=str(run_dir), errors=errors)

        manifest_info = payload.get("manifest") or {}
        manifest_path = Path(str(manifest_info.get("path", "")))
        if not _require_file(manifest_path, errors):
            continue
        if manifest_info.get("sha256") and manifest_file_sha256(manifest_path) != manifest_info["sha256"]:
            errors.append(f"manifest file hash mismatch for {manifest_path}")
        manifest = load_manifest(manifest_path)
        if str(manifest.get("env_id")) != cell[0]:
            errors.append(f"manifest env mismatch for {run_dir}")
        if int(manifest.get("seed", -1)) != cell[1]:
            errors.append(f"manifest seed mismatch for {run_dir}")
        if len(manifest.get("pairs", [])) != int(payload.get("episodes", 0)):
            errors.append(f"manifest pair count mismatch for {run_dir}")
        _validate_dependency_refs(manifest.get("dependency_shas", {}), label=f"manifest {manifest_path}", errors=errors)

        manifest_hash = str(manifest_info.get("manifest_sha256", manifest.get("manifest_sha256", "")))
        manifest_by_cell[cell] = manifest_hash
        manifest_by_env_seed.setdefault((cell[0], cell[1]), set()).add(manifest_hash)

        diagnostics = payload.get("planning_diagnostics", {})
        summary_diag = diagnostics.get("summary", {}) if isinstance(diagnostics, dict) else {}
        if int(summary_diag.get("cem_cost_calls", 0)) <= 0 or int(summary_diag.get("candidate_action_values", 0)) <= 0:
            errors.append(f"MWM run missing nonzero CEM work diagnostics: {run_dir}")
        checkpoint_dir = Path(str(payload.get("checkpoint_run_dir", "")))
        checkpoint_metadata = load_checkpoint_metadata_for_benchmark(checkpoint_dir, errors)
        if checkpoint_metadata:
            validate_benchmark_role_checkpoint_contract(row, checkpoint_metadata, errors)

    for key, hashes in sorted(manifest_by_env_seed.items()):
        if len(hashes) != 1:
            errors.append(f"benchmark roles do not share one manifest for {key}: {sorted(hashes)}")

    required_plots = required_plots_for_benchmark(cfg, roles={cell[2] for cell in expected})
    plot_dir = output_dir / "plots"
    for name in required_plots:
        _require_file(plot_dir / name, errors)
    summary_plot_names = {Path(str(plot)).name for plot in summary.get("plots", [])}
    missing_summary_plots = sorted(required_plots - summary_plot_names)
    if missing_summary_plots:
        errors.append(f"summary.json missing required plot refs: {missing_summary_plots}")

    review_text = review_path.read_text(encoding="utf-8") if review_path.is_file() else ""
    for token in ("Benchmark Status", "Outcome Summary", "Plots", "Paired Seed Comparison", "Run Drilldown", "Review Notes"):
        if token not in review_text:
            errors.append(f"review.html missing section {token!r}")
    for name in required_plots:
        if f"plots/{name}" not in review_text:
            errors.append(f"review.html does not embed/link plots/{name}")
    for row in rows:
        run_output = Path(str(row.get("output_json", "")))
        eval_href = _review_href(run_output, output_dir)
        diagnostics_href = _review_href(run_output.parent / "planning_diagnostics.json", output_dir)
        if eval_href and eval_href not in review_text:
            errors.append(f"review.html missing eval drilldown link {eval_href}")
        if diagnostics_href and diagnostics_href not in review_text:
            errors.append(f"review.html missing diagnostics drilldown link {diagnostics_href}")

    per_env_rows = _read_csv(per_env_path)
    per_env_cells = {(row.get("env_id", ""), row.get("role", "")) for row in per_env_rows}
    for env, _, role in sorted(expected):
        if (str(env), str(role)) not in per_env_cells:
            errors.append(f"per-env table missing {(str(env), str(role))}")

    errors = list(dict.fromkeys(errors))
    if errors:
        raise ValueError("Benchmark output failed verification:\n- " + "\n- ".join(errors))
    return {
        "output_dir": str(output_dir),
        "runs": len(rows),
        "cells": sorted(manifest_by_cell),
        "plots": sorted(required_plots),
    }


__all__ = ["verify_benchmark_output"]
