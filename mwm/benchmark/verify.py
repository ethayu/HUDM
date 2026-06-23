from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.config import (
    DEFAULTS as BENCHMARK_DEFAULTS,
    filter_resolved_by_roles,
    load_manifest_config,
    merged_run_config,
    role,
    validate_benchmark_matrix,
)
from mwm.adapters.constants import LEWM_BASE_ADAPTER_ARCH
from mwm.adapters.builder import STABLE_CONFIG_TARGET
from mwm.checkpoint_contract import checkpoint_full_latent_dim
from mwm.checkpoint_io import METADATA_FILENAME, validate_checkpoint_directory
from mwm.data.manifest import load_manifest, manifest_file_sha256
from mwm.io import file_sha256, load_json


REQUIRED_PLOTS = {
    "efficiency_ratios.png",
    "paired_success_delta.png",
    "schedule_usage_by_role.png",
    "success_vs_compute.png",
    "success_by_env_role.png",
    "success_vs_wall_time.png",
    "schedule_level_usage.png",
}


def required_plots_for_benchmark(cfg: Any, roles: set[str] | None = None) -> set[str]:
    if roles is None:
        roles = {str(run.get("role", run.get("name", ""))) for run in cfg.get("runs", [])}
    required = {"success_vs_compute.png", "success_vs_wall_time.png", "success_by_env_role.png"}
    comparison_roles = roles - {"upstream_lewm_converted"}
    if "upstream_lewm_converted" in roles and comparison_roles:
        required.update({"efficiency_ratios.png", "paired_success_delta.png"})
    if roles & {"mwm_scheduled", "mwm_dense"}:
        required.update({"schedule_level_usage.png", "schedule_usage_by_role.png"})
    if not roles:
        return set(REQUIRED_PLOTS)
    return required


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


def _expected_cells_from_resolved(resolved: list[tuple[Any, Any]]) -> set[tuple[str, int, str]]:
    return {
        (str(run_cfg.get("env_id", "")), int(run_cfg.eval.seed), role(run, run_cfg))
        for run, run_cfg in resolved
    }


def _load_expected(cfg: Any, *, roles: Any = None) -> list[tuple[Any, Any]]:
    resolved = []
    for run in cfg.runs:
        _, run_cfg = merged_run_config(cfg, run)
        resolved.append((run, run_cfg))
    resolved = filter_resolved_by_roles(cfg, resolved, roles)
    validate_benchmark_matrix(cfg, resolved)
    return resolved


def _metric_identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row.get("env_id", "")), int(row.get("seed", -1)), str(row.get("role", ""))


def _row_success(row: dict[str, Any]) -> float | None:
    try:
        value = float(row.get("success_rate"))
    except (TypeError, ValueError):
        return None
    return value


def _mean_success(rows: list[dict[str, Any]]) -> float | None:
    vals = [v for v in (_row_success(row) for row in rows) if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _episode_count(rows: list[dict[str, Any]]) -> int | None:
    total = 0
    for row in rows:
        try:
            episodes = int(row.get("episodes", 0))
        except (TypeError, ValueError):
            continue
        if episodes > 0:
            total += episodes
    return total or None


def _effective_tolerance_pp(configured_pp: float, *row_groups: list[dict[str, Any]]) -> float:
    if configured_pp <= 0.0:
        return configured_pp
    episode_counts = [_episode_count(list(group)) for group in row_groups]
    usable_counts = [count for count in episode_counts if count]
    if len(usable_counts) != len(row_groups):
        return configured_pp
    episodes = min(usable_counts)
    count_allowance = max(1, math.ceil((configured_pp / 100.0) * episodes))
    return max(configured_pp, count_allowance * 100.0 / episodes)


def append_paper_target_errors(
    cfg: Any,
    rows: list[dict[str, Any]],
    errors: list[str],
    *,
    roles: set[str] | None = None,
) -> None:
    targets = cfg.get("paper_targets", {})
    if not bool(targets.get("enabled", False)):
        return
    expected = targets.get("success_rate", {})
    if not isinstance(expected, dict) and not OmegaConf.is_config(expected):
        errors.append("paper_targets.success_rate must be a mapping")
        return
    expected = dict(OmegaConf.to_container(expected, resolve=True) if OmegaConf.is_config(expected) else expected)
    upstream_tol = float(targets.get("tolerance_pp", targets.get("upstream_tolerance_pp", 1.0)))
    match_tol = float(targets.get("retrained_match_tolerance_pp", 5.0))
    benchmark_roles = roles if roles is not None else {str(run.get("role", run.get("name", ""))) for run in cfg.get("runs", [])}
    require_retrained = not benchmark_roles or "retrained_lewm_identity" in benchmark_roles
    for env_id, target in sorted((str(k), float(v)) for k, v in expected.items()):
        upstream_rows = [
            row
            for row in rows
            if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == "upstream_lewm_converted"
        ]
        retrained_rows = [
            row
            for row in rows
            if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == "retrained_lewm_identity"
        ]
        upstream = _mean_success(upstream_rows)
        retrained = _mean_success(retrained_rows)
        if upstream is None:
            errors.append(f"paper target check missing upstream_lewm_converted rows for {env_id}")
            continue
        effective_upstream_tol = _effective_tolerance_pp(upstream_tol, upstream_rows)
        if abs(upstream - target) > effective_upstream_tol:
            errors.append(
                f"paper target check failed for {env_id}: upstream success {upstream:.2f} "
                f"differs from paper target {target:.2f} by more than {effective_upstream_tol:.2f} pp"
            )
        if retrained is None:
            if not require_retrained:
                continue
            errors.append(f"paper target check missing retrained_lewm_identity rows for {env_id}")
            continue
        effective_match_tol = _effective_tolerance_pp(match_tol, upstream_rows, retrained_rows)
        if abs(retrained - upstream) > effective_match_tol:
            errors.append(
                f"retrained match check failed for {env_id}: retrained success {retrained:.2f} "
                f"differs from upstream {upstream:.2f} by more than {effective_match_tol:.2f} pp"
            )


def validate_paper_targets(rows: list[dict[str, Any]], cfg: Any) -> list[str]:
    config = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    errors: list[str] = []
    append_paper_target_errors(config, rows, errors)
    return errors


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


def load_checkpoint_metadata_for_benchmark(checkpoint_dir: Path, errors: list[str]) -> dict[str, Any]:
    try:
        _, metadata = validate_checkpoint_directory(checkpoint_dir, strict_artifacts=True, strict_metadata=True)
    except Exception as exc:  # noqa: BLE001 - verifier should aggregate failures
        errors.append(f"MWM checkpoint contract failed: {checkpoint_dir}: {exc}")
        try:
            return load_json(checkpoint_dir / METADATA_FILENAME)
        except Exception:
            return {}
    return metadata


def validate_benchmark_role_checkpoint_contract(row: dict[str, Any], metadata: dict[str, Any], errors: list[str]) -> None:
    role = str(row.get("role", ""))
    checkpoint_dir = Path(str(row.get("checkpoint_run_dir", "")))
    levels = [int(k) for k in metadata.get("levels", [])] if isinstance(metadata.get("levels", []), list) else []
    model_meta = metadata.get("model", {})
    target = str(model_meta.get("target", "")) if isinstance(model_meta, dict) else ""
    backend = str(metadata.get("training_backend", ""))
    try:
        d: int | None = checkpoint_full_latent_dim(metadata)
    except ValueError as exc:
        errors.append(f"MWM checkpoint contract failed: {checkpoint_dir}: {exc}")
        d = None
    if role == "upstream_lewm_converted":
        if metadata.get("role") != "upstream_lewm_converted":
            errors.append(f"upstream role checkpoint missing upstream_lewm_converted metadata role: {checkpoint_dir}")
        if d is not None and levels != [d]:
            errors.append(f"upstream role checkpoint must be identity-parity K=[D={d}], got {levels}: {checkpoint_dir}")
        if target != STABLE_CONFIG_TARGET:
            errors.append(f"upstream role checkpoint must export the generic base-adaptive target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"upstream role checkpoint missing corrected architecture version: {checkpoint_dir}")
    elif role == "retrained_lewm_identity":
        if d is not None and levels != [d]:
            errors.append(f"retrained identity checkpoint must be K=[D={d}], got {levels}: {checkpoint_dir}")
        if backend != "stable_worldmodel_lewm":
            errors.append(
                f"retrained identity checkpoint must use the Le-WM base-adapter backend, got {backend!r}: {checkpoint_dir}"
            )
        if target != STABLE_CONFIG_TARGET:
            errors.append(f"retrained identity checkpoint must export the generic base-adaptive target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"retrained identity checkpoint missing corrected architecture version: {checkpoint_dir}")
    elif role == "mwm_scheduled":
        if levels != [48, 96, 144]:
            errors.append(f"scheduled MWM checkpoint must be K=[48,96,144], got {levels}: {checkpoint_dir}")
        if target != STABLE_CONFIG_TARGET:
            errors.append(f"scheduled MWM checkpoint must export the generic base-adaptive target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"scheduled MWM checkpoint missing corrected architecture version: {checkpoint_dir}")
    elif role == "mwm_dense":
        if levels != [6, 12, 48, 96, 144, 192]:
            errors.append(f"dense MWM checkpoint must be K=[6,12,48,96,144,192], got {levels}: {checkpoint_dir}")
        if target != STABLE_CONFIG_TARGET:
            errors.append(f"dense MWM checkpoint must export the generic base-adaptive target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"dense MWM checkpoint missing corrected architecture version: {checkpoint_dir}")


def verify_benchmark_output(
    cfg_path: str | Path = "configs/benchmark/scheduled_pusht.yaml",
    *,
    roles: Any = None,
) -> dict[str, Any]:
    cfg = OmegaConf.merge(BENCHMARK_DEFAULTS, OmegaConf.load(str(cfg_path)))
    resolved = _load_expected(cfg, roles=roles)
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

    expected = _expected_cells_from_resolved(resolved)
    identities = [_metric_identity(row) for row in rows]
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
        cell = _metric_identity(row)
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


def verify_benchmark_static(
    cfg_path: str | Path = "configs/benchmark/scheduled_pusht.yaml",
    *,
    roles: Any = None,
    check_checkpoints: bool = True,
) -> dict[str, Any]:
    cfg = OmegaConf.merge(BENCHMARK_DEFAULTS, OmegaConf.load(str(cfg_path)))
    resolved = _load_expected(cfg, roles=roles)
    errors: list[str] = []
    targets = cfg.get("paper_targets", {})
    paper_targets: dict[str, Any] = {}
    if bool(targets.get("enabled", False)):
        success_rate = targets.get("success_rate", {})
        if not isinstance(success_rate, dict) and not OmegaConf.is_config(success_rate):
            raise ValueError("paper_targets.success_rate must be a mapping")
        success_rate = dict(
            OmegaConf.to_container(success_rate, resolve=True) if OmegaConf.is_config(success_rate) else success_rate
        )
        missing_targets = [str(cfg.env_id)] if str(cfg.env_id) not in {str(key) for key in success_rate} else []
        if missing_targets:
            raise ValueError(f"paper_targets.success_rate missing benchmark env: {missing_targets}")
        paper_targets = {
            "tolerance_pp": float(targets.get("tolerance_pp", targets.get("upstream_tolerance_pp", 1.0))),
            "retrained_match_tolerance_pp": float(
                targets.get("retrained_match_tolerance_pp", targets.get("retrained_match_tolerance_pp", 5.0))
            ),
            "success_rate": {str(key): float(value) for key, value in success_rate.items()},
        }

    checked_checkpoints: set[str] = set()
    if check_checkpoints:
        for run, run_cfg in resolved:
            checkpoint_ref = OmegaConf.select(run_cfg, "checkpoint.run_dir")
            run_role = role(run, run_cfg)
            if checkpoint_ref is None:
                errors.append(f"static benchmark run {run.get('name', '<unnamed>')} missing checkpoint.run_dir")
                continue
            checkpoint_dir = Path(str(checkpoint_ref))
            checkpoint_metadata = load_checkpoint_metadata_for_benchmark(checkpoint_dir, errors)
            if checkpoint_metadata:
                validate_benchmark_role_checkpoint_contract(
                    {"role": run_role, "checkpoint_run_dir": str(checkpoint_dir)},
                    checkpoint_metadata,
                    errors,
                )
                checked_checkpoints.add(str(checkpoint_dir))
    errors = list(dict.fromkeys(errors))
    if errors:
        raise ValueError("Benchmark static verification failed:\n- " + "\n- ".join(errors))

    return {
        "config": str(cfg_path),
        "output_dir": str(cfg.output_dir),
        "runs": len(resolved),
        "env_id": str(cfg.env_id),
        "seed": int(cfg.seed),
        "manifest": load_manifest_config(cfg),
        "expected_cells": sorted(_expected_cells_from_resolved(resolved)),
        "paper_targets": paper_targets,
        "checkpoint_contracts": sorted(checked_checkpoints),
        "check_checkpoints": bool(check_checkpoints),
        "static_only": True,
    }


def main(
    cfg_path: str,
    *,
    static_only: bool = False,
    roles: Any = None,
    check_checkpoints: bool = True,
) -> None:
    report = (
        verify_benchmark_static(cfg_path, roles=roles, check_checkpoints=check_checkpoints)
        if static_only
        else verify_benchmark_output(cfg_path, roles=roles)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify MWM benchmark artifacts.")
    parser.add_argument("config", nargs="?", default="configs/benchmark/scheduled_pusht.yaml", help="Benchmark YAML config")
    parser.add_argument("--static-only", action="store_true", help="Validate the config matrix and input checkpoint contracts")
    parser.add_argument("--no-checkpoints", action="store_true", help="Skip checkpoint contract checks in --static-only mode")
    parser.add_argument("--roles", nargs="+", help="Optional role filter, e.g. upstream_lewm_converted")
    args = parser.parse_args()
    main(args.config, static_only=args.static_only, roles=args.roles, check_checkpoints=not args.no_checkpoints)


__all__ = [
    "append_paper_target_errors",
    "load_checkpoint_metadata_for_benchmark",
    "main",
    "required_plots_for_benchmark",
    "validate_benchmark_role_checkpoint_contract",
    "validate_paper_targets",
    "verify_benchmark_output",
    "verify_benchmark_static",
]
