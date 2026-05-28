from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from benchmark_mwm import DEFAULTS as BENCHMARK_DEFAULTS
from benchmark_mwm import _filter_resolved_by_roles, _merged_run_config, _role, _validate_gate_matrix
from mwm.benchmark.artifacts import file_sha256, load_json
from mwm.checkpoints import (
    CHECKPOINT_FORMAT,
    CONFIG_FILENAME,
    LEWM_BASE_ADAPTER_ARCH,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    validate_checkpoint_contract,
)
from mwm.data.manifest import load_manifest, manifest_file_sha256
from mwm.eval.reference import REFERENCE_ROLE, needs_reference_evaluator


REQUIRED_PLOTS = {
    "efficiency_ratios.png",
    "paired_success_delta.png",
    "schedule_usage_by_role.png",
    "success_vs_compute.png",
    "success_by_env_role.png",
    "success_vs_wall_time.png",
    "schedule_level_usage.png",
}


def _required_plots_for_gate(cfg: Any) -> set[str]:
    roles = {str(role) for role in cfg.get("gate", {}).get("roles", [])}
    required = {"success_vs_compute.png", "success_vs_wall_time.png", "success_by_env_role.png"}
    comparison_roles = roles - {"upstream_lewm_converted"}
    if "upstream_lewm_converted" in roles and comparison_roles:
        required.update({"efficiency_ratios.png", "paired_success_delta.png"})
    if "mwm_scheduled" in roles:
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


def _expected_cells(cfg: Any) -> set[tuple[str, int, str]]:
    gate = cfg.get("gate", {})
    return {
        (str(env), int(seed), str(role))
        for env in gate.get("env_ids", [])
        for seed in gate.get("seeds", [])
        for role in gate.get("roles", [])
    }


def _load_expected(cfg: Any, *, roles: Any = None) -> list[tuple[Any, Any]]:
    resolved = []
    for run in cfg.runs:
        _, run_cfg = _merged_run_config(run)
        resolved.append((run, run_cfg))
    resolved = _filter_resolved_by_roles(cfg, resolved, roles)
    _validate_gate_matrix(cfg, resolved)
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


def _validate_paper_targets(cfg: Any, rows: list[dict[str, Any]], errors: list[str]) -> None:
    targets = cfg.get("paper_targets", {})
    if not bool(targets.get("enabled", False)):
        return
    expected = targets.get("success_rate", {})
    if not isinstance(expected, dict) and not OmegaConf.is_config(expected):
        errors.append("paper_targets.success_rate must be a mapping")
        return
    expected = dict(OmegaConf.to_container(expected, resolve=True) if OmegaConf.is_config(expected) else expected)
    upstream_tol = float(targets.get("tolerance_pp", targets.get("upstream_tolerance_pp", 1.0)))
    match_tol = float(targets.get("single_level_tolerance_pp", targets.get("retrained_match_tolerance_pp", 5.0)))
    gate_roles = {str(role) for role in cfg.get("gate", {}).get("roles", [])}
    require_retrained = not gate_roles or "retrained_lewm_single" in gate_roles
    for env_id, target in sorted((str(k), float(v)) for k, v in expected.items()):
        upstream_rows = [
            row
            for row in rows
            if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == "upstream_lewm_converted"
        ]
        reference_rows = [
            row for row in rows if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == REFERENCE_ROLE
        ]
        retrained_rows = [
            row
            for row in rows
            if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == "retrained_lewm_single"
        ]
        upstream = _mean_success(upstream_rows)
        reference = _mean_success(reference_rows)
        retrained = _mean_success(retrained_rows)
        if upstream is None:
            errors.append(f"paper target check missing upstream_lewm_converted rows for {env_id}")
            continue
        if needs_reference_evaluator(upstream, target, upstream_tol):
            if reference is None:
                errors.append(
                    f"paper target check failed for {env_id}: upstream success {upstream:.2f} "
                    f"differs from paper target {target:.2f} by more than {upstream_tol:.2f} pp; "
                    f"run {REFERENCE_ROLE} fallback before accepting MWM results"
                )
            elif not needs_reference_evaluator(reference, target, upstream_tol):
                errors.append(
                    f"MWM evaluator discrepancy for {env_id}: upstream success {upstream:.2f} "
                    f"misses paper target {target:.2f} by more than {upstream_tol:.2f} pp while "
                    f"{REFERENCE_ROLE} success {reference:.2f} is within tolerance; correct evaluator/solver parameters"
                )
            else:
                errors.append(
                    f"paper target investigation required for {env_id}: MWM evaluator upstream success {upstream:.2f} "
                    f"and {REFERENCE_ROLE} success {reference:.2f} both miss paper target {target:.2f}; "
                    "check data/checkpoint/protocol mismatch"
                )
        if retrained is None:
            if not require_retrained:
                continue
            errors.append(f"paper target check missing retrained_lewm_single rows for {env_id}")
            continue
        if abs(retrained - upstream) > match_tol:
            errors.append(
                f"single-level match check failed for {env_id}: retrained success {retrained:.2f} "
                f"differs from upstream {upstream:.2f} by more than {match_tol:.2f} pp"
            )


def _single_level_env_ids(cfg: Any, targets: Any) -> list[str]:
    expected = targets.get("success_rate", {})
    if isinstance(expected, dict) or OmegaConf.is_config(expected):
        expected = dict(OmegaConf.to_container(expected, resolve=True) if OmegaConf.is_config(expected) else expected)
        if expected:
            return sorted(str(env_id) for env_id in expected)
    return sorted(str(env_id) for env_id in cfg.get("gate", {}).get("env_ids", []))


def _validate_single_level_matches(cfg: Any, rows: list[dict[str, Any]], errors: list[str]) -> None:
    targets = cfg.get("paper_targets", {})
    match_tol = float(targets.get("single_level_tolerance_pp", targets.get("retrained_match_tolerance_pp", 5.0)))
    for env_id in _single_level_env_ids(cfg, targets):
        upstream = _mean_success(
            [
                row
                for row in rows
                if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == "upstream_lewm_converted"
            ]
        )
        retrained = _mean_success(
            [
                row
                for row in rows
                if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == "retrained_lewm_single"
            ]
        )
        if upstream is None:
            errors.append(f"single-level match check missing upstream_lewm_converted rows for {env_id}")
        if retrained is None:
            errors.append(f"single-level match check missing retrained_lewm_single rows for {env_id}")
        if upstream is None or retrained is None:
            continue
        if abs(retrained - upstream) > match_tol:
            errors.append(
                f"single-level match check failed for {env_id}: retrained success {retrained:.2f} "
                f"differs from upstream {upstream:.2f} by more than {match_tol:.2f} pp"
            )


def validate_paper_targets(rows: list[dict[str, Any]], cfg: Any) -> list[str]:
    config = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    errors: list[str] = []
    _validate_paper_targets(config, rows, errors)
    return errors


def validate_single_level_matches(rows: list[dict[str, Any]], cfg: Any) -> list[str]:
    config = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    errors: list[str] = []
    _validate_single_level_matches(config, rows, errors)
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


def _to_int(value: Any, *, label: str, errors: list[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} is not an integer: {value!r}")
        return None


def _validate_checkpoint_metadata(checkpoint_dir: Path, errors: list[str]) -> dict[str, Any]:
    files = (CONFIG_FILENAME, WEIGHTS_FILENAME, METADATA_FILENAME)
    file_status = [_require_file(checkpoint_dir / name, errors) for name in files]
    if not all(file_status):
        return {}

    config = load_json(checkpoint_dir / CONFIG_FILENAME)
    metadata = load_json(checkpoint_dir / METADATA_FILENAME)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        errors.append(f"MWM checkpoint has format={metadata.get('format')!r}, expected {CHECKPOINT_FORMAT!r}: {checkpoint_dir}")

    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        errors.append(f"MWM checkpoint artifacts are not a mapping: {checkpoint_dir}")
        artifacts = {}
    for label, filename in (("config", CONFIG_FILENAME), ("weights", WEIGHTS_FILENAME)):
        artifact = artifacts.get(label, {})
        if not isinstance(artifact, dict):
            errors.append(f"MWM checkpoint artifact {label!r} is not a mapping: {checkpoint_dir}")
            continue
        if artifact.get("path") != filename:
            errors.append(f"MWM checkpoint artifact {label!r} path mismatch for {checkpoint_dir}")
        sha = artifact.get("sha256")
        if not sha:
            errors.append(f"MWM checkpoint missing {label} sha256: {checkpoint_dir}")
        elif file_sha256(checkpoint_dir / filename) != sha:
            errors.append(f"MWM checkpoint {label} sha256 mismatch: {checkpoint_dir}")

    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, dict):
        errors.append(f"MWM checkpoint config kwargs are not a mapping: {checkpoint_dir}")
        kwargs = {}
    if not config.get("target"):
        errors.append(f"MWM checkpoint config missing import target: {checkpoint_dir}")
    else:
        try:
            validate_checkpoint_contract(config, metadata)
        except Exception as exc:  # noqa: BLE001 - verifier should aggregate failures
            errors.append(f"MWM checkpoint contract failed: {checkpoint_dir}: {exc}")

    action_spec = metadata.get("action_spec")
    if not isinstance(action_spec, dict):
        errors.append(f"MWM checkpoint missing action_spec mapping: {checkpoint_dir}")
    else:
        dim = _to_int(action_spec.get("dim"), label=f"{checkpoint_dir} action_spec.dim", errors=errors)
        base_dim = _to_int(action_spec.get("base_dim"), label=f"{checkpoint_dir} action_spec.base_dim", errors=errors)
        block = _to_int(action_spec.get("block"), label=f"{checkpoint_dir} action_spec.block", errors=errors)
        cfg_dim = _to_int(kwargs.get("action_dim"), label=f"{checkpoint_dir} config action_dim", errors=errors)
        cfg_block = _to_int(kwargs.get("action_block", metadata.get("action_block", 1)), label=f"{checkpoint_dir} config action_block", errors=errors)
        meta_base_dim = _to_int(metadata.get("action_dim"), label=f"{checkpoint_dir} metadata action_dim", errors=errors)
        meta_block = _to_int(metadata.get("action_block", 1), label=f"{checkpoint_dir} metadata action_block", errors=errors)
        if dim is not None and cfg_dim is not None and dim != cfg_dim:
            errors.append(f"MWM checkpoint action_spec.dim does not match config action_dim: {checkpoint_dir}")
        if block is not None and cfg_block is not None and block != cfg_block:
            errors.append(f"MWM checkpoint action_spec.block does not match config action_block: {checkpoint_dir}")
        if block is not None and meta_block is not None and block != meta_block:
            errors.append(f"MWM checkpoint action_spec.block does not match metadata action_block: {checkpoint_dir}")
        if base_dim is not None and meta_base_dim is not None and base_dim != meta_base_dim:
            errors.append(f"MWM checkpoint action_spec.base_dim does not match metadata action_dim: {checkpoint_dir}")
        if dim is not None and base_dim is not None and block is not None and base_dim * block != dim:
            errors.append(f"MWM checkpoint action spec is internally inconsistent: {checkpoint_dir}")

    cfg_levels = [int(k) for k in kwargs.get("K", [])] if "K" in kwargs else []
    meta_levels = [int(k) for k in metadata.get("levels", [])] if "levels" in metadata else []
    if cfg_levels and meta_levels and cfg_levels != meta_levels:
        errors.append(f"MWM checkpoint levels do not match config K: {checkpoint_dir}")
    return metadata


def _validate_role_checkpoint_contract(row: dict[str, Any], metadata: dict[str, Any], errors: list[str]) -> None:
    role = str(row.get("role", ""))
    checkpoint_dir = Path(str(row.get("checkpoint_run_dir", "")))
    levels = [int(k) for k in metadata.get("levels", [])] if isinstance(metadata.get("levels", []), list) else []
    model_meta = metadata.get("model", {})
    target = str(model_meta.get("target", "")) if isinstance(model_meta, dict) else ""
    backend = str(metadata.get("training_backend", ""))
    trainable_lewm_targets = ("build_mwm_lewm_from_stable_config",)
    if role == "upstream_lewm_converted":
        if metadata.get("role") != "upstream_lewm_converted":
            errors.append(f"upstream role checkpoint missing upstream_lewm_converted metadata role: {checkpoint_dir}")
        if levels != [192]:
            errors.append(f"upstream role checkpoint must be single-fidelity K=[192], got {levels}: {checkpoint_dir}")
        if not target.endswith("build_mwm_lewm_from_upstream_object"):
            errors.append(f"upstream role checkpoint must load through the normal converted Le-WM MWM target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"upstream role checkpoint missing corrected architecture version: {checkpoint_dir}")
    elif role == "retrained_lewm_single":
        if levels != [192]:
            errors.append(f"retrained single checkpoint must be K=[192], got {levels}: {checkpoint_dir}")
        if backend != "stable_worldmodel_lewm":
            errors.append(
                f"retrained single checkpoint must use the Le-WM base-adapter backend, got {backend!r}: {checkpoint_dir}"
            )
        if not target.endswith(trainable_lewm_targets):
            errors.append(f"retrained single checkpoint must export the Le-WM base-adapter target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"retrained single checkpoint missing corrected architecture version: {checkpoint_dir}")
    elif role == "mwm_scheduled":
        if levels != [48, 96, 144]:
            errors.append(f"scheduled MWM checkpoint must be K=[48,96,144], got {levels}: {checkpoint_dir}")
        if not target.endswith(trainable_lewm_targets):
            errors.append(f"scheduled MWM checkpoint must export the Le-WM base-adapter target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"scheduled MWM checkpoint missing corrected architecture version: {checkpoint_dir}")


def verify_benchmark_output(
    cfg_path: str | Path = "configs/benchmark_mwm.yaml",
    *,
    roles: Any = None,
    single_level_only: bool = False,
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

    expected = _expected_cells(cfg)
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
    if single_level_only:
        _validate_single_level_matches(cfg, rows, errors)
    else:
        _validate_paper_targets(cfg, rows, errors)

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
        checkpoint_metadata = _validate_checkpoint_metadata(checkpoint_dir, errors)
        if checkpoint_metadata:
            _validate_role_checkpoint_contract(row, checkpoint_metadata, errors)

    for key, hashes in sorted(manifest_by_env_seed.items()):
        if len(hashes) != 1:
            errors.append(f"benchmark roles do not share one manifest for {key}: {sorted(hashes)}")

    required_plots = _required_plots_for_gate(cfg)
    plot_dir = output_dir / "plots"
    for name in required_plots:
        _require_file(plot_dir / name, errors)
    summary_plot_names = {Path(str(plot)).name for plot in summary.get("plots", [])}
    missing_summary_plots = sorted(required_plots - summary_plot_names)
    if missing_summary_plots:
        errors.append(f"summary.json missing required plot refs: {missing_summary_plots}")

    review_text = review_path.read_text(encoding="utf-8") if review_path.is_file() else ""
    for token in ("Gate Status", "Outcome Summary", "Plots", "Paired Seed Comparison", "Run Drilldown", "Review Notes"):
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
    for env in cfg.gate.env_ids:
        for role in cfg.gate.roles:
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
        "single_level_only": bool(single_level_only),
    }


def verify_benchmark_static(cfg_path: str | Path = "configs/benchmark_mwm.yaml", *, roles: Any = None) -> dict[str, Any]:
    cfg = OmegaConf.merge(BENCHMARK_DEFAULTS, OmegaConf.load(str(cfg_path)))
    resolved = _load_expected(cfg, roles=roles)
    targets = cfg.get("paper_targets", {})
    paper_targets: dict[str, Any] = {}
    if bool(targets.get("enabled", False)):
        success_rate = targets.get("success_rate", {})
        if not isinstance(success_rate, dict) and not OmegaConf.is_config(success_rate):
            raise ValueError("paper_targets.success_rate must be a mapping")
        success_rate = dict(
            OmegaConf.to_container(success_rate, resolve=True) if OmegaConf.is_config(success_rate) else success_rate
        )
        gate_envs = [str(env) for env in cfg.get("gate", {}).get("env_ids", [])]
        missing_targets = sorted(env for env in gate_envs if env not in {str(key) for key in success_rate})
        if missing_targets:
            raise ValueError(f"paper_targets.success_rate missing gate envs: {missing_targets}")
        paper_targets = {
            "tolerance_pp": float(targets.get("tolerance_pp", targets.get("upstream_tolerance_pp", 1.0))),
            "single_level_tolerance_pp": float(
                targets.get("single_level_tolerance_pp", targets.get("retrained_match_tolerance_pp", 5.0))
            ),
            "success_rate": {str(key): float(value) for key, value in success_rate.items()},
        }
    return {
        "config": str(cfg_path),
        "output_dir": str(cfg.output_dir),
        "runs": len(resolved),
        "expected_cells": sorted(_expected_cells(cfg)),
        "paper_targets": paper_targets,
        "static_only": True,
    }


def main(cfg_path: str, *, static_only: bool = False, roles: Any = None, single_level_only: bool = False) -> None:
    report = (
        verify_benchmark_static(cfg_path, roles=roles)
        if static_only
        else verify_benchmark_output(cfg_path, roles=roles, single_level_only=single_level_only)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify MWM benchmark artifacts.")
    parser.add_argument("config", nargs="?", default="configs/benchmark_mwm.yaml", help="Benchmark YAML config")
    parser.add_argument("--static-only", action="store_true", help="Only validate the config matrix")
    parser.add_argument("--roles", nargs="+", help="Optional role filter, e.g. upstream_lewm_converted")
    parser.add_argument(
        "--single-level-only",
        action="store_true",
        help="Validate retrained K=[D] MWM against upstream without applying paper-target thresholds",
    )
    args = parser.parse_args()
    main(args.config, static_only=args.static_only, roles=args.roles, single_level_only=args.single_level_only)
