from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from benchmark_mwm import DEFAULTS as BENCHMARK_DEFAULTS
from benchmark_mwm import _filter_resolved_by_roles, _load_manifest_config, _merged_run_config, _role, _validate_benchmark_matrix
from mwm.benchmark.artifacts import file_sha256, load_json
from mwm.adapters.builder import STABLE_CONFIG_TARGET
from mwm.checkpoints import (
    CHECKPOINT_FORMAT,
    CONFIG_FILENAME,
    LEWM_BASE_ADAPTER_ARCH,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    validate_checkpoint_contract,
)
from mwm.data.manifest import load_manifest, manifest_file_sha256


REQUIRED_PLOTS = {
    "efficiency_ratios.png",
    "paired_success_delta.png",
    "schedule_usage_by_role.png",
    "success_vs_compute.png",
    "success_by_env_role.png",
    "success_vs_wall_time.png",
    "schedule_level_usage.png",
}


def _required_plots_for_benchmark(cfg: Any, roles: set[str] | None = None) -> set[str]:
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
        (str(run_cfg.get("env_id", "")), int(run_cfg.eval.seed), _role(run, run_cfg))
        for run, run_cfg in resolved
    }


def _load_expected(cfg: Any, *, roles: Any = None) -> list[tuple[Any, Any]]:
    resolved = []
    for run in cfg.runs:
        _, run_cfg = _merged_run_config(cfg, run)
        resolved.append((run, run_cfg))
    resolved = _filter_resolved_by_roles(cfg, resolved, roles)
    _validate_benchmark_matrix(cfg, resolved)
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


def _validate_paper_targets(
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
        if abs(upstream - target) > upstream_tol:
            errors.append(
                f"paper target check failed for {env_id}: upstream success {upstream:.2f} "
                f"differs from paper target {target:.2f} by more than {upstream_tol:.2f} pp"
            )
        if retrained is None:
            if not require_retrained:
                continue
            errors.append(f"paper target check missing retrained_lewm_identity rows for {env_id}")
            continue
        if abs(retrained - upstream) > match_tol:
            errors.append(
                f"retrained match check failed for {env_id}: retrained success {retrained:.2f} "
                f"differs from upstream {upstream:.2f} by more than {match_tol:.2f} pp"
            )


def validate_paper_targets(rows: list[dict[str, Any]], cfg: Any) -> list[str]:
    config = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    errors: list[str] = []
    _validate_paper_targets(config, rows, errors)
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


def _checkpoint_full_latent_dim(metadata: dict[str, Any], checkpoint_dir: Path, errors: list[str]) -> int | None:
    model_meta = metadata.get("model", {})
    model_kwargs = model_meta.get("kwargs", {}) if isinstance(model_meta, dict) else {}
    candidates = [
        metadata.get("D"),
        model_meta.get("D") if isinstance(model_meta, dict) else None,
        model_kwargs.get("D") if isinstance(model_kwargs, dict) else None,
        model_kwargs.get("expected_D") if isinstance(model_kwargs, dict) else None,
    ]
    dims: list[int] = []
    for idx, value in enumerate(candidates):
        if value is None:
            continue
        dim = _to_int(value, label=f"{checkpoint_dir} D candidate {idx}", errors=errors)
        if dim is not None:
            dims.append(dim)
    if not dims:
        errors.append(f"MWM checkpoint missing full latent dimension D: {checkpoint_dir}")
        return None
    if len(set(dims)) != 1:
        errors.append(f"MWM checkpoint has inconsistent full latent dimension D values {dims}: {checkpoint_dir}")
        return None
    return dims[0]


def _validate_role_checkpoint_contract(row: dict[str, Any], metadata: dict[str, Any], errors: list[str]) -> None:
    role = str(row.get("role", ""))
    checkpoint_dir = Path(str(row.get("checkpoint_run_dir", "")))
    levels = [int(k) for k in metadata.get("levels", [])] if isinstance(metadata.get("levels", []), list) else []
    model_meta = metadata.get("model", {})
    target = str(model_meta.get("target", "")) if isinstance(model_meta, dict) else ""
    backend = str(metadata.get("training_backend", ""))
    d = _checkpoint_full_latent_dim(metadata, checkpoint_dir, errors)
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
    _validate_paper_targets(cfg, rows, errors, roles={cell[2] for cell in expected})

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

    required_plots = _required_plots_for_benchmark(cfg, roles={cell[2] for cell in expected})
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
            role = _role(run, run_cfg)
            if checkpoint_ref is None:
                errors.append(f"static benchmark run {run.get('name', '<unnamed>')} missing checkpoint.run_dir")
                continue
            checkpoint_dir = Path(str(checkpoint_ref))
            checkpoint_metadata = _validate_checkpoint_metadata(checkpoint_dir, errors)
            if checkpoint_metadata:
                _validate_role_checkpoint_contract(
                    {"role": role, "checkpoint_run_dir": str(checkpoint_dir)},
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
        "manifest": _load_manifest_config(cfg),
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
