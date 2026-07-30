from __future__ import annotations

from typing import Any

from mwm.benchmark.config import cell_id, filter_resolved_by_roles, merged_run_config, validate_benchmark_matrix
from mwm.benchmark.sweep import expand_benchmark_runs


def expected_cells_from_resolved(resolved: list[tuple[Any, Any]]) -> set[tuple[str, int, str]]:
    return {
        (str(run_cfg.get("env_id", "")), int(run_cfg.eval.seed), cell_id(run, run_cfg))
        for run, run_cfg in resolved
    }


def load_expected_resolved(cfg: Any, *, roles: Any = None) -> list[tuple[Any, Any]]:
    resolved = []
    for run in expand_benchmark_runs(cfg):
        _, run_cfg = merged_run_config(cfg, run)
        resolved.append((run, run_cfg))
    resolved = filter_resolved_by_roles(cfg, resolved, roles)
    validate_benchmark_matrix(cfg, resolved)
    return resolved


def metric_identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row.get("env_id", "")), int(row.get("seed", -1)), str(row.get("cell_id", row.get("role", "")))


__all__ = ["expected_cells_from_resolved", "load_expected_resolved", "metric_identity"]
