from __future__ import annotations

import math
from typing import Any

from omegaconf import OmegaConf


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


def normalize_paper_target_config(cfg: Any) -> dict[str, Any]:
    config = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    targets = config.get("paper_targets", {})
    if not bool(targets.get("enabled", False)):
        return {}
    success_rate = targets.get("success_rate", {})
    if not isinstance(success_rate, dict) and not OmegaConf.is_config(success_rate):
        raise ValueError("paper_targets.success_rate must be a mapping")
    success_rate = dict(
        OmegaConf.to_container(success_rate, resolve=True) if OmegaConf.is_config(success_rate) else success_rate
    )
    env_id = str(config.env_id)
    missing_targets = [env_id] if env_id not in {str(key) for key in success_rate} else []
    if missing_targets:
        raise ValueError(f"paper_targets.success_rate missing benchmark env: {missing_targets}")
    return {
        "tolerance_pp": float(targets.get("tolerance_pp", targets.get("upstream_tolerance_pp", 1.0))),
        "retrained_match_tolerance_pp": float(targets.get("retrained_match_tolerance_pp", 5.0)),
        "success_rate": {str(key): float(value) for key, value in success_rate.items()},
    }


__all__ = ["append_paper_target_errors", "normalize_paper_target_config", "validate_paper_targets"]
