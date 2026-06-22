from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def float_metric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_metric(values: Iterable[Any]) -> float:
    vals = [float_metric(v, float("nan")) for v in values]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def role_label(role: str) -> str:
    labels = {
        "upstream_lewm_converted": "Upstream Le-WM",
        "retrained_lewm_identity": "Retrained Le-WM",
        "mwm_scheduled": "MWM scheduled",
        "mwm_dense": "MWM dense",
    }
    return labels.get(str(role), str(role))


def env_label(env_id: str) -> str:
    return str(env_id).removeprefix("swm/").removesuffix("-v1")


def sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    role_order = {"upstream_lewm_converted": 0, "retrained_lewm_identity": 1, "mwm_scheduled": 2, "mwm_dense": 3}
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("env_id", "")),
            int(r.get("seed", 0)),
            role_order.get(str(r.get("role", "")), 99),
            str(r.get("name", "")),
        ),
    )


def row_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (str(row.get("env_id", "")), int(row.get("seed", 0)), str(row.get("role", ""))): row
        for row in rows
    }


def comparison_roles(rows: Iterable[dict[str, Any]]) -> list[str]:
    role_order = {"retrained_lewm_identity": 0, "mwm_scheduled": 1, "mwm_dense": 2}
    roles = {
        str(row.get("role", ""))
        for row in rows
        if str(row.get("role", "")) and str(row.get("role", "")) != "upstream_lewm_converted"
    }
    return sorted(roles, key=lambda role: (role_order.get(role, 99), role))


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = row_index(rows)
    pairs: list[dict[str, Any]] = []
    env_seed = sorted({(env, seed) for env, seed, _ in indexed})
    roles = comparison_roles(rows)
    for env_id, seed in env_seed:
        baseline = indexed.get((env_id, seed, "upstream_lewm_converted"))
        if not baseline:
            continue
        for role in roles:
            comparison = indexed.get((env_id, seed, role))
            if not comparison:
                continue
            base_success = float_metric(baseline.get("success_rate"), float("nan"))
            comparison_success = float_metric(comparison.get("success_rate"), float("nan"))
            base_wall = float_metric(baseline.get("wall_time_sec"), float("nan"))
            comparison_wall = float_metric(comparison.get("wall_time_sec"), float("nan"))
            base_bits = float_metric(baseline.get("bits_used_total"), float("nan"))
            comparison_bits = float_metric(comparison.get("bits_used_total"), float("nan"))
            pairs.append(
                {
                    "env_id": env_id,
                    "seed": seed,
                    "baseline": baseline,
                    "comparison": comparison,
                    "comparison_role": role,
                    "mwm": comparison,
                    "delta_success": comparison_success - base_success,
                    "wall_ratio": comparison_wall / base_wall if base_wall > 0 else float("nan"),
                    "compute_ratio": comparison_bits / base_bits if base_bits > 0 else float("nan"),
                    "same_manifest": str(baseline.get("manifest_sha256", ""))
                    == str(comparison.get("manifest_sha256", "")),
                }
            )
    return pairs


def outcome_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = paired_rows(rows)
    envs = sorted({str(row.get("env_id", "")) for row in rows})
    roles = comparison_roles(rows)
    out: list[dict[str, Any]] = []
    for env_id in envs:
        env_rows = [r for r in rows if str(r.get("env_id", "")) == env_id]
        base = [r for r in env_rows if str(r.get("role", "")) == "upstream_lewm_converted"]
        for role in roles:
            comparison = [r for r in env_rows if str(r.get("role", "")) == role]
            if not comparison:
                continue
            env_pairs = [p for p in pairs if p["env_id"] == env_id and p["comparison_role"] == role]
            base_success = mean_metric(r.get("success_rate") for r in base)
            comparison_success = mean_metric(r.get("success_rate") for r in comparison)
            out.append(
                {
                    "env_id": env_id,
                    "comparison_role": role,
                    "baseline_success": base_success,
                    "comparison_success": comparison_success,
                    "delta_success": comparison_success - base_success,
                    "baseline_wall": mean_metric(r.get("wall_time_sec") for r in base),
                    "comparison_wall": mean_metric(r.get("wall_time_sec") for r in comparison),
                    "baseline_compute": mean_metric(r.get("bits_used_total") for r in base),
                    "comparison_compute": mean_metric(r.get("bits_used_total") for r in comparison),
                    "same_manifests": sum(1 for p in env_pairs if p["same_manifest"]),
                    "pairs": len(env_pairs),
                }
            )
    return out


__all__ = [
    "comparison_roles",
    "env_label",
    "float_metric",
    "mean_metric",
    "outcome_rows",
    "paired_rows",
    "role_label",
    "row_index",
    "sorted_rows",
]
