from __future__ import annotations

from typing import Any


EFFICIENCY_RATIOS_PLOT = "efficiency_ratios.png"
PAIRED_SUCCESS_DELTA_PLOT = "paired_success_delta.png"
SCHEDULE_USAGE_BY_ROLE_PLOT = "schedule_usage_by_role.png"
SUCCESS_VS_COMPUTE_PLOT = "success_vs_compute.png"
SUCCESS_BY_ENV_ROLE_PLOT = "success_by_env_role.png"
SUCCESS_VS_WALL_TIME_PLOT = "success_vs_wall_time.png"
SCHEDULE_LEVEL_USAGE_PLOT = "schedule_level_usage.png"

BASE_REQUIRED_PLOTS = {
    SUCCESS_VS_COMPUTE_PLOT,
    SUCCESS_BY_ENV_ROLE_PLOT,
    SUCCESS_VS_WALL_TIME_PLOT,
}
PAIR_REQUIRED_PLOTS = {
    EFFICIENCY_RATIOS_PLOT,
    PAIRED_SUCCESS_DELTA_PLOT,
}
SCHEDULE_REQUIRED_PLOTS = {
    SCHEDULE_LEVEL_USAGE_PLOT,
    SCHEDULE_USAGE_BY_ROLE_PLOT,
}
REQUIRED_PLOTS = BASE_REQUIRED_PLOTS | PAIR_REQUIRED_PLOTS | SCHEDULE_REQUIRED_PLOTS


def required_plots_for_benchmark(cfg: Any, roles: set[str] | None = None) -> set[str]:
    if roles is None:
        roles = {str(run.get("role", run.get("name", ""))) for run in cfg.get("runs", [])}
    required = set(BASE_REQUIRED_PLOTS)
    comparison_roles = roles - {"upstream_lewm_converted"}
    if "upstream_lewm_converted" in roles and comparison_roles:
        required.update(PAIR_REQUIRED_PLOTS)
    if roles & {"mwm_scheduled", "mwm_dense"}:
        required.update(SCHEDULE_REQUIRED_PLOTS)
    if not roles:
        return set(REQUIRED_PLOTS)
    return required


__all__ = [
    "BASE_REQUIRED_PLOTS",
    "EFFICIENCY_RATIOS_PLOT",
    "PAIRED_SUCCESS_DELTA_PLOT",
    "PAIR_REQUIRED_PLOTS",
    "REQUIRED_PLOTS",
    "SCHEDULE_LEVEL_USAGE_PLOT",
    "SCHEDULE_REQUIRED_PLOTS",
    "SCHEDULE_USAGE_BY_ROLE_PLOT",
    "SUCCESS_BY_ENV_ROLE_PLOT",
    "SUCCESS_VS_COMPUTE_PLOT",
    "SUCCESS_VS_WALL_TIME_PLOT",
    "required_plots_for_benchmark",
]
