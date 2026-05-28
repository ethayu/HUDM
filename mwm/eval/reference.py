from __future__ import annotations

from typing import Any


REFERENCE_ROLE = "stable_wm_reference"


def needs_reference_evaluator(upstream_success: float, target_success: float, tolerance_pp: float = 1.0) -> bool:
    return abs(float(upstream_success) - float(target_success)) > float(tolerance_pp)


def reference_role_name() -> str:
    return REFERENCE_ROLE


def build_stable_wm_reference_policy(model: Any, plan_config: Any, *, cem_kwargs: dict[str, Any]) -> Any:
    from stable_worldmodel.policy import WorldModelPolicy
    from stable_worldmodel.solver import CEMSolver

    solver = CEMSolver(model=model, **cem_kwargs)
    return WorldModelPolicy(solver=solver, config=plan_config)


__all__ = [
    "REFERENCE_ROLE",
    "build_stable_wm_reference_policy",
    "needs_reference_evaluator",
    "reference_role_name",
]
