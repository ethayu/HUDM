from __future__ import annotations

import time
from typing import Any

import torch


REFERENCE_ROLE = "stable_wm_reference"


def needs_reference_evaluator(upstream_success: float, target_success: float, tolerance_pp: float = 1.0) -> bool:
    return abs(float(upstream_success) - float(target_success)) > float(tolerance_pp)


def reference_role_name() -> str:
    return REFERENCE_ROLE


def build_stable_wm_reference_policy(
    model: Any,
    plan_config: Any,
    *,
    cem_kwargs: dict[str, Any],
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> Any:
    from stable_worldmodel.policy import WorldModelPolicy
    from stable_worldmodel.solver import CEMSolver

    class StableWMReferencePolicy(WorldModelPolicy):
        def __init__(self) -> None:
            super().__init__(solver=solver, config=plan_config, process=process, transform=transform)
            self._action_calls = 0
            self._policy_time_sec = 0.0
            self._estimated_replans = 0

        def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> Any:
            before = time.perf_counter()
            action_buffer = getattr(self, "_action_buffer", None)
            if action_buffer is None:
                estimated_replans = 1
            else:
                terminated = info_dict.get("terminated")
                dead = (
                    torch.as_tensor(terminated, dtype=torch.bool).cpu().numpy()
                    if terminated is not None
                    else [False for _ in action_buffer]
                )
                estimated_replans = sum(1 for idx, buf in enumerate(action_buffer) if not len(buf) and not bool(dead[idx]))
            try:
                return super().get_action(info_dict, **kwargs)
            finally:
                self._action_calls += 1
                self._estimated_replans += int(bool(estimated_replans))
                self._policy_time_sec += time.perf_counter() - before

        def reset_trace(self) -> None:
            self._action_calls = 0
            self._policy_time_sec = 0.0
            self._estimated_replans = 0
            for buf in getattr(self, "_action_buffer", []) or []:
                buf.clear()
            if hasattr(self, "_next_init"):
                self._next_init = None

        def diagnostics(self) -> dict[str, Any]:
            replans = int(self._estimated_replans)
            actions = int(self._action_calls)
            cost_calls = replans * int(getattr(self.solver, "n_steps", 0))
            candidate_values = (
                cost_calls
                * int(getattr(self.solver, "num_samples", 0))
                * int(getattr(self.cfg, "horizon", 0))
                * int(getattr(self.solver, "action_dim", 0) if hasattr(self.solver, "_action_dim") else 0)
            )
            return {
                "trace": [],
                "summary": {
                    "actions_recorded": actions,
                    "replans": replans,
                    "total_plan_time_sec": 0.0,
                    "mean_plan_time_sec": 0.0,
                    "total_policy_time_sec": float(self._policy_time_sec),
                    "mean_policy_time_sec": self._policy_time_sec / actions if actions else 0.0,
                    "total_bits_used_estimate": 0,
                    "latent_work_total": 0,
                    "cem_cost_calls": int(cost_calls),
                    "candidate_action_values": int(candidate_values),
                    "stable_wm_reference": True,
                },
            }

    solver = CEMSolver(model=model, **cem_kwargs)
    return StableWMReferencePolicy()


__all__ = [
    "REFERENCE_ROLE",
    "build_stable_wm_reference_policy",
    "needs_reference_evaluator",
    "reference_role_name",
]
