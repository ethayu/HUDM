from __future__ import annotations

import time
from typing import Any

try:
    from stable_worldmodel.policy import WorldModelPolicy
except Exception:  # pragma: no cover - optional dependency
    class WorldModelPolicy:  # type: ignore[no-redef]
        def __init__(self, model: Any = None, solver: Any = None, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.model = model
            self.solver = solver


def model_accounting(model: Any) -> dict[str, Any]:
    k_values = [int(k) for k in getattr(model, "K", [])]
    params = 0
    if hasattr(model, "parameters"):
        params = int(sum(p.numel() for p in model.parameters()))
    return {
        "K": k_values,
        "num_levels": int(len(k_values)),
        "D": int(getattr(model, "D", max(k_values) if k_values else 0)),
        "parameters": params,
    }


class MWMWorldModelPolicy(WorldModelPolicy):
    """Stable-WM policy wrapper exposing MWM solver diagnostics."""

    def __init__(self, model: Any = None, solver: Any = None, config: Any = None, **kwargs: Any) -> None:
        self.model = model
        self.config = config
        self._action_calls = 0
        self._policy_time_sec = 0.0
        if config is None:
            raise ValueError("MWMWorldModelPolicy requires a Stable-WM PlanConfig.")
        super().__init__(solver=solver, config=config, **kwargs)
        if getattr(self, "cfg", None) is None:
            raise RuntimeError("Stable-WM WorldModelPolicy initialization did not produce a usable policy config.")
        self.model = model

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return super().get_action(info_dict, **kwargs)
        finally:
            self._action_calls += 1
            self._policy_time_sec += time.perf_counter() - start

    def set_env(self, env: Any) -> None:
        super().set_env(env)

    def reset_trace(self) -> None:
        if hasattr(self.solver, "reset_history"):
            self.solver.reset_history()
        self._action_calls = 0
        self._policy_time_sec = 0.0
        for buf in getattr(self, "_action_buffer", []) or []:
            buf.clear()
        if hasattr(self, "_next_init"):
            self._next_init = None

    def diagnostics(self) -> dict[str, Any]:
        history = list(getattr(self.solver, "solve_history", []))
        traces = [diag for item in history for diag in item.get("mwm_diagnostics", [])]
        total_time = float(sum(float(item.get("solve_time_sec", 0.0)) for item in history))
        total_work = int(
            sum(
                int(diag.get("model_latent_work", diag.get("model_latent_work_total", 0)))
                for item in history
                for diag in item.get("mwm_diagnostics", [])
            )
        )
        total_dynamics_flops = int(sum(int(diag.get("model_dynamics_flops", 0)) for diag in traces))
        flop_audit_errors = [str(diag["model_flop_audit_error"]) for diag in traces if diag.get("model_flop_audit_error")]
        total_cem_cost_calls = int(sum(int(diag.get("cem_cost_calls", 1)) for diag in traces))
        total_candidate_action_values = int(sum(int(diag.get("candidate_action_values", 0)) for diag in traces))
        return {
            "trace": traces,
            "summary": {
                "actions_recorded": int(self._action_calls),
                "replans": int(len(history)),
                "total_plan_time_sec": total_time,
                "mean_plan_time_sec": total_time / len(history) if history else 0.0,
                "total_policy_time_sec": float(self._policy_time_sec),
                "mean_policy_time_sec": self._policy_time_sec / self._action_calls if self._action_calls else 0.0,
                "total_bits_used_estimate": total_work * 32,
                "latent_work_total": total_work,
                "dynamics_flops_total": total_dynamics_flops,
                "flop_audit_error_count": int(len(flop_audit_errors)),
                "cem_cost_calls": total_cem_cost_calls,
                "candidate_action_values": total_candidate_action_values,
            },
        }


__all__ = [
    "MWMWorldModelPolicy",
    "WorldModelPolicy",
    "model_accounting",
]
