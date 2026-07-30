from __future__ import annotations

import time
from typing import Any

import numpy as np

from mwm.io import jsonable

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
    supports_arbitrary_k = bool(getattr(model, "supports_arbitrary_k", False))
    min_k = int(getattr(model, "min_k", min(k_values) if k_values else 0))
    max_k = int(getattr(model, "D", max(k_values) if k_values else 0))
    params = 0
    if hasattr(model, "parameters"):
        params = int(sum(p.numel() for p in model.parameters()))
    return {
        "K": k_values,
        "num_levels": int(len(k_values)),
        "D": int(getattr(model, "D", max(k_values) if k_values else 0)),
        "supports_arbitrary_k": supports_arbitrary_k,
        "supported_k": (
            {"min": min_k, "max": max_k, "arbitrary": True}
            if supports_arbitrary_k
            else {"values": k_values, "arbitrary": False}
        ),
        "parameters": params,
    }


class _TracingActionTransform:
    """Capture model-space actions at the exact inverse-transform boundary."""

    def __init__(self, transform: Any, capture: Any) -> None:
        self._transform = transform
        self._capture = capture

    def transform(self, value: Any) -> Any:
        return self._transform.transform(value)

    def inverse_transform(self, value: Any) -> Any:
        self._capture(np.asarray(value).copy())
        return self._transform.inverse_transform(value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transform, name)


class MWMWorldModelPolicy(WorldModelPolicy):
    """Stable-WM policy wrapper exposing MWM solver diagnostics."""

    def __init__(self, model: Any = None, solver: Any = None, config: Any = None, **kwargs: Any) -> None:
        self.model = model
        self.config = config
        self._action_calls = 0
        self._policy_time_sec = 0.0
        self._action_trace: list[list[Any]] = []
        self._model_action_trace: list[list[Any]] = []
        self._pending_model_action: Any = None
        if config is None:
            raise ValueError("MWMWorldModelPolicy requires a Stable-WM PlanConfig.")
        super().__init__(solver=solver, config=config, **kwargs)
        if getattr(self, "cfg", None) is None:
            raise RuntimeError("Stable-WM WorldModelPolicy initialization did not produce a usable policy config.")
        self.model = model
        if "action" in self.process:
            self.process = dict(self.process)
            self.process["action"] = _TracingActionTransform(
                self.process["action"],
                self._capture_model_action,
            )

    def _capture_model_action(self, action: Any) -> None:
        self._pending_model_action = np.asarray(action).copy()

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> Any:
        start = time.perf_counter()
        action: Any = None
        recorded = False
        self._pending_model_action = None
        try:
            action = super().get_action(info_dict, **kwargs)
            recorded = True
            return action
        finally:
            if recorded:
                self._record_trace(self._action_trace, action)
                self._record_trace(
                    self._model_action_trace,
                    action if self._pending_model_action is None else self._pending_model_action,
                )
            self._action_calls += 1
            self._policy_time_sec += time.perf_counter() - start

    @staticmethod
    def _record_trace(trace: list[list[Any]], action: Any) -> None:
        data = jsonable(action)
        if isinstance(data, list) and data and all(isinstance(row, list) for row in data):
            rows = data
        else:
            rows = [data]
        if len(trace) < len(rows):
            trace.extend([] for _ in range(len(rows) - len(trace)))
        for idx, row in enumerate(rows):
            trace[idx].append(row)

    def set_env(self, env: Any) -> None:
        super().set_env(env)

    def reset_trace(self) -> None:
        if hasattr(self.solver, "reset_history"):
            self.solver.reset_history()
        self._action_calls = 0
        self._policy_time_sec = 0.0
        self._action_trace = []
        self._model_action_trace = []
        self._pending_model_action = None
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

    def review_trace(self) -> dict[str, Any]:
        return {
            "action_trace": jsonable(self._action_trace),
            "model_action_trace": jsonable(self._model_action_trace),
        }


__all__ = [
    "MWMWorldModelPolicy",
    "WorldModelPolicy",
    "model_accounting",
]
