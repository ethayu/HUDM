from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


REFERENCE_ROLE = "stable_wm_reference"


def needs_reference_evaluator(upstream_success: float, target_success: float, tolerance_pp: float = 1.0) -> bool:
    return abs(float(upstream_success) - float(target_success)) > float(tolerance_pp)


def reference_role_name() -> str:
    return REFERENCE_ROLE


def _sample_expand_goal_emb(emb: torch.Tensor, num_samples: int) -> torch.Tensor:
    if emb.ndim == 2:
        emb = emb[:, None, None, :]
    elif emb.ndim == 3:
        emb = emb[:, None, :, :]
    elif emb.ndim != 4:
        raise ValueError(f"Stable-WM goal embedding must be 2D, 3D, or 4D; got {tuple(emb.shape)}")
    if emb.shape[1] == num_samples:
        return emb
    if emb.shape[1] != 1:
        raise ValueError(
            f"Stable-WM goal embedding sample dimension must be 1 or {num_samples}; got {tuple(emb.shape)}"
        )
    return emb.expand(-1, num_samples, *([-1] * (emb.ndim - 2)))


class SampleExpandedGoalCostModel(nn.Module):
    def __init__(self, wrapped_model: nn.Module) -> None:
        super().__init__()
        self.wrapped_model = wrapped_model

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_model, name)

    def _ensure_goal_emb(self, info_dict: dict[str, Any], action_candidates: torch.Tensor) -> None:
        num_samples = int(action_candidates.shape[1])
        existing = info_dict.get("goal_emb")
        if torch.is_tensor(existing):
            info_dict["goal_emb"] = _sample_expand_goal_emb(existing, num_samples)
            return
        if "goal" not in info_dict or not hasattr(self.wrapped_model, "encode"):
            return
        goal: dict[str, Any] = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                goal[key] = value[:, 0]
        goal["pixels"] = goal["goal"]
        for key in list(goal):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        encoded = self.wrapped_model.encode(goal)
        info_dict["goal_emb"] = _sample_expand_goal_emb(encoded["emb"].detach(), num_samples)

    def get_cost(self, info_dict: dict[str, Any], action_candidates: torch.Tensor) -> torch.Tensor:
        self._ensure_goal_emb(info_dict, action_candidates)
        return self.wrapped_model.get_cost(info_dict, action_candidates)


def _needs_sample_expanded_goal_wrapper(model: Any) -> bool:
    try:
        from stable_worldmodel.wm.pldm.pldm import PLDM
    except Exception:
        return False
    return isinstance(model, PLDM)


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

    solver_model = SampleExpandedGoalCostModel(model) if _needs_sample_expanded_goal_wrapper(model) else model
    solver = CEMSolver(model=solver_model, **cem_kwargs)
    return StableWMReferencePolicy()


__all__ = [
    "REFERENCE_ROLE",
    "build_stable_wm_reference_policy",
    "needs_reference_evaluator",
    "reference_role_name",
]
