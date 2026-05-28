from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F


try:
    from stable_worldmodel.policy import WorldModelPolicy
except Exception:  # pragma: no cover - optional dependency
    class WorldModelPolicy:  # type: ignore[no-redef]
        def __init__(self, model: Any = None, solver: Any = None, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.model = model
            self.solver = solver


_MWM_IMAGE_SIZE = (224, 224)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def mwm_image_input_transform(image: torch.Tensor) -> torch.Tensor:
    """Stable-WM policy transform for MWM image inputs.

    The model owns Le-WM/ImageNet normalization in its preprocess module; this
    policy transform only standardizes layout, range, and image size.
    """

    tensor = torch.as_tensor(image)
    if tensor.ndim == 3 and tensor.shape[0] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
    else:
        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() and torch.isfinite(tensor).all().item() and tensor.max().item() > 2.0:
            tensor = tensor / 255.0
    if tuple(tensor.shape[-2:]) != _MWM_IMAGE_SIZE:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=_MWM_IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    return tensor


def imagenet_image_input_transform(image: torch.Tensor) -> torch.Tensor:
    """Policy transform for imported Stable-WM models trained with ImageNet normalization."""

    tensor = mwm_image_input_transform(image)
    if tensor.numel() and torch.isfinite(tensor).all().item() and tensor.min().item() < -0.5:
        return tensor
    mean = tensor.new_tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(_IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


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
                "cem_cost_calls": total_cem_cost_calls,
                "candidate_action_values": total_candidate_action_values,
            },
        }


__all__ = [
    "MWMWorldModelPolicy",
    "WorldModelPolicy",
    "imagenet_image_input_transform",
    "model_accounting",
    "mwm_image_input_transform",
]
