from __future__ import annotations

from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hudm.swm_envs import validate_continuous_box_action_space
from planning.swm_latent_cem import SWMLatentCEMPlanner


@dataclass(frozen=True)
class SWMPlannerConfig:
    horizon: int = 16
    receding_horizon: int = 1
    pop_size: int = 256
    elite_frac: float = 0.1
    n_iter: int = 5
    init_std: float = 1.0
    warm_start: bool = True
    fidelity: dict[str, Any] | None = None


def _latest_image_array(value: Any, key: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 5:
        arr = arr[:, -1]
    elif arr.ndim == 4:
        pass
    elif arr.ndim == 3:
        arr = arr[None]
    else:
        raise ValueError(f"{key} must have shape (N,T,H,W,C), (N,H,W,C), or (H,W,C); got {arr.shape}")
    return arr


def images_to_tensor(value: Any, key: str, device: torch.device) -> torch.Tensor:
    arr = _latest_image_array(value, key)
    if arr.shape[-1] == 3:
        arr = np.moveaxis(arr, -1, 1)
    elif arr.shape[1] != 3:
        raise ValueError(f"HUDM SWM eval supports RGB-only {key}; got shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.size and float(np.nanmax(arr)) > 2.0:
        arr = arr / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=torch.float32)


class HUDMLatentCEMPolicy:
    """SWM-compatible policy backed by HUDM latent CEM planning."""

    type = "hudm_latent_cem"

    def __init__(
        self,
        world_model,
        config: SWMPlannerConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self.world_model = world_model.eval()
        self.cfg = config
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.action_dim = int(self.action_low.size)
        self.device = device or torch.device("cpu")
        self.seed = seed
        self.env = None
        self._planners: list[SWMLatentCEMPlanner] = []
        self._buffers: list[deque[torch.Tensor]] = []
        self._plan_calls = 0
        self.latest_infos: list[Any] = []
        self.trace_buffer: list[dict[str, Any]] = []

    def set_env(self, env: Any) -> None:
        self.env = env
        n_envs = int(getattr(env, "num_envs", 1))
        low, high = validate_continuous_box_action_space(env.single_action_space, "SWM env")
        if low.shape != self.action_low.shape or high.shape != self.action_high.shape:
            raise ValueError(
                f"Checkpoint action bounds shape {self.action_low.shape} does not match env bounds {low.shape}."
            )
        if not np.allclose(low, self.action_low) or not np.allclose(high, self.action_high):
            raise ValueError(
                "Checkpoint action bounds do not match env bounds: "
                f"checkpoint low/high={self.action_low.tolist()}/{self.action_high.tolist()}, "
                f"env low/high={low.tolist()}/{high.tolist()}."
            )
        self._planners = [
            SWMLatentCEMPlanner(
                self.world_model,
                horizon=self.cfg.horizon,
                action_dim=self.action_dim,
                action_low=self.action_low,
                action_high=self.action_high,
                pop_size=self.cfg.pop_size,
                elite_frac=self.cfg.elite_frac,
                n_iter=self.cfg.n_iter,
                init_std=self.cfg.init_std,
                fidelity_cfg=self.cfg.fidelity,
                warm_start=self.cfg.warm_start,
                device=self.device,
            )
            for _ in range(n_envs)
        ]
        self._buffers = [deque(maxlen=max(1, int(self.cfg.receding_horizon))) for _ in range(n_envs)]
        self._plan_calls = 0
        self.trace_buffer = []

    def set_seed(self, seed: int | None) -> None:
        self.seed = None if seed is None else int(seed)

    def reset_trace(self) -> None:
        self.trace_buffer = []

    @staticmethod
    def _plan_info_dict(info: Any) -> dict[str, Any]:
        if info is None:
            return {}
        if hasattr(info, "__dataclass_fields__"):
            out = asdict(info)
        elif isinstance(info, dict):
            out = dict(info)
        else:
            out = dict(getattr(info, "__dict__", {}))
        if "rollout_level_indices" in out:
            out["rollout_level_indices"] = [int(x) for x in out["rollout_level_indices"]]
        for key in ("level_idx", "level_k", "bits_used_estimate"):
            if key in out:
                out[key] = int(out[key])
        out.pop("best_cost", None)
        for key in ("plan_time_sec",):
            if key in out:
                out[key] = float(out[key])
        return out

    def diagnostics(self) -> dict[str, Any]:
        replans = [entry for entry in self.trace_buffer if entry["replanned"]]
        plan_times = [float(entry["plan_info"].get("plan_time_sec", 0.0)) for entry in replans]
        bits = [int(entry["plan_info"].get("bits_used_estimate", 0)) for entry in replans]
        return {
            "trace": list(self.trace_buffer),
            "summary": {
                "action_calls": int(self._plan_calls),
                "actions_recorded": int(len(self.trace_buffer)),
                "replans": int(len(replans)),
                "total_plan_time_sec": float(np.sum(plan_times)) if plan_times else 0.0,
                "mean_plan_time_sec": float(np.mean(plan_times)) if plan_times else 0.0,
                "total_bits_used_estimate": int(np.sum(bits)) if bits else 0,
            },
        }

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.world_model.encode(images)

    @torch.no_grad()
    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
        del kwargs
        if self.env is None:
            raise RuntimeError("HUDMLatentCEMPolicy.set_env must be called before get_action.")
        if "pixels" not in info_dict:
            raise KeyError("SWM info_dict must include 'pixels'.")
        if "goal" not in info_dict:
            raise KeyError("SWM info_dict must include 'goal'.")

        pixels = images_to_tensor(info_dict["pixels"], "pixels", self.device)
        goals = images_to_tensor(info_dict["goal"], "goal", self.device)
        if pixels.shape[0] != goals.shape[0]:
            raise ValueError(f"pixels batch {pixels.shape[0]} does not match goal batch {goals.shape[0]}")
        n_envs = int(pixels.shape[0])
        if n_envs != len(self._planners):
            raise ValueError(f"Policy was configured for {len(self._planners)} envs but got {n_envs} observations.")

        z = self._encode(pixels)
        z_goal = self._encode(goals)
        actions = np.zeros((n_envs, self.action_dim), dtype=np.float32)
        plan_infos: list[Any] = []
        for env_i in range(n_envs):
            replanned = False
            plan_info = None
            buffer_len_before = len(self._buffers[env_i])
            if len(self._buffers[env_i]) == 0:
                seed = None if self.seed is None else int(self.seed) + self._plan_calls * n_envs + env_i
                plan, plan_info = self._planners[env_i].plan(
                    z[env_i : env_i + 1],
                    z_goal[env_i : env_i + 1],
                    mpc_progress=0.0,
                    warm_start_steps=1,
                    seed=seed,
                )
                keep = max(1, min(int(self.cfg.receding_horizon), int(plan.shape[0])))
                for row in plan[:keep]:
                    self._buffers[env_i].append(row)
                plan_infos.append(plan_info)
                replanned = True
            raw_action = self._buffers[env_i].popleft().detach().cpu().numpy()
            actions[env_i] = np.clip(raw_action, self.action_low, self.action_high)
            plan_info_dict = self._plan_info_dict(plan_info)
            self.trace_buffer.append(
                {
                    "action_call": int(self._plan_calls),
                    "env_index": int(env_i),
                    "replanned": bool(replanned),
                    "buffer_len_before": int(buffer_len_before),
                    "buffer_len_after": int(len(self._buffers[env_i])),
                    "action": actions[env_i].astype(float).tolist(),
                    "plan_info": plan_info_dict,
                    "fidelity_levels": {
                        "level_idx": plan_info_dict.get("level_idx"),
                        "level_k": plan_info_dict.get("level_k"),
                        "rollout_level_indices": plan_info_dict.get("rollout_level_indices"),
                    },
                    "bits_used_estimate": int(plan_info_dict.get("bits_used_estimate", 0)),
                    "plan_time_sec": float(plan_info_dict.get("plan_time_sec", 0.0)),
                }
            )
        self._plan_calls += 1
        self.latest_infos = plan_infos
        return actions
