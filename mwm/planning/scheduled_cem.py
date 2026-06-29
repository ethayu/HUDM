from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from mwm.fidelity import FidelityDecision, FidelityScheduler
from mwm.models.flops import FLOP_ACCOUNTING_NONE, normalize_flop_accounting


class MWMScheduledCEMSolver:
    """Stable-WM-compatible CEM solver with explicit MWM fidelity decisions."""

    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        scheduler: FidelityScheduler | dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        clamp_actions: bool = False,
        std_unbiased: bool = True,
        callbacks: list[Any] | None = None,
        pop_schedule: dict[str, Any] | None = None,
        elite_frac: float | None = None,
        max_replans: int | None = None,
        flop_accounting: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.var_scale = float(var_scale)
        self.num_samples = int(num_samples)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.scheduler_spec = scheduler
        self.pop_schedule = dict(pop_schedule) if pop_schedule else None
        self.elite_frac = float(elite_frac) if elite_frac is not None else None
        self.max_replans = max(1, int(max_replans)) if max_replans is not None else 1
        self._replan_idx = 0
        self.flop_accounting = normalize_flop_accounting(flop_accounting)
        self.device = torch.device(device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.clamp_actions = bool(clamp_actions)
        self.std_unbiased = bool(std_unbiased)
        self.callbacks = list(callbacks or [])
        self._configured = False
        self.solve_history: list[dict[str, Any]] = []
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        if not isinstance(action_space, Box):
            raise TypeError(f"MWMScheduledCEMSolver requires continuous Box action_space, got {type(action_space)}")
        single_shape = getattr(action_space, "shape", ())
        if len(single_shape) >= 2 and int(single_shape[0]) == int(n_envs):
            base_shape = single_shape[1:]
        else:
            base_shape = getattr(action_space, "single_action_space", action_space).shape
        self._base_action_dim = int(np.prod(base_shape))
        action_block = int(getattr(config, "action_block", 1))
        self._action_block = action_block
        self._action_dim = self._base_action_dim * action_block
        low = torch.as_tensor(action_space.low, dtype=self.dtype)
        high = torch.as_tensor(action_space.high, dtype=self.dtype)
        if low.ndim > 1 and low.shape[0] == int(n_envs):
            low = low.reshape(low.shape[0], -1)[0]
            high = high.reshape(high.shape[0], -1)[0]
        else:
            low = low.reshape(-1)
            high = high.reshape(-1)
        self._action_low = low.repeat(action_block).to(self.device)
        self._action_high = high.repeat(action_block).to(self.device)
        horizon = int(getattr(config, "horizon"))
        num_levels = int(getattr(self.model, "num_levels", len(getattr(self.model, "K"))))
        if isinstance(self.scheduler_spec, FidelityScheduler):
            self.scheduler = self.scheduler_spec
        else:
            self.scheduler = FidelityScheduler.from_config(self.scheduler_spec, num_levels=num_levels, horizon=horizon)
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def horizon(self) -> int:
        return int(self._config.horizon)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.solve(*args, **kwargs)

    def init_action_distrib(self, n_envs: int, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        var = self.var_scale * torch.ones([n_envs, self.horizon, self.action_dim], dtype=self.dtype)
        mean = torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype) if actions is None else actions
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            mean = torch.cat([mean, torch.zeros([n_envs, remaining, self.action_dim], dtype=self.dtype)], dim=1)
        return mean, var

    def _expanded_infos(
        self, info_dict: dict[str, Any], start_idx: int, end_idx: int, current_bs: int, num_samples: int
    ) -> dict[str, Any]:
        expanded: dict[str, Any] = {}
        for key, value in info_dict.items():
            value_batch = value[start_idx:end_idx]
            if torch.is_tensor(value):
                target_dtype = self.dtype if value_batch.is_floating_point() else None
                expanded[key] = (
                    value_batch.to(device=self.device, dtype=target_dtype)
                    .unsqueeze(1)
                    .expand(current_bs, num_samples, *value_batch.shape[1:])
                )
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(value_batch[:, None, ...], num_samples, axis=1)
            else:
                expanded[key] = value_batch
        return expanded

    def _resolve_num_samples(self, cem_progress: float) -> int:
        if self.pop_schedule is None:
            return self.num_samples
        start = int(self.pop_schedule.get("start", self.num_samples))
        end = int(self.pop_schedule.get("end", self.num_samples))
        value = float(start) + (float(end) - float(start)) * max(0.0, min(1.0, float(cem_progress)))
        return max(1, int(round(value)))

    def _resolve_topk(self, num_samples: int) -> int:
        if self.elite_frac is not None:
            return max(1, min(num_samples, int(round(self.elite_frac * num_samples))))
        return max(1, min(self.topk, num_samples))

    def _cost(self, infos: dict[str, Any], candidates: torch.Tensor, decision: FidelityDecision) -> torch.Tensor:
        if not hasattr(self.model, "get_cost_with_fidelity"):
            raise TypeError("MWMScheduledCEMSolver requires get_cost_with_fidelity(...); legacy get_cost fallback is disabled.")
        if self.flop_accounting != FLOP_ACCOUNTING_NONE:
            metadata = dict(decision.metadata)
            metadata["flop_accounting"] = self.flop_accounting
            decision = replace(decision, metadata=metadata)
        return self.model.get_cost_with_fidelity(infos, candidates, decision)

    def _current_mpc_progress(self) -> float:
        if self.max_replans <= 1:
            return 0.0
        return max(0.0, min(1.0, float(self._replan_idx) / float(self.max_replans - 1)))

    @torch.inference_mode()
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        if not self._configured:
            raise RuntimeError("MWMScheduledCEMSolver.configure must be called before solve")
        start_time = time.perf_counter()
        mpc_progress = self._current_mpc_progress()
        replan_idx = int(self._replan_idx)
        total_envs = len(next(iter(info_dict.values())))
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)
        outputs: dict[str, Any] = {"costs": [], "mean": [], "var": [], "mwm_diagnostics": []}
        for cb in self.callbacks:
            cb.reset()
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            final_batch_cost: list[float] | None = None
            for cb in self.callbacks:
                cb.start_batch()
            for step in range(self.n_steps):
                decision = self.scheduler.decision(
                    cem_iter=step,
                    n_iter=self.n_steps,
                    mpc_progress=mpc_progress,
                    context={"batch_start": start_idx, "batch_end": end_idx},
                )
                current_num_samples = self._resolve_num_samples(decision.cem_progress)
                current_topk = self._resolve_topk(current_num_samples)
                expanded_infos = self._expanded_infos(info_dict, start_idx, end_idx, current_bs, current_num_samples)
                candidates = torch.randn(
                    current_bs,
                    current_num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                if self.clamp_actions:
                    candidates = torch.minimum(torch.maximum(candidates, self._action_low), self._action_high)
                    candidates[:, 0] = torch.minimum(torch.maximum(batch_mean, self._action_low), self._action_high)
                else:
                    candidates[:, 0] = batch_mean
                cost_t0 = time.perf_counter()
                costs = self._cost(expanded_infos, candidates, decision)
                cost_time = time.perf_counter() - cost_t0
                if costs.ndim != 2 or tuple(costs.shape) != (current_bs, current_num_samples):
                    raise ValueError(f"Expected costs shape {(current_bs, current_num_samples)}, got {tuple(costs.shape)}")
                topk_vals, topk_inds = torch.topk(costs, k=current_topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, current_topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                prev_mean = batch_mean
                prev_var = batch_var
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1, unbiased=self.std_unbiased)
                for cb in self.callbacks:
                    cb(
                        step=step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=topk_vals,
                        topk_inds=topk_inds,
                        topk_candidates=topk_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=prev_mean,
                        prev_var=prev_var,
                    )
                diagnostics = {
                    "batch_start": int(start_idx),
                    "batch_end": int(end_idx),
                    "cem_iter": int(step),
                    "cem_progress": float(decision.cem_progress),
                    "mpc_iter": int(replan_idx),
                    "mpc_progress": float(decision.mpc_progress),
                    "base_level_idx": int(decision.base_level_idx),
                    "mpc_level_idx": int(decision.metadata.get("mpc_level_idx", decision.base_level_idx)),
                    "terminal_level_idx": int(
                        decision.metadata.get("terminal_level_idx", decision.rollout_level_indices[-1])
                    ),
                    "rollout_level_indices": [int(x) for x in decision.rollout_level_indices],
                    "cost_time_sec": float(cost_time),
                    "cem_cost_calls": 1,
                    "num_samples": int(current_num_samples),
                    "topk": int(current_topk),
                    "horizon": int(self.horizon),
                    "action_dim": int(self.action_dim),
                    "candidate_action_values": int(current_bs * current_num_samples * self.horizon * self.action_dim),
                }
                model_diag = getattr(self.model, "_last_cost_diagnostics", None)
                if isinstance(model_diag, dict):
                    diagnostics.update({f"model_{k}": v for k, v in model_diag.items()})
                outputs["mwm_diagnostics"].append(diagnostics)
                final_batch_cost = topk_vals.mean(dim=1).detach().cpu().tolist()
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost or [])
        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        outputs["solve_time_sec"] = float(time.perf_counter() - start_time)
        if self.callbacks:
            outputs["callbacks"] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs["callbacks"][cb.output_key] = cb.history
        self.solve_history.append(
            {
                "solve_time_sec": outputs["solve_time_sec"],
                "costs": list(outputs["costs"]),
                "mwm_diagnostics": list(outputs["mwm_diagnostics"]),
            }
        )
        self._replan_idx += 1
        return outputs

    def reset_history(self) -> None:
        self.solve_history = []
        self._replan_idx = 0


__all__ = ["MWMScheduledCEMSolver"]
