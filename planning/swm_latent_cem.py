from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from planning.cem_core import SharedCEMCore


@dataclass
class SWMLatentCEMInfo:
    level_idx: int
    level_k: int
    rollout_level_indices: list[int]
    bits_used_estimate: int
    plan_time_sec: float
    best_cost: float


class SWMLatentCEMPlanner:
    """CEM planner over HUDM learned latent dynamics for SWM policies."""

    def __init__(
        self,
        world_model,
        horizon: int,
        action_dim: int,
        action_low: Any,
        action_high: Any,
        pop_size: int = 256,
        elite_frac: float = 0.1,
        n_iter: int = 5,
        init_std: float = 1.0,
        fidelity_cfg: dict[str, Any] | None = None,
        drop_tail_on_coarsen: bool = True,
        warm_start: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.world_model = world_model.eval()
        self.K = [int(k) for k in world_model.K]
        self.D = int(world_model.D)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.drop_tail_on_coarsen = bool(drop_tail_on_coarsen)
        self.warm_start = bool(warm_start)
        self.device = device or torch.device("cpu")
        self._last_costs: torch.Tensor | None = None

        self.core = SharedCEMCore(
            horizon=self.horizon,
            action_dim=self.action_dim,
            pop_size=int(pop_size),
            elite_frac=float(elite_frac),
            n_iter=int(n_iter),
            init_std=float(init_std),
            action_low=action_low,
            action_high=action_high,
            fidelity_cfg=fidelity_cfg,
            num_levels=len(self.K),
            rollout_modes={"fixed", "linear"},
            device=self.device,
        )

    @torch.no_grad()
    def _evaluate_population(
        self,
        actions: torch.Tensor,
        z0: torch.Tensor,
        z_goal: torch.Tensor,
        base_level_idx: int,
        rollout_levels: list[int],
    ) -> tuple[torch.Tensor, list[int], int]:
        p = int(actions.shape[0])
        z = z0.expand(p, -1).clone()
        bits = 0
        for t, level_idx in enumerate(rollout_levels):
            k = self.K[int(level_idx)]
            z_next_k = self.world_model.predict_next(int(level_idx), z, actions[:, t, :])
            z_next = z.clone()
            z_next[:, :k] = z_next_k
            if self.drop_tail_on_coarsen and k < self.D:
                z_next[:, k:] = 0.0
            z = z_next
            bits += int(p * k * 32)

        k_terminal = self.K[int(base_level_idx)]
        diff = z[:, :k_terminal] - z_goal.expand(p, -1)[:, :k_terminal]
        costs = torch.linalg.vector_norm(diff, dim=1)
        self._last_costs = costs.detach()
        return costs, [int(x) for x in rollout_levels], bits

    @torch.no_grad()
    def plan(
        self,
        z0: torch.Tensor,
        z_goal: torch.Tensor,
        mpc_progress: float = 0.0,
        warm_start_steps: int = 1,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, SWMLatentCEMInfo]:
        if z0.shape[-1] != self.D or z_goal.shape[-1] != self.D:
            raise ValueError(
                f"Latent dim mismatch: expected D={self.D}, got z0={tuple(z0.shape)} z_goal={tuple(z_goal.shape)}"
            )
        z0 = z0.to(self.device)
        z_goal = z_goal.to(self.device)
        t0 = time.perf_counter()

        def evaluate(actions: torch.Tensor, base_level_idx: int, rollout_levels: list[int], iter_idx: int):
            del iter_idx
            return self._evaluate_population(actions, z0, z_goal, base_level_idx, rollout_levels)

        action_seq, level_idx, rollout_levels, bits = self.core.optimize(
            mpc_progress=float(mpc_progress),
            evaluate_population=evaluate,
            warm_start=self.warm_start,
            shift_steps=int(warm_start_steps),
            rng_seed=seed,
        )
        best_cost = float("nan")
        if self._last_costs is not None and self._last_costs.numel():
            best_cost = float(self._last_costs.min().item())
        info = SWMLatentCEMInfo(
            level_idx=int(level_idx),
            level_k=int(self.K[int(level_idx)]),
            rollout_level_indices=[int(x) for x in rollout_levels],
            bits_used_estimate=int(bits),
            plan_time_sec=float(time.perf_counter() - t0),
            best_cost=best_cost,
        )
        return action_seq, info
