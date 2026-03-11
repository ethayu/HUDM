from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from planning.cem_core import SharedCEMCore

@dataclass
class LatentCEMInfo:
    base_level_idx: int
    base_k: int
    rollout_level_indices: List[int]
    bits_used_estimate: int = 0
    plan_time_sec: float = 0.0


class LatentCEMPlanner:
    """
    CEM planner over latent-space world-model rollouts.

    Planning/CEM/fidelity orchestration is shared in `SharedCEMCore`; this
    class only implements world-model-specific rollout and objective logic.
    """

    def __init__(
        self,
        world_model,
        horizon: int,
        action_dim: int,
        pop_size: int = 256,
        elite_frac: float = 0.1,
        n_iter: int = 5,
        init_std: float = 1.0,
        action_low: Optional[float] = None,
        action_high: Optional[float] = None,
        objective_cfg: Optional[Dict[str, Any]] = None,
        fidelity_cfg: Optional[Dict[str, Any]] = None,
        drop_tail_on_coarsen: bool = True,
        warm_start: bool = True,
        device: Optional[torch.device] = None,
    ):

        self.ACTION_MEAN = torch.tensor([-0.0087, 0.0068]).to(device)    
        self.ACTION_STD = torch.tensor([0.2019, 0.2002]).to(device)
        self.world_model = world_model
        self.world_model.eval()
        self.K = list(world_model.K)
        self.num_levels = len(self.K)
        self.D = int(world_model.D)
        self.num_members = int(getattr(world_model, "num_members", 1))

        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.pop_size = int(pop_size)
        self.n_iter = int(n_iter)
        self.init_std = float(init_std)
        self.action_low = action_low
        self.action_high = action_high
        self.drop_tail_on_coarsen = bool(drop_tail_on_coarsen)
        self.warm_start = bool(warm_start)

        self.device = device or torch.device("cpu")

        self.objective_cfg = objective_cfg or {}
        self.metric = str(self.objective_cfg.get("latent_metric", "l2")).lower()
        self.terminal_weight = float(self.objective_cfg.get("terminal_weight", 1.0))
        self.running_weight = float(self.objective_cfg.get("running_weight", 0.0))
        self.action_l2_weight = float(self.objective_cfg.get("action_l2_weight", 0.0))

        self.core = SharedCEMCore(
            horizon=self.horizon,
            action_dim=self.action_dim,
            pop_size=self.pop_size,
            elite_frac=float(elite_frac),
            n_iter=self.n_iter,
            init_std=self.init_std,
            action_low=self.action_low,
            action_high=self.action_high,
            fidelity_cfg=fidelity_cfg,
            num_levels=self.num_levels,
            rollout_modes={"fixed", "linear", "uncertainty_downshift"},
            device=self.device,
        )

        self.rollout_mode = self.core.rollout_mode
        uncertainty_cfg = SharedCEMCore.as_cfg_dict(
            self.core.rollout_cfg.get("uncertainty", {}),
            "fidelity.rollout.uncertainty",
        )
        self.uncertainty_criterion = str(uncertainty_cfg.get("criterion", "mean")).lower()
        self.uncertainty_threshold = float(uncertainty_cfg.get("threshold", 0.05))
        self.uncertainty_percentile = float(uncertainty_cfg.get("percentile", 0.8))
        self.uncertainty_min_level = self.core.resolve_level_spec(
            uncertainty_cfg.get("min_level", "coarsest"),
            base_level_idx=None,
            field_name="fidelity.rollout.uncertainty.min_level",
        )
        self.max_downshifts_per_step = int(uncertainty_cfg.get("max_downshifts_per_step", 1))

        if self.metric not in {"l1", "l2"}:
            raise ValueError(f"Unsupported latent_metric '{self.metric}'. Use 'l1' or 'l2'.")
        if self.rollout_mode == "uncertainty_downshift" and self.num_members < 2:
            raise ValueError(
                "rollout.mode=uncertainty_downshift requires a world-model ensemble "
                "with at least 2 members."
            )
        if self.uncertainty_criterion not in {"mean", "percentile"}:
            raise ValueError(
                f"Unknown uncertainty criterion '{self.uncertainty_criterion}'. "
                "Use 'mean' or 'percentile'."
            )
        if not (0.0 <= self.uncertainty_percentile <= 1.0):
            raise ValueError(
                f"uncertainty percentile must be in [0,1], got {self.uncertainty_percentile}"
            )
        if self.max_downshifts_per_step <= 0:
            raise ValueError(
                f"max_downshifts_per_step must be > 0, got {self.max_downshifts_per_step}"
            )

    def _predict_next_stats(
        self,
        level: int,
        z: torch.Tensor,
        a_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.world_model, "predict_next_stats"):
            mu, var = self.world_model.predict_next_stats(level, z, a_t)
            return mu, torch.clamp_min(var, 0.0)
        #a_t = (a_t - self.ACTION_MEAN) / self.ACTION_STD
        mu = self.world_model.predict_next(level, z, a_t)
        var = torch.zeros_like(mu)
        return mu, var

    def _uncertainty_score(
        self,
        var_cur: torch.Tensor,  # (P,k_cur)
        k_next: int,
        k_cur: int,
    ) -> torch.Tensor:
        if k_next >= k_cur:
            return torch.tensor(0.0, device=var_cur.device)
        tail = var_cur[:, k_next:k_cur]  # (P, dropped_dims)
        per_candidate = tail.mean(dim=1)
        if self.uncertainty_criterion == "mean":
            return per_candidate.mean()
        return torch.quantile(per_candidate, q=self.uncertainty_percentile)

    def _latent_distance(self, z: torch.Tensor, z_goal: torch.Tensor, k: int) -> torch.Tensor:
        diff = z[:, :k] - z_goal[:, :k]
        if self.metric == "l1":
            return diff.abs().mean(dim=1)
        # RMS distance keeps scale more stable across changing latent prefix sizes.
        return torch.sqrt(diff.pow(2).mean(dim=1) + 1e-8)

    def reset(self) -> None:
        self.core.reset_distribution()

    @torch.no_grad()
    def _evaluate_population(
        self,
        actions: torch.Tensor,          # (P,H,A)
        z0: torch.Tensor,               # (1,D)
        z_goal: torch.Tensor,           # (1,D)
        base_level_idx: int,
        rollout_levels: List[int],
    ) -> tuple[torch.Tensor, List[int], int]:
        if self.rollout_mode == "uncertainty_downshift":
            return self._evaluate_population_uncertainty(actions, z0, z_goal, base_level_idx)

        P = actions.shape[0]
        z = z0.expand(P, -1).clone()
        z_goal_exp = z_goal.expand(P, -1)
        running = torch.zeros(P, device=self.device)
        bits_used = 0

        for t in range(self.horizon):
            li = rollout_levels[t]
            k = self.K[li]
            z_next_k, _ = self._predict_next_stats(li, z, actions[:, t, :])
            bits_used += int(P * k * 32 * max(1, self.num_members))

            z_next = z.clone()
            z_next[:, :k] = z_next_k
            if self.drop_tail_on_coarsen and k < self.D:
                z_next[:, k:] = 0.0
            z = z_next

            if self.running_weight > 0.0:
                running = running + self.running_weight * self._latent_distance(z, z_goal_exp, k)

        k_terminal = self.K[base_level_idx]
        terminal = self.terminal_weight * self._latent_distance(z, z_goal_exp, k_terminal)
        cost = terminal + running

        if self.action_l2_weight > 0.0:
            action_penalty = actions.pow(2).mean(dim=(1, 2))
            cost = cost + self.action_l2_weight * action_penalty

        return cost, rollout_levels, bits_used

    @torch.no_grad()
    def _evaluate_population_uncertainty(
        self,
        actions: torch.Tensor,          # (P,H,A)
        z0: torch.Tensor,               # (1,D)
        z_goal: torch.Tensor,           # (1,D)
        base_level_idx: int,
    ) -> tuple[torch.Tensor, List[int], int]:
        P = actions.shape[0]
        z = z0.expand(P, -1).clone()
        z_goal_exp = z_goal.expand(P, -1)
        running = torch.zeros(P, device=self.device)
        bits_used = 0

        current_level = base_level_idx
        min_level = self.uncertainty_min_level
        rollout_levels: List[int] = []

        for t in range(self.horizon):
            a_t = actions[:, t, :]

            for _ in range(self.max_downshifts_per_step):
                if current_level <= min_level:
                    break
                k_cur = self.K[current_level]
                k_next = self.K[current_level - 1]
                _, var_cur = self._predict_next_stats(current_level, z, a_t)
                bits_used += int(P * k_cur * 32 * max(1, self.num_members))
                score = self._uncertainty_score(var_cur, k_next=k_next, k_cur=k_cur)
                if float(score.item()) > self.uncertainty_threshold:
                    current_level -= 1
                else:
                    break

            k = self.K[current_level]
            z_next_k, _ = self._predict_next_stats(current_level, z, a_t)
            bits_used += int(P * k * 32 * max(1, self.num_members))

            z_next = z.clone()
            z_next[:, :k] = z_next_k
            if self.drop_tail_on_coarsen and k < self.D:
                z_next[:, k:] = 0.0
            z = z_next

            if self.running_weight > 0.0:
                running = running + self.running_weight * self._latent_distance(z, z_goal_exp, k)
            rollout_levels.append(current_level)

        k_terminal = self.K[current_level]
        terminal = self.terminal_weight * self._latent_distance(z, z_goal_exp, k_terminal)
        cost = terminal + running

        if self.action_l2_weight > 0.0:
            action_penalty = actions.pow(2).mean(dim=(1, 2))
            cost = cost + self.action_l2_weight * action_penalty

        return cost, rollout_levels, bits_used

    @torch.no_grad()
    def plan(
        self,
        z0: torch.Tensor,       # (1,D)
        z_goal: torch.Tensor,   # (1,D)
        mpc_progress: float = 0.0,
        warm_start_steps: int = 0,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, LatentCEMInfo]:
        t0 = time.perf_counter()
        if z0.shape[-1] != self.D or z_goal.shape[-1] != self.D:
            raise ValueError(
                f"Latent dim mismatch: expected D={self.D}, got z0={tuple(z0.shape)}, z_goal={tuple(z_goal.shape)}"
            )

        def _evaluate(
            actions: torch.Tensor,
            base_level_idx: int,
            rollout_levels: List[int],
            iter_idx: int,
        ) -> tuple[torch.Tensor, List[int], int]:
            del iter_idx
            return self._evaluate_population(actions, z0, z_goal, base_level_idx, rollout_levels)

        action_seq, final_level_idx, final_rollout_levels, total_bits = self.core.optimize(
            mpc_progress=mpc_progress,
            evaluate_population=_evaluate,
            warm_start=self.warm_start,
            shift_steps=int(warm_start_steps),
            rng_seed=None if seed is None else int(seed),
        )

        info = LatentCEMInfo(
            base_level_idx=final_level_idx,
            base_k=self.K[final_level_idx],
            rollout_level_indices=final_rollout_levels,
            bits_used_estimate=int(total_bits),
            plan_time_sec=float(time.perf_counter() - t0),
        )
        return action_seq, info
