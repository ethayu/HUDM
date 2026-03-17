from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from planning.latent_cem import LatentCEMInfo, LatentCEMPlanner


@dataclass
class BatchedLatentPlanResult:
    action_seq: torch.Tensor
    info: LatentCEMInfo


class BatchedLatentCEMPlanner:
    """
    Evaluate multiple fidelity schedules against the same world-model state in one pass.

    Each schedule keeps an independent CEM distribution/state while sharing batched world
    model calls whenever schedules are at the same latent level.
    """

    def __init__(
        self,
        world_model,
        fidelity_cfgs: Sequence[Dict[str, Any]],
        horizon: int,
        action_dim: int,
        pop_size: int = 256,
        elite_frac: float = 0.1,
        n_iter: int = 5,
        init_std: float = 1.0,
        action_low: Optional[float] = None,
        action_high: Optional[float] = None,
        objective_cfg: Optional[Dict[str, Any]] = None,
        drop_tail_on_coarsen: bool = True,
        warm_start: bool = True,
        device: Optional[torch.device] = None,
    ):
        if len(fidelity_cfgs) <= 0:
            raise ValueError("BatchedLatentCEMPlanner requires at least one fidelity config.")
        self.planners = [
            LatentCEMPlanner(
                world_model=world_model,
                horizon=horizon,
                action_dim=action_dim,
                pop_size=pop_size,
                elite_frac=elite_frac,
                n_iter=n_iter,
                init_std=init_std,
                action_low=action_low,
                action_high=action_high,
                objective_cfg=objective_cfg,
                fidelity_cfg=fidelity_cfg,
                drop_tail_on_coarsen=drop_tail_on_coarsen,
                warm_start=warm_start,
                device=device,
            )
            for fidelity_cfg in fidelity_cfgs
        ]
        self.world_model = world_model
        self.device = self.planners[0].device
        self.horizon = self.planners[0].horizon
        self.pop_size = self.planners[0].pop_size
        self.n_iter = self.planners[0].n_iter
        self.num_schedules = len(self.planners)
        self.K = list(self.planners[0].K)
        self.D = int(self.planners[0].D)
        self.num_members = int(getattr(world_model, "num_members", 1))
        self.terminal_weight = float(self.planners[0].terminal_weight)
        self.running_weight = float(self.planners[0].running_weight)
        self.action_l2_weight = float(self.planners[0].action_l2_weight)
        self.metric = str(self.planners[0].metric)
        self.drop_tail_on_coarsen = bool(drop_tail_on_coarsen)

        for planner in self.planners[1:]:
            if planner.horizon != self.horizon or planner.pop_size != self.pop_size or planner.n_iter != self.n_iter:
                raise ValueError("All batched schedules must share horizon, pop_size, and n_iter.")
            if planner.K != self.K or planner.D != self.D:
                raise ValueError("All batched schedules must share world-model latent structure.")
            if (
                float(planner.terminal_weight) != self.terminal_weight
                or float(planner.running_weight) != self.running_weight
                or float(planner.action_l2_weight) != self.action_l2_weight
                or str(planner.metric) != self.metric
            ):
                raise ValueError("Batched schedules must share objective weights and metric.")

    def reset(self) -> None:
        for planner in self.planners:
            planner.reset()

    def _latent_distance(self, z: torch.Tensor, z_goal: torch.Tensor, k: int) -> torch.Tensor:
        diff = z[:, :k] - z_goal[:, :k]
        if self.metric == "l1":
            return diff.abs().mean(dim=1)
        return torch.sqrt(diff.pow(2).mean(dim=1) + 1e-8)

    @torch.no_grad()
    def _evaluate_standard_batch(
        self,
        schedule_indices: List[int],
        actions: torch.Tensor,   # (S,P,H,A)
        z0: torch.Tensor,        # (S,D)
        z_goal: torch.Tensor,    # (S,D)
        rollout_levels_by_schedule: List[List[int]],
        base_levels: List[int],
    ) -> tuple[torch.Tensor, List[List[int]], List[int]]:
        s_count = len(schedule_indices)
        costs = torch.zeros((s_count, self.pop_size), device=self.device)
        bits_used = [0 for _ in range(s_count)]
        z = z0.unsqueeze(1).expand(s_count, self.pop_size, -1).clone()
        running = torch.zeros((s_count, self.pop_size), device=self.device)

        for t in range(self.horizon):
            by_level: dict[int, List[int]] = {}
            for local_idx, rollout_levels in enumerate(rollout_levels_by_schedule):
                level_idx = int(rollout_levels[t])
                by_level.setdefault(level_idx, []).append(local_idx)

            for level_idx, local_scheds in by_level.items():
                sched_tensor = torch.as_tensor(local_scheds, device=self.device, dtype=torch.long)
                z_group = z.index_select(0, sched_tensor).reshape(-1, self.D)
                a_group = actions.index_select(0, sched_tensor)[:, :, t, :].reshape(-1, actions.shape[-1])
                mu_group, _ = self.planners[0]._predict_next_stats(level_idx, z_group, a_group)
                k = self.K[level_idx]
                mu_group = mu_group.reshape(len(local_scheds), self.pop_size, k)
                for offset, local_idx in enumerate(local_scheds):
                    z_next = z[local_idx].clone()
                    z_next[:, :k] = mu_group[offset]
                    if self.drop_tail_on_coarsen and k < self.D:
                        z_next[:, k:] = 0.0
                    z[local_idx] = z_next
                    bits_used[local_idx] += int(self.pop_size * k * 32 * max(1, self.num_members))
                    if self.running_weight > 0.0:
                        running[local_idx] = running[local_idx] + self.running_weight * self._latent_distance(
                            z_next,
                            z_goal[local_idx].unsqueeze(0).expand(self.pop_size, -1),
                            k,
                        )

        for local_idx, base_level_idx in enumerate(base_levels):
            k_terminal = self.K[int(base_level_idx)]
            terminal = self.terminal_weight * self._latent_distance(
                z[local_idx],
                z_goal[local_idx].unsqueeze(0).expand(self.pop_size, -1),
                k_terminal,
            )
            costs[local_idx] = terminal + running[local_idx]

        if self.action_l2_weight > 0.0:
            action_penalty = actions.pow(2).mean(dim=(2, 3))
            costs = costs + self.action_l2_weight * action_penalty

        return costs, rollout_levels_by_schedule, bits_used

    @torch.no_grad()
    def plan_batch(
        self,
        z0: torch.Tensor,
        z_goal: torch.Tensor,
        mpc_progress: float = 0.0,
        warm_start_steps: int = 0,
        seeds: Optional[Sequence[Optional[int]]] = None,
    ) -> List[BatchedLatentPlanResult]:
        t0 = time.perf_counter()
        if z0.shape[-1] != self.D or z_goal.shape[-1] != self.D:
            raise ValueError(
                f"Latent dim mismatch: expected D={self.D}, got z0={tuple(z0.shape)}, z_goal={tuple(z_goal.shape)}"
            )
        if z0.ndim != 2 or z_goal.ndim != 2:
            raise ValueError(
                f"plan_batch expects z0 and z_goal with rank 2, got z0={tuple(z0.shape)}, z_goal={tuple(z_goal.shape)}"
            )
        if z0.shape[0] not in {1, self.num_schedules}:
            raise ValueError(
                f"z0 batch dimension must be 1 or num_schedules={self.num_schedules}, got {z0.shape[0]}"
            )
        if z_goal.shape[0] not in {1, self.num_schedules}:
            raise ValueError(
                f"z_goal batch dimension must be 1 or num_schedules={self.num_schedules}, got {z_goal.shape[0]}"
            )
        z0_sched = z0.expand(self.num_schedules, -1) if z0.shape[0] == 1 else z0
        z_goal_sched = z_goal.expand(self.num_schedules, -1) if z_goal.shape[0] == 1 else z_goal
        if seeds is None:
            seeds = [None] * self.num_schedules
        if len(seeds) != self.num_schedules:
            raise ValueError(f"Expected {self.num_schedules} seeds, got {len(seeds)}")

        generators: List[Optional[torch.Generator]] = []
        total_bits = [0 for _ in range(self.num_schedules)]
        eff_rollout_levels: List[List[int]] = [[] for _ in range(self.num_schedules)]

        for planner, seed in zip(self.planners, seeds):
            planner.core.initialize_distribution(
                warm_start=planner.warm_start,
                shift_steps=int(warm_start_steps),
            )
            generators.append(planner.core.make_generator(seed))

        for it in range(self.n_iter):
            cem_progress = 1.0 if self.n_iter == 1 else it / (self.n_iter - 1)
            base_levels = [planner.core.base_level_index(mpc_progress, cem_progress) for planner in self.planners]
            rollout_levels = [planner.core.rollout_level_indices(base_levels[idx]) for idx, planner in enumerate(self.planners)]
            actions = torch.stack(
                [planner.core.sample_population(generator=generators[idx]) for idx, planner in enumerate(self.planners)],
                dim=0,
            )

            costs = torch.zeros((self.num_schedules, self.pop_size), device=self.device)
            active_rollout_levels = [list(levels) for levels in rollout_levels]

            standard_idxs = [
                idx for idx, planner in enumerate(self.planners)
                if planner.rollout_mode != "uncertainty_downshift"
            ]
            uncertainty_idxs = [
                idx for idx, planner in enumerate(self.planners)
                if planner.rollout_mode == "uncertainty_downshift"
            ]

            if standard_idxs:
                costs_std, used_levels_std, bits_std = self._evaluate_standard_batch(
                    schedule_indices=standard_idxs,
                    actions=actions.index_select(0, torch.as_tensor(standard_idxs, device=self.device)),
                    z0=z0_sched.index_select(0, torch.as_tensor(standard_idxs, device=self.device)),
                    z_goal=z_goal_sched.index_select(0, torch.as_tensor(standard_idxs, device=self.device)),
                    rollout_levels_by_schedule=[rollout_levels[idx] for idx in standard_idxs],
                    base_levels=[base_levels[idx] for idx in standard_idxs],
                )
                for local_idx, sched_idx in enumerate(standard_idxs):
                    costs[sched_idx] = costs_std[local_idx]
                    active_rollout_levels[sched_idx] = list(used_levels_std[local_idx])
                    total_bits[sched_idx] += int(bits_std[local_idx])

            for sched_idx in uncertainty_idxs:
                planner = self.planners[sched_idx]
                c, used_levels, bits = planner._evaluate_population_uncertainty(
                    actions[sched_idx],
                    z0_sched[sched_idx : sched_idx + 1],
                    z_goal_sched[sched_idx : sched_idx + 1],
                    base_levels[sched_idx],
                )
                costs[sched_idx] = c
                active_rollout_levels[sched_idx] = list(used_levels)
                total_bits[sched_idx] += int(bits)

            for sched_idx, planner in enumerate(self.planners):
                planner.core.update_distribution(actions[sched_idx], costs[sched_idx])
                eff_rollout_levels[sched_idx] = [int(x) for x in active_rollout_levels[sched_idx]]

        results: List[BatchedLatentPlanResult] = []
        elapsed = float(time.perf_counter() - t0)
        for sched_idx, planner in enumerate(self.planners):
            final_level_idx = planner.core.base_level_index(mpc_progress, 1.0)
            if planner.rollout_mode == "uncertainty_downshift":
                final_rollout_levels = list(eff_rollout_levels[sched_idx])
                if len(final_rollout_levels) > 0:
                    final_level_idx = int(final_rollout_levels[-1])
            else:
                final_rollout_levels = planner.core.rollout_level_indices(final_level_idx)

            z_cur = z0_sched[sched_idx : sched_idx + 1].clone()
            z_goal_cur = z_goal_sched[sched_idx : sched_idx + 1]
            action_seq = planner.core.mu.detach().to(self.device)
            per_step_losses: List[float] = []
            for t in range(self.horizon):
                li = int(final_rollout_levels[t]) if t < len(final_rollout_levels) else int(final_level_idx)
                k = planner.K[li]
                a_t = action_seq[t : t + 1, :]
                z_next_k, _ = planner._predict_next_stats(li, z_cur, a_t)
                z_next = z_cur.clone()
                z_next[:, :k] = z_next_k
                if planner.drop_tail_on_coarsen and k < planner.D:
                    z_next[:, k:] = 0.0
                z_cur = z_next
                per_step_losses.append(float(planner._latent_distance(z_cur, z_goal_cur, k).item()))
            info = LatentCEMInfo(
                base_level_idx=int(final_level_idx),
                base_k=self.K[int(final_level_idx)],
                rollout_level_indices=[int(x) for x in final_rollout_levels],
                rollout_latent_losses=per_step_losses,
                bits_used_estimate=int(total_bits[sched_idx]),
                plan_time_sec=elapsed,
            )
            results.append(
                BatchedLatentPlanResult(
                    action_seq=planner.core.mu.detach().cpu(),
                    info=info,
                )
            )
        return results
