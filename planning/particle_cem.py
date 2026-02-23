from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from planning.cem_core import SharedCEMCore

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


@dataclass
class ParticleCEMInfo:
    base_level_idx: int
    rollout_level_indices: List[int]
    bits_used_estimate: int
    plan_time_sec: float
    base_spacing: float
    base_num_particles: int


class ParticleCEMPlanner:
    """
    CEM planner that evaluates candidate action sequences in Warp particle simulation.
    """

    def __init__(
        self,
        particle_backend,
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
        particle_env_cfg: Optional[Dict[str, Any]] = None,
        warm_start: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.backend = particle_backend
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.pop_size = int(pop_size)
        self.n_iter = int(n_iter)
        self.init_std = float(init_std)
        self.action_low = action_low
        self.action_high = action_high
        self.warm_start = bool(warm_start)
        self.device = device or torch.device("cpu")

        self.objective_cfg = objective_cfg or {}
        self.metric = str(self.objective_cfg.get("latent_metric", "l2")).lower()
        self.terminal_weight = float(self.objective_cfg.get("terminal_weight", 1.0))
        self.running_weight = float(self.objective_cfg.get("running_weight", 0.0))
        self.action_l2_weight = float(self.objective_cfg.get("action_l2_weight", 0.0))

        self.eef_weight = float(self.objective_cfg.get("eef_weight", 1.0))
        self.block_pos_weight = float(self.objective_cfg.get("block_pos_weight", 1.0))
        self.block_angle_weight = float(self.objective_cfg.get("block_angle_weight", 0.1))
        self.state_l2_weight = float(self.objective_cfg.get("state_l2_weight", 0.0))

        fidelity_cfg = fidelity_cfg or {}
        num_levels = int(fidelity_cfg.get("num_levels", 4))
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
            num_levels=num_levels,
            rollout_modes={"fixed"},
            device=self.device,
        )
        if self.core.rollout_mode != "fixed":
            raise ValueError(
                "particle_sim backend currently supports only fidelity.rollout.mode='fixed'."
            )

        self.particle_env_cfg = particle_env_cfg or {}
        self.rollout_samples = int(self.particle_env_cfg.get("rollout_samples", 1))
        self.objective_space = str(self.particle_env_cfg.get("objective_space", "state")).lower()
        self.progress = bool(self.particle_env_cfg.get("progress", True))
        self.progress_leave = bool(self.particle_env_cfg.get("progress_leave", False))

        if self.rollout_samples <= 0:
            raise ValueError(f"particle_env.rollout_samples must be > 0, got {self.rollout_samples}")
        if self.objective_space not in {"image", "state"}:
            raise ValueError(
                f"particle_env.objective_space must be 'image' or 'state', got {self.objective_space}"
            )
        if self.metric not in {"l1", "l2"}:
            raise ValueError(f"Unsupported objective latent_metric '{self.metric}'. Use 'l1' or 'l2'.")

    @staticmethod
    def _angle_delta(a: float, b: float) -> float:
        d = a - b
        return abs(math.atan2(math.sin(d), math.cos(d)))

    def _state_cost(self, final_state: np.ndarray, goal_state: np.ndarray, actions: np.ndarray) -> float:
        eef = float(np.linalg.norm(final_state[:2] - goal_state[:2]))
        block_pos = float(np.linalg.norm(final_state[2:4] - goal_state[2:4]))
        block_ang = self._angle_delta(float(final_state[4]), float(goal_state[4]))
        cost = (
            self.eef_weight * eef
            + self.block_pos_weight * block_pos
            + self.block_angle_weight * block_ang
        )
        if self.state_l2_weight > 0.0:
            d = min(final_state.shape[0], goal_state.shape[0])
            cost += self.state_l2_weight * float(np.linalg.norm(final_state[:d] - goal_state[:d]))
        if self.action_l2_weight > 0.0:
            cost += self.action_l2_weight * float(np.mean(actions ** 2))
        return cost

    @staticmethod
    def _to_float_image(x: np.ndarray | torch.Tensor) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        img = np.asarray(x, dtype=np.float32)
        if img.ndim != 3:
            raise ValueError(f"Expected image shape (H,W,C), got {tuple(img.shape)}")
        if float(img.max()) > 1.5:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    def _image_distance(self, img: np.ndarray, goal_img: np.ndarray) -> float:
        if img.shape != goal_img.shape:
            raise ValueError(
                f"Image shape mismatch in particle objective: img={img.shape}, goal={goal_img.shape}"
            )
        diff = img - goal_img
        if self.metric == "l1":
            return float(np.mean(np.abs(diff)))
        return float(np.sqrt(np.mean(diff ** 2) + 1e-8))

    def _goal_visuals_by_level(
        self,
        goal_state: np.ndarray,
        rollout_levels: List[int],
        seed: int,
    ) -> Dict[int, np.ndarray]:
        by_level: Dict[int, np.ndarray] = {}
        unique_levels = sorted({int(li) for li in rollout_levels})
        for li in unique_levels:
            self.backend.set_planning_fidelity_level(li)
            obs, _ = self.backend.prepare(seed=seed + 17 * li, init_state=goal_state, goal_state=goal_state)
            visual = obs.get("visual", None)
            if visual is None:
                raise ValueError("particle_sim backend requires observations to include 'visual'.")
            by_level[li] = self._to_float_image(visual)
        return by_level

    def _image_cost(
        self,
        final_dist: float,
        running_dists: List[float],
        actions: np.ndarray,
    ) -> float:
        cost = self.terminal_weight * float(final_dist)
        if self.running_weight > 0.0 and len(running_dists) > 0:
            cost += self.running_weight * float(np.sum(running_dists))
        if self.action_l2_weight > 0.0:
            cost += self.action_l2_weight * float(np.mean(actions ** 2))
        return cost

    def _make_rollout_progress_bar(self, total_rollouts: int, iter_idx: int):
        if not self.progress or total_rollouts <= 0:
            return None
        if _tqdm is None:
            print(
                f"[cem] iter {iter_idx + 1}/{self.n_iter}: simulating "
                f"{total_rollouts} rollouts"
            )
            return None
        return _tqdm(
            total=total_rollouts,
            desc=f"CEM {iter_idx + 1}/{self.n_iter} rollouts",
            leave=self.progress_leave,
            dynamic_ncols=True,
            mininterval=0.1,
            disable=not sys.stderr.isatty(),
        )

    def _candidate_rollout(
        self,
        init_state: np.ndarray,
        goal_state: np.ndarray,
        actions: np.ndarray,
        rollout_levels: List[int],
        seed: int,
        goal_visual_by_level: Dict[int, np.ndarray],
    ) -> tuple[float, int]:
        # Fixed rollout mode: all entries are the same level.
        level_idx = int(rollout_levels[0])
        self.backend.set_planning_fidelity_level(level_idx)
        obs, state = self.backend.prepare(seed=seed, init_state=init_state, goal_state=goal_state)
        del obs

        bits_used = 0
        running_dists: List[float] = []
        final_dist = 0.0

        for t in range(self.horizon):
            obs_t, _, done, info = self.backend.step(actions[t])
            state = np.asarray(info["state"], dtype=np.float32)
            bits_used += int(self.backend.num_particles(level_idx=level_idx) * 2 * 32)

            if self.objective_space == "image":
                visual = obs_t.get("visual", None)
                if visual is None:
                    raise ValueError("particle_sim image objective requires visual observations.")
                img_t = self._to_float_image(visual)
                d = self._image_distance(img_t, goal_visual_by_level[int(level_idx)])
                final_dist = float(d)
                if self.running_weight > 0.0:
                    running_dists.append(float(d))

            if done:
                break

        if self.objective_space == "image":
            cost = self._image_cost(final_dist=final_dist, running_dists=running_dists, actions=actions)
        else:
            cost = self._state_cost(np.asarray(state), np.asarray(goal_state), actions)
        return cost, bits_used

    @torch.no_grad()
    def plan(
        self,
        init_state: np.ndarray,
        goal_state: np.ndarray,
        mpc_progress: float = 0.0,
        seed: int = 0,
        warm_start_steps: int = 0,
    ) -> tuple[torch.Tensor, ParticleCEMInfo]:
        t0 = time.perf_counter()

        init_state_np = np.asarray(init_state, dtype=np.float32)
        goal_state_np = np.asarray(goal_state, dtype=np.float32)

        def _evaluate(
            actions: torch.Tensor,
            base_level_idx: int,
            rollout_levels: List[int],
            iter_idx: int,
        ) -> tuple[torch.Tensor, List[int], int]:
            del base_level_idx
            actions_np = actions.detach().cpu().numpy()
            pop = actions_np.shape[0]

            if self.objective_space == "image":
                goal_visual_by_level = self._goal_visuals_by_level(
                    goal_state=goal_state_np,
                    rollout_levels=rollout_levels,
                    seed=int(seed + 700001 * iter_idx),
                )
            else:
                goal_visual_by_level = {}

            costs = np.zeros((pop,), dtype=np.float32)
            bits_iter = 0
            total_rollouts = int(pop * self.rollout_samples)
            pbar = self._make_rollout_progress_bar(total_rollouts, iter_idx)
            try:
                for p in range(pop):
                    sample_costs = []
                    for s_idx in range(self.rollout_samples):
                        c, b = self._candidate_rollout(
                            init_state=init_state_np,
                            goal_state=goal_state_np,
                            actions=actions_np[p],
                            rollout_levels=rollout_levels,
                            seed=int(seed + 1000003 * iter_idx + 1009 * p + 7 * s_idx),
                            goal_visual_by_level=goal_visual_by_level,
                        )
                        sample_costs.append(c)
                        bits_iter += int(b)
                    costs[p] = float(np.mean(sample_costs))
                    if pbar is not None:
                        pbar.update(self.rollout_samples)
            finally:
                if pbar is not None:
                    pbar.close()

            return torch.as_tensor(costs, device=self.device), rollout_levels, bits_iter

        action_seq, final_level_idx, final_rollout_levels, total_bits = self.core.optimize(
            mpc_progress=mpc_progress,
            evaluate_population=_evaluate,
            warm_start=self.warm_start,
            shift_steps=int(warm_start_steps),
        )

        base_spacing = float(self.backend.spacing(final_level_idx))
        base_num_particles = int(self.backend.num_particles(final_level_idx))
        info = ParticleCEMInfo(
            base_level_idx=int(final_level_idx),
            rollout_level_indices=final_rollout_levels,
            bits_used_estimate=int(total_bits),
            plan_time_sec=float(time.perf_counter() - t0),
            base_spacing=base_spacing,
            base_num_particles=base_num_particles,
        )
        return action_seq, info
