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
    start_level_idx: int = -1
    batch_impl: str = "serial"


class BatchedParticlePlannerUnavailable(RuntimeError):
    """Raised when the particle backend cannot support the batched planner path."""


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
        num_levels: Optional[int] = None,
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
        if num_levels is None:
            num_levels = getattr(self.backend, "num_levels", None)
        if num_levels is None:
            num_levels = getattr(self.backend, "_planning_fidelity_num_levels", None)
        if num_levels is None:
            raise ValueError("ParticleCEMPlanner requires the particle backend to define num_levels.")
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
            num_levels=int(num_levels),
            rollout_modes={"fixed", "linear"},
            device=self.device,
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

    def reset(self) -> None:
        self.core.reset_distribution()

    @staticmethod
    def _prepare_inject_actions_tensor(
        gt_action_trajectory: Any,
        horizon: int,
        action_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        arr = np.asarray(gt_action_trajectory, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != int(action_dim):
            raise ValueError(
                f"gt_action_trajectory must have shape (T, {action_dim}), got {arr.shape}"
            )
        out = np.zeros((int(horizon), int(action_dim)), dtype=np.float32)
        n = min(int(horizon), int(arr.shape[0]))
        if n > 0:
            out[:n] = arr[:n]
        return torch.as_tensor(out, device=device, dtype=torch.float32)

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
    def _particle_angle_delta(
        cur_particles_xy: np.ndarray,
        goal_particles_xy: np.ndarray,
    ) -> float:
        cur = np.asarray(cur_particles_xy, dtype=np.float32).reshape(-1, 2)
        goal = np.asarray(goal_particles_xy, dtype=np.float32).reshape(-1, 2)
        if cur.shape != goal.shape:
            raise ValueError(
                f"Particle-cloud shape mismatch in particle objective: cur={cur.shape}, goal={goal.shape}"
            )
        if cur.shape[0] <= 1:
            return 0.0
        cur_centered = cur - cur.mean(axis=0, keepdims=True)
        goal_centered = goal - goal.mean(axis=0, keepdims=True)
        if float(np.linalg.norm(cur_centered)) <= 1e-8 or float(np.linalg.norm(goal_centered)) <= 1e-8:
            return 0.0
        a = float(np.sum(goal_centered[:, 0] * cur_centered[:, 1] - goal_centered[:, 1] * cur_centered[:, 0]))
        b = float(np.sum(goal_centered[:, 0] * cur_centered[:, 0] + goal_centered[:, 1] * cur_centered[:, 1]))
        return abs(math.atan2(a, b))

    @classmethod
    def _particle_state_terms(
        cls,
        cur_cloud: Dict[str, np.ndarray],
        goal_cloud: Dict[str, np.ndarray],
    ) -> tuple[float, float, float, float]:
        cur_pusher = np.asarray(cur_cloud["pusher_xy"], dtype=np.float32).reshape(2)
        goal_pusher = np.asarray(goal_cloud["pusher_xy"], dtype=np.float32).reshape(2)
        cur_particles = np.asarray(cur_cloud["particle_xy"], dtype=np.float32).reshape(-1, 2)
        goal_particles = np.asarray(goal_cloud["particle_xy"], dtype=np.float32).reshape(-1, 2)
        if cur_particles.shape != goal_particles.shape:
            raise ValueError(
                f"Particle-cloud shape mismatch in particle objective: cur={cur_particles.shape}, goal={goal_particles.shape}"
            )

        eef = float(np.linalg.norm(cur_pusher - goal_pusher))
        cur_center = cur_particles.mean(axis=0) if cur_particles.shape[0] > 0 else np.zeros((2,), dtype=np.float32)
        goal_center = goal_particles.mean(axis=0) if goal_particles.shape[0] > 0 else np.zeros((2,), dtype=np.float32)
        block_pos = float(np.linalg.norm(cur_center - goal_center))
        block_ang = cls._particle_angle_delta(cur_particles, goal_particles)

        flat_cur = np.concatenate([cur_pusher.reshape(-1), cur_particles.reshape(-1)], axis=0)
        flat_goal = np.concatenate([goal_pusher.reshape(-1), goal_particles.reshape(-1)], axis=0)
        state_l2 = float(np.sqrt(np.mean((flat_cur - flat_goal) ** 2)))
        return eef, block_pos, block_ang, state_l2

    def _particle_state_cost(
        self,
        cur_cloud: Dict[str, np.ndarray],
        goal_cloud: Dict[str, np.ndarray],
        actions: np.ndarray,
    ) -> float:
        eef, block_pos, block_ang, state_l2 = self._particle_state_terms(cur_cloud, goal_cloud)
        cost = (
            self.eef_weight * eef
            + self.block_pos_weight * block_pos
            + self.block_angle_weight * block_ang
        )
        if self.state_l2_weight > 0.0:
            cost += self.state_l2_weight * state_l2
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
            obs, _ = self.backend.prepare(
                seed=seed + 17 * li,
                init_state=goal_state,
                goal_state=goal_state,
                with_visual=True,
            )
            visual = obs.get("visual", None)
            if visual is None:
                raise ValueError("particle_sim backend requires observations to include 'visual'.")
            by_level[li] = self._to_float_image(visual)
        return by_level

    def _goal_particle_clouds_by_level(
        self,
        goal_state: np.ndarray,
        rollout_levels: List[int],
        seed: int,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        by_level: Dict[int, Dict[str, np.ndarray]] = {}
        unique_levels = sorted({int(li) for li in rollout_levels})
        for li in unique_levels:
            self.backend.set_planning_fidelity_level(li)
            self.backend.prepare(
                seed=seed + 17 * li,
                init_state=goal_state,
                goal_state=goal_state,
                with_visual=False,
            )
            by_level[li] = self.backend.current_particle_cloud_state(level_idx=li, pixel=True)
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
        goal_particle_cloud_by_level: Dict[int, Dict[str, np.ndarray]],
    ) -> tuple[float, int]:
        # Fixed rollout mode: all entries are the same level.
        level_idx = int(rollout_levels[0])
        self.backend.set_planning_fidelity_level(level_idx)
        need_visual = self.objective_space == "image"
        obs, state = self.backend.prepare(
            seed=seed,
            init_state=init_state,
            goal_state=goal_state,
            with_visual=need_visual,
        )
        del obs

        bits_used = 0
        running_dists: List[float] = []
        final_dist = 0.0
        cur_particle_cloud = (
            self.backend.current_particle_cloud_state(level_idx=level_idx, pixel=True)
            if self.objective_space == "state"
            else {}
        )

        for t in range(self.horizon):
            level_idx = int(rollout_levels[t])
            self.backend.set_planning_fidelity_level(level_idx)
            obs_t, _, done, info = self.backend.step(actions[t], with_visual=need_visual)
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
            else:
                cur_particle_cloud = self.backend.current_particle_cloud_state(level_idx=level_idx, pixel=True)

            if done:
                break

        if self.objective_space == "image":
            cost = self._image_cost(final_dist=final_dist, running_dists=running_dists, actions=actions)
        else:
            cost = self._particle_state_cost(
                cur_particle_cloud,
                goal_particle_cloud_by_level[int(level_idx)],
                actions,
            )
        return cost, bits_used

    @torch.no_grad()
    def plan(
        self,
        init_state: np.ndarray,
        goal_state: np.ndarray,
        mpc_progress: float = 0.0,
        seed: int = 0,
        warm_start_steps: int = 0,
        rng_seed: Optional[int] = None,
        gt_action_trajectory: Optional[Any] = None,
        gt_inject_count: int = 1,
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
                goal_particle_cloud_by_level = {}
            else:
                goal_visual_by_level = {}
                goal_particle_cloud_by_level = self._goal_particle_clouds_by_level(
                    goal_state=goal_state_np,
                    rollout_levels=rollout_levels,
                    seed=int(seed + 700001 * iter_idx),
                )

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
                            goal_particle_cloud_by_level=goal_particle_cloud_by_level,
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

        start_level_idx = self.core.base_level_index(mpc_progress, 0.0)
        inject_tensor: Optional[torch.Tensor] = None
        if gt_action_trajectory is not None:
            inject_tensor = self._prepare_inject_actions_tensor(
                gt_action_trajectory,
                self.horizon,
                self.action_dim,
                self.device,
            )
        action_seq, final_level_idx, final_rollout_levels, total_bits = self.core.optimize(
            mpc_progress=mpc_progress,
            evaluate_population=_evaluate,
            warm_start=self.warm_start,
            shift_steps=int(warm_start_steps),
            rng_seed=rng_seed,
            inject_actions=inject_tensor,
            inject_count=int(gt_inject_count),
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
            start_level_idx=int(start_level_idx),
            batch_impl="serial",
        )
        return action_seq, info


class BatchedParticleCEMPlanner(ParticleCEMPlanner):
    """
    Particle planner variant that evaluates the population through the backend's
    additive batch APIs while preserving the scalar planner contract.
    """

    @staticmethod
    def _image_distance_batch(
        imgs: np.ndarray,
        goal_img: np.ndarray,
        *,
        metric: str,
    ) -> np.ndarray:
        img_arr = np.asarray(imgs, dtype=np.float32)
        goal_arr = np.asarray(goal_img, dtype=np.float32)
        if img_arr.ndim != 4:
            raise ValueError(f"Expected image batch with rank 4, got {tuple(img_arr.shape)}")
        if goal_arr.shape != img_arr.shape[1:]:
            raise ValueError(
                f"Image shape mismatch in particle objective batch: imgs={img_arr.shape}, goal={goal_arr.shape}"
            )
        lane_needs_rescale = (np.max(img_arr, axis=(1, 2, 3), keepdims=True) > 1.5).astype(np.float32)
        img_arr = np.where(lane_needs_rescale > 0.0, img_arr / 255.0, img_arr)
        img_arr = np.clip(img_arr, 0.0, 1.0)
        if float(goal_arr.max()) > 1.5:
            goal_arr = goal_arr / 255.0
        goal_arr = np.clip(goal_arr, 0.0, 1.0)
        diff = img_arr - goal_arr[None, ...]
        if str(metric).lower() == "l1":
            return np.mean(np.abs(diff), axis=(1, 2, 3), dtype=np.float32).astype(np.float32)
        return np.sqrt(np.mean(diff ** 2, axis=(1, 2, 3), dtype=np.float32) + 1e-8).astype(np.float32)

    @classmethod
    def _particle_state_cost_batch(
        cls,
        cur_cloud_batch: Dict[str, np.ndarray],
        goal_cloud: Dict[str, np.ndarray],
        actions: np.ndarray,
        *,
        eef_weight: float,
        block_pos_weight: float,
        block_angle_weight: float,
        state_l2_weight: float,
        action_l2_weight: float,
    ) -> np.ndarray:
        cur_pusher = np.asarray(cur_cloud_batch["pusher_xy"], dtype=np.float32)
        cur_particles = np.asarray(cur_cloud_batch["particle_xy"], dtype=np.float32)
        goal_pusher = np.asarray(goal_cloud["pusher_xy"], dtype=np.float32).reshape(1, 2)
        goal_particles = np.asarray(goal_cloud["particle_xy"], dtype=np.float32).reshape(1, -1, 2)
        if cur_pusher.ndim != 2 or cur_particles.ndim != 3:
            raise ValueError(
                f"Unexpected particle-cloud batch shapes: pusher={cur_pusher.shape}, particles={cur_particles.shape}"
            )
        if cur_particles.shape[1:] != goal_particles.shape[1:]:
            raise ValueError(
                f"Particle-cloud shape mismatch in particle objective batch: cur={cur_particles.shape}, goal={goal_particles.shape}"
            )

        eef = np.linalg.norm(cur_pusher - goal_pusher, axis=1).astype(np.float32)
        cur_center = cur_particles.mean(axis=1)
        goal_center = goal_particles.mean(axis=1)
        block_pos = np.linalg.norm(cur_center - goal_center, axis=1).astype(np.float32)

        if cur_particles.shape[1] <= 1:
            block_ang = np.zeros((cur_particles.shape[0],), dtype=np.float32)
        else:
            cur_centered = cur_particles - cur_particles.mean(axis=1, keepdims=True)
            goal_centered = goal_particles - goal_particles.mean(axis=1, keepdims=True)
            cur_norm = np.linalg.norm(cur_centered.reshape(cur_centered.shape[0], -1), axis=1)
            goal_norm = float(np.linalg.norm(goal_centered.reshape(-1)))
            singular = (cur_norm <= 1e-8) | (goal_norm <= 1e-8)
            a = np.sum(
                goal_centered[:, :, 0] * cur_centered[:, :, 1]
                - goal_centered[:, :, 1] * cur_centered[:, :, 0],
                axis=1,
                dtype=np.float32,
            )
            b = np.sum(
                goal_centered[:, :, 0] * cur_centered[:, :, 0]
                + goal_centered[:, :, 1] * cur_centered[:, :, 1],
                axis=1,
                dtype=np.float32,
            )
            block_ang = np.abs(np.arctan2(a, b)).astype(np.float32)
            block_ang[singular] = 0.0

        cost = (
            float(eef_weight) * eef
            + float(block_pos_weight) * block_pos
            + float(block_angle_weight) * block_ang
        ).astype(np.float32)

        if float(state_l2_weight) > 0.0:
            goal_flat = np.concatenate([goal_pusher.reshape(1, -1), goal_particles.reshape(1, -1)], axis=1)
            cur_flat = np.concatenate([cur_pusher.reshape(cur_pusher.shape[0], -1), cur_particles.reshape(cur_particles.shape[0], -1)], axis=1)
            state_l2 = np.sqrt(np.mean((cur_flat - goal_flat) ** 2, axis=1, dtype=np.float32)).astype(np.float32)
            cost = cost + float(state_l2_weight) * state_l2
        if float(action_l2_weight) > 0.0:
            action_penalty = np.mean(np.asarray(actions, dtype=np.float32) ** 2, axis=(1, 2), dtype=np.float32)
            cost = cost + float(action_l2_weight) * action_penalty.astype(np.float32)
        return cost.astype(np.float32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        missing = [
            name
            for name in ("prepare_batch", "step_batch", "current_particle_cloud_state_batch")
            if not callable(getattr(self.backend, name, None))
        ]
        if missing:
            raise BatchedParticlePlannerUnavailable(
                "particle backend does not expose required batch APIs: " + ", ".join(missing)
            )

    def _backend_supports_device_batch(self) -> bool:
        supports = getattr(self.backend, "supports_cuda_native_batch", None)
        if not callable(supports):
            return False
        if not bool(supports()):
            return False
        required = (
            "prepare_batch_device",
            "step_batch_device",
            "current_particle_cloud_state_batch_device",
        )
        return all(callable(getattr(self.backend, name, None)) for name in required)

    def _evaluate_population_device_batch(
        self,
        *,
        init_state_np: np.ndarray,
        goal_state_np: np.ndarray,
        actions_np: np.ndarray,
        rollout_levels: List[int],
        iter_idx: int,
        seed: int,
    ) -> tuple[np.ndarray, int]:
        pop = int(actions_np.shape[0])
        lane_count = int(pop * self.rollout_samples)
        lane_actions = np.repeat(actions_np, self.rollout_samples, axis=0)
        lane_candidate_indices = np.repeat(np.arange(pop, dtype=np.int64), self.rollout_samples)
        lane_init_states = np.repeat(init_state_np[None, :], lane_count, axis=0)
        lane_goal_states = np.repeat(goal_state_np[None, :], lane_count, axis=0)

        if self.objective_space == "image":
            goal_visual_by_level = self._goal_visuals_by_level(
                goal_state=goal_state_np,
                rollout_levels=rollout_levels,
                seed=int(seed + 700001 * iter_idx),
            )
            goal_particle_cloud_by_level = {}
        else:
            goal_visual_by_level = {}
            goal_particle_cloud_by_level = self._goal_particle_clouds_by_level(
                goal_state=goal_state_np,
                rollout_levels=rollout_levels,
                seed=int(seed + 700001 * iter_idx),
            )

        initial_level_idx = int(rollout_levels[0])
        _obs0, lane_states, context = self.backend.prepare_batch_device(
            level_idx=initial_level_idx,
            init_states=lane_init_states,
            goal_states=lane_goal_states,
            with_visual=self.objective_space == "image",
        )
        del _obs0

        done_mask = np.zeros((lane_count,), dtype=bool)
        last_level_idx = np.full((lane_count,), initial_level_idx, dtype=np.int32)
        final_dists = np.zeros((lane_count,), dtype=np.float32)
        running_sum = np.zeros((lane_count,), dtype=np.float32)
        bits_iter = 0
        current_level_idx = initial_level_idx

        total_rollouts = int(pop * self.rollout_samples)
        pbar = self._make_rollout_progress_bar(total_rollouts, iter_idx)
        try:
            for t in range(self.horizon):
                active_mask = ~done_mask
                active_idx = np.flatnonzero(active_mask)
                if active_idx.size <= 0:
                    break
                level_idx = int(rollout_levels[t])
                if level_idx != current_level_idx:
                    context = self.backend._relevel_batch_device_context(context, level_idx)
                    current_level_idx = level_idx
                last_level_idx[active_idx] = level_idx

                obs_t, _rewards, done_t, _infos = self.backend.step_batch_device(
                    context=context,
                    actions=lane_actions[:, t, :],
                    active_mask=active_mask,
                    with_visual=self.objective_space == "image",
                )
                lane_states = np.asarray(obs_t["state"], dtype=np.float32).copy()
                done_mask[active_idx] = np.asarray(done_t, dtype=bool)[active_idx]
                bits_iter += int(active_idx.size) * int(self.backend.num_particles(level_idx=level_idx) * 2 * 32)

                if self.objective_space == "image":
                    visual_batch = obs_t.get("visual", None)
                    if visual_batch is None:
                        raise ValueError("particle_sim image objective requires batched visual observations.")
                    dists = self._image_distance_batch(
                        np.asarray(visual_batch, dtype=np.float32),
                        goal_visual_by_level[int(level_idx)],
                        metric=self.metric,
                    )
                    final_dists[active_idx] = dists[active_idx]
                    if self.running_weight > 0.0:
                        running_sum[active_idx] = running_sum[active_idx] + dists[active_idx]

            lane_costs = np.zeros((lane_count,), dtype=np.float32)
            if self.objective_space == "image":
                lane_costs = (
                    float(self.terminal_weight) * final_dists
                    + float(self.running_weight) * running_sum
                ).astype(np.float32)
                if self.action_l2_weight > 0.0:
                    lane_costs = lane_costs + float(self.action_l2_weight) * np.mean(
                        lane_actions ** 2,
                        axis=(1, 2),
                        dtype=np.float32,
                    )
            else:
                for level_idx in sorted({int(li) for li in last_level_idx.tolist()}):
                    level_lane_idx = np.flatnonzero(last_level_idx == int(level_idx))
                    if level_idx != current_level_idx:
                        context = self.backend._relevel_batch_device_context(context, int(level_idx))
                        current_level_idx = int(level_idx)
                    cloud_batch = self.backend.current_particle_cloud_state_batch_device(
                        context=context,
                        pixel=True,
                        lane_indices=level_lane_idx,
                    )
                    lane_costs[level_lane_idx] = self._particle_state_cost_batch(
                        cloud_batch,
                        goal_particle_cloud_by_level[int(level_idx)],
                        lane_actions[level_lane_idx],
                        eef_weight=self.eef_weight,
                        block_pos_weight=self.block_pos_weight,
                        block_angle_weight=self.block_angle_weight,
                        state_l2_weight=self.state_l2_weight,
                        action_l2_weight=self.action_l2_weight,
                    )
        finally:
            if pbar is not None:
                pbar.update(total_rollouts)
                pbar.close()

        costs = np.zeros((pop,), dtype=np.float32)
        for candidate_idx in range(pop):
            costs[candidate_idx] = float(
                np.mean(lane_costs[lane_candidate_indices == candidate_idx], dtype=np.float32)
            )
        return costs.astype(np.float32), int(bits_iter)

    @torch.no_grad()
    def plan(
        self,
        init_state: np.ndarray,
        goal_state: np.ndarray,
        mpc_progress: float = 0.0,
        seed: int = 0,
        warm_start_steps: int = 0,
        rng_seed: Optional[int] = None,
        gt_action_trajectory: Optional[Any] = None,
        gt_inject_count: int = 1,
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
            actions_np = np.asarray(actions.detach().cpu().numpy(), dtype=np.float32)
            if self._backend_supports_device_batch():
                costs, bits_iter = self._evaluate_population_device_batch(
                    init_state_np=init_state_np,
                    goal_state_np=goal_state_np,
                    actions_np=actions_np,
                    rollout_levels=rollout_levels,
                    iter_idx=iter_idx,
                    seed=seed,
                )
                return torch.as_tensor(costs, device=self.device), rollout_levels, int(bits_iter)
            pop = int(actions_np.shape[0])
            lane_count = int(pop * self.rollout_samples)
            lane_actions = np.repeat(actions_np, self.rollout_samples, axis=0)
            lane_candidate_indices = np.repeat(np.arange(pop, dtype=np.int64), self.rollout_samples)
            lane_init_states = np.repeat(init_state_np[None, :], lane_count, axis=0)
            lane_goal_states = np.repeat(goal_state_np[None, :], lane_count, axis=0)

            if self.objective_space == "image":
                goal_visual_by_level = self._goal_visuals_by_level(
                    goal_state=goal_state_np,
                    rollout_levels=rollout_levels,
                    seed=int(seed + 700001 * iter_idx),
                )
                goal_particle_cloud_by_level = {}
            else:
                goal_visual_by_level = {}
                goal_particle_cloud_by_level = self._goal_particle_clouds_by_level(
                    goal_state=goal_state_np,
                    rollout_levels=rollout_levels,
                    seed=int(seed + 700001 * iter_idx),
                )

            initial_level_idx = int(rollout_levels[0])
            lane_seeds = [
                int(seed + 1000003 * iter_idx + 1009 * int(lane_candidate_indices[i]) + 7 * (i % self.rollout_samples))
                for i in range(lane_count)
            ]
            _obs0, lane_states, lane_snapshots = self.backend.prepare_batch(
                level_idx=initial_level_idx,
                seeds=lane_seeds,
                init_states=lane_init_states,
                goal_states=lane_goal_states,
                with_visual=self.objective_space == "image",
            )
            del _obs0

            done_mask = np.zeros((lane_count,), dtype=bool)
            last_level_idx = np.full((lane_count,), initial_level_idx, dtype=np.int32)
            final_dists = np.zeros((lane_count,), dtype=np.float32)
            running_sum = np.zeros((lane_count,), dtype=np.float32)
            bits_iter = 0

            total_rollouts = int(pop * self.rollout_samples)
            pbar = self._make_rollout_progress_bar(total_rollouts, iter_idx)
            try:
                for t in range(self.horizon):
                    active_idx = np.flatnonzero(~done_mask)
                    if active_idx.size <= 0:
                        break
                    level_idx = int(rollout_levels[t])
                    last_level_idx[active_idx] = int(level_idx)

                    active_obs, _rewards, active_done, active_infos, active_snapshots = self.backend.step_batch(
                        level_idx=level_idx,
                        snapshots=[lane_snapshots[int(idx)] for idx in active_idx],
                        goal_states=lane_goal_states[active_idx],
                        actions=lane_actions[active_idx, t, :],
                        with_visual=self.objective_space == "image",
                    )
                    bits_iter += int(active_idx.size) * int(self.backend.num_particles(level_idx=level_idx) * 2 * 32)
                    for local_idx, lane_idx in enumerate(active_idx.tolist()):
                        lane_snapshots[lane_idx] = active_snapshots[local_idx]
                        lane_states[lane_idx] = np.asarray(active_infos[local_idx]["state"], dtype=np.float32).copy()
                        done_mask[lane_idx] = bool(active_done[local_idx])

                    if self.objective_space == "image":
                        visual_batch = active_obs.get("visual", None)
                        if visual_batch is None:
                            raise ValueError("particle_sim image objective requires batched visual observations.")
                        dists = self._image_distance_batch(
                            np.asarray(visual_batch, dtype=np.float32),
                            goal_visual_by_level[int(level_idx)],
                            metric=self.metric,
                        )
                        final_dists[active_idx] = dists
                        if self.running_weight > 0.0:
                            running_sum[active_idx] = running_sum[active_idx] + dists

                lane_costs = np.zeros((lane_count,), dtype=np.float32)
                if self.objective_space == "image":
                    lane_costs = (
                        float(self.terminal_weight) * final_dists
                        + float(self.running_weight) * running_sum
                    ).astype(np.float32)
                    if self.action_l2_weight > 0.0:
                        lane_costs = lane_costs + float(self.action_l2_weight) * np.mean(
                            lane_actions ** 2,
                            axis=(1, 2),
                            dtype=np.float32,
                        )
                else:
                    for level_idx in sorted({int(li) for li in last_level_idx.tolist()}):
                        level_lane_idx = np.flatnonzero(last_level_idx == int(level_idx))
                        cloud_batch = self.backend.current_particle_cloud_state_batch(
                            level_idx=int(level_idx),
                            snapshots=[lane_snapshots[int(idx)] for idx in level_lane_idx],
                            pixel=True,
                        )
                        lane_costs[level_lane_idx] = self._particle_state_cost_batch(
                            cloud_batch,
                            goal_particle_cloud_by_level[int(level_idx)],
                            lane_actions[level_lane_idx],
                            eef_weight=self.eef_weight,
                            block_pos_weight=self.block_pos_weight,
                            block_angle_weight=self.block_angle_weight,
                            state_l2_weight=self.state_l2_weight,
                            action_l2_weight=self.action_l2_weight,
                        )
            finally:
                if pbar is not None:
                    pbar.update(total_rollouts)
                    pbar.close()

            costs = np.zeros((pop,), dtype=np.float32)
            for candidate_idx in range(pop):
                costs[candidate_idx] = float(np.mean(lane_costs[lane_candidate_indices == candidate_idx], dtype=np.float32))
            return torch.as_tensor(costs, device=self.device), rollout_levels, int(bits_iter)

        start_level_idx = self.core.base_level_index(mpc_progress, 0.0)
        inject_tensor: Optional[torch.Tensor] = None
        if gt_action_trajectory is not None:
            inject_tensor = self._prepare_inject_actions_tensor(
                gt_action_trajectory,
                self.horizon,
                self.action_dim,
                self.device,
            )
        action_seq, final_level_idx, final_rollout_levels, total_bits = self.core.optimize(
            mpc_progress=mpc_progress,
            evaluate_population=_evaluate,
            warm_start=self.warm_start,
            shift_steps=int(warm_start_steps),
            rng_seed=rng_seed,
            inject_actions=inject_tensor,
            inject_count=int(gt_inject_count),
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
            start_level_idx=int(start_level_idx),
            batch_impl="cuda_native" if self._backend_supports_device_batch() else "host_batch",
        )
        return action_seq, info
