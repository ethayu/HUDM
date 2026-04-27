from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from planning.particle_cem import BatchedParticleCEMPlanner, ParticleCEMPlanner


class TinyParticleBackend:
    def __init__(self):
        self.num_levels = 3
        self._planning_fidelity_level_idx = 2
        self._state = np.zeros((7,), dtype=np.float32)
        self._goal_state = np.zeros((7,), dtype=np.float32)

    def set_planning_fidelity_level(self, level_idx: int) -> None:
        self._planning_fidelity_level_idx = max(0, min(int(level_idx), self.num_levels - 1))

    def num_particles(self, level_idx: int | None = None) -> int:
        li = self._planning_fidelity_level_idx if level_idx is None else int(level_idx)
        return [1, 4, 16][li]

    def spacing(self, level_idx: int | None = None) -> float:
        li = self._planning_fidelity_level_idx if level_idx is None else int(level_idx)
        return [1.0, 0.5, 0.25][li]

    @staticmethod
    def _ensure_state(state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 7:
            arr = np.concatenate([arr, np.zeros((7 - arr.shape[0],), dtype=np.float32)], axis=0)
        return arr[:7].astype(np.float32)

    def _render(self, state: np.ndarray, goal_state: np.ndarray, level_idx: int) -> np.ndarray:
        state_arr = self._ensure_state(state)
        goal_arr = self._ensure_state(goal_state)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        vals = np.array(
            [
                state_arr[0] + 10.0 * float(level_idx),
                state_arr[2] + goal_arr[2],
                state_arr[4] * 50.0 + goal_arr[4] * 50.0,
            ],
            dtype=np.float32,
        )
        img[0, 0, :] = np.clip(np.rint(vals), 0, 255).astype(np.uint8)
        img[1, 1, :] = np.clip(np.rint(np.abs(vals[::-1])), 0, 255).astype(np.uint8)
        return img

    def _cloud(self, state: np.ndarray, level_idx: int) -> dict[str, np.ndarray]:
        state_arr = self._ensure_state(state)
        center = np.asarray(state_arr[2:4], dtype=np.float32)
        offsets = {
            0: np.asarray([[0.0, 0.0]], dtype=np.float32),
            1: np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            2: np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        }[int(level_idx)]
        return {
            "pusher_xy": np.asarray(state_arr[:2], dtype=np.float32).copy(),
            "particle_xy": center[None, :] + offsets,
        }

    @staticmethod
    def _next_state(state: np.ndarray, action: np.ndarray, level_idx: int) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32).copy()
        action_arr = np.asarray(action, dtype=np.float32).reshape(2)
        scale = 0.1 * float(level_idx + 1)
        state_arr[0:2] = state_arr[0:2] + action_arr * scale
        state_arr[2:4] = state_arr[2:4] + action_arr[::-1] * (0.05 * float(level_idx + 1))
        state_arr[4] = state_arr[4] + float(action_arr.sum()) * 0.02 * float(level_idx + 1)
        return state_arr.astype(np.float32)

    def prepare(self, seed: int, init_state: np.ndarray, goal_state: np.ndarray, with_visual: bool = True):
        del seed
        self._state = self._ensure_state(init_state)
        self._goal_state = self._ensure_state(goal_state)
        obs = {
            "visual": self._render(self._state, self._goal_state, self._planning_fidelity_level_idx) if bool(with_visual) else None,
            "state": self._state.copy(),
        }
        return obs, self._state.copy()

    def step(self, action: np.ndarray, with_visual: bool = True):
        self._state = self._next_state(self._state, action, self._planning_fidelity_level_idx)
        obs = {
            "visual": self._render(self._state, self._goal_state, self._planning_fidelity_level_idx) if bool(with_visual) else None,
            "state": self._state.copy(),
        }
        info = {"state": self._state.copy()}
        return obs, 0.0, False, info

    def current_particle_cloud_state(self, *, level_idx: int | None = None, pixel: bool = True) -> dict[str, np.ndarray]:
        del pixel
        li = self._planning_fidelity_level_idx if level_idx is None else int(level_idx)
        return self._cloud(self._state, li)

    def prepare_batch(
        self,
        *,
        level_idx: int,
        seeds,
        init_states: np.ndarray,
        goal_states: np.ndarray,
        with_visual: bool = True,
    ):
        del seeds
        li = int(level_idx)
        states = np.asarray(init_states, dtype=np.float32).copy()
        goals = np.asarray(goal_states, dtype=np.float32).copy()
        visuals = []
        snapshots = []
        for idx in range(states.shape[0]):
            snapshots.append({"state": states[idx].copy(), "goal_state": goals[idx].copy()})
            if bool(with_visual):
                visuals.append(self._render(states[idx], goals[idx], li))
        obs = {
            "visual": np.stack(visuals, axis=0) if bool(with_visual) else None,
            "state": states.copy(),
        }
        return obs, states.copy(), snapshots

    def step_batch(
        self,
        *,
        level_idx: int,
        snapshots,
        goal_states: np.ndarray,
        actions: np.ndarray,
        with_visual: bool = True,
    ):
        li = int(level_idx)
        goals = np.asarray(goal_states, dtype=np.float32)
        action_arr = np.asarray(actions, dtype=np.float32)
        states = np.zeros((action_arr.shape[0], 7), dtype=np.float32)
        rewards = np.zeros((action_arr.shape[0],), dtype=np.float32)
        dones = np.zeros((action_arr.shape[0],), dtype=bool)
        visuals = []
        infos = []
        next_snapshots = []
        for idx, snapshot in enumerate(snapshots):
            cur_state = self._next_state(snapshot["state"], action_arr[idx], li)
            states[idx] = cur_state
            infos.append({"state": cur_state.copy()})
            next_snapshots.append({"state": cur_state.copy(), "goal_state": goals[idx].copy()})
            if bool(with_visual):
                visuals.append(self._render(cur_state, goals[idx], li))
        obs = {
            "visual": np.stack(visuals, axis=0) if bool(with_visual) else None,
            "state": states.copy(),
        }
        return obs, rewards, dones, infos, next_snapshots

    def current_particle_cloud_state_batch(
        self,
        *,
        level_idx: int,
        snapshots,
        pixel: bool = True,
    ) -> dict[str, np.ndarray]:
        del pixel
        li = int(level_idx)
        pusher = []
        particles = []
        for snapshot in snapshots:
            cloud = self._cloud(snapshot["state"], li)
            pusher.append(cloud["pusher_xy"])
            particles.append(cloud["particle_xy"])
        return {
            "pusher_xy": np.stack(pusher, axis=0).astype(np.float32),
            "particle_xy": np.stack(particles, axis=0).astype(np.float32),
        }


class TinyCudaParticleBackend(TinyParticleBackend):
    def __init__(self):
        super().__init__()
        self.device_prepare_calls = 0
        self.device_step_calls = 0
        self.device_cloud_calls = 0
        self.host_prepare_calls = 0
        self.host_step_calls = 0
        self.host_cloud_calls = 0

    def supports_cuda_native_batch(self) -> bool:
        return True

    def prepare_batch(self, *args, **kwargs):
        self.host_prepare_calls += 1
        return super().prepare_batch(*args, **kwargs)

    def step_batch(self, *args, **kwargs):
        self.host_step_calls += 1
        return super().step_batch(*args, **kwargs)

    def current_particle_cloud_state_batch(self, *args, **kwargs):
        self.host_cloud_calls += 1
        return super().current_particle_cloud_state_batch(*args, **kwargs)

    def prepare_batch_device(
        self,
        *,
        level_idx: int,
        init_states: np.ndarray,
        goal_states: np.ndarray,
        with_visual: bool = True,
    ):
        self.device_prepare_calls += 1
        li = int(level_idx)
        states = np.asarray(init_states, dtype=np.float32).copy()
        goals = np.asarray(goal_states, dtype=np.float32).copy()
        visuals = None
        if bool(with_visual):
            visuals = np.stack([self._render(states[idx], goals[idx], li) for idx in range(states.shape[0])], axis=0)
        obs = {
            "visual": visuals,
            "state": states.copy(),
        }
        ctx = SimpleNamespace(
            level_idx=li,
            goal_states=goals.copy(),
            current_states=states.copy(),
        )
        return obs, states.copy(), ctx

    def _relevel_batch_device_context(self, context, level_idx: int):
        context.level_idx = int(level_idx)
        return context

    def step_batch_device(
        self,
        *,
        context,
        actions: np.ndarray,
        active_mask: np.ndarray,
        with_visual: bool = True,
    ):
        self.device_step_calls += 1
        active = np.asarray(active_mask, dtype=bool).reshape(-1)
        action_arr = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(context.current_states, dtype=np.float32).copy()
        for idx in np.flatnonzero(active):
            next_states[idx] = self._next_state(next_states[idx], action_arr[idx], context.level_idx)
        context.current_states = next_states.copy()

        visuals = None
        if bool(with_visual):
            visuals = np.stack(
                [self._render(next_states[idx], context.goal_states[idx], context.level_idx) for idx in range(next_states.shape[0])],
                axis=0,
            )
        obs = {
            "visual": visuals,
            "state": next_states.copy(),
        }
        rewards = np.zeros((next_states.shape[0],), dtype=np.float32)
        dones = np.zeros((next_states.shape[0],), dtype=bool)
        infos = [{"state": next_states[idx].copy()} for idx in range(next_states.shape[0])]
        return obs, rewards, dones, infos

    def current_particle_cloud_state_batch_device(
        self,
        *,
        context,
        pixel: bool = True,
        lane_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        del pixel
        self.device_cloud_calls += 1
        idx = (
            np.arange(context.current_states.shape[0], dtype=np.int64)
            if lane_indices is None
            else np.asarray(lane_indices, dtype=np.int64).reshape(-1)
        )
        pusher = []
        particles = []
        for lane_idx in idx:
            cloud = self._cloud(context.current_states[int(lane_idx)], int(context.level_idx))
            pusher.append(cloud["pusher_xy"])
            particles.append(cloud["particle_xy"])
        return {
            "pusher_xy": np.stack(pusher, axis=0).astype(np.float32),
            "particle_xy": np.stack(particles, axis=0).astype(np.float32),
        }


class BatchedParticlePlannerTests(unittest.TestCase):
    def setUp(self):
        self.common_kwargs = {
            "horizon": 4,
            "action_dim": 2,
            "pop_size": 24,
            "elite_frac": 0.25,
            "n_iter": 3,
            "init_std": 0.4,
            "warm_start": True,
            "device": torch.device("cpu"),
            "fidelity_cfg": {
                "enabled": True,
                "mpc": {"mode": "fixed", "level": "finest"},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "linear", "start_level": "base", "end_level": "coarsest"},
            },
        }
        self.init_state = np.asarray([5.0, 7.0, 10.0, 20.0, 0.1, 0.0, 0.0], dtype=np.float32)
        self.goal_state = np.asarray([4.0, 3.0, 14.0, 18.0, -0.2, 0.0, 0.0], dtype=np.float32)

    def _assert_planners_match(self, *, objective_cfg: dict, particle_env_cfg: dict):
        serial = ParticleCEMPlanner(
            particle_backend=TinyParticleBackend(),
            objective_cfg=objective_cfg,
            particle_env_cfg=particle_env_cfg,
            **self.common_kwargs,
        )
        batched = BatchedParticleCEMPlanner(
            particle_backend=TinyParticleBackend(),
            objective_cfg=objective_cfg,
            particle_env_cfg=particle_env_cfg,
            **self.common_kwargs,
        )

        action_seq_serial, info_serial = serial.plan(
            init_state=self.init_state,
            goal_state=self.goal_state,
            mpc_progress=0.3,
            seed=11,
            warm_start_steps=0,
            rng_seed=123,
        )
        action_seq_batched, info_batched = batched.plan(
            init_state=self.init_state,
            goal_state=self.goal_state,
            mpc_progress=0.3,
            seed=11,
            warm_start_steps=0,
            rng_seed=123,
        )

        self.assertTrue(torch.allclose(action_seq_serial, action_seq_batched, atol=1e-6, rtol=1e-6))
        self.assertEqual(info_serial.base_level_idx, info_batched.base_level_idx)
        self.assertEqual(info_serial.start_level_idx, info_batched.start_level_idx)
        self.assertEqual(info_serial.rollout_level_indices, info_batched.rollout_level_indices)
        self.assertEqual(info_serial.bits_used_estimate, info_batched.bits_used_estimate)

    def test_batched_matches_serial_for_state_objective(self):
        self._assert_planners_match(
            objective_cfg={
                "action_l2_weight": 0.05,
                "eef_weight": 1.0,
                "block_pos_weight": 1.0,
                "block_angle_weight": 0.1,
                "state_l2_weight": 0.2,
            },
            particle_env_cfg={
                "rollout_samples": 2,
                "objective_space": "state",
                "progress": False,
                "progress_leave": False,
            },
        )

    def test_batched_matches_serial_for_image_objective(self):
        self._assert_planners_match(
            objective_cfg={
                "latent_metric": "l2",
                "terminal_weight": 1.0,
                "running_weight": 0.2,
                "action_l2_weight": 0.05,
            },
            particle_env_cfg={
                "rollout_samples": 2,
                "objective_space": "image",
                "progress": False,
                "progress_leave": False,
            },
        )

    def test_image_distance_batch_matches_scalar_per_lane_normalization(self):
        planner = BatchedParticleCEMPlanner(
            particle_backend=TinyParticleBackend(),
            objective_cfg={
                "latent_metric": "l2",
            },
            particle_env_cfg={
                "rollout_samples": 1,
                "objective_space": "image",
                "progress": False,
                "progress_leave": False,
            },
            **self.common_kwargs,
        )

        goal = np.asarray(
            [
                [[0.25, 0.5, 0.75], [0.0, 0.25, 0.5]],
                [[0.1, 0.2, 0.3], [0.9, 1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        imgs = np.stack(
            [
                goal.copy(),
                np.clip(np.rint(goal * 255.0), 0, 255).astype(np.uint8),
            ],
            axis=0,
        )

        batch_dist = planner._image_distance_batch(imgs, goal, metric="l2")
        scalar_dist = np.asarray(
            [
                planner._image_distance(planner._to_float_image(imgs[0]), planner._to_float_image(goal)),
                planner._image_distance(planner._to_float_image(imgs[1]), planner._to_float_image(goal)),
            ],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(batch_dist, scalar_dist, atol=1e-7, rtol=1e-7))

    def test_batched_planner_prefers_device_batch_apis_when_available(self):
        backend = TinyCudaParticleBackend()
        planner = BatchedParticleCEMPlanner(
            particle_backend=backend,
            objective_cfg={
                "action_l2_weight": 0.05,
                "eef_weight": 1.0,
                "block_pos_weight": 1.0,
                "block_angle_weight": 0.1,
                "state_l2_weight": 0.2,
            },
            particle_env_cfg={
                "rollout_samples": 2,
                "objective_space": "state",
                "progress": False,
                "progress_leave": False,
            },
            **self.common_kwargs,
        )

        action_seq, info = planner.plan(
            init_state=self.init_state,
            goal_state=self.goal_state,
            mpc_progress=0.3,
            seed=11,
            warm_start_steps=0,
            rng_seed=123,
        )

        self.assertEqual(tuple(action_seq.shape), (self.common_kwargs["horizon"], self.common_kwargs["action_dim"]))
        self.assertEqual(info.batch_impl, "cuda_native")
        self.assertGreater(backend.device_prepare_calls, 0)
        self.assertGreater(backend.device_step_calls, 0)
        self.assertGreater(backend.device_cloud_calls, 0)
        self.assertEqual(backend.host_prepare_calls, 0)
        self.assertEqual(backend.host_step_calls, 0)
        self.assertEqual(backend.host_cloud_calls, 0)


if __name__ == "__main__":
    unittest.main()
