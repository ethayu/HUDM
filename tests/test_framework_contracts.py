from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import pymunk
import torch
from omegaconf import OmegaConf

from hudm.config import _plan_defaults, _prune_inactive_backend, plan_spec_to_runtime_cfg
from hudm.metrics import pose_metrics
from hudm.runtime import build_plan_runtime
from hudm.session import save_plan_result
from hudm.session_exec import run_closed_loop
from hudm.session_helpers import sample_init_goal_states
from hudm.world_io import checkpoint_epochs, latest_checkpoint_epoch, load_world_checkpoint, save_world_checkpoint
from models.world.model import HierWorldModel
from planning.particle_cem import (
    BatchedParticleCEMPlanner,
    BatchedParticlePlannerUnavailable,
    ParticleCEMPlanner,
)
from pusht.pusht_env import PushTEnv
from pusht.pusht_particle_backend import PushTParticleBackend
from pusht.pusht_particle_warp import PushTWarpEnv
from validate_cfg import validate_plan_cfg


ROOT = os.path.dirname(os.path.dirname(__file__))


class FrameworkContractTests(unittest.TestCase):
    @staticmethod
    def _expected_gt_tee_center_of_gravity(scale: float) -> np.ndarray:
        return np.asarray([0.0, 1.5 * float(scale)], dtype=np.float32)

    @staticmethod
    def _expected_gt_tee_moment(scale: float) -> float:
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        return float(inertia1 + inertia2)

    def test_world_checkpoint_roundtrip_uses_epoch_suffixes(self):
        model = HierWorldModel(
            K=[4],
            D=4,
            action_dim=2,
            input="state",
            decoder_mode="per_level",
            dynamics_mode="per_level",
        )
        for param in model.parameters():
            torch.nn.init.uniform_(param, a=-0.1, b=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_world_checkpoint(model, tmpdir, epoch=7)
            self.assertEqual(checkpoint_epochs(tmpdir), [7])
            self.assertEqual(latest_checkpoint_epoch(tmpdir), 7)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "encoder_epoch7.pt")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "decoder_l0_epoch7.pt")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "dyn_l0_epoch7.pt")))

            restored = HierWorldModel(
                K=[4],
                D=4,
                action_dim=2,
                input="state",
                decoder_mode="per_level",
                dynamics_mode="per_level",
            )
            load_world_checkpoint(restored, tmpdir, epoch=7, device=torch.device("cpu"))
            for p_a, p_b in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.allclose(p_a, p_b))

    def test_pose_metrics_use_shared_thresholds(self):
        goal = torch.tensor([0.0, 0.0, 100.0, 100.0, 0.0]).numpy()
        cur = torch.tensor([0.0, 0.0, 105.0, 104.0, 0.1]).numpy()
        metrics = pose_metrics(goal, cur)
        self.assertIn("success", metrics)
        self.assertAlmostEqual(metrics["pos_diff"], 6.403124, places=5)

    def test_gt_add_tee_matches_current_com_and_inertia_convention(self):
        env = PushTEnv.__new__(PushTEnv)
        env.space = pymunk.Space()

        scale = 30
        body = env.add_tee((256, 300), 0, scale=scale)

        np.testing.assert_allclose(
            np.asarray(tuple(body.center_of_gravity), dtype=np.float32),
            self._expected_gt_tee_center_of_gravity(scale),
            atol=1e-6,
        )
        self.assertAlmostEqual(float(body.moment), self._expected_gt_tee_moment(scale), places=6)

    def test_runtime_files_do_not_contain_live_pdb_breakpoints(self):
        for rel_path in ("plan.py", "planning/cem_core.py"):
            with open(os.path.join(ROOT, rel_path), "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("pdb.set_trace()", content, msg=rel_path)

    def test_sample_init_goal_states_random_without_dataset_uses_none_seed(self):
        class DummyEnv:
            def sample_random_init_goal_states(self, seed=None):
                return np.asarray([seed], dtype=object), np.asarray([seed], dtype=object)

        cfg = OmegaConf.create({"init_goal": {"source": "random"}})
        init_state, goal_state, meta = sample_init_goal_states(DummyEnv(), cfg, wm_cfg=None)

        self.assertIsNone(init_state[0])
        self.assertIsNone(goal_state[0])
        self.assertEqual(meta, {"source": "random"})

    def test_sample_init_goal_states_random_with_dataset_uses_dataset_seed(self):
        class DummyEnv:
            def sample_random_init_goal_states(self, seed=None):
                return np.asarray([seed], dtype=np.int64), np.asarray([seed], dtype=np.int64)

        cfg = OmegaConf.create({"init_goal": {"source": "random", "dataset": {"seed": 17}}})
        init_state, goal_state, meta = sample_init_goal_states(DummyEnv(), cfg, wm_cfg=None)

        self.assertEqual(int(init_state[0]), 17)
        self.assertEqual(int(goal_state[0]), 17)
        self.assertEqual(meta, {"source": "random"})

    def test_particle_backend_done_ignores_pose_metric_success(self):
        class DummySim:
            pusher_pos = np.zeros(2, dtype=np.float32)
            num_particles = 1
            spacing = 0.1

            def step(self, a_world):
                return None, 0.0, False, {"final_coverage": 0.0}

        sim = DummySim()
        backend = SimpleNamespace(
            relative=True,
            action_scale=1.0,
            _goal_state=np.zeros(7, dtype=np.float32),
            _active_sim=lambda: sim,
            _pix_delta_to_world=lambda delta_px: np.asarray(delta_px, dtype=np.float32),
            _sim_state=lambda active_sim: np.zeros(7, dtype=np.float32),
            _capture_sim_snapshot=lambda active_sim: None,
            _proprio_from_state=lambda state: np.asarray(state[:2], dtype=np.float32),
            render=lambda mode, include_start_pose=False: np.zeros((4, 4, 3), dtype=np.uint8),
            eval_state=lambda goal_state, cur_state: {
                "success": True,
                "pos_diff": 0.0,
                "angle_diff": 0.0,
                "eef_diff": 0.0,
                "state_dist": 0.0,
            },
            num_particles=lambda: 1,
            spacing=lambda: 0.1,
        )
        backend._obs_from_sim = lambda sim, cur_state, with_visual: {
            "visual": None,
            "proprio": np.asarray(cur_state[:2], dtype=np.float32),
            "state": np.asarray(cur_state, dtype=np.float32).copy(),
        }
        backend._goal_state_for_sim = lambda sim, goal_state=None: np.zeros(7, dtype=np.float32)
        backend._step_sim = lambda sim, **kwargs: PushTParticleBackend._step_sim(backend, sim, **kwargs)

        _, _, done, _ = PushTParticleBackend.step(backend, np.zeros(2, dtype=np.float32), with_visual=False)
        self.assertFalse(done)

    def test_particle_backend_particle_cloud_state_uses_pusht_pixel_coordinates(self):
        class DummySim:
            pusher_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

            def get_particle_positions(self):
                return np.asarray(
                    [
                        [-0.25, -0.25, 0.0],
                        [0.25, 0.25, 0.0],
                    ],
                    dtype=np.float32,
                )

        backend = SimpleNamespace(
            _planning_fidelity_level_idx=0,
            _sims=[DummySim()],
            xmin=-0.25,
            ymin=-0.25,
            _xrange=0.5,
            _yrange=0.5,
        )
        backend._sim_for_level = lambda level_idx=None: PushTParticleBackend._sim_for_level(backend, level_idx)
        backend._world_xy_to_pix = lambda xy_world: PushTParticleBackend._world_xy_to_pix(backend, xy_world)
        backend._pusher_position_from_sim = PushTParticleBackend._pusher_position_from_sim
        backend._particle_positions_from_sim = PushTParticleBackend._particle_positions_from_sim
        backend.current_pusher_position = lambda **kwargs: PushTParticleBackend.current_pusher_position(backend, **kwargs)
        backend.current_particle_positions = lambda **kwargs: PushTParticleBackend.current_particle_positions(backend, **kwargs)

        cloud = PushTParticleBackend.current_particle_cloud_state(backend)

        np.testing.assert_allclose(cloud["pusher_xy"], np.asarray([256.0, 256.0], dtype=np.float32))
        np.testing.assert_allclose(
            cloud["particle_xy"],
            np.asarray(
                [
                    [0.0, 0.0],
                    [512.0, 512.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_particle_backend_world_to_img_xy_allows_out_of_frame_positions(self):
        backend = SimpleNamespace(
            render_size=96,
            xmin=-0.25,
            ymin=-0.25,
            _xrange=0.5,
            _yrange=0.5,
        )
        backend._world_to_pix_xy = lambda xy: PushTParticleBackend._world_to_pix_xy(backend, xy)

        off_right = PushTParticleBackend._world_to_img_xy(backend, np.asarray([0.30, 0.0], dtype=np.float32))
        off_left = PushTParticleBackend._world_to_img_xy(backend, np.asarray([-0.30, 0.0], dtype=np.float32))

        self.assertGreater(off_right[0], backend.render_size - 1)
        self.assertLess(off_left[0], 0)

    def test_particle_backend_switch_level_restores_snapshot_instead_of_pose_only_reset(self):
        class DummySim:
            def __init__(self, obj_pose, obj_twist, pusher_xy, pusher_velocity):
                self._obj_pose = np.asarray(obj_pose, dtype=np.float32)
                self._obj_twist = np.asarray(obj_twist, dtype=np.float32)
                self.pusher_pos = np.asarray([pusher_xy[0], pusher_xy[1], 0.0], dtype=np.float32)
                self._pusher_velocity = np.asarray(pusher_velocity, dtype=np.float32)
                self.goal_pose = np.zeros(3, dtype=np.float32)
                self.restored = None

            def capture_state(self):
                return {
                    "pusher_xy": self.pusher_pos[:2].copy(),
                    "pusher_velocity": self._pusher_velocity.copy(),
                    "obj_pose": self._obj_pose.copy(),
                    "obj_twist": self._obj_twist.copy(),
                    "goal_pose": self.goal_pose.copy(),
                }

            def restore_state(self, snapshot):
                self.restored = {
                    k: np.asarray(v, dtype=np.float32).copy()
                    for k, v in snapshot.items()
                }
                self._obj_pose = self.restored["obj_pose"].copy()
                self._obj_twist = self.restored["obj_twist"].copy()
                self.pusher_pos = np.asarray(
                    [self.restored["pusher_xy"][0], self.restored["pusher_xy"][1], 0.0],
                    dtype=np.float32,
                )
                self._pusher_velocity = self.restored["pusher_velocity"].copy()
                self.goal_pose = self.restored["goal_pose"].copy()
                return {}

            def get_object_pose(self):
                return self._obj_pose.copy()

            def get_object_twist(self):
                return self._obj_twist.copy()

            def get_pusher_velocity(self):
                return self._pusher_velocity.copy()

        sim0 = DummySim(
            obj_pose=[0.25, 0.50, 0.75],
            obj_twist=[0.10, -0.05, 1.25],
            pusher_xy=[0.2, 0.4],
            pusher_velocity=[0.03, -0.02],
        )
        sim1 = DummySim(
            obj_pose=[-0.4, -0.4, -0.4],
            obj_twist=[0.0, 0.0, 0.0],
            pusher_xy=[-0.3, -0.3],
            pusher_velocity=[0.0, 0.0],
        )

        backend = PushTParticleBackend.__new__(PushTParticleBackend)
        backend.state_dim = 7
        backend._planning_fidelity_num_levels = 2
        backend._planning_fidelity_level_idx = 0
        backend._sims = [sim0, sim1]
        backend._current_state = np.zeros(7, dtype=np.float32)
        backend._goal_state = np.zeros(7, dtype=np.float32)
        backend._current_snapshot = None
        backend.xmin = 0.0
        backend.ymin = 0.0
        backend._xrange = 1.0
        backend._yrange = 1.0

        PushTParticleBackend.set_planning_fidelity_level(backend, 1)

        self.assertIsNotNone(sim1.restored)
        np.testing.assert_allclose(sim1.restored["obj_pose"], sim0.get_object_pose())
        np.testing.assert_allclose(sim1.restored["obj_twist"], sim0.get_object_twist())
        np.testing.assert_allclose(sim1.restored["pusher_velocity"], sim0.get_pusher_velocity())
        self.assertAlmostEqual(float(backend._current_state[4]), 0.75, places=6)

    def test_particle_backend_set_sim_from_state_preserves_pusher_velocity(self):
        class DummySim:
            def __init__(self):
                self.last_kwargs = None

            def set_state(self, **kwargs):
                self.last_kwargs = kwargs
                return {}

            def _make_obs(self):
                return {}

        sim = DummySim()
        backend = SimpleNamespace(
            _ensure_state_dim=lambda state: np.asarray(state, dtype=np.float32),
            _pix_to_world_xy=lambda xy: np.asarray(xy, dtype=np.float32) / 10.0,
            _pix_vel_to_world=lambda vel: np.asarray(vel, dtype=np.float32) / 20.0,
            _goal_pose_world=lambda goal_state: np.asarray([5.0, 6.0, -0.25], dtype=np.float32),
            _gt_pose_to_internal_pose=lambda pose_world, level_idx=None: np.asarray(pose_world, dtype=np.float32),
        )

        PushTParticleBackend._set_sim_from_state(
            backend,
            sim,
            np.asarray([10.0, 20.0, 30.0, 40.0, 0.5, 6.0, -4.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 50.0, 60.0, -0.25, 0.0, 0.0], dtype=np.float32),
        )

        np.testing.assert_allclose(sim.last_kwargs["pusher_xy"], np.asarray([1.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(sim.last_kwargs["obj_xy"], np.asarray([3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(sim.last_kwargs["pusher_velocity"], np.asarray([0.3, -0.2], dtype=np.float32))
        np.testing.assert_allclose(sim.last_kwargs["goal_pose"], np.asarray([5.0, 6.0, -0.25], dtype=np.float32))

    def test_particle_plan_validation_accepts_gt_pusher_controller_keys(self):
        spec_cfg = OmegaConf.create(_plan_defaults())
        spec_cfg.backend.kind = "particle_sim"
        runtime_cfg = plan_spec_to_runtime_cfg(_prune_inactive_backend(spec_cfg))

        validate_plan_cfg(runtime_cfg)

    def test_particle_plan_validation_rejects_non_divisible_pusher_rates(self):
        spec_cfg = OmegaConf.create(_plan_defaults())
        spec_cfg.backend.kind = "particle_sim"
        spec_cfg.backend.particle_sim.fidelity_env.sim_hz = 95
        spec_cfg.backend.particle_sim.fidelity_env.control_hz = 10

        with self.assertRaisesRegex(ValueError, "sim_hz must be divisible by control_hz"):
            plan_spec_to_runtime_cfg(_prune_inactive_backend(spec_cfg))

    def test_plan_spec_to_runtime_cfg_rejects_legacy_particle_pusher_keys(self):
        cfg = OmegaConf.create(_plan_defaults())
        cfg.backend.kind = "particle_sim"
        cfg.backend.wm = None
        cfg.backend.gt_env = None
        cfg.backend.particle_sim.fidelity_env.pusher_speed = 0.6

        with self.assertRaisesRegex(ValueError, "pusher_speed"):
            plan_spec_to_runtime_cfg(cfg)

    def test_plan_spec_to_runtime_cfg_rejects_legacy_particle_fidelity_keys(self):
        cfg = OmegaConf.create(_plan_defaults())
        cfg.backend.kind = "particle_sim"
        cfg.backend.wm = None
        cfg.backend.gt_env = None
        cfg.backend.particle_sim.fidelity_env.canonical_spacing = 0.01

        with self.assertRaisesRegex(ValueError, "canonical_spacing"):
            plan_spec_to_runtime_cfg(cfg)

    def test_particle_warp_pusher_path_matches_gt_pd_controller(self):
        gt_env = PushTEnv(with_velocity=True, action_scale=1.0, relative=True)
        gt_env.goal_pose = np.asarray([256.0, 256.0, np.pi / 4.0], dtype=np.float32)
        gt_env.reset()

        init_state = np.asarray([256.0, 128.0, 300.0, 320.0, 0.35, -30.0, 20.0], dtype=np.float32)
        action = np.asarray([15.0, -8.0], dtype=np.float32)
        gt_env._set_state(np.asarray([256.0, 128.0, 300.0, 320.0, 0.35, 0.0, 0.0], dtype=np.float32))
        gt_env.agent.velocity = (float(init_state[5]), float(init_state[6]))
        gt_env.step(action)

        env = PushTWarpEnv.__new__(PushTWarpEnv)
        env.pusher_pos = np.asarray([init_state[0], init_state[1], 0.0], dtype=np.float32)
        env.pusher_velocity = np.asarray(init_state[5:7], dtype=np.float32).copy()
        env.pusher_k_p = float(gt_env.k_p)
        env.pusher_k_v = float(gt_env.k_v)
        env.sim_dt = 1.0 / float(gt_env.sim_hz)
        env.controller_steps = int(gt_env.sim_hz // gt_env.control_hz)

        path, final_velocity = PushTWarpEnv._build_pusher_path(env, init_state[:2] + action)

        np.testing.assert_allclose(
            path[-1, :2],
            np.asarray(gt_env.agent.position, dtype=np.float32),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            final_velocity,
            np.asarray(gt_env.agent.velocity, dtype=np.float32),
            atol=1e-4,
        )

    def test_particle_backend_contact_fit_normalization_preserves_global_scalar_knobs(self):
        cfg = OmegaConf.create(_plan_defaults())
        cfg.backend.kind = "particle_sim"
        runtime_cfg = plan_spec_to_runtime_cfg(_prune_inactive_backend(cfg))

        self.assertIn("contact_alpha", runtime_cfg.particle_env.fidelity_env)
        self.assertIn("alpha_rigid", runtime_cfg.particle_env.fidelity_env)
        self.assertAlmostEqual(float(runtime_cfg.particle_env.fidelity_env.contact_alpha), 0.35, places=6)
        self.assertAlmostEqual(float(runtime_cfg.particle_env.fidelity_env.alpha_rigid), 1.0, places=6)

    def test_particle_backend_prepare_reuses_live_snapshot_when_state_matches(self):
        state = np.asarray([256.0, 128.0, 300.0, 320.0, 0.35, 12.0, -6.0], dtype=np.float32)
        goal_state = np.asarray([0.0, 0.0, 350.0, 360.0, -0.40, 0.0, 0.0], dtype=np.float32)
        sim = mock.Mock()

        backend = PushTParticleBackend.__new__(PushTParticleBackend)
        backend.with_velocity = True
        backend._planning_fidelity_level_idx = 0
        backend._sims = [sim]
        backend._current_state = state.copy()
        backend._goal_state = np.zeros(7, dtype=np.float32)
        backend._start_state = np.zeros(7, dtype=np.float32)
        backend._current_snapshot = {"obj_pose": np.zeros(3, dtype=np.float32)}
        backend.xmin = -0.25
        backend.ymin = -0.25
        backend._xrange = 0.5
        backend._yrange = 0.5
        backend._capture_sim_snapshot = mock.Mock(return_value={"live": np.asarray([1.0], dtype=np.float32)})
        backend._set_sim_from_state = mock.Mock(side_effect=AssertionError("prepare should not reset live state"))

        backend._active_sim = lambda: PushTParticleBackend._active_sim(backend)
        backend._ensure_state_dim = lambda x: PushTParticleBackend._ensure_state_dim(backend, x)
        backend._goal_pose_world = lambda x: PushTParticleBackend._goal_pose_world(backend, x)
        backend._state_matches_current = lambda x, atol=1e-3: PushTParticleBackend._state_matches_current(
            backend, x, atol
        )
        backend._sim_state = lambda active_sim: state.copy()
        backend._proprio_from_state = lambda s: PushTParticleBackend._proprio_from_state(backend, s)

        obs, cur_state = PushTParticleBackend.prepare(
            backend,
            seed=0,
            init_state=state.copy(),
            goal_state=goal_state.copy(),
            with_visual=False,
        )

        sim.set_goal_pose.assert_called_once()
        np.testing.assert_allclose(sim.set_goal_pose.call_args.args[0], backend._goal_pose_world(goal_state))
        np.testing.assert_allclose(cur_state, state)
        np.testing.assert_allclose(obs["state"], state)
        np.testing.assert_allclose(obs["proprio"], np.asarray([256.0, 128.0, 12.0, -6.0], dtype=np.float32))
        self.assertIsNone(obs["visual"])
        np.testing.assert_allclose(backend._start_state, state)
        np.testing.assert_allclose(backend._goal_state, goal_state)
        self.assertEqual(list(backend._current_snapshot.keys()), ["live"])

    def test_particle_backend_batch_size_one_matches_scalar_contract(self):
        try:
            backend = PushTParticleBackend(
                with_velocity=True,
                with_target=True,
                render_size=64,
                relative=True,
                action_scale=100.0,
                device="cpu",
                particle_counts=[1, 4],
                warp_cfg={
                    "particle_radius": 0.01,
                    "sim_hz": 20,
                    "control_hz": 10,
                    "substeps": 1,
                    "iters": 1,
                    "min_particles": 1,
                },
                seed=0,
            )
        except ImportError as exc:
            self.skipTest(str(exc))

        init_state = np.asarray([180.0, 170.0, 260.0, 250.0, 0.15, 4.0, -2.0], dtype=np.float32)
        goal_state = np.asarray([0.0, 0.0, 300.0, 280.0, -0.2, 0.0, 0.0], dtype=np.float32)
        action = np.asarray([0.015, -0.01], dtype=np.float32)

        backend.set_planning_fidelity_level(1)
        obs_scalar_0, state_scalar_0 = backend.prepare(
            seed=0,
            init_state=init_state,
            goal_state=goal_state,
            with_visual=True,
        )
        frame_scalar_0 = backend.render("rgb_array", include_start_pose=True)
        obs_scalar_1, reward_scalar, done_scalar, info_scalar = backend.step(action, with_visual=True)
        cloud_scalar_1 = backend.current_particle_cloud_state(level_idx=1, pixel=True)
        frame_scalar_1 = backend.render("rgb_array", include_start_pose=True)

        obs_batch_0, state_batch_0, snapshots = backend.prepare_batch(
            level_idx=1,
            seeds=[0],
            init_states=init_state[None, :],
            goal_states=goal_state[None, :],
            with_visual=True,
        )
        frame_batch_0 = backend.render_batch(
            level_idx=1,
            snapshots=snapshots,
            include_start_pose=True,
        )
        obs_batch_1, reward_batch, done_batch, info_batch, next_snapshots = backend.step_batch(
            level_idx=1,
            snapshots=snapshots,
            goal_states=goal_state[None, :],
            actions=action[None, :],
            with_visual=True,
        )
        cloud_batch_1 = backend.current_particle_cloud_state_batch(
            level_idx=1,
            snapshots=next_snapshots,
            pixel=True,
        )
        frame_batch_1 = backend.render_batch(
            level_idx=1,
            snapshots=next_snapshots,
            include_start_pose=True,
        )

        np.testing.assert_allclose(state_scalar_0, state_batch_0[0], atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(obs_scalar_0["visual"], obs_batch_0["visual"][0])
        np.testing.assert_array_equal(frame_scalar_0, frame_batch_0[0])
        np.testing.assert_allclose(obs_scalar_1["state"], obs_batch_1["state"][0], atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(obs_scalar_1["visual"], obs_batch_1["visual"][0])
        self.assertAlmostEqual(float(reward_scalar), float(reward_batch[0]), places=6)
        self.assertEqual(bool(done_scalar), bool(done_batch[0]))
        np.testing.assert_allclose(info_scalar["state"], info_batch[0]["state"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cloud_scalar_1["pusher_xy"], cloud_batch_1["pusher_xy"][0], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cloud_scalar_1["particle_xy"], cloud_batch_1["particle_xy"][0], atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(frame_scalar_1, frame_batch_1[0])

    def test_particle_state_terms_use_particle_coordinates_for_pose_like_losses(self):
        cur_cloud = {
            "pusher_xy": np.asarray([3.0, 0.0], dtype=np.float32),
            "particle_xy": np.asarray(
                [
                    [1.0, -1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        }
        goal_cloud = {
            "pusher_xy": np.asarray([0.0, 0.0], dtype=np.float32),
            "particle_xy": np.asarray(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                ],
                dtype=np.float32,
            ),
        }

        eef, block_pos, block_ang, state_l2 = ParticleCEMPlanner._particle_state_terms(cur_cloud, goal_cloud)

        self.assertAlmostEqual(eef, 3.0, places=5)
        self.assertAlmostEqual(block_pos, 0.0, places=5)
        self.assertAlmostEqual(block_ang, np.pi / 2.0, places=5)
        self.assertGreater(state_l2, 0.0)

    def test_particle_state_terms_ignore_angle_when_single_particle_has_no_orientation_signal(self):
        cur_cloud = {
            "pusher_xy": np.asarray([0.0, 0.0], dtype=np.float32),
            "particle_xy": np.asarray([[1.0, 1.0]], dtype=np.float32),
        }
        goal_cloud = {
            "pusher_xy": np.asarray([0.0, 0.0], dtype=np.float32),
            "particle_xy": np.asarray([[1.0, 1.0]], dtype=np.float32),
        }

        _eef, _block_pos, block_ang, _state_l2 = ParticleCEMPlanner._particle_state_terms(cur_cloud, goal_cloud)
        self.assertEqual(block_ang, 0.0)

    def test_build_plan_runtime_resolves_random_particle_seed(self):
        cfg = OmegaConf.create(
            {
                "env_id": "pusht",
                "env": {
                    "with_velocity": True,
                    "with_target": True,
                },
                "backend": "particle_sim",
                "world_model": {
                    "device": "cpu",
                },
                "mpc": {
                    "horizon": 2,
                },
                "cem": {
                    "pop_size": 4,
                    "elite_frac": 0.5,
                    "n_iter": 1,
                    "init_std": 1.0,
                    "warm_start": False,
                    "action_low": None,
                    "action_high": None,
                },
                "objective": {
                    "action_l2_weight": 0.0,
                },
                "fidelity": {},
                "particle_env": {
                    "batch_mode": "off",
                    "fidelity_env": {
                        "particle_counts": [1, 8, 32],
                        "device": "cpu",
                    },
                },
                "init_goal": {
                    "dataset": {
                        "seed": "random",
                    },
                },
            }
        )

        dummy_env = SimpleNamespace(
            action_dim=2,
            render_size=96,
            relative=True,
            action_scale=100.0,
        )

        with mock.patch("hudm.runtime.register_plan_env"):
            with mock.patch("hudm.runtime.gym_make_versioned", return_value=dummy_env):
                with mock.patch("hudm.runtime.unwrap_env", return_value=dummy_env):
                    with mock.patch("hudm.runtime.PushTParticleBackend") as backend_cls:
                        with mock.patch("hudm.runtime.ParticleCEMPlanner") as planner_cls:
                            backend_cls.return_value.num_levels = 4
                            runtime = build_plan_runtime(cfg)

        backend_seed = backend_cls.call_args.kwargs["seed"]
        self.assertIsInstance(backend_seed, int)
        self.assertNotEqual(backend_seed, "random")
        self.assertEqual(runtime["cfg"].init_goal.dataset.seed, backend_seed)
        self.assertEqual(runtime["cfg"].fidelity.num_levels, 4)
        self.assertEqual(cfg.init_goal.dataset.seed, "random")
        self.assertIsNone(OmegaConf.select(cfg, "fidelity.num_levels"))
        planner_cls.assert_called_once()

    def test_build_plan_runtime_uses_batched_particle_planner_by_default(self):
        cfg = OmegaConf.create(
            {
                "env_id": "pusht",
                "env": {
                    "with_velocity": True,
                    "with_target": True,
                },
                "backend": "particle_sim",
                "world_model": {
                    "device": "cpu",
                },
                "mpc": {
                    "horizon": 2,
                },
                "cem": {
                    "pop_size": 4,
                    "elite_frac": 0.5,
                    "n_iter": 1,
                    "init_std": 1.0,
                    "warm_start": False,
                    "action_low": None,
                    "action_high": None,
                },
                "objective": {
                    "action_l2_weight": 0.0,
                },
                "fidelity": {},
                "particle_env": {
                    "batch_mode": "auto",
                    "fidelity_env": {
                        "particle_counts": [1, 8, 32],
                        "device": "cpu",
                    },
                },
                "init_goal": {
                    "dataset": {
                        "seed": 0,
                    },
                },
            }
        )
        dummy_env = SimpleNamespace(
            action_dim=2,
            render_size=96,
            relative=True,
            action_scale=100.0,
        )

        with mock.patch("hudm.runtime.register_plan_env"):
            with mock.patch("hudm.runtime.gym_make_versioned", return_value=dummy_env):
                with mock.patch("hudm.runtime.unwrap_env", return_value=dummy_env):
                    with mock.patch("hudm.runtime.PushTParticleBackend") as backend_cls:
                        with mock.patch("hudm.runtime.BatchedParticleCEMPlanner") as batched_cls:
                            with mock.patch("hudm.runtime.ParticleCEMPlanner") as serial_cls:
                                backend_cls.return_value.num_levels = 4
                                build_plan_runtime(cfg)

        batched_cls.assert_called_once()
        serial_cls.assert_not_called()

    def test_build_plan_runtime_falls_back_to_serial_particle_planner_when_auto_batch_init_fails(self):
        cfg = OmegaConf.create(
            {
                "env_id": "pusht",
                "env": {
                    "with_velocity": True,
                    "with_target": True,
                },
                "backend": "particle_sim",
                "world_model": {
                    "device": "cpu",
                },
                "mpc": {
                    "horizon": 2,
                },
                "cem": {
                    "pop_size": 4,
                    "elite_frac": 0.5,
                    "n_iter": 1,
                    "init_std": 1.0,
                    "warm_start": False,
                    "action_low": None,
                    "action_high": None,
                },
                "objective": {
                    "action_l2_weight": 0.0,
                },
                "fidelity": {},
                "particle_env": {
                    "batch_mode": "auto",
                    "fidelity_env": {
                        "particle_counts": [1, 8, 32],
                        "device": "cpu",
                    },
                },
                "init_goal": {
                    "dataset": {
                        "seed": 0,
                    },
                },
            }
        )
        dummy_env = SimpleNamespace(
            action_dim=2,
            render_size=96,
            relative=True,
            action_scale=100.0,
        )

        with mock.patch("hudm.runtime.register_plan_env"):
            with mock.patch("hudm.runtime.gym_make_versioned", return_value=dummy_env):
                with mock.patch("hudm.runtime.unwrap_env", return_value=dummy_env):
                    with mock.patch("hudm.runtime.PushTParticleBackend") as backend_cls:
                        with mock.patch(
                            "hudm.runtime.BatchedParticleCEMPlanner",
                            side_effect=BatchedParticlePlannerUnavailable("batch init failed"),
                        ) as batched_cls:
                            with mock.patch("hudm.runtime.ParticleCEMPlanner") as serial_cls:
                                backend_cls.return_value.num_levels = 4
                                build_plan_runtime(cfg)

        batched_cls.assert_called_once()
        serial_cls.assert_called_once()

    def test_build_plan_runtime_raises_when_force_batch_init_fails(self):
        cfg = OmegaConf.create(
            {
                "env_id": "pusht",
                "env": {
                    "with_velocity": True,
                    "with_target": True,
                },
                "backend": "particle_sim",
                "world_model": {
                    "device": "cpu",
                },
                "mpc": {
                    "horizon": 2,
                },
                "cem": {
                    "pop_size": 4,
                    "elite_frac": 0.5,
                    "n_iter": 1,
                    "init_std": 1.0,
                    "warm_start": False,
                    "action_low": None,
                    "action_high": None,
                },
                "objective": {
                    "action_l2_weight": 0.0,
                },
                "fidelity": {},
                "particle_env": {
                    "batch_mode": "force",
                    "fidelity_env": {
                        "particle_counts": [1, 8, 32],
                        "device": "cpu",
                    },
                },
                "init_goal": {
                    "dataset": {
                        "seed": 0,
                    },
                },
            }
        )
        dummy_env = SimpleNamespace(
            action_dim=2,
            render_size=96,
            relative=True,
            action_scale=100.0,
        )

        with mock.patch("hudm.runtime.register_plan_env"):
            with mock.patch("hudm.runtime.gym_make_versioned", return_value=dummy_env):
                with mock.patch("hudm.runtime.unwrap_env", return_value=dummy_env):
                    with mock.patch("hudm.runtime.PushTParticleBackend") as backend_cls:
                        with mock.patch(
                            "hudm.runtime.BatchedParticleCEMPlanner",
                            side_effect=BatchedParticlePlannerUnavailable("batch init failed"),
                        ) as batched_cls:
                            with mock.patch("hudm.runtime.ParticleCEMPlanner") as serial_cls:
                                backend_cls.return_value.num_levels = 4
                                with self.assertRaisesRegex(RuntimeError, "batch init failed"):
                                    build_plan_runtime(cfg)

        batched_cls.assert_called_once()
        serial_cls.assert_not_called()

    def test_build_plan_runtime_auto_batch_does_not_swallow_arbitrary_batched_errors(self):
        cfg = OmegaConf.create(
            {
                "env_id": "pusht",
                "env": {
                    "with_velocity": True,
                    "with_target": True,
                },
                "backend": "particle_sim",
                "world_model": {
                    "device": "cpu",
                },
                "mpc": {
                    "horizon": 2,
                },
                "cem": {
                    "pop_size": 4,
                    "elite_frac": 0.5,
                    "n_iter": 1,
                    "init_std": 1.0,
                    "warm_start": False,
                    "action_low": None,
                    "action_high": None,
                },
                "objective": {
                    "action_l2_weight": 0.0,
                },
                "fidelity": {},
                "particle_env": {
                    "batch_mode": "auto",
                    "fidelity_env": {
                        "particle_counts": [1, 8, 32],
                        "device": "cpu",
                    },
                },
                "init_goal": {
                    "dataset": {
                        "seed": 0,
                    },
                },
            }
        )
        dummy_env = SimpleNamespace(
            action_dim=2,
            render_size=96,
            relative=True,
            action_scale=100.0,
        )

        with mock.patch("hudm.runtime.register_plan_env"):
            with mock.patch("hudm.runtime.gym_make_versioned", return_value=dummy_env):
                with mock.patch("hudm.runtime.unwrap_env", return_value=dummy_env):
                    with mock.patch("hudm.runtime.PushTParticleBackend") as backend_cls:
                        with mock.patch(
                            "hudm.runtime.BatchedParticleCEMPlanner",
                            side_effect=RuntimeError("unexpected bug"),
                        ) as batched_cls:
                            with mock.patch("hudm.runtime.ParticleCEMPlanner") as serial_cls:
                                backend_cls.return_value.num_levels = 4
                                with self.assertRaisesRegex(RuntimeError, "unexpected bug"):
                                    build_plan_runtime(cfg)

        batched_cls.assert_called_once()
        serial_cls.assert_not_called()

    def test_plan_spec_to_runtime_cfg_allows_missing_particle_num_levels(self):
        cfg = OmegaConf.create(_plan_defaults())
        cfg.backend.kind = "particle_sim"
        cfg.backend.wm = None
        cfg.backend.gt_env = None
        if "num_levels" in cfg.planner.fidelity:
            del cfg.planner.fidelity["num_levels"]
        cfg.backend.particle_sim.fidelity_env.particle_counts = [1, 4, 16, 64]

        runtime_cfg = plan_spec_to_runtime_cfg(cfg)

        self.assertIsNone(OmegaConf.select(runtime_cfg, "fidelity.num_levels"))
        self.assertEqual(list(runtime_cfg.particle_env.fidelity_env.particle_counts), [1, 4, 16, 64])

    def test_particle_planner_accepts_linear_rollout_fidelity(self):
        planner = ParticleCEMPlanner(
            particle_backend=SimpleNamespace(num_levels=4),
            horizon=4,
            action_dim=2,
            pop_size=8,
            elite_frac=0.25,
            n_iter=1,
            init_std=1.0,
            fidelity_cfg={
                "enabled": True,
                "rollout": {
                    "mode": "linear",
                    "start_level": "base",
                    "end_level": "coarsest",
                },
            },
        )

        self.assertEqual(planner.core.rollout_mode, "linear")
        self.assertEqual(planner.core.rollout_level_indices(3), [3, 2, 1, 0])

    def test_run_closed_loop_terminates_on_env_done_even_without_metric_success(self):
        class DummyPlanner:
            def plan(self, **kwargs):
                info = SimpleNamespace(
                    base_level_idx=0,
                    rollout_level_indices=[0],
                    bits_used_estimate=0,
                    plan_time_sec=0.0,
                    base_k=None,
                    base_spacing=None,
                    base_num_particles=None,
                )
                return torch.zeros((1, 2), dtype=torch.float32), info

        class DummyEnv:
            def __init__(self):
                self.cur_state = np.zeros(7, dtype=np.float32)
                self._planning_fidelity_num_levels = 1

            def set_task_start(self, pose):
                del pose

            def set_planning_fidelity_level(self, level_idx):
                del level_idx

            def prepare(self, seed, init_state, goal_state=None):
                del seed, goal_state
                self.cur_state = np.asarray(init_state, dtype=np.float32).copy()
                obs = {"visual": np.zeros((8, 8, 3), dtype=np.uint8)}
                return obs, self.cur_state.copy()

            def render(self, mode, include_start_pose=False):
                del mode, include_start_pose
                return np.zeros((8, 8, 3), dtype=np.uint8)

            def step(self, action):
                del action
                self.cur_state = np.asarray([0, 0, 200, 200, 0, 0, 0], dtype=np.float32)
                obs = {"visual": np.zeros((8, 8, 3), dtype=np.uint8)}
                info = {"state": self.cur_state.copy(), "final_coverage": 0.99}
                return obs, 1.0, True, info

            def eval_termination(self, goal_state, cur_state, done=None, info=None):
                del goal_state, cur_state
                coverage = None if info is None else float(info.get("final_coverage", 0.0))
                done_flag = False if done is None else bool(done)
                return {
                    "success": False,
                    "pos_diff": 50.0,
                    "angle_diff": 1.0,
                    "eef_diff": 0.0,
                    "state_dist": 50.0,
                    "done": done_flag,
                    "coverage": coverage,
                    "success_and_done": False,
                }

        cfg = OmegaConf.create(
            {
                "save": False,
                "render": False,
                "mpc": {
                    "steps": 1,
                    "horizon": 1,
                    "replan_every": 1,
                },
            }
        )
        env = DummyEnv()
        success, _, _, _, run_stats, _ = run_closed_loop(
            env=env,
            wm=None,
            planner=DummyPlanner(),
            backend="gt_env",
            cfg=cfg,
            init_state=np.zeros(7, dtype=np.float32),
            goal_state=np.zeros(7, dtype=np.float32),
            device=torch.device("cpu"),
        )

        self.assertTrue(success)
        self.assertEqual(run_stats["termination_reason"], "env_done")
        self.assertTrue(run_stats["termination_done"])
        self.assertFalse(run_stats["termination_metric_success"])

    def test_save_plan_result_persists_action_overlay_metadata(self):
        cfg = OmegaConf.create({"save": False, "render": False})
        runtime = {
            "backend": "gt_env",
            "env": SimpleNamespace(relative=True, action_scale=100.0, window_size=512.0),
        }
        result = {
            "cfg": cfg,
            "runtime": runtime,
            "success": False,
            "trajectory": [np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32)],
            "frames": [],
            "planner_frames": [],
            "run_stats": {
                "plans": 1,
                "bits_used_total": 0,
                "flops_used_total": 0,
                "plan_time_total_sec": 0.0,
                "termination_reason": "max_steps",
                "termination_step": 1,
                "termination_metric_success": False,
                "termination_done": False,
                "termination_pos_diff": None,
                "termination_angle_diff": None,
                "termination_eef_diff": None,
                "termination_coverage": None,
            },
            "trace": {
                "executed_actions": [[0.25, 0.0]],
                "trajectory": [[256.0, 256.0, 0.0, 0.0, 0.0], [281.0, 256.0, 0.0, 0.0, 0.0]],
                "pos_diffs": [],
                "angle_diffs": [],
                "eef_diffs": [],
                "coverages": [],
                "metric_success_flags": [],
                "done_flags": [],
                "state_dists": [],
                "replans": [],
            },
            "init_state": np.zeros(5, dtype=np.float32),
            "goal_state": np.zeros(5, dtype=np.float32),
            "sample_meta": {"source": "random"},
            "schedule_name": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_plan_result(result, tmpdir, save_media=False)
            with open(os.path.join(tmpdir, "trace.json"), "r", encoding="utf-8") as f:
                trace_meta = json.load(f)
            with open(os.path.join(tmpdir, "metadata.json"), "r", encoding="utf-8") as f:
                metadata = json.load(f)

        for payload in (trace_meta, metadata):
            self.assertEqual(payload["action_format"], "env_input")
            self.assertTrue(payload["action_relative"])
            self.assertEqual(payload["action_scale"], 100.0)

    def test_run_closed_loop_continues_recording_saved_media_after_env_done(self):
        class DummyPlanner:
            def plan(self, **kwargs):
                del kwargs
                info = SimpleNamespace(
                    base_level_idx=0,
                    rollout_level_indices=[0, 0],
                    bits_used_estimate=0,
                    plan_time_sec=0.0,
                    base_k=None,
                    base_spacing=None,
                    base_num_particles=None,
                )
                return torch.zeros((2, 2), dtype=torch.float32), info

        class DummyEnv:
            def __init__(self):
                self.cur_state = np.zeros(7, dtype=np.float32)
                self._planning_fidelity_num_levels = 1
                self.relative = True
                self.action_scale = 1.0
                self.window_size = 512.0
                self._step_count = 0

            def set_task_start(self, pose):
                del pose

            def set_planning_fidelity_level(self, level_idx):
                del level_idx

            def prepare(self, seed, init_state, goal_state=None):
                del seed, goal_state
                self.cur_state = np.asarray(init_state, dtype=np.float32).copy()
                obs = {"visual": np.zeros((8, 8, 3), dtype=np.uint8)}
                return obs, self.cur_state.copy()

            def render(self, mode, include_start_pose=False):
                del mode, include_start_pose
                return np.zeros((8, 8, 3), dtype=np.uint8)

            def step(self, action):
                del action
                self._step_count += 1
                self.cur_state = np.asarray([self._step_count, 0, 200, 200, 0, 0, 0], dtype=np.float32)
                obs = {"visual": np.zeros((8, 8, 3), dtype=np.uint8)}
                done = self._step_count == 1
                info = {"state": self.cur_state.copy(), "final_coverage": 0.99 if done else 0.0}
                return obs, 1.0, done, info

            def eval_termination(self, goal_state, cur_state, done=None, info=None):
                del goal_state, cur_state
                coverage = None if info is None else float(info.get("final_coverage", 0.0))
                done_flag = False if done is None else bool(done)
                return {
                    "success": False,
                    "pos_diff": 50.0,
                    "angle_diff": 1.0,
                    "eef_diff": 0.0,
                    "state_dist": 50.0,
                    "done": done_flag,
                    "coverage": coverage,
                    "success_and_done": False,
                }

        cfg = OmegaConf.create(
            {
                "save": True,
                "render": False,
                "mpc": {
                    "steps": 2,
                    "horizon": 2,
                    "replan_every": 2,
                },
            }
        )
        env = DummyEnv()
        success, _, frames, planner_frames, run_stats, trace = run_closed_loop(
            env=env,
            wm=None,
            planner=DummyPlanner(),
            backend="gt_env",
            cfg=cfg,
            init_state=np.zeros(7, dtype=np.float32),
            goal_state=np.zeros(7, dtype=np.float32),
            device=torch.device("cpu"),
        )

        self.assertTrue(success)
        self.assertEqual(run_stats["termination_reason"], "env_done")
        self.assertEqual(len(trace["executed_actions"]), 1)
        self.assertEqual(len(frames), 3)
        self.assertEqual(len(planner_frames), 3)


if __name__ == "__main__":
    unittest.main()
