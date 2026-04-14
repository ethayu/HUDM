from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "pusht" / "pusht_particle_warp.py"
_SPEC = importlib.util.spec_from_file_location("pusht_particle_warp_local", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

PushTWarpEnv = _MOD.PushTWarpEnv
_points_in_t_grid = _MOD._points_in_t_grid
_tee_structure_metrics = _MOD._tee_structure_metrics
build_t_particle_hierarchy = _MOD.build_t_particle_hierarchy
GT_PUSHER_RADIUS = _MOD.GT_PUSHER_RADIUS
GT_T_BAR_H = _MOD.GT_T_BAR_H
GT_T_BAR_W = _MOD.GT_T_BAR_W
GT_T_STEM_H = _MOD.GT_T_STEM_H
GT_T_STEM_W = _MOD.GT_T_STEM_W


class ParticleFidelityHierarchyTests(unittest.TestCase):
    def test_hierarchy_counts_are_monotone_and_centered(self):
        levels = build_t_particle_hierarchy(
            particle_counts=[1, 4, 16, 64],
            stem_w=0.05,
            stem_h=0.10,
            bar_w=0.12,
            bar_h=0.04,
            min_particles=1,
        )

        counts = [int(level.rest_offsets.shape[0]) for level in levels]
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(int(levels[-1].target_particle_count), 64)
        self.assertLessEqual(abs(int(counts[-1]) - 64), 20)

        for level in levels:
            np.testing.assert_allclose(
                level.rest_offsets[:, :2].mean(axis=0),
                np.zeros((2,), dtype=np.float32),
                atol=1e-6,
            )

    def test_hierarchy_preserves_t_structure_above_low_count_regime(self):
        levels = build_t_particle_hierarchy(
            particle_counts=[1, 4, 16, 64],
            stem_w=0.05,
            stem_h=0.10,
            bar_w=0.12,
            bar_h=0.04,
            min_particles=1,
        )

        canonical_bbox = np.ptp(levels[-1].rest_offsets[:, :2], axis=0)
        for level in levels:
            if int(level.rest_offsets.shape[0]) <= 4:
                continue
            missing, _ = _tee_structure_metrics(
                level.rest_offsets[:, :2],
                canonical_bbox=canonical_bbox,
                stem_w=0.05,
                stem_h=0.10,
                bar_w=0.12,
                bar_h=0.04,
            )
            self.assertEqual(missing, 0)

    def test_hierarchy_rejects_unsorted_particle_counts(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_t_particle_hierarchy(
                particle_counts=[1, 16, 4],
                stem_w=0.05,
                stem_h=0.10,
                bar_w=0.12,
                bar_h=0.04,
                min_particles=1,
            )

    def test_pd_pusher_path_matches_gt_recurrence(self):
        env = PushTWarpEnv.__new__(PushTWarpEnv)
        env.pusher_pos = np.asarray([0.15, -0.05, 0.0], dtype=np.float32)
        env.pusher_velocity = np.asarray([-0.04, 0.07], dtype=np.float32)
        env.pusher_k_p = 100.0
        env.pusher_k_v = 20.0
        env.sim_dt = 0.01
        env.controller_steps = 10

        target = np.asarray([0.22, 0.03], dtype=np.float32)
        path, final_velocity = PushTWarpEnv._build_pusher_path(env, target)

        pos = env.pusher_pos[:2].astype(np.float64).copy()
        vel = env.pusher_velocity.astype(np.float64).copy()
        expected = np.zeros_like(path)
        expected[0, :2] = pos.astype(np.float32)
        for step_idx in range(env.controller_steps):
            acc = env.pusher_k_p * (target.astype(np.float64) - pos) - env.pusher_k_v * vel
            vel = vel + acc * env.sim_dt
            pos = pos + vel * env.sim_dt
            expected[step_idx + 1, :2] = pos.astype(np.float32)

        np.testing.assert_allclose(path, expected, atol=1e-6)
        np.testing.assert_allclose(final_velocity, vel.astype(np.float32), atol=1e-6)

    def test_simulate_frame_replays_each_controller_step_in_path(self):
        env = PushTWarpEnv.__new__(PushTWarpEnv)
        env.control_dt = 0.1
        env.controller_steps = 10
        env.substeps = 3
        env.iters = 2
        env.N = 1
        env.device = "cpu"
        env.x = object()
        env.v = object()
        env.inv_m = object()
        env.x_pred = object()
        env.r0 = object()
        env.xmin = -1.0
        env.xmax = 1.0
        env.ymin = -1.0
        env.ymax = 1.0
        env.pr = 0.01
        env.pusher_r = 0.02
        env.mu = 0.6
        env.contact_alpha = 0.35
        env.alpha_rigid = 1.0
        env.lin_damp = 0.995
        env.vel_damp = 0.999
        env.ground_friction_accel = 2.0
        env.rest_speed_eps = 0.01
        env._compute_com_and_theta = lambda: (SimpleNamespace(), 0.0)

        path = np.zeros((env.controller_steps + 1, 3), dtype=np.float32)
        path[:, 0] = np.arange(env.controller_steps + 1, dtype=np.float32)

        calls = []

        class FakeWP:
            @staticmethod
            def vec3(x, y, z):
                return (x, y, z)

            @staticmethod
            def launch(kernel, dim, inputs, device):
                calls.append((kernel, list(inputs), dim, device))

        old_wp = _MOD.wp
        _MOD.wp = FakeWP
        try:
            PushTWarpEnv._simulate_frame(env, pusher_path=path)
        finally:
            _MOD.wp = old_wp

        predict_dts = [float(inputs[4]) for kernel, inputs, _dim, _device in calls if kernel is _MOD._predict_positions]
        self.assertEqual(len(predict_dts), env.controller_steps)
        self.assertTrue(all(abs(dt - (env.control_dt / env.controller_steps)) < 1e-8 for dt in predict_dts))

        contact_positions = [inputs[3] for kernel, inputs, _dim, _device in calls if kernel is _MOD._pusher_contact]
        self.assertEqual(len(contact_positions), env.controller_steps * env.iters)
        self.assertEqual(contact_positions[0], (1.0, 0.0, 0.0))
        self.assertEqual(contact_positions[-1], (10.0, 0.0, 0.0))

    def test_get_pusher_velocity_returns_authoritative_state(self):
        env = PushTWarpEnv.__new__(PushTWarpEnv)
        env.pusher_velocity = np.asarray([0.5, -0.25], dtype=np.float32)
        np.testing.assert_allclose(
            PushTWarpEnv.get_pusher_velocity(env),
            np.asarray([0.5, -0.25], dtype=np.float32),
        )

    def test_gt_matched_t_geometry_uses_gt_bbox_and_cog_convention(self):
        pts = _points_in_t_grid(
            stem_w=GT_T_STEM_W,
            stem_h=GT_T_STEM_H,
            bar_w=GT_T_BAR_W,
            bar_h=GT_T_BAR_H,
            spacing=_MOD.PUSHT_WORLD_PER_PIXEL,
            min_particles=1,
        )
        px = pts[:, :2] * (_MOD.PUSHT_RENDER_PIXELS / _MOD.PUSHT_WORLD_SIZE)

        self.assertAlmostEqual(float(np.ptp(px[:, 0])), 160.0, places=4)
        self.assertAlmostEqual(float(np.ptp(px[:, 1])), 160.0, places=4)
        self.assertAlmostEqual(float(px[:, 0].min()), -80.0, places=4)
        self.assertAlmostEqual(float(px[:, 0].max()), 80.0, places=4)
        self.assertAlmostEqual(float(px[:, 1].min()), 0.0, places=4)
        self.assertAlmostEqual(float(px[:, 1].max()), 160.0, places=4)

    def test_gt_matched_pusher_radius_uses_agent_radius(self):
        px_radius = float(GT_PUSHER_RADIUS * (_MOD.PUSHT_RENDER_PIXELS / _MOD.PUSHT_WORLD_SIZE))
        self.assertAlmostEqual(px_radius, 15.0, places=6)


if __name__ == "__main__":
    unittest.main()
