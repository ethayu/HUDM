from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from hudm.metrics import pose_metrics
from hudm.session import save_plan_result
from hudm.session_exec import run_closed_loop
from hudm.session_helpers import sample_init_goal_states
from hudm.world_io import checkpoint_epochs, latest_checkpoint_epoch, load_world_checkpoint, save_world_checkpoint
from models.world.model import HierWorldModel
from pusht.pusht_particle_backend import PushTParticleBackend


ROOT = os.path.dirname(os.path.dirname(__file__))


class FrameworkContractTests(unittest.TestCase):
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

        _, _, done, _ = PushTParticleBackend.step(backend, np.zeros(2, dtype=np.float32), with_visual=False)
        self.assertFalse(done)

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
