from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from hudm.artifacts import (
    draw_action_target_cross,
    infer_action_overlay_spec,
    overlay_action_targets_on_frames,
    resolve_action_target,
    write_video_mp4,
)
from scripts import planning_media


class PlanningMediaTests(unittest.TestCase):
    def test_write_video_mp4_uses_imageio_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "writer_smoke.mp4")
            frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
            ffmpeg_exe = shutil.which("ffmpeg")
            old_ffmpeg_exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
            try:
                if ffmpeg_exe:
                    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
                write_video_mp4(out_path, frames, fps=5, output_size=32)
            finally:
                if old_ffmpeg_exe is None:
                    os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
                else:
                    os.environ["IMAGEIO_FFMPEG_EXE"] = old_ffmpeg_exe

            self.assertTrue(os.path.isfile(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_render_returns_existing_media_without_regeneration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "trace.json"), "w", encoding="utf-8") as f:
                json.dump({"plan_config": {}, "sample": {}, "init_state": [], "goal_state": []}, f)
            np.savez_compressed(os.path.join(tmpdir, "trace.npz"), executed_actions=np.zeros((0, 2), dtype=np.float32))
            existing_path = os.path.join(tmpdir, "planned.mp4")
            with open(existing_path, "wb") as f:
                f.write(b"")

            outputs = planning_media.render_media(tmpdir, schedule=None, rollout_id=None, media=["closed_loop_replay"])
            self.assertEqual(outputs, [existing_path])

    def test_resolve_action_target_uses_relative_semantics(self):
        target = resolve_action_target(
            agent_pos=np.asarray([10.0, 20.0], dtype=np.float32),
            action=np.asarray([1.5, -0.5], dtype=np.float32),
            action_format="env_input",
            action_relative=True,
            action_scale=100.0,
        )
        np.testing.assert_allclose(target, np.asarray([160.0, -30.0], dtype=np.float32))

    def test_infer_action_overlay_spec_uses_global_fallback_for_old_traces(self):
        spec = infer_action_overlay_spec(
            trace_meta={},
            actions=np.asarray([[64.0, 128.0], [200.0, 300.0]], dtype=np.float32),
            env=type("Env", (), {"relative": True, "action_scale": 100.0, "window_size": 512})(),
        )
        self.assertEqual(spec["action_format"], "absolute_target")
        self.assertFalse(spec["action_relative"])
        self.assertEqual(spec["action_scale"], 1.0)

    def test_draw_action_target_cross_marks_target(self):
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        out = draw_action_target_cross(frame, np.asarray([256.0, 256.0], dtype=np.float32))
        self.assertGreater(int(np.sum(out[:, :, 0])), 0)
        self.assertTrue(np.array_equal(out[:, :, 1], np.zeros((96, 96), dtype=np.uint8)))

    def test_overlay_action_targets_leaves_terminal_frame_unmodified(self):
        frames = [np.zeros((96, 96, 3), dtype=np.uint8) for _ in range(2)]
        states = np.asarray([[256.0, 256.0, 0, 0, 0], [260.0, 260.0, 0, 0, 0]], dtype=np.float32)
        actions = np.asarray([[0.5, 0.0]], dtype=np.float32)
        out = overlay_action_targets_on_frames(
            frames,
            states,
            actions,
            {
                "action_format": "env_input",
                "action_relative": True,
                "action_scale": 100.0,
                "reference_size": 512.0,
            },
        )
        self.assertGreater(int(np.sum(out[0][:, :, 0])), 0)
        self.assertEqual(int(np.sum(out[1])), 0)

    def test_overlay_action_targets_draws_on_eef_when_no_action_exists(self):
        frames = [np.zeros((96, 96, 3), dtype=np.uint8)]
        states = np.asarray([[256.0, 256.0, 0, 0, 0]], dtype=np.float32)
        out = overlay_action_targets_on_frames(
            frames,
            states,
            np.zeros((0, 2), dtype=np.float32),
            {
                "action_format": "env_input",
                "action_relative": True,
                "action_scale": 100.0,
                "reference_size": 512.0,
            },
        )
        self.assertGreater(int(np.sum(out[0][:, :, 0])), 0)

    def test_closed_loop_replay_overlays_action_target(self):
        class DummyEnv:
            def __init__(self):
                self.relative = True
                self.action_scale = 100.0
                self.window_size = 512.0
                self._state = np.asarray([256.0, 256.0, 0, 0, 0], dtype=np.float32)

            def prepare(self, seed, init_state, goal_state=None):
                del seed, goal_state
                self._state = np.asarray(init_state, dtype=np.float32).copy()
                return {"visual": np.zeros((96, 96, 3), dtype=np.uint8)}, self._state.copy()

            def render(self, mode, include_start_pose=False):
                del mode, include_start_pose
                return np.zeros((96, 96, 3), dtype=np.uint8)

            def step(self, action):
                action = np.asarray(action, dtype=np.float32)
                self._state[:2] = self._state[:2] + action * self.action_scale
                return {"visual": np.zeros((96, 96, 3), dtype=np.uint8)}, 0.0, False, {"state": self._state.copy()}

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "trace.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "plan_config": {},
                        "sample": {},
                        "init_state": [256.0, 256.0, 0, 0, 0],
                        "goal_state": [300.0, 256.0, 0, 0, 0],
                        "action_format": "env_input",
                        "action_relative": True,
                        "action_scale": 100.0,
                    },
                    f,
                )
            np.savez_compressed(
                os.path.join(tmpdir, "trace.npz"),
                executed_actions=np.asarray([[0.5, 0.0]], dtype=np.float32),
                trajectory=np.asarray([[256.0, 256.0, 0, 0, 0], [306.0, 256.0, 0, 0, 0]], dtype=np.float32),
            )
            captured = {}

            def _capture_video(path, frames, fps=15, output_size=256):
                del fps, output_size
                captured["path"] = path
                captured["frames"] = [np.asarray(fr).copy() for fr in frames]

            with mock.patch.object(planning_media, "_build_runtime", return_value=(OmegaConf.create({"env": {"render_size": 96}}), {"env": DummyEnv(), "backend": "gt_env"})):
                with mock.patch.object(planning_media.single_plan, "_write_video_mp4", side_effect=_capture_video):
                    outputs = planning_media.render_media(tmpdir, schedule=None, rollout_id=None, media=["closed_loop_replay"])

            self.assertEqual(outputs, [os.path.join(tmpdir, "closed_loop_replay.mp4")])
            self.assertIn("frames", captured)
            self.assertGreater(int(np.sum(captured["frames"][0][:, :, 0])), 0)
            self.assertEqual(int(np.sum(captured["frames"][-1])), 0)

    def test_planner_view_replay_gt_env_uses_single_execution_replay(self):
        class SettlingEnv:
            def __init__(self):
                self.relative = True
                self.action_scale = 1.0
                self.window_size = 512.0
                self._state = np.zeros((5,), dtype=np.float32)
                self._planning_fidelity_level_idx = 0

            def prepare(self, seed, init_state, goal_state=None):
                del seed, goal_state
                self._state = np.asarray(init_state, dtype=np.float32).copy()
                self._state[0] += 10.0
                return {"visual": self.render("rgb_array", include_start_pose=False)}, self._state.copy()

            def render(self, mode, include_start_pose=False):
                del mode
                frame = np.zeros((4, 4, 3), dtype=np.uint8)
                frame[:, :, 0] = int(round(float(self._state[0])))
                if include_start_pose:
                    frame[0, 0, 1] = 255
                return frame

            def step(self, action):
                action = np.asarray(action, dtype=np.float32)
                self._state[0] += float(action[0])
                return {"visual": self.render("rgb_array", include_start_pose=False)}, 0.0, False, {"state": self._state.copy()}

            def set_planning_fidelity_level(self, level_idx):
                self._planning_fidelity_level_idx = int(level_idx)

            def _apply_planning_fidelity_visual(self, img):
                return np.asarray(img).copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "trace.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "plan_config": {"env": {"render_size": 4}},
                        "sample": {},
                        "init_state": [0.0, 0, 0, 0, 0],
                        "goal_state": [0.0, 0, 0, 0, 0],
                    },
                    f,
                )
            np.savez_compressed(
                os.path.join(tmpdir, "trace.npz"),
                executed_actions=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
                trajectory=np.asarray(
                    [
                        [0.0, 0, 0, 0, 0],
                        [2.0, 0, 0, 0, 0],
                        [4.0, 0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                replan_rollout_levels=np.asarray([[0, 0]], dtype=np.int32),
                replan_rollout_lengths=np.asarray([2], dtype=np.int32),
                replan_step_starts=np.asarray([0], dtype=np.int32),
            )
            captured = {}

            def _capture_video(path, frames, fps=15, output_size=256):
                del path, fps, output_size
                captured["frames"] = [np.asarray(fr).copy() for fr in frames]

            with mock.patch.object(
                planning_media,
                "_build_runtime",
                return_value=(OmegaConf.create({"env": {"render_size": 4}}), {"env": SettlingEnv(), "backend": "gt_env"}),
            ):
                with mock.patch.object(planning_media, "overlay_action_targets_on_frames", side_effect=lambda frames, states, actions, overlay_spec: frames):
                    with mock.patch.object(planning_media.single_plan, "_write_video_mp4", side_effect=_capture_video):
                        planning_media.render_media(tmpdir, schedule=None, rollout_id=None, media=["planner_view_replay"])

            self.assertEqual([int(fr[1, 1, 0]) for fr in captured["frames"]], [10, 11, 12])
            self.assertTrue(all(int(fr[0, 0, 1]) == 255 for fr in captured["frames"]))

    def test_predicted_backend_replay_gt_env_uses_trajectory_start_state(self):
        class SettlingEnv:
            def __init__(self):
                self.relative = True
                self.action_scale = 1.0
                self.window_size = 512.0
                self._state = np.zeros((5,), dtype=np.float32)
                self._planning_fidelity_level_idx = 0

            def prepare(self, seed, init_state, goal_state=None):
                del seed, goal_state
                self._state = np.asarray(init_state, dtype=np.float32).copy()
                self._state[0] += 10.0
                return {"visual": self.render("rgb_array", include_start_pose=False)}, self._state.copy()

            def render(self, mode, include_start_pose=False):
                del mode
                frame = np.zeros((4, 4, 3), dtype=np.uint8)
                frame[:, :, 0] = int(round(float(self._state[0])))
                if include_start_pose:
                    frame[0, 0, 1] = 255
                return frame

            def step(self, action):
                del action
                return {"visual": self.render("rgb_array", include_start_pose=False)}, 0.0, False, {"state": self._state.copy()}

            def set_planning_fidelity_level(self, level_idx):
                self._planning_fidelity_level_idx = int(level_idx)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "trace.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "plan_config": {"env": {"render_size": 4}},
                        "sample": {},
                        "init_state": [0.0, 0, 0, 0, 0],
                        "goal_state": [0.0, 0, 0, 0, 0],
                    },
                    f,
                )
            np.savez_compressed(
                os.path.join(tmpdir, "trace.npz"),
                replan_action_seqs=np.asarray([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
                replan_start_states=np.asarray([[10.0, 0, 0, 0, 0]], dtype=np.float32),
                replan_rollout_levels=np.asarray([[0, 0]], dtype=np.int32),
                replan_rollout_lengths=np.asarray([2], dtype=np.int32),
                replan_step_starts=np.asarray([0], dtype=np.int32),
                trajectory=np.asarray(
                    [
                        [0.0, 0, 0, 0, 0],
                        [1.0, 0, 0, 0, 0],
                        [2.0, 0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                executed_actions=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            )
            captured = {}

            def _capture_video(path, frames, fps=15, output_size=256):
                del path, fps, output_size
                captured["frames"] = [np.asarray(fr).copy() for fr in frames]

            with mock.patch.object(
                planning_media,
                "_build_runtime",
                return_value=(OmegaConf.create({"env": {"render_size": 4}}), {"env": SettlingEnv(), "backend": "gt_env"}),
            ):
                with mock.patch.object(planning_media, "overlay_action_targets_on_frames", side_effect=lambda frames, states, actions, overlay_spec: frames):
                    with mock.patch.object(planning_media.single_plan, "_write_video_mp4", side_effect=_capture_video):
                        planning_media.render_media(tmpdir, schedule=None, rollout_id=None, media=["predicted_backend_replay"])

            self.assertEqual([int(fr[1, 1, 0]) for fr in captured["frames"]], [10, 10, 10])

    def test_predicted_backend_replay_particle_uses_info_state_not_reward(self):
        class DummyRuntimeEnv:
            relative = True
            action_scale = 1.0
            window_size = 512.0

        class DummyParticleBackend:
            def __init__(self):
                self._state = np.asarray([10.0, 20.0, 0, 0, 0], dtype=np.float32)

            def set_planning_fidelity_level(self, level_idx):
                del level_idx

            def prepare(self, seed, init_state, goal_state=None, with_visual=True):
                del seed, goal_state, with_visual
                self._state = np.asarray(init_state, dtype=np.float32).copy()
                obs = {
                    "visual": np.zeros((4, 4, 3), dtype=np.uint8),
                    "state": self._state.copy(),
                }
                return obs, self._state.copy()

            def step(self, action, with_visual=True):
                del action, with_visual
                self._state = self._state.copy()
                self._state[0] += 1.0
                obs = {
                    "visual": np.zeros((4, 4, 3), dtype=np.uint8),
                    "state": self._state.copy(),
                }
                return obs, 123.0, False, {"state": self._state.copy()}

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "trace.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "plan_config": {"env": {"render_size": 4}},
                        "sample": {},
                        "init_state": [10.0, 20.0, 0, 0, 0],
                        "goal_state": [15.0, 20.0, 0, 0, 0],
                        "action_format": "env_input",
                        "action_relative": True,
                        "action_scale": 1.0,
                    },
                    f,
                )
            np.savez_compressed(
                os.path.join(tmpdir, "trace.npz"),
                replan_action_seqs=np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
                replan_start_states=np.asarray([[10.0, 20.0, 0, 0, 0]], dtype=np.float32),
                replan_rollout_levels=np.asarray([[0, 0]], dtype=np.int32),
                replan_rollout_lengths=np.asarray([2], dtype=np.int32),
            )
            captured = {}

            def _capture_video(path, frames, fps=15, output_size=256):
                del path, fps, output_size
                captured["frames"] = [np.asarray(fr).copy() for fr in frames]

            runtime = {
                "env": DummyRuntimeEnv(),
                "backend": "particle_sim",
                "planner": type("Planner", (), {"backend": DummyParticleBackend()})(),
            }
            with mock.patch.object(
                planning_media,
                "_build_runtime",
                return_value=(OmegaConf.create({"env": {"render_size": 4}}), runtime),
            ):
                with mock.patch.object(planning_media.single_plan, "_write_video_mp4", side_effect=_capture_video):
                    outputs = planning_media.render_media(tmpdir, schedule=None, rollout_id=None, media=["predicted_backend_replay"])

            self.assertEqual(outputs, [os.path.join(tmpdir, "predicted_backend_replay.mp4")])
            self.assertEqual(len(captured["frames"]), 3)
            self.assertGreater(int(np.sum(captured["frames"][0][:, :, 0])), 0)


if __name__ == "__main__":
    unittest.main()
