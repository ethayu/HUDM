from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from gymnasium import spaces

from datasets.swm_hdf5 import SWMHDF5Episodes
from hudm.swm_envs import validate_continuous_box_action_space
from hudm.swm_restore import eval_callables_for_env, restore_spec_for_env, validate_restore_columns
from hudm.world_io import (
    build_world_model_from_metadata,
    load_world_metadata,
    save_world_checkpoint,
)
from models.world.model import HierWorldModel
from plan_swm import _validate_dataset_metadata
from collect_swm import _record_dataset_to_path
from planning.cem_core import SharedCEMCore
from planning.swm_latent_cem import SWMLatentCEMPlanner


def alias_restore_spec_builder(env_id, columns):
    del columns
    return {
        "spec_id": "alias_restore",
        "env_ids": [env_id],
        "required_columns": ["restore_state"],
        "callables": [
            {
                "method": "set_restore_state",
                "args": {"restore_state": {"value": "restore_state", "in_dataset": True}},
            }
        ],
    }


class SWMFrameworkTests(unittest.TestCase):
    def _write_h5(self, path: str) -> None:
        lengths = np.asarray([5, 5, 5], dtype=np.int32)
        offsets = np.asarray([0, 5, 10], dtype=np.int64)
        pixels = np.arange(15 * 4 * 4 * 3, dtype=np.uint8).reshape(15, 4, 4, 3)
        action = np.zeros((15, 2), dtype=np.float32)
        for i in range(15):
            action[i] = [float(i), float(i + 100)]
        state = np.arange(15 * 2, dtype=np.float32).reshape(15, 2)
        with h5py.File(path, "w") as f:
            f.create_dataset("ep_len", data=lengths)
            f.create_dataset("ep_offset", data=offsets)
            f.create_dataset("pixels", data=pixels)
            f.create_dataset("action", data=action)
            f.create_dataset("state", data=state)

    def test_swm_hdf5_dataset_uses_swm_left_shifted_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tiny.h5")
            self._write_h5(path)

            ds = SWMHDF5Episodes(path, horizon=3, split="all")
            item = ds[0]

            self.assertEqual(tuple(item["x"].shape), (3, 3, 4, 4))
            self.assertEqual(tuple(item["a"].shape), (2, 2))
            np.testing.assert_allclose(item["a"].numpy(), np.asarray([[0, 100], [1, 101]], dtype=np.float32))
            self.assertTrue(torch.all(item["mask"]))

    def test_swm_hdf5_dataset_supports_swm_eval_load_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tiny.h5")
            self._write_h5(path)

            ds = SWMHDF5Episodes(path, horizon=3, split="all")
            chunk = ds.load_chunk(np.asarray([0]), np.asarray([1]), np.asarray([4]))

            self.assertNotIn("ep_len", ds.column_names)
            self.assertNotIn("ep_offset", ds.column_names)
            self.assertEqual(len(chunk), 1)
            self.assertEqual(tuple(chunk[0]["pixels"].shape), (3, 3, 4, 4))
            self.assertEqual(tuple(chunk[0]["action"].shape), (3, 2))
            np.testing.assert_allclose(chunk[0]["state"][0].numpy(), np.asarray([2.0, 3.0], dtype=np.float32))

    def test_eval_pair_sampling_matches_swm_goal_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tiny.h5")
            self._write_h5(path)

            ds = SWMHDF5Episodes(path, horizon=2, split="all")
            pair = ds.sample_eval_start_goal_pairs(count=1, goal_offset_steps=3, seed=0)[0]

            self.assertEqual(pair.goal_step - pair.start_step, 2)
            self.assertEqual(pair.goal_row - pair.start_row, 2)

    def test_restore_registry_builds_eval_callables(self):
        spec = restore_spec_for_env("swm/TwoRoom-v1")
        self.assertEqual(spec.spec_id, "point_state_goal_state")
        validate_restore_columns("swm/TwoRoom-v1", {"state", "goal_state", "pixels", "action"})
        self.assertTrue(restore_spec_for_env("swm/OGBPointMaze-v0").needs_restore_recorder)

        spec_id, callables = eval_callables_for_env(
            "swm/TwoRoom-v1",
            {"state", "goal_state", "pixels", "action"},
        )

        self.assertEqual(spec_id, "point_state_goal_state")
        self.assertEqual([c["method"] for c in callables], ["_set_state", "_set_goal_state"])
        self.assertEqual(callables[1]["args"]["goal_state"]["value"], "goal_state")

        with self.assertRaisesRegex(ValueError, "goal_state"):
            validate_restore_columns("swm/TwoRoom-v1", {"state", "pixels", "action"})

    def test_continuous_box_action_validation_rejects_discrete(self):
        with self.assertRaisesRegex(ValueError, "continuous Box"):
            validate_continuous_box_action_space(spaces.Discrete(3), "swm/Discrete-v0")

    def test_user_restore_adapter_import_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, "adapter.py")
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        def build_restore_spec(env_id, columns):
                            return {
                                "spec_id": "custom_restore",
                                "env_ids": [env_id],
                                "required_columns": ["restore_state"],
                                "eval_callables": [
                                    {
                                        "method": "set_restore_state",
                                        "args": {
                                            "restore_state": {"value": "restore_state", "in_dataset": True}
                                        },
                                    }
                                ],
                            }
                        """
                    )
                )
            sys.path.insert(0, tmpdir)
            try:
                spec_id, callables = eval_callables_for_env(
                    "swm/Custom-v0",
                    {"pixels", "action", "restore_state"},
                    import_path="adapter:build_restore_spec",
                )
            finally:
                sys.path.remove(tmpdir)

        self.assertEqual(spec_id, "custom_restore")
        self.assertEqual(callables[0]["method"], "set_restore_state")

    def test_user_restore_adapter_accepts_callables_alias(self):
        spec_id, callables = eval_callables_for_env(
            "swm/Custom-v0",
            {"pixels", "action", "restore_state"},
            import_path="tests.test_swm_framework:alias_restore_spec_builder",
        )

        self.assertEqual(spec_id, "alias_restore")
        self.assertEqual(callables[0]["method"], "set_restore_state")

    def test_world_model_rejects_non_rgb_input_mode(self):
        with self.assertRaisesRegex(ValueError, "RGB-only"):
            HierWorldModel(K=[4], D=4, action_dim=2, input="state", image_shape=(32, 32))

    def test_collect_refuses_existing_hdf5_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "existing.h5")
            with open(path, "wb") as f:
                f.write(b"")

            with self.assertRaisesRegex(FileExistsError, "Refusing to append"):
                _record_dataset_to_path(object(), Path(path), episodes=1, seed=0)

    def test_world_checkpoint_metadata_roundtrip(self):
        model = HierWorldModel(K=[4], D=4, action_dim=2, image_shape=(32, 32))
        metadata = {
            "env_id": "swm/PushT-v1",
            "restore_spec": "pusht_state_goal_state",
            "image_shape": [32, 32],
            "action_dim": 2,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
            "model": {
                "input": "images",
                "D": 4,
                "K": [4],
                "decoder_mode": "per_level",
                "dynamics_mode": "per_level",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_world_checkpoint(model, tmpdir, epoch=3, metadata=metadata)
            loaded = load_world_metadata(tmpdir)
            self.assertEqual(loaded["epoch"], 3)
            restored = build_world_model_from_metadata(loaded)
            self.assertEqual(restored.image_shape, (32, 32))
            self.assertEqual(restored.K, [4])

    def test_planner_rejects_dataset_checkpoint_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tiny.h5")
            self._write_h5(path)
            with open(path + ".metadata.json", "w", encoding="utf-8") as f:
                f.write(
                    '{"format":"swm_hdf5","env_id":"swm/TwoRoom-v1","restore_spec":"point_target_state",'
                    '"image_shape":[4,4],"action_dim":2,'
                    '"action_low":[-1.0,-1.0],"action_high":[1.0,1.0],'
                    '"dataset":{"pixels_key":"pixels","action_key":"action"}}'
                )
            ds = SWMHDF5Episodes(path, horizon=2, split="all")
            checkpoint_metadata = {
                "format": "swm_hdf5",
                "env_id": "swm/PushT-v1",
                "restore_spec": "pusht_state_goal_state",
                "image_shape": [4, 4],
                "action_dim": 2,
                "action_low": [-1.0, -1.0],
                "action_high": [1.0, 1.0],
                "dataset": {"pixels_key": "pixels", "action_key": "action"},
                "model": {"input": "images"},
            }
            with self.assertRaisesRegex(ValueError, "env_id"):
                _validate_dataset_metadata(ds, checkpoint_metadata)

    def test_cem_clamps_injected_action_plan_without_rank_broadcast(self):
        core = SharedCEMCore(
            action_dim=2,
            horizon=3,
            action_low=np.asarray([-1.0, -2.0], dtype=np.float32),
            action_high=np.asarray([1.0, 2.0], dtype=np.float32),
            pop_size=4,
            elite_frac=0.5,
            n_iter=1,
            init_std=1.0,
            fidelity_cfg={},
            num_levels=1,
            rollout_modes=("fixed",),
            device=torch.device("cpu"),
        )
        plan = torch.tensor([[3.0, -3.0], [0.5, 0.5], [-4.0, 4.0]], dtype=torch.float32)
        clamped = core._clamp_action_tensor(plan)
        self.assertEqual(tuple(clamped.shape), (3, 2))
        np.testing.assert_allclose(
            clamped.numpy(),
            np.asarray([[1.0, -2.0], [0.5, 0.5], [-1.0, 2.0]], dtype=np.float32),
        )

    def test_swm_latent_cost_uses_terminal_l2_prefix(self):
        class DummyModel:
            K = [2]
            D = 2

            def eval(self):
                return self

            def predict_next(self, level, z, a):
                del level
                return z[..., :2] + a[..., :2]

        planner = SWMLatentCEMPlanner(
            DummyModel(),
            horizon=2,
            action_dim=2,
            action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_high=np.asarray([1.0, 1.0], dtype=np.float32),
            pop_size=2,
            elite_frac=0.5,
            n_iter=1,
            device=torch.device("cpu"),
        )
        actions = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        costs, _, _ = planner._evaluate_population(
            actions,
            z0=torch.zeros(1, 2),
            z_goal=torch.zeros(1, 2),
            base_level_idx=0,
            rollout_levels=[0, 0],
        )

        self.assertAlmostEqual(float(costs[0]), 0.0, places=6)
        self.assertAlmostEqual(float(costs[1]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
