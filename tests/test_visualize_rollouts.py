from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from scripts import visualize_rollouts


class VisualizeRolloutsTests(unittest.TestCase):
    def test_collect_episodes_to_show_supports_real_source(self):
        data = OmegaConf.create(
            {
                "zarr_path": "real.zarr",
                "split_ratio": 0.8,
                "synthetic": {"zarr_path": "synthetic.zarr"},
            }
        )
        real_episode = np.zeros((3, 4, 4, 3), dtype=np.float32)
        with mock.patch.object(
            visualize_rollouts,
            "load_episodes_from_zarr",
            side_effect=lambda path, **kwargs: [real_episode] if path == "real.zarr" else [],
        ) as loader:
            episodes, titles = visualize_rollouts.collect_episodes_to_show(
                data,
                split_name="valid",
                source="real",
                count=1,
            )

        self.assertEqual(len(episodes), 1)
        self.assertEqual(titles, ["Real Episode 1/1"])
        self.assertEqual(loader.call_count, 1)

    def test_collect_episodes_to_show_supports_mixed_source(self):
        data = OmegaConf.create(
            {
                "zarr_path": "real.zarr",
                "split_ratio": 0.8,
                "synthetic": {"zarr_path": "synthetic.zarr"},
            }
        )
        real_episode = np.zeros((3, 4, 4, 3), dtype=np.float32)
        synth_episode = np.ones((2, 4, 4, 3), dtype=np.float32)

        def fake_loader(path, **kwargs):
            del kwargs
            if path == "real.zarr":
                return [real_episode]
            if path == "synthetic.zarr":
                return [synth_episode]
            raise AssertionError(path)

        with mock.patch.object(visualize_rollouts, "load_episodes_from_zarr", side_effect=fake_loader):
            episodes, titles = visualize_rollouts.collect_episodes_to_show(
                data,
                split_name="valid",
                source="mixed",
                count=2,
            )

        self.assertEqual(len(episodes), 2)
        self.assertEqual(titles, ["Real Episode 1/1", "Synthetic Episode 1/1"])


if __name__ == "__main__":
    unittest.main()
