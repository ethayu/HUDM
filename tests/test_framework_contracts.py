from __future__ import annotations

import os
import tempfile
import unittest

import torch

from hudm.metrics import pose_metrics
from hudm.world_io import checkpoint_epochs, latest_checkpoint_epoch, load_world_checkpoint, save_world_checkpoint
from models.world.model import HierWorldModel


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


if __name__ == "__main__":
    unittest.main()
