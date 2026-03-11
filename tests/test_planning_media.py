from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from scripts import planning_media


class PlanningMediaTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
