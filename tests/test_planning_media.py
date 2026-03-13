from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from hudm.artifacts import write_video_mp4
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


if __name__ == "__main__":
    unittest.main()
