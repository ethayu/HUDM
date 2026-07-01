from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mwm.io import load_json, write_json


class ReviewMediaTests(unittest.TestCase):
    def test_final_cem_trace_expands_rollout_levels_by_action_block(self) -> None:
        from mwm.benchmark.review_media import fidelity_trace_from_planning_trace

        trace = fidelity_trace_from_planning_trace(
            planning_trace=[
                {
                    "mpc_iter": 0,
                    "cem_iter": 0,
                    "batch_start": 0,
                    "batch_end": 1,
                    "rollout_level_indices": [0, 0, 0],
                },
                {
                    "mpc_iter": 0,
                    "cem_iter": 2,
                    "batch_start": 0,
                    "batch_end": 1,
                    "rollout_level_indices": [3, 2, 1],
                },
                {
                    "mpc_iter": 1,
                    "cem_iter": 1,
                    "batch_start": 0,
                    "batch_end": 1,
                    "rollout_level_indices": [2, 1, 0],
                },
            ],
            batch_env=0,
            eval_budget=8,
            action_block=2,
            replan_interval=4,
            k_values=[8, 16, 32, 64],
        )

        self.assertEqual([row["level_idx"] for row in trace], [3, 3, 2, 2, 2, 2, 1, 1])
        self.assertEqual([row["K"] for row in trace], [64, 64, 32, 32, 32, 32, 16, 16])
        self.assertEqual(trace[4]["replan_idx"], 1)
        self.assertEqual(trace[4]["block_idx"], 0)

    def test_review_media_update_augments_eval_json_without_changing_metrics(self) -> None:
        from mwm.benchmark.review_media import record_review_media

        with tempfile.TemporaryDirectory() as tmp:
            eval_path = Path(tmp) / "run" / "eval.json"
            payload = {
                "env_id": "swm/PushT-v1",
                "episodes": 1,
                "swm_results": {"success_rate": 0.0},
                "review_rollouts": [{"episode_index": 0, "success": False}],
            }
            write_json(eval_path, payload)

            media_path = eval_path.parent / "review_media" / "episode_0000" / "latent_reconstruction.mp4"
            record_review_media(
                eval_path,
                episode_index=0,
                kind="latent_reconstruction",
                path=media_path,
                source_trace_type="fidelity_trace",
                warnings=["decoder unavailable"],
            )

            updated = load_json(eval_path)
            self.assertEqual(updated["swm_results"], payload["swm_results"])
            media = updated["review_media"]["rollouts"]["episode_0000"]["latent_reconstruction"]
            self.assertEqual(media["path"], str(media_path))
            self.assertEqual(media["source_trace_type"], "fidelity_trace")
            self.assertEqual(media["warnings"], ["decoder unavailable"])

    def test_environment_video_requires_action_trace(self) -> None:
        from mwm.benchmark.review_media import ReviewMediaUnsupported, render_environment_video

        with tempfile.TemporaryDirectory() as tmp:
            eval_path = Path(tmp) / "eval.json"
            write_json(
                eval_path,
                {
                    "review_rollouts": [
                        {
                            "episode_index": 0,
                            "batch": 0,
                            "batch_env": 0,
                            "start_step": 0,
                            "goal_step": 1,
                        }
                    ]
                },
            )

            with self.assertRaisesRegex(ReviewMediaUnsupported, "action_trace"):
                render_environment_video(eval_path, episode_index=0, force=False)

    def test_collect_video_paths_uses_swm_env_outputs(self) -> None:
        from mwm.eval.videos import collect_video_paths

        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp)
            for name in ["env_1.mp4", "env_0.mp4", "rollout_0000.mp4", "notes.txt"]:
                (video_dir / name).write_bytes(b"video")

            self.assertEqual([path.name for path in collect_video_paths(video_dir)], ["env_0.mp4", "env_1.mp4"])

    def test_dataset_frame_uses_indexed_access_for_pixels(self) -> None:
        from mwm.benchmark.review_media import _dataset_frame

        class Dataset:
            def __init__(self) -> None:
                self.indexed_rows: list[int] = []

            def __getitem__(self, row: int) -> dict[str, object]:
                self.indexed_rows.append(row)
                return {"pixels": ["frame0", "frame1"]}

            def get_col_data(self, key: str) -> object:
                raise AssertionError(f"should not materialize column {key}")

        dataset = Dataset()

        self.assertEqual(_dataset_frame(dataset, 7, "pixels"), "frame0")
        self.assertEqual(dataset.indexed_rows, [7])

    def test_server_rejects_eval_path_outside_review_root(self) -> None:
        from mwm.benchmark.review_server import resolve_eval_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmark"
            root.mkdir()
            inside = root / "000_run" / "eval.json"
            inside.parent.mkdir()
            inside.write_text("{}", encoding="utf-8")
            outside = Path(tmp) / "elsewhere" / "eval.json"
            outside.parent.mkdir()
            outside.write_text("{}", encoding="utf-8")

            self.assertEqual(resolve_eval_path(root, "000_run/eval.json"), inside.resolve())
            with self.assertRaisesRegex(ValueError, "outside"):
                resolve_eval_path(root, str(outside))

    def test_review_html_links_rollout_pages_and_embeds_recorded_media(self) -> None:
        from mwm.benchmark.html import write_review_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "000_run"
            run_dir.mkdir()
            eval_path = run_dir / "eval.json"
            video_path = run_dir / "review_media" / "episode_0000" / "latent_reconstruction.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"mp4")
            row = {
                "name": "run",
                "env_id": "swm/PushT-v1",
                "seed": 0,
                "role": "mwm_scheduled",
                "success_rate": 0.0,
                "episodes": 1,
                "wall_time_sec": 1.0,
                "bits_used_total": 1,
                "plans": 1,
                "manifest_sha256": "manifest",
                "config_sha256": "config",
                "output_json": str(eval_path),
            }
            payload = {
                "review_rollouts": [{"episode_index": 0, "success": False}],
                "review_media": {
                    "rollouts": {
                        "episode_0000": {
                            "latent_reconstruction": {
                                "path": str(video_path),
                                "kind": "latent_reconstruction",
                            }
                        }
                    }
                },
            }

            out = root / "review.html"
            write_review_html(out, "Review", [row], [payload], plots=[], expected_cells=1)

            text = out.read_text(encoding="utf-8")
            self.assertIn("rollouts/000_run/episode_0000.html", text)
            self.assertIn("<video", text)
            self.assertIn("review_media/episode_0000/latent_reconstruction.mp4", text)


if __name__ == "__main__":
    unittest.main()
