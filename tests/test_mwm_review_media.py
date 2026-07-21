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

    def test_dataset_frame_uses_global_row_access_for_pixels(self) -> None:
        from mwm.benchmark.review_media import _dataset_frame

        class Dataset:
            def __init__(self) -> None:
                self.indexed_rows: list[int] = []

            def get_row_data(self, row: int) -> dict[str, object]:
                self.indexed_rows.append(row)
                return {"pixels": ["global_frame"]}

            def __getitem__(self, row: int) -> dict[str, object]:
                raise AssertionError(f"sample index {row} is not a global Lance row")

            def get_col_data(self, key: str) -> object:
                raise AssertionError(f"should not materialize column {key}")

        dataset = Dataset()

        self.assertEqual(_dataset_frame(dataset, 7, "pixels"), "global_frame")
        self.assertEqual(dataset.indexed_rows, [7])

    def test_executed_action_prefix_drops_masked_vector_steps(self) -> None:
        from mwm.eval.review_trace import executed_action_prefix

        trace = [[0.1, 0.2], [0.3, 0.4], [float("nan"), float("nan")], [0.9, 1.0]]

        self.assertEqual(executed_action_prefix(trace), trace[:2])

    def test_targeted_lance_dataset_reads_only_requested_global_rows(self) -> None:
        import io

        import lancedb
        import numpy as np
        from PIL import Image
        import pyarrow as pa

        from mwm.benchmark.replay_runtime import TargetedLanceReviewDataset

        def encoded(value: int) -> bytes:
            buffer = io.BytesIO()
            Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(buffer, format="JPEG")
            return buffer.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            table = pa.table(
                {
                    "episode_idx": pa.array([0, 0, 0, 0], type=pa.int64()),
                    "step_idx": pa.array([0, 1, 2, 3], type=pa.int64()),
                    "action": pa.array([[0.0], [1.0], [2.0], [3.0]], type=pa.list_(pa.float32(), 1)),
                    "pixels": pa.array([encoded(10), encoded(20), encoded(30), encoded(40)], type=pa.binary()),
                    "state": pa.array(
                        [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]],
                        type=pa.list_(pa.float32(), 2),
                    ),
                }
            )
            lancedb.connect(tmp).create_table("tiny", data=table)
            dataset = TargetedLanceReviewDataset(
                Path(tmp) / "tiny.lance",
                start_row=1,
                pixels_key="pixels",
                action_key="action",
                env_id="swm/PushT-v1",
                restore_import_path=None,
            )

            chunk = dataset.load_chunk([0], [1], [3])[0]

            self.assertEqual(dataset.column_names, ["pixels", "state"])
            self.assertEqual(tuple(chunk["pixels"].shape), (2, 3, 8, 8))
            self.assertEqual(chunk["state"].tolist(), [[1.0, 1.5], [2.0, 2.5]])
            self.assertEqual(tuple(dataset.get_frame(3).shape), (3, 8, 8))

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
            self.assertIn("Rollout Review", text)
            self.assertIn("Start with failures", text)
            self.assertIn("--serve", text)
            self.assertIn("data-filter=\"failure\"", text)
            self.assertIn("class=\"table-scroll\"", text)

    def test_rollout_page_explains_media_and_links_aligned_runs(self) -> None:
        from mwm.benchmark.review_server import rollout_page_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_name, success in (("000_upstream", False), ("001_candidate", True)):
                eval_path = root / run_name / "eval.json"
                video_path = eval_path.parent / "review_media" / "episode_0000" / "env.mp4"
                video_path.parent.mkdir(parents=True)
                video_path.write_bytes(b"mp4")
                write_json(
                    eval_path,
                    {
                        "benchmark_name": run_name,
                        "review_rollouts": [
                            {
                                "episode_index": 0,
                                "dataset_episode": 7,
                                "start_step": 3,
                                "goal_step": 9,
                                "success": success,
                                "action_trace": [[0.0], [float("nan")]],
                                "fidelity_trace": [{"t": 0, "level_idx": 0, "K": 1}],
                            }
                        ],
                        "review_media": {
                            "rollouts": {
                                "episode_0000": {
                                    "env": {"path": str(video_path), "kind": "env"}
                                }
                            }
                        },
                    },
                )

            text = rollout_page_html(root, root / "001_candidate" / "eval.json", 0)

            self.assertIn("What to look for", text)
            self.assertIn("not a predicted planner rollout", text)
            self.assertIn("Compare this episode across runs", text)
            self.assertIn("000_upstream · failure", text)
            self.assertIn("Env ready", text)
            self.assertIn("1 executable", text)
            self.assertIn("2 actions max", text)
            self.assertIn("later vector slots were masked", text)
            self.assertIn("Re-render existing media", text)
            self.assertIn("window.location.reload(), 700", text)

    def test_server_address_validation_reports_conflicts(self) -> None:
        import socket

        from mwm.benchmark.review_server import validate_server_address

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        try:
            with self.assertRaisesRegex(OSError, "port may already be in use"):
                validate_server_address("127.0.0.1", listener.getsockname()[1])
            with self.assertRaisesRegex(ValueError, "localhost"):
                validate_server_address("0.0.0.0", 8765)
        finally:
            listener.close()

    def test_rollout_page_treats_stale_media_entry_as_missing(self) -> None:
        from mwm.benchmark.review_server import rollout_page_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "000_run" / "eval.json"
            missing_video = eval_path.parent / "review_media" / "episode_0000" / "env.mp4"
            write_json(
                eval_path,
                {
                    "review_rollouts": [
                        {"episode_index": 0, "success": False, "action_trace": [[0.0]]}
                    ],
                    "review_media": {
                        "rollouts": {"episode_0000": {"env": {"path": str(missing_video)}}}
                    },
                },
            )

            text = rollout_page_html(root, eval_path, 0)

            self.assertIn(">Render Env</button>", text)
            self.assertIn("No media rendered yet.", text)
            self.assertNotIn("data-media-kind='env'", text)

    def test_review_server_status_and_unsupported_render_response(self) -> None:
        import json
        from http.server import ThreadingHTTPServer
        import threading
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        from mwm.benchmark.review_server import ReviewRequestHandler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "000_run" / "eval.json"
            write_json(eval_path, {"review_rollouts": [{"episode_index": 0, "success": False}]})

            class QuietReviewRequestHandler(ReviewRequestHandler):
                def log_message(self, format: str, *args: object) -> None:
                    del format, args

            def handler(*args: object, **kwargs: object) -> ReviewRequestHandler:
                return QuietReviewRequestHandler(*args, root=root, **kwargs)

            from mwm.benchmark.review_server import ReviewRenderManager

            render_manager = ReviewRenderManager()

            def managed_handler(*args: object, **kwargs: object) -> ReviewRequestHandler:
                return QuietReviewRequestHandler(
                    *args,
                    root=root,
                    render_manager=render_manager,
                    **kwargs,
                )

            server = ThreadingHTTPServer(("127.0.0.1", 0), managed_handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(f"{base}/api/status") as response:
                    self.assertEqual(json.load(response)["ok"], True)

                body = json.dumps(
                    {"eval_path": "000_run/eval.json", "episode_index": 0, "sources": ["bogus"]}
                ).encode("utf-8")
                request = Request(
                    f"{base}/api/render-rollout",
                    data=body,
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                    job = json.load(response)
                for _ in range(100):
                    with urlopen(f"{base}/api/render-status?job_id={job['job_id']}") as response:
                        status = json.load(response)
                    if status["status"] not in {"queued", "running"}:
                        break
                    import time

                    time.sleep(0.01)
                self.assertEqual(status["status"], "failed")
                self.assertIn("unknown media source", status["error"])
            finally:
                server.shutdown()
                server.server_close()
                render_manager.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
