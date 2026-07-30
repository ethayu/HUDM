from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mwm.io import load_json, write_json


class ReviewMediaTests(unittest.TestCase):
    def test_render_all_expands_to_every_media_mode(self) -> None:
        from unittest.mock import patch

        from mwm.benchmark.review_media import RenderedMedia, _render_rollout_media_unlocked

        calls: list[str] = []

        def rendered(kind: str) -> RenderedMedia:
            calls.append(kind)
            return RenderedMedia(kind, f"/{kind}.mp4", "trace", [])

        with (
            patch("mwm.benchmark.review_media.render_environment_video", side_effect=lambda *a, **k: rendered("env")),
            patch("mwm.benchmark.review_media.render_latent_reconstruction_video", side_effect=lambda *a, **k: rendered("latent_reconstruction")),
            patch("mwm.benchmark.review_media.render_latent_predictive_rollout_video", side_effect=lambda *a, **k: rendered("latent_predictive_rollout")),
        ):
            result = _render_rollout_media_unlocked(
                "/tmp/eval.json",
                episode_index=4,
                sources=["all"],
            )

        self.assertEqual(calls, ["env", "latent_reconstruction", "latent_predictive_rollout"])
        self.assertEqual([item["kind"] for item in result["rendered"]], calls)

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

    def test_review_rollout_persists_aligned_model_action_trace(self) -> None:
        from mwm.eval.review_trace import review_rollouts_for_batches

        rows = review_rollouts_for_batches(
            batches=[
                {
                    "pairs": [
                        {
                            "episode": 2,
                            "start_step": 1,
                            "goal_step": 6,
                            "start_row": 10,
                            "goal_row": 15,
                        }
                    ],
                    "review_trace": {
                        "action_trace": [[[10.0], [20.0], [float("nan")]]],
                        "model_action_trace": [[[1.0], [2.0], [float("nan")]]],
                    },
                    "planning_diagnostics": {
                        "trace": [
                            {
                                "mpc_iter": 0,
                                "cem_iter": 1,
                                "batch_start": 0,
                                "batch_end": 1,
                                "rollout_level_indices": [1],
                            }
                        ]
                    },
                }
            ],
            successes=[True],
            eval_budget=3,
            action_block=1,
            receding_horizon=3,
            k_values=[8, 16],
        )

        self.assertEqual(rows[0]["action_trace"], [[10.0], [20.0]])
        self.assertEqual(rows[0]["model_action_trace"], [[1.0], [2.0]])
        self.assertEqual(len(rows[0]["fidelity_trace"]), 2)

    def test_level_fallback_prefers_rollout_then_cem_base_then_mpc(self) -> None:
        from mwm.eval.review_trace import final_cem_rollout_levels

        self.assertEqual(
            final_cem_rollout_levels(
                {"rollout_level_indices": [3, 2], "base_level_idx": 1, "mpc_level_idx": 0}
            ),
            ([3, 2], "rollout_level_indices"),
        )
        self.assertEqual(
            final_cem_rollout_levels({"base_level_idx": 2, "mpc_level_idx": 1}),
            ([2], "base_level_idx"),
        )
        self.assertEqual(
            final_cem_rollout_levels({"mpc_level_idx": 1}),
            ([1], "mpc_level_idx"),
        )
        with self.assertRaisesRegex(ValueError, "no rollout_level_indices"):
            final_cem_rollout_levels({})

    def test_predictive_segments_map_blocks_replans_levels_and_early_tail(self) -> None:
        from mwm.benchmark.review_media import predictive_replan_segments

        trace = []
        levels = [[3, 2, 1], [2, 1, 0]]
        for replan_idx, replan_levels in enumerate(levels):
            for block_idx, level in enumerate(replan_levels):
                for primitive in range(2):
                    trace.append(
                        {
                            "t": replan_idx * 6 + block_idx * 2 + primitive,
                            "replan_idx": replan_idx,
                            "block_idx": block_idx,
                            "level_idx": level,
                            "K": [8, 16, 32, 64][level],
                        }
                    )

        segments = predictive_replan_segments(
            trace,
            action_count=13,
            action_block=2,
            receding_horizon=3,
        )

        self.assertEqual([segment["anchor_step"] for segment in segments], [0, 6])
        self.assertEqual(
            [[block["level_idx"] for block in segment["blocks"]] for segment in segments],
            levels,
        )
        self.assertEqual(segments[1]["blocks"][-1]["primitive_end"], 12)
        self.assertEqual(segments[1]["blocks"][-1]["distance_since_anchor"], 6)
        self.assertFalse(any(block["primitive_end"] > 12 for segment in segments for block in segment["blocks"]))

    def test_piecewise_predictions_are_autoregressive_and_reanchor(self) -> None:
        import numpy as np
        import torch

        from mwm.benchmark.review_media import (
            piecewise_predictive_latents,
            predictive_replan_segments,
        )

        class FakeScheduledModel:
            action_dim = 2

            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def rollout_with_schedule(self, infos, action_sequence, levels):
                anchor = float(infos["pixels"].mean())
                blocks = action_sequence.sum(dim=-1)[0, 0]
                endpoints = anchor + torch.cumsum(blocks, dim=0)
                predicted = torch.tensor([anchor], dtype=torch.float32).reshape(1, 1, 1, 1)
                predicted = torch.cat([predicted, endpoints.reshape(1, 1, -1, 1)], dim=2)
                self.calls.append(
                    {
                        "anchor": anchor,
                        "actions": action_sequence.detach().cpu().clone(),
                        "levels": list(levels),
                    }
                )
                return {"predicted_emb": predicted}

        trace = [
            {
                "replan_idx": replan,
                "block_idx": block,
                "level_idx": replan + block,
                "K": 10 * (replan + block + 1),
            }
            for replan in range(2)
            for block in range(2)
        ]
        segments = predictive_replan_segments(
            trace,
            action_count=8,
            action_block=2,
            receding_horizon=2,
        )
        observations = {
            0: {"pixels": np.full((1, 1, 2, 2, 3), 10, dtype=np.uint8)},
            4: {"pixels": np.full((1, 1, 2, 2, 3), 100, dtype=np.uint8)},
        }
        model = FakeScheduledModel()
        outputs = piecewise_predictive_latents(
            model,
            observations,
            np.arange(1, 9, dtype=np.float32).reshape(-1, 1),
            segments,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["levels"], [0, 1])
        self.assertEqual(model.calls[1]["levels"], [1, 2])
        first_anchor = float(model.calls[0]["anchor"])
        second_anchor = float(model.calls[1]["anchor"])
        self.assertAlmostEqual(first_anchor, 10 / 255.0, places=5)
        self.assertAlmostEqual(second_anchor, 100 / 255.0, places=5)
        self.assertAlmostEqual(float(outputs[1]["latent"]), first_anchor + 1 + 2 + 3 + 4, places=5)
        self.assertAlmostEqual(float(outputs[2]["latent"]), second_anchor + 5 + 6, places=5)

    def test_legacy_action_stats_scan_only_action_and_cache_by_version(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        import pyarrow as pa

        from mwm.benchmark import replay_runtime

        scans: list[list[str]] = []
        arrow = pa.table(
            {"action": pa.array([[1.0, 3.0], [3.0, 7.0]], type=pa.list_(pa.float32(), 2))}
        )

        class FakeLance:
            def scanner(self, *, columns):
                scans.append(list(columns))
                return SimpleNamespace(to_table=lambda: arrow)

        fake_table = SimpleNamespace(
            version=7,
            schema=SimpleNamespace(names=["action", "pixels", "state"]),
            to_lance=lambda: FakeLance(),
        )
        replay_runtime._ACTION_STATS_CACHE.clear()
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            replay_runtime,
            "_open_lance_review_table",
            return_value=fake_table,
        ):
            first, first_hit = replay_runtime.load_lance_action_stats(Path(tmp) / "tiny.lance", "action")
            second, second_hit = replay_runtime.load_lance_action_stats(Path(tmp) / "tiny.lance", "action")

        self.assertEqual(scans, [["action"]])
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertIs(first, second)
        self.assertEqual(first.mean.tolist(), [2.0, 5.0])
        self.assertEqual(first.scale.tolist(), [1.0, 2.0])
        self.assertEqual(first.transform([[3.0, 7.0]]).tolist(), [[1.0, 1.0]])

    def test_model_action_trace_prefers_saved_values_and_transforms_legacy_actions(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np
        from omegaconf import OmegaConf

        from mwm.benchmark.replay_runtime import ActionStats
        from mwm.benchmark.review_media import model_actions_for_rollout

        runtime = SimpleNamespace(
            model=object(),
            metadata={"action_preprocessing": "standard_scaler"},
            cfg=OmegaConf.create(
                {
                    "eval": {},
                    "data": {"action_preprocessing": "standard_scaler", "action_key": "action"},
                }
            ),
            dataset=SimpleNamespace(path="dataset.lance"),
        )
        saved, source = model_actions_for_rollout(
            {
                "action_trace": [[12.0, 26.0], [14.0, 29.0]],
                "model_action_trace": [[1.0, 2.0], [3.0, 4.0]],
            },
            runtime,
        )
        self.assertEqual(source, "model_action_trace")
        self.assertEqual(saved.tolist(), [[1.0, 2.0], [3.0, 4.0]])

        stats = ActionStats(
            mean=np.array([10.0, 20.0]),
            scale=np.array([2.0, 3.0]),
            dataset_path="dataset.lance",
            dataset_version=4,
            action_key="action",
        )
        with patch(
            "mwm.benchmark.replay_runtime.load_lance_action_stats",
            return_value=(stats, False),
        ) as load_stats:
            legacy, legacy_source = model_actions_for_rollout(
                {"action_trace": [[12.0, 26.0], [14.0, 29.0]]},
                runtime,
            )

        load_stats.assert_called_once_with("dataset.lance", "action", progress=None)
        self.assertEqual(legacy.tolist(), [[1.0, 2.0], [2.0, 3.0]])
        self.assertIn("Lance action statistics v4", legacy_source)

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
            self.assertIn("Latent predictive rollout", text)
            self.assertIn("complete action-block endpoints", text)
            self.assertIn("data-source='predictive'", text)
            self.assertIn("data-source='all'", text)
            self.assertIn(">Render all</button>", text)
            self.assertNotIn("Env + reconstruction", text)
            self.assertIn("Compare this episode across runs", text)
            self.assertIn("000_upstream · failure", text)
            self.assertIn("Env ready", text)
            self.assertIn("1 executable", text)
            self.assertIn("2 actions max", text)
            self.assertIn("later vector slots were masked", text)
            self.assertIn("Re-render existing media", text)
            self.assertIn("window.location.reload(), 700", text)
            self.assertIn("environment and CUDA checkpoint warm-up", text)

    def test_review_warmup_deduplicates_checkpoints_and_initializes_env(self) -> None:
        from unittest.mock import patch

        from mwm.benchmark.review_server import warm_review_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("000_a", "001_b"):
                run_dir = root / name
                run_dir.mkdir()
                (run_dir / "resolved_config.yaml").write_text("device: cuda\n", encoding="utf-8")
                write_json(
                    run_dir / "eval.json",
                    {
                        "checkpoint_run_dir": "shared_checkpoint",
                        "review_rollouts": [
                            {"episode_index": 0, "start_row": 10, "goal_row": 20}
                        ],
                    },
                )

            with patch(
                "mwm.benchmark.replay_runtime.warm_review_runtime",
                return_value={"checkpoint": "shared_checkpoint", "device": "cuda", "env_id": "PushT"},
            ) as warm:
                report = warm_review_assets(root)

            warm.assert_called_once()
            self.assertEqual(report["checkpoint_count"], 1)
            self.assertEqual(report["warmed"][0]["run"], "000_a")
            self.assertEqual(report["warnings"], [])

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
                    status_payload = json.load(response)
                    self.assertEqual(status_payload["ok"], True)
                    self.assertEqual(status_payload["warmup"]["status"], "idle")

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
