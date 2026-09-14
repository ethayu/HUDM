from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from omegaconf import OmegaConf

from mwm.benchmark.live_review import (
    LiveBenchmark,
    _snapshot_fingerprint,
    collect_completed_cells,
    render_live_snapshot,
)
from mwm.benchmark.matrix import _configure_run_paths, _run_dir
from mwm.io import file_sha256, write_json, write_metrics_jsonl


class LiveReviewTests(unittest.TestCase):
    def _benchmark(self, root: Path) -> LiveBenchmark:
        output_dir = root / "benchmark"
        output_dir.mkdir()
        manifest_path = root / "manifest.json"
        write_json(manifest_path, {"manifest_sha256": "semantic-manifest"})
        run = OmegaConf.create(
            {
                "name": "schedule_a",
                "cell_id": "schedule_a__pop20__elite0p1__iter5",
                "role": "mwm_dense",
                "matrix_index": 0,
            }
        )
        run_cfg = OmegaConf.create(
            {
                "env_id": "swm/Test-v1",
                "checkpoint": {"run_dir": "checkpoint", "epoch": None},
                "env": {"max_steps": 10},
                "eval": {"seed": 42, "episodes": 1},
                "planner": {"pop_size": 20, "elite_frac": 0.1, "n_iter": 5},
            }
        )
        cfg = OmegaConf.create(
            {
                "title": "Live test",
                "seed": 42,
                "sweep": {"planner.pop_size": [20]},
            }
        )
        return LiveBenchmark(
            config_path=root / "benchmark.yaml",
            cfg=cfg,
            output_dir=output_dir,
            resolved=[(run, run_cfg)],
            manifest_path=manifest_path,
            manifest_info={"group": "test", "path": str(manifest_path)},
        )

    def _write_completed_cell(
        self,
        benchmark: LiveBenchmark,
        *,
        success_rate: float = 0.5,
    ) -> tuple[Path, dict[str, object]]:
        run, run_cfg = benchmark.resolved[0]
        run_dir = _run_dir(benchmark.output_dir, run, 0)
        run_dir.mkdir(parents=True)
        _configure_run_paths(run_cfg, run_dir, benchmark.manifest_path)
        resolved = run_dir / "resolved_config.yaml"
        resolved.write_text(OmegaConf.to_yaml(run_cfg), encoding="utf-8")
        row: dict[str, object] = {
            "name": "schedule_a",
            "cell_id": str(run.cell_id),
            "base_name": "schedule_a",
            "strategy": "mwm_dense",
            "sweep_key": '{"planner.pop_size":20}',
            "sweep_params": '{"planner.pop_size":20}',
            "env_id": str(run_cfg.env_id),
            "config_sha256": file_sha256(resolved),
            "manifest_sha256": "semantic-manifest",
            "manifest_file_sha256": file_sha256(benchmark.manifest_path),
            "episodes": 1,
            "success_rate": success_rate,
            "plans": 1,
            "steps": 1,
            "latent_work_total": 10,
            "bits_used_total": 10,
            "dynamics_flops_total": 100,
            "cem_cost_calls": 1,
            "candidate_action_values": 20,
            "pop_size": 20,
            "elite_frac": 0.1,
            "topk": 2,
            "n_iter": 5,
            "plan_time_total_sec": 0.1,
            "wall_time_sec": 1.0,
            "schedule_level_counts": "{}",
            "schedule_k_counts": "{}",
            "schedule": "Rollout=fine -> coarse",
            "role": "mwm_dense",
            "seed": 42,
            "output_json": str(run_dir / "eval.json"),
        }
        write_json(run_dir / "eval.json", {"result": "complete"})
        write_json(run_dir / "summary.json", {"run": row})
        write_json(run_dir / "planning_diagnostics.json", {})
        write_metrics_jsonl(run_dir / "metrics.jsonl", [row])
        write_metrics_jsonl(run_dir / "episode_traces.jsonl", [{"episode_index": 0}])
        return run_dir, row

    def test_zero_completed_cells_renders_isolated_live_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = self._benchmark(Path(tmp))

            report = render_live_snapshot(benchmark, refresh_seconds=5)

            self.assertEqual(report["completed_cells"], 0)
            self.assertTrue((benchmark.output_dir / "review.live.html").is_file())
            self.assertTrue((benchmark.output_dir / "summary.live.json").is_file())
            self.assertFalse((benchmark.output_dir / "review.html").exists())
            self.assertFalse((benchmark.output_dir / "summary.json").exists())
            html = (benchmark.output_dir / "review.live.html").read_text(encoding="utf-8")
            self.assertIn("0/1 cells complete", html)
            self.assertIn("Provisional Interactive Pareto Frontier", html)

    def test_collector_accepts_complete_cell_and_rejects_stale_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = self._benchmark(Path(tmp))
            run_dir, _ = self._write_completed_cell(benchmark)

            rows, missing = collect_completed_cells(benchmark)
            self.assertEqual(len(rows), 1)
            self.assertEqual(missing, [])

            saved = OmegaConf.load(run_dir / "resolved_config.yaml")
            saved.planner.pop_size = 999
            (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(saved), encoding="utf-8")
            rows, missing = collect_completed_cells(benchmark)
            self.assertEqual(rows, [])
            self.assertEqual(missing, [str(benchmark.resolved[0][0].cell_id)])

    def test_collector_rejects_manifest_changed_at_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = self._benchmark(Path(tmp))
            self._write_completed_cell(benchmark)
            write_json(benchmark.manifest_path, {"manifest_sha256": "replacement"})

            rows, missing = collect_completed_cells(benchmark)

            self.assertEqual(rows, [])
            self.assertEqual(len(missing), 1)

    def test_malformed_partial_sidecar_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = self._benchmark(Path(tmp))
            run_dir, row = self._write_completed_cell(benchmark)
            (run_dir / "summary.json").write_text('{"run":', encoding="utf-8")
            self.assertEqual(collect_completed_cells(benchmark)[0], [])

            write_json(run_dir / "summary.json", {"run": row})
            self.assertEqual(len(collect_completed_cells(benchmark)[0]), 1)

    def test_fingerprint_changes_when_metrics_change_without_cell_count_change(self) -> None:
        first = [{"cell_id": "a", "success_rate": 0.1}]
        second = [{"cell_id": "a", "success_rate": 0.9}]
        self.assertNotEqual(_snapshot_fingerprint(first, 1), _snapshot_fingerprint(second, 1))

    def test_unchanged_cell_sidecars_reuse_validation_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = self._benchmark(Path(tmp))
            self._write_completed_cell(benchmark)
            from mwm.benchmark import matrix

            with mock.patch.object(matrix, "_completed_row", wraps=matrix._completed_row) as completed:
                self.assertEqual(len(collect_completed_cells(benchmark)[0]), 1)
                self.assertEqual(len(collect_completed_cells(benchmark)[0]), 1)

            self.assertEqual(completed.call_count, 1)

    def test_unchanged_snapshot_is_not_rebuilt_but_metric_change_is(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = self._benchmark(Path(tmp))
            run_dir, row = self._write_completed_cell(benchmark, success_rate=0.2)
            first = render_live_snapshot(benchmark, refresh_seconds=5)
            second = render_live_snapshot(benchmark, refresh_seconds=5)
            self.assertTrue(first["updated"])
            self.assertFalse(second["updated"])

            row["success_rate"] = 0.8
            write_json(run_dir / "summary.json", {"run": row})
            write_metrics_jsonl(run_dir / "metrics.jsonl", [row])
            third = render_live_snapshot(benchmark, refresh_seconds=5)
            self.assertTrue(third["updated"])

            html = (benchmark.output_dir / "review.live.html").read_text(encoding="utf-8")
            self.assertIn("liveInitialFingerprint", html)
            self.assertIn("Complete snapshot", html)
            self.assertLess(html.index("Interactive Pareto Frontier"), html.index("Supporting Views"))
            summary = json.loads((benchmark.output_dir / "summary.live.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {Path(path).name for path in summary["plots"]},
                {"success_vs_compute.png", "success_vs_wall_time.png", "strategy_legend.png"},
            )


class LiveReviewServerTests(unittest.TestCase):
    def test_live_server_can_skip_cuda_warmup_and_print_live_page(self) -> None:
        from mwm.benchmark import review_server

        fake_server = mock.MagicMock()
        fake_server.server_port = 8765
        manager = mock.MagicMock()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            review_server, "ThreadingHTTPServer", return_value=fake_server
        ), mock.patch.object(review_server, "ReviewRenderManager", return_value=manager), mock.patch(
            "builtins.print"
        ) as print_mock:
            review_server.serve_review(
                tmp,
                review_page="review.live.html",
                warmup=False,
            )

        manager.start_warmup.assert_not_called()
        print_mock.assert_any_call(
            "Serving benchmark review at http://127.0.0.1:8765/review.live.html",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
