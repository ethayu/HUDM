from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from mwm.benchmark import matrix as benchmark_matrix
from mwm.benchmark.config import DEFAULTS, merged_run_config, validate_benchmark_matrix
from mwm.benchmark.matrix import _compact_review_payload
from mwm.benchmark.matrix_identity import expected_cells_from_resolved
from mwm.benchmark.pareto import _strategy_color_map, pareto_frontier, write_pareto_html
from mwm.benchmark.sweep import expand_benchmark_runs


class BenchmarkSweepTests(unittest.TestCase):
    def _config(self, root: Path, **overrides: object):
        eval_path = root / "eval.yaml"
        eval_path.write_text(
            """
env_id: swm/PushT-v1
checkpoint: {run_dir: checkpoints_mwm/example, epoch: null}
data: {path: data/pusht_swm.lance, format: lance}
env: {max_steps: 100}
eval: {seed: 99, episodes: 2}
planner:
  pop_size: 25
  elite_frac: 0.2
  topk: 30
  n_iter: 3
  flop_accounting: none
""",
            encoding="utf-8",
        )
        base = {
            "env_id": "swm/PushT-v1",
            "seed": 7,
            "eval_config": str(eval_path),
            "manifest": {"group": "sweep_test", "path": str(root / "manifest.json")},
            "runs": [
                {
                    "name": "00_upstream_fixed_finest",
                    "role": "upstream_lewm_converted",
                    "checkpoint": "checkpoints_mwm/upstream_lewm_pusht",
                },
                {
                    "name": "01_dense_schedule",
                    "role": "mwm_scheduled",
                    "checkpoint": "checkpoints_mwm/dense_pusht",
                },
            ],
        }
        return OmegaConf.merge(DEFAULTS, base, overrides)

    def test_cartesian_sweep_expands_every_base_run_including_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(
                Path(tmp),
                sweep={
                    "planner.pop_size": [100, 300],
                    "planner.elite_frac": [0.05, 0.1],
                    "planner.n_iter": [5, 10],
                },
            )

            expanded = expand_benchmark_runs(cfg)

        self.assertEqual(len(expanded), 2 * 2 * 2 * 2)
        self.assertEqual(
            Counter(str(run.role) for run in expanded),
            {"upstream_lewm_converted": 8, "mwm_scheduled": 8},
        )
        upstream_combinations = {
            (int(run.planner.pop_size), float(run.planner.elite_frac), int(run.planner.n_iter))
            for run in expanded
            if run.role == "upstream_lewm_converted"
        }
        self.assertEqual(
            upstream_combinations,
            {
                (population, elite_frac, iterations)
                for population in (100, 300)
                for elite_frac in (0.05, 0.1)
                for iterations in (5, 10)
            },
        )

    def test_compact_review_payload_uses_trace_sidecar_and_drops_heavy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "000_run"
            run_dir.mkdir()
            (run_dir / "eval.json").write_text(
                json.dumps(
                    {
                        "benchmark_name": "dense run",
                        "role": "mwm_scheduled",
                        "batches": [{"large": "x" * 10_000}],
                        "review_rollouts": [
                            {"episode_index": 0, "success": False, "action_trace": [1] * 10_000}
                        ],
                        "review_media": {"rollouts": {}},
                        "videos": ["video.mp4"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "episode_traces.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "success": True,
                        "dataset_episode": 7,
                        "start_step": 3,
                        "goal_step": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            payload = _compact_review_payload(run_dir, {"name": "fallback", "role": "fallback"})

        self.assertNotIn("batches", payload)
        self.assertEqual(payload["benchmark_name"], "dense run")
        self.assertEqual(payload["videos"], ["video.mp4"])
        self.assertEqual(
            payload["review_rollouts"],
            [
                {
                    "episode_index": 0,
                    "success": True,
                    "dataset_episode": 7,
                    "start_step": 3,
                    "goal_step": 8,
                }
            ],
        )
        self.assertNotIn("action_trace", payload["review_rollouts"][0])

    def test_dense_finalize_skips_eval_payloads_and_uses_compact_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = OmegaConf.create({"title": "Dense", "seed": 7, "sweep": {}})
            run = OmegaConf.create({"matrix_index": 0, "name": "run", "cell_id": "cell"})
            run_cfg = OmegaConf.create({"eval": {}})
            row = {"name": "run", "manifest_sha256": "manifest"}
            pareto_path = root / "plots" / "pareto.html"
            with (
                mock.patch.object(benchmark_matrix, "DENSE_REVIEW_CELL_LIMIT", 0),
                mock.patch.object(benchmark_matrix, "_completed_row", return_value=row),
                mock.patch.object(
                    benchmark_matrix,
                    "_compact_review_payload",
                    side_effect=AssertionError("dense finalization must not load eval payloads"),
                ),
                mock.patch("mwm.benchmark.summary.write_summary_csv"),
                mock.patch("mwm.benchmark.summary.write_per_env_table", return_value="per_env.csv"),
                mock.patch("mwm.benchmark.plots.write_default_plots", return_value=[]) as write_plots,
                mock.patch("mwm.benchmark.pareto.write_pareto_html", return_value=str(pareto_path)),
                mock.patch("mwm.benchmark.html.write_review_html") as write_review,
                mock.patch("mwm.io.write_json"),
                mock.patch("mwm.io.write_metrics_jsonl"),
            ):
                benchmark_matrix._finalize(
                    cfg=cfg,
                    output_dir=root,
                    resolved=[(run, run_cfg)],
                    manifest_path=root / "manifest.json",
                    manifest_info={"group": "dense"},
                )

        self.assertTrue(write_plots.call_args.kwargs["compact"])
        self.assertEqual(write_review.call_args.args[3], [])
        self.assertFalse(write_review.call_args.kwargs["include_rollouts"])

    def test_expanded_cell_ids_and_matrix_identities_are_unique_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(
                Path(tmp),
                sweep={
                    "planner.pop_size": [100, 300],
                    "planner.elite_frac": [0.1],
                    "planner.n_iter": [5],
                },
            )
            first = expand_benchmark_runs(cfg)
            second = expand_benchmark_runs(cfg)
            first_ids = [str(run.cell_id) for run in first]
            second_ids = [str(run.cell_id) for run in second]

            resolved = [(run, merged_run_config(cfg, run)[1]) for run in first]
            validate_benchmark_matrix(cfg, resolved)
            identities = expected_cells_from_resolved(resolved)

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertEqual(len(identities), len(first_ids))
        self.assertIn("00_upstream_fixed_finest__pop100__elite0p1__iter5", first_ids)
        self.assertIn("01_dense_schedule__pop300__elite0p1__iter5", first_ids)

    def test_elite_fraction_sweep_clears_inherited_fixed_topk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(
                Path(tmp),
                sweep={
                    "planner.pop_size": [100],
                    "planner.elite_frac": [0.1],
                    "planner.n_iter": [5],
                },
            )

            run = expand_benchmark_runs(cfg)[0]
            _, run_cfg = merged_run_config(cfg, run)

        self.assertIn("topk", run.planner)
        self.assertIsNone(run.planner.topk)
        self.assertIsNone(run_cfg.planner.topk)
        self.assertEqual(run_cfg.planner.elite_frac, 0.1)

    def test_explicit_topk_sweep_is_not_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(
                Path(tmp),
                sweep={
                    "planner.elite_frac": [0.1],
                    "planner.topk": [4, 8],
                },
            )

            expanded = expand_benchmark_runs(cfg)

        self.assertEqual({int(run.planner.topk) for run in expanded}, {4, 8})

    def test_run_defaults_merge_before_run_and_sweep_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(
                Path(tmp),
                run_defaults={
                    "planner": {"flop_accounting": "dynamics_audit", "n_iter": 4},
                    "eval": {"episodes": 6},
                    "env": {"max_steps": 250},
                },
                sweep={"planner.n_iter": [9]},
            )
            cfg.runs[1].planner = {"flop_accounting": "none"}
            upstream, scheduled = expand_benchmark_runs(cfg)

            _, upstream_cfg = merged_run_config(cfg, upstream)
            _, scheduled_cfg = merged_run_config(cfg, scheduled)

        self.assertEqual(upstream_cfg.planner.flop_accounting, "dynamics_audit")
        self.assertEqual(scheduled_cfg.planner.flop_accounting, "none")
        self.assertEqual(upstream_cfg.planner.n_iter, 9)
        self.assertEqual(upstream_cfg.eval.episodes, 6)
        self.assertEqual(upstream_cfg.eval.seed, 7)
        self.assertEqual(upstream_cfg.env.max_steps, 250)


class ParetoFrontierTests(unittest.TestCase):
    def test_strategy_palette_assigns_27_distinct_fixed_colors(self) -> None:
        first = _strategy_color_map(f"strategy-{index:02d}" for index in range(27))
        second = _strategy_color_map(reversed(list(first)))

        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), 27)
        self.assertEqual(first["strategy-00"], "#e41a1c")
        self.assertEqual(first["strategy-26"], "#f06ab6")

    def test_frontier_excludes_cost_and_success_dominated_rows(self) -> None:
        rows = [
            {"cell_id": "cheap", "dynamics_flops_total": 10, "success_rate": 0.4},
            {"cell_id": "same_cost_better", "dynamics_flops_total": 20, "success_rate": 0.7},
            {"cell_id": "same_cost_worse", "dynamics_flops_total": 20, "success_rate": 0.5},
            {"cell_id": "higher_cost_same_success", "dynamics_flops_total": 30, "success_rate": 0.7},
            {"cell_id": "best", "dynamics_flops_total": 40, "success_rate": 0.9},
            {"cell_id": "expensive_worse", "dynamics_flops_total": 50, "success_rate": 0.8},
        ]

        frontier = pareto_frontier(rows)

        self.assertEqual(
            [row["cell_id"] for row in frontier],
            ["cheap", "same_cost_better", "best"],
        )

    def test_plot_legend_uses_human_readable_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pareto.html"
            write_pareto_html(
                out,
                [
                    {
                        "name": "internal_cell_id",
                        "role": "dense_schedule_22_internal_role",
                        "strategy": "dense_schedule_22_internal_role",
                        "schedule": "MPC=coarse -> fine | CEM=coarse -> base | Rollout=fine -> base",
                        "dynamics_flops_total": 10,
                        "success_rate": 0.5,
                    }
                ],
            )
            html = out.read_text(encoding="utf-8")

        self.assertIn(r"MPC=coarse -\u003e fine", html)
        self.assertNotIn('"name":"dense_schedule_22_internal_role"', html)

    def test_plot_uses_percent_scale_and_emphasizes_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pareto.html"
            write_pareto_html(
                out,
                [
                    {"schedule": "cheap", "dynamics_flops_total": 10, "success_rate": 20},
                    {"schedule": "frontier", "dynamics_flops_total": 20, "success_rate": 80},
                    {"schedule": "dominated", "dynamics_flops_total": 30, "success_rate": 50},
                ],
                provisional=True,
            )
            html = out.read_text(encoding="utf-8")

        self.assertIn('"range":[0,102]', html)
        self.assertIn('"mode":"lines+markers"', html)
        self.assertIn('"name":"Provisional global Pareto frontier (reference)"', html)
        self.assertIn('"name":"Dominated cells"', html)
        self.assertIn('"label":"dominated"', html)
        self.assertIn('"opacity":0.78', html)
        self.assertIn('"dash":"dot"', html)


if __name__ == "__main__":
    unittest.main()
