from __future__ import annotations

import math
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from mwm.benchmark.config import DEFAULTS, load_manifest_config, merged_run_config
from mwm.benchmark.sweep import expand_benchmark_runs
from mwm.data.manifest import load_manifest
from mwm.eval.runtime import effective_goal_offset
from mwm.eval.validation import validate_manifest


class BenchmarkSweepExclusionTests(unittest.TestCase):
    def test_partial_exclusion_removes_only_matching_combinations(self) -> None:
        cfg = OmegaConf.create(
            {
                "sweep": {
                    "planner.pop_size": [20, 50],
                    "planner.elite_frac": [0.05, 0.1],
                    "planner.n_iter": [5, 10],
                },
                "sweep_exclude": [
                    {"planner.pop_size": 20, "planner.elite_frac": 0.05}
                ],
                "runs": [{"name": "scheduled", "checkpoint": "checkpoint"}],
            }
        )

        expanded = expand_benchmark_runs(cfg)
        combinations = {
            (
                int(run.planner.pop_size),
                float(run.planner.elite_frac),
                int(run.planner.n_iter),
            )
            for run in expanded
        }

        self.assertEqual(len(expanded), 6)
        self.assertNotIn((20, 0.05, 5), combinations)
        self.assertNotIn((20, 0.05, 10), combinations)
        self.assertIn((50, 0.05, 5), combinations)
        self.assertIn((20, 0.1, 10), combinations)

    def test_exclusion_rejects_unknown_sweep_parameter(self) -> None:
        cfg = OmegaConf.create(
            {
                "sweep": {"planner.pop_size": [20]},
                "sweep_exclude": [{"planner.elite_frac": 0.05}],
                "runs": [{"name": "scheduled", "checkpoint": "checkpoint"}],
            }
        )

        with self.assertRaisesRegex(ValueError, "not present in sweep"):
            expand_benchmark_runs(cfg)

    def test_exclusion_cannot_remove_entire_sweep(self) -> None:
        cfg = OmegaConf.create(
            {
                "sweep": {"planner.pop_size": [20]},
                "sweep_exclude": [{"planner.pop_size": 20}],
                "runs": [{"name": "scheduled", "checkpoint": "checkpoint"}],
            }
        )

        with self.assertRaisesRegex(ValueError, "removes every sweep combination"):
            expand_benchmark_runs(cfg)


class ReleaseScheduleConfigTests(unittest.TestCase):
    def test_release_sweeps_use_exact_goal25_and_plan25_execute10(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for env in ("pusht", "reacher", "ogb_cube", "tworoom"):
            with self.subTest(env=env):
                cfg = OmegaConf.merge(
                    DEFAULTS,
                    OmegaConf.load(
                        root
                        / "configs"
                        / "research"
                        / f"release20260728_dense_{env}_all_fidelity_schedules.yaml"
                    ),
                )
                self.assertEqual(
                    list(cfg.sweep["planner.n_iter"]),
                    [5, 10, 15, 20, 30],
                )
                _, run_cfg = merged_run_config(cfg, cfg.runs[0])

                self.assertEqual(int(run_cfg.eval.episodes), 250)
                self.assertEqual(int(run_cfg.eval.num_envs), 50)
                self.assertEqual(int(run_cfg.eval.goal_offset), 25)
                self.assertEqual(str(run_cfg.eval.goal_indexing), "exact")
                self.assertEqual(
                    effective_goal_offset(
                        int(run_cfg.eval.goal_offset), str(run_cfg.eval.goal_indexing)
                    ),
                    25,
                )

                plan_actions = int(run_cfg.planner.horizon) * int(run_cfg.planner.action_block)
                execute_actions = int(run_cfg.planner.receding_horizon) * int(
                    run_cfg.planner.action_block
                )
                max_replans = math.ceil(int(run_cfg.eval.budget) / execute_actions)
                self.assertEqual(int(run_cfg.planner.horizon), 5)
                self.assertEqual(int(run_cfg.planner.receding_horizon), 2)
                self.assertEqual(plan_actions, 25)
                self.assertEqual(execute_actions, 10)
                self.assertEqual(max_replans, 5)

                manifest_path = root / load_manifest_config(cfg)["path"]
                manifest = load_manifest(manifest_path)
                self.assertEqual(len(manifest["pairs"]), 250)
                self.assertEqual(
                    {
                        int(pair["goal_step"]) - int(pair["start_step"])
                        for pair in manifest["pairs"]
                    },
                    {25},
                )
                validate_manifest(
                    manifest,
                    path=str(manifest_path),
                    dataset=object(),
                    cfg=run_cfg,
                    env_id=str(run_cfg.env_id),
                    restore_spec_id=str(manifest["restore_spec"]),
                )

    def test_goal50_sweeps_plan50_execute20_with_separate_manifests(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for env in ("pusht", "reacher", "ogb_cube", "tworoom"):
            with self.subTest(env=env):
                cfg = OmegaConf.merge(
                    DEFAULTS,
                    OmegaConf.load(
                        root
                        / "configs"
                        / "research"
                        / (
                            f"release20260728_dense_{env}_goal50_plan50_execute20_"
                            "all_fidelity_schedules.yaml"
                        )
                    ),
                )
                self.assertEqual(len(cfg.runs), 26)
                self.assertEqual(
                    list(cfg.sweep["planner.n_iter"]),
                    [5, 10, 15, 20, 30],
                )
                self.assertIn("goal50_plan50_execute20", str(cfg.output_dir))
                self.assertIn("goal50_exact", str(cfg.manifest.config))
                _, run_cfg = merged_run_config(cfg, cfg.runs[0])

                self.assertEqual(int(run_cfg.eval.episodes), 250)
                self.assertEqual(int(run_cfg.eval.num_envs), 50)
                self.assertEqual(int(run_cfg.eval.goal_offset), 50)
                self.assertEqual(str(run_cfg.eval.goal_indexing), "exact")
                self.assertEqual(int(run_cfg.eval.budget), 100)
                self.assertEqual(
                    effective_goal_offset(
                        int(run_cfg.eval.goal_offset), str(run_cfg.eval.goal_indexing)
                    ),
                    50,
                )

                plan_actions = int(run_cfg.planner.horizon) * int(run_cfg.planner.action_block)
                execute_actions = int(run_cfg.planner.receding_horizon) * int(
                    run_cfg.planner.action_block
                )
                max_replans = math.ceil(int(run_cfg.eval.budget) / execute_actions)
                self.assertEqual(int(run_cfg.planner.horizon), 10)
                self.assertEqual(int(run_cfg.planner.receding_horizon), 4)
                self.assertEqual(plan_actions, 50)
                self.assertEqual(execute_actions, 20)
                self.assertEqual(max_replans, 5)

                manifest_path = root / load_manifest_config(cfg)["path"]
                manifest = load_manifest(manifest_path)
                self.assertEqual(len(manifest["pairs"]), 250)
                self.assertEqual(manifest["goal_indexing"], "exact")
                self.assertEqual(int(manifest["effective_goal_offset"]), 50)
                self.assertEqual(int(manifest["eval_budget"]), 100)
                self.assertEqual(
                    {
                        int(pair["goal_step"]) - int(pair["start_step"])
                        for pair in manifest["pairs"]
                    },
                    {50},
                )
                validate_manifest(
                    manifest,
                    path=str(manifest_path),
                    dataset=object(),
                    cfg=run_cfg,
                    env_id=str(run_cfg.env_id),
                    restore_spec_id=str(manifest["restore_spec"]),
                )


if __name__ == "__main__":
    unittest.main()
