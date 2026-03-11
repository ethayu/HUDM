from __future__ import annotations

import unittest

from omegaconf import OmegaConf

import plan as single_plan
import planner_eval
from validate_cfg import validate_planner_eval_cfg


class PlannerEvalConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_cfg = single_plan.load_plan_cfg("configs/plan.yaml")

    def test_validate_rejects_duplicate_schedule_names(self):
        cfg = OmegaConf.create(
            {
                "plan_config": "configs/plan.yaml",
                "seed": 0,
                "num_rollouts": 2,
                "sample_without_replacement": True,
                "output_root": "rollouts",
                "parallel": {"mode": "auto", "max_workers": 1, "wm_schedule_batch_size": 1},
                "schedules": [
                    {"name": "dup", "fidelity": {"mpc": {"mode": "fixed", "level": "finest"}}},
                    {"name": "dup", "fidelity": {"mpc": {"mode": "fixed", "level": "coarsest"}}},
                ],
            }
        )
        with self.assertRaises(ValueError):
            validate_planner_eval_cfg(cfg, base_plan_cfg=self.base_cfg)

    def test_select_rollouts_is_deterministic(self):
        cfg = OmegaConf.create(
            {
                "plan_config": "configs/plan.yaml",
                "seed": 17,
                "num_rollouts": 3,
                "sample_without_replacement": True,
                "output_root": "rollouts",
                "parallel": {"mode": "auto", "max_workers": 1, "wm_schedule_batch_size": 1},
                "schedules": [
                    {"name": "sched_a", "fidelity": {"mpc": {"mode": "fixed", "level": "finest"}}},
                ],
            }
        )
        validate_planner_eval_cfg(cfg, base_plan_cfg=self.base_cfg)
        candidates = planner_eval.enumerate_rollout_candidates(self.base_cfg)
        sel_a = planner_eval.select_rollouts(cfg, candidates)
        sel_b = planner_eval.select_rollouts(cfg, candidates)
        self.assertEqual(sel_a, sel_b)
        self.assertEqual(len({item["rollout_id"] for item in sel_a}), len(sel_a))

    def test_select_rollouts_rejects_impossible_without_replacement(self):
        cfg = OmegaConf.create(
            {
                "plan_config": "configs/plan.yaml",
                "seed": 0,
                "num_rollouts": 10_000_000,
                "sample_without_replacement": True,
                "output_root": "rollouts",
                "parallel": {"mode": "auto", "max_workers": 1, "wm_schedule_batch_size": 1},
                "schedules": [
                    {"name": "sched_a", "fidelity": {"mpc": {"mode": "fixed", "level": "finest"}}},
                ],
            }
        )
        candidates = planner_eval.enumerate_rollout_candidates(self.base_cfg)
        with self.assertRaises(ValueError):
            planner_eval.select_rollouts(cfg, candidates)

    def test_aggregate_summary_uses_first_schedule_as_baseline(self):
        rows = [
            {
                "schedule_name": "zeta",
                "rollout_id": "r0",
                "success": 1,
                "success_and_done": 1,
                "termination_reason": "success",
                "executed_steps": 5,
                "plans": 1,
                "final_pos_diff": 1.0,
                "final_angle_diff": 0.1,
                "final_eef_diff": 0.2,
                "best_pos_diff": 1.0,
                "best_angle_diff": 0.1,
                "best_eef_diff": 0.2,
                "final_coverage": 0.9,
                "auc_pos_diff": 10.0,
                "auc_angle_diff": 1.0,
                "auc_eef_diff": 2.0,
                "bits_used_total": 100,
                "bits_used_per_step": 20.0,
                "flops_used_total": 200,
                "flops_used_per_step": 40.0,
                "plan_time_total_sec": 0.5,
                "plan_time_per_replan_sec": 0.5,
            },
            {
                "schedule_name": "alpha",
                "rollout_id": "r0",
                "success": 0,
                "success_and_done": 0,
                "termination_reason": "timeout",
                "executed_steps": 5,
                "plans": 1,
                "final_pos_diff": 3.0,
                "final_angle_diff": 0.3,
                "final_eef_diff": 0.4,
                "best_pos_diff": 3.0,
                "best_angle_diff": 0.3,
                "best_eef_diff": 0.4,
                "final_coverage": 0.1,
                "auc_pos_diff": 30.0,
                "auc_angle_diff": 3.0,
                "auc_eef_diff": 4.0,
                "bits_used_total": 300,
                "bits_used_per_step": 60.0,
                "flops_used_total": 400,
                "flops_used_per_step": 80.0,
                "plan_time_total_sec": 1.5,
                "plan_time_per_replan_sec": 1.5,
            },
        ]
        summary_rows, paired_rows = planner_eval.aggregate_summary(
            rows,
            self.base_cfg,
            baseline_schedule="zeta",
            schedule_order=["zeta", "alpha"],
        )
        self.assertEqual(summary_rows[0]["schedule_name"], "zeta")
        self.assertEqual(summary_rows[1]["schedule_name"], "alpha")
        self.assertEqual(paired_rows[0]["baseline_schedule"], "zeta")
        self.assertEqual(paired_rows[0]["success_delta"], -1)


if __name__ == "__main__":
    unittest.main()
