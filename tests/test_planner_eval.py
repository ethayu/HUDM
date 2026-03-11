from __future__ import annotations

import os
import tempfile
import unittest

from hudm.benchmark import load_benchmark_spec
from hudm.config import resolve_experiment_spec, resolve_plan_spec
from hudm.experiment import aggregate_summary
from hudm.task_sampling import enumerate_rollout_candidates, select_rollouts


ROOT = os.path.dirname(os.path.dirname(__file__))


class ExperimentConfigTests(unittest.TestCase):
    def test_resolve_plan_spec_prunes_inactive_backend_blocks(self):
        spec = resolve_plan_spec(os.path.join(ROOT, "configs/plan_smoke_gt_env.yaml"))
        self.assertEqual(spec.active_backend_kind(), "gt_env")
        self.assertNotIn("wm", spec.clean_cfg.backend)
        self.assertNotIn("particle_sim", spec.clean_cfg.backend)
        self.assertEqual(str(spec.runtime_cfg.backend), "gt_env")

    def test_select_rollouts_is_deterministic(self):
        spec = resolve_experiment_spec(os.path.join(ROOT, "configs/planner_eval_smoke_gt_env.yaml"))
        candidates = enumerate_rollout_candidates(spec.shared_plan)
        sel_a = select_rollouts(spec.rollouts, candidates)
        sel_b = select_rollouts(spec.rollouts, candidates)
        self.assertEqual(sel_a, sel_b)
        self.assertEqual(len({item["rollout_id"] for item in sel_a}), len(sel_a))

    def test_variant_rejects_task_override(self):
        task_cfg = os.path.join(ROOT, "configs/task/pusht_smoke_dataset.yaml")
        planner_cfg = os.path.join(ROOT, "configs/planner/smoke.yaml")
        backend_cfg = os.path.join(ROOT, "configs/backend/gt_env_state.yaml")
        fidelity_cfg = os.path.join(ROOT, "configs/fidelity/finest.yaml")
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(
                f"""
experiment:
  name: "bad_variant"
  shared_plan:
    imports:
      - "{task_cfg}"
      - "{planner_cfg}"
    plan:
      artifacts:
        render: false
        save: false
  rollouts:
    seed: 0
    num_rollouts: 1
    sample_without_replacement: true
  execution:
    mode: "serial"
    max_workers: 1
  reporting:
    output_root: "rollouts"
  baseline: "bad"
  variants:
    - name: "bad"
      imports:
        - "{backend_cfg}"
        - "{fidelity_cfg}"
      overrides:
        task:
          env:
            render_size: 64
"""
            )
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                resolve_experiment_spec(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_aggregate_summary_uses_explicit_baseline(self):
        rows = [
            {
                "variant_name": "baseline",
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
                "variant_name": "challenger",
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
        summary_rows, paired_rows = aggregate_summary(
            rows,
            baseline_variant="baseline",
            variant_order=["baseline", "challenger"],
        )
        self.assertEqual(summary_rows[0]["variant_name"], "baseline")
        self.assertEqual(summary_rows[1]["variant_name"], "challenger")
        self.assertEqual(paired_rows[0]["baseline_variant"], "baseline")
        self.assertEqual(paired_rows[0]["success_delta"], -1)

    def test_benchmark_spec_resolves_entries(self):
        spec = load_benchmark_spec(os.path.join(ROOT, "configs/benchmark.yaml"))
        self.assertGreaterEqual(len(spec.entries), 1)
        self.assertTrue(spec.entries[0].experiment_config.endswith(".yaml"))


if __name__ == "__main__":
    unittest.main()
