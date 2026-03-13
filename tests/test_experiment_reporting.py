from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from hudm import experiment
from hudm.experiment_report import write_experiment_report


class ExperimentReportingTests(unittest.TestCase):
    def test_run_variant_task_writes_run_log(self):
        fake_result = {
            "cfg": mock.Mock(save=False),
            "runtime": {"backend": "gt_env"},
            "success": True,
            "trajectory": [],
            "frames": [],
            "planner_frames": [],
            "run_stats": {},
            "trace": {},
            "init_state": [],
            "goal_state": [],
            "sample_meta": {},
        }
        task = {
            "cfg": {"save": False},
            "selection": {"rollout_id": "r0", "rollout_index": 0},
            "variant_name": "variant_a",
            "run_dir": "",
            "terminal_mode": "compact",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            task["run_dir"] = os.path.join(tmpdir, "traces", "variant_a", "r0")
            with mock.patch.object(experiment, "run_plan_session", return_value=fake_result):
                with mock.patch.object(experiment, "save_plan_result", side_effect=lambda *args, **kwargs: print("saved")):
                    with mock.patch.object(experiment, "result_row", return_value={"variant_name": "variant_a", "rollout_id": "r0"}):
                        row = experiment._run_variant_task(task)

            self.assertEqual(row["variant_name"], "variant_a")
            log_path = os.path.join(task["run_dir"], "run.log")
            self.assertTrue(os.path.isfile(log_path))
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("saved", content)

    def test_write_experiment_report_creates_summary_and_detail_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "experiment_demo")
            detail_dir = os.path.join(run_dir, "traces", "variant_a", "rollout_0")
            os.makedirs(detail_dir, exist_ok=True)
            with open(os.path.join(detail_dir, "run.log"), "w", encoding="utf-8") as f:
                f.write("step log\n")
            with open(os.path.join(detail_dir, "planned.mp4"), "wb") as f:
                f.write(b"not_a_real_video")

            rows = [
                {
                    "variant_name": "variant_a",
                    "rollout_id": "rollout_0",
                    "success": 1,
                    "termination_reason": "env_done",
                    "final_pos_diff": 1.0,
                    "final_coverage": 0.95,
                    "bits_used_total": 100.0,
                    "plan_time_total_sec": 0.25,
                    "run_dir": detail_dir,
                }
            ]
            summary_rows = [
                {
                    "variant_name": "variant_a",
                    "n_rollouts": 1,
                    "success_rate": 1.0,
                    "mean_final_pos_diff": 1.0,
                    "mean_final_coverage": 0.95,
                    "mean_bits_used_total": 100.0,
                    "mean_plan_time_total_sec": 0.25,
                }
            ]

            report_path = write_experiment_report(
                run_dir,
                summary_rows,
                rows,
                experiment_name="demo",
                baseline_variant="variant_a",
            )

            self.assertTrue(os.path.isfile(report_path))
            self.assertTrue(os.path.isfile(os.path.join(detail_dir, "index.html")))
            with open(report_path, "r", encoding="utf-8") as f:
                summary_html = f.read()
            self.assertIn("rollout_0", summary_html)
            self.assertIn("open", summary_html)
            with open(os.path.join(detail_dir, "index.html"), "r", encoding="utf-8") as f:
                detail_html = f.read()
            self.assertIn("planned.mp4", detail_html)
            self.assertIn("Run Log Tail", detail_html)


if __name__ == "__main__":
    unittest.main()
