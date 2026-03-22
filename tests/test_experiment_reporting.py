from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from unittest import mock

from hudm import experiment
from hudm.specs import ExperimentSpec, ExperimentVariant
from omegaconf import OmegaConf
from scripts.create_dummy_multivariant_bundle import FAKE_VARIANTS, build_dummy_bundle
from hudm.experiment_bundle import (
    EXPERIMENT_JSON,
    PAIRED_VS_BASELINE_CSV,
    RUNS_CSV,
    SELECTED_ROLLOUTS_JSON,
    VARIANTS_CSV,
    migrate_legacy_experiment_dir,
    write_experiment_bundle,
)
from hudm.experiment_review import load_experiment_review_data


class ExperimentReportingTests(unittest.TestCase):
    def test_dummy_multivariant_bundle_reuses_rollout_ids_across_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = os.path.join(tmpdir, "source_bundle")
            trace_root = os.path.join(source_dir, "traces", "variant_a")
            os.makedirs(os.path.join(trace_root, "rollout_0"), exist_ok=True)
            os.makedirs(os.path.join(trace_root, "rollout_1"), exist_ok=True)
            for rollout_id in ("rollout_0", "rollout_1"):
                rollout_dir = os.path.join(trace_root, rollout_id)
                with open(os.path.join(rollout_dir, "trace.json"), "w", encoding="utf-8") as f:
                    json.dump({"ok": True}, f)
                with open(os.path.join(rollout_dir, "trace.npz"), "wb") as f:
                    f.write(b"npz")
                with open(os.path.join(rollout_dir, "run.log"), "w", encoding="utf-8") as f:
                    f.write("log\n")

            write_experiment_bundle(
                source_dir,
                experiment_payload={
                    "schema_version": 1,
                    "reviewer_version": 1,
                    "experiment_name": "source_demo",
                    "variant_order": ["variant_a"],
                    "num_rollouts": 2,
                },
                selected_rollouts=[
                    {"rollout_id": "rollout_0", "rollout_index": 0},
                    {"rollout_id": "rollout_1", "rollout_index": 1},
                ],
                run_rows=[
                    {
                        "variant_name": "variant_a",
                        "rollout_id": "rollout_0",
                        "rollout_index": 0,
                        "success": 1,
                        "success_and_done": 1,
                        "termination_reason": "env_done",
                        "executed_steps": 1,
                        "plans": 1,
                        "final_pos_diff": 1.0,
                        "final_angle_diff": 0.1,
                        "final_eef_diff": 0.2,
                        "best_pos_diff": 1.0,
                        "best_angle_diff": 0.1,
                        "best_eef_diff": 0.2,
                        "final_coverage": 0.9,
                        "auc_pos_diff": 1.0,
                        "auc_angle_diff": 0.1,
                        "auc_eef_diff": 0.2,
                        "bits_used_total": 100.0,
                        "bits_used_per_step": 100.0,
                        "flops_used_total": 200.0,
                        "flops_used_per_step": 200.0,
                        "plan_time_total_sec": 0.5,
                        "plan_time_per_replan_sec": 0.5,
                    },
                    {
                        "variant_name": "variant_a",
                        "rollout_id": "rollout_1",
                        "rollout_index": 1,
                        "success": 0,
                        "success_and_done": 0,
                        "termination_reason": "max_steps",
                        "executed_steps": 2,
                        "plans": 1,
                        "final_pos_diff": 2.0,
                        "final_angle_diff": 0.2,
                        "final_eef_diff": 0.3,
                        "best_pos_diff": 2.0,
                        "best_angle_diff": 0.2,
                        "best_eef_diff": 0.3,
                        "final_coverage": 0.4,
                        "auc_pos_diff": 2.0,
                        "auc_angle_diff": 0.2,
                        "auc_eef_diff": 0.3,
                        "bits_used_total": 150.0,
                        "bits_used_per_step": 75.0,
                        "flops_used_total": 300.0,
                        "flops_used_per_step": 150.0,
                        "plan_time_total_sec": 0.7,
                        "plan_time_per_replan_sec": 0.7,
                    },
                ],
                variant_rows=[{"variant_name": "variant_a", "n_rollouts": 2, "success_rate": 0.5}],
                paired_rows=[],
            )

            output_dir = os.path.join(tmpdir, "dummy_bundle")
            build_dummy_bundle(source_dir, output_dir, overwrite=False)

            with open(os.path.join(output_dir, RUNS_CSV), "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            by_variant: dict[str, set[str]] = {}
            for row in rows:
                by_variant.setdefault(str(row["variant_name"]), set()).add(str(row["rollout_id"]))

            self.assertEqual(set(by_variant.keys()), set(FAKE_VARIANTS))
            self.assertTrue(all(rollout_ids == {"rollout_0", "rollout_1"} for rollout_ids in by_variant.values()))

            with open(os.path.join(output_dir, EXPERIMENT_JSON), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["num_rollouts"], 2)

    def test_run_experiment_accepts_plain_dict_runtime_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_plan = mock.Mock(clean_cfg=OmegaConf.create({"task": {"name": "demo"}}))
            variant_plan = mock.Mock(
                runtime_cfg=OmegaConf.create({"artifacts": {"save": False}}),
                clean_cfg=OmegaConf.create({"backend": {"kind": "gt_env"}}),
            )
            spec = ExperimentSpec(
                name="demo",
                config_path=None,
                shared_plan=shared_plan,
                variants=[ExperimentVariant(name="variant_a", plan=variant_plan)],
                rollouts={"seed": 0, "num_rollouts": 1, "sample_without_replacement": True},
                execution={"mode": "serial", "max_workers": 1},
                terminal={"mode": "quiet"},
                reporting={"output_root": tmpdir},
            )
            fake_row = {
                "variant_name": "variant_a",
                "rollout_id": "r0",
                "rollout_index": 0,
                "success": 1,
                "success_and_done": 1,
                "termination_reason": "env_done",
                "executed_steps": 1,
                "plans": 1,
                "final_pos_diff": 1.0,
                "final_angle_diff": 0.1,
                "final_eef_diff": 0.2,
                "best_pos_diff": 1.0,
                "best_angle_diff": 0.1,
                "best_eef_diff": 0.2,
                "final_coverage": 0.9,
                "auc_pos_diff": 1.0,
                "auc_angle_diff": 0.1,
                "auc_eef_diff": 0.2,
                "bits_used_total": 100.0,
                "bits_used_per_step": 100.0,
                "flops_used_total": 200.0,
                "flops_used_per_step": 200.0,
                "plan_time_total_sec": 0.5,
                "plan_time_per_replan_sec": 0.5,
            }
            with mock.patch.object(experiment, "enumerate_rollout_candidates", return_value=[{"rollout_id": "r0"}]):
                with mock.patch.object(experiment, "select_rollouts", return_value=[{"rollout_id": "r0", "rollout_index": 0}]):
                    with mock.patch.object(experiment, "_group_wm_variants", return_value=([], spec.variants)):
                        with mock.patch.object(experiment, "_run_variant_task", return_value=fake_row):
                            run_dir = experiment.run_experiment(spec)

            with open(os.path.join(run_dir, EXPERIMENT_JSON), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["rollouts"]["num_rollouts"], 1)
            self.assertEqual(payload["execution"]["mode"], "serial")
            self.assertEqual(payload["terminal"]["mode"], "quiet")
            self.assertEqual(payload["reporting"]["output_root"], tmpdir)

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

    def test_write_experiment_bundle_emits_canonical_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "experiment_demo")
            write_experiment_bundle(
                run_dir,
                experiment_payload={
                    "schema_version": 1,
                    "reviewer_version": 1,
                    "experiment_name": "demo",
                    "variant_order": ["variant_a"],
                },
                selected_rollouts=[{"rollout_id": "rollout_0"}],
                run_rows=[{"variant_name": "variant_a", "rollout_id": "rollout_0", "success": 1}],
                variant_rows=[{"variant_name": "variant_a", "n_rollouts": 1, "success_rate": 1.0}],
                paired_rows=[{"variant_name": "variant_a", "rollout_id": "rollout_0", "success_delta": 0}],
            )

            for filename in (
                EXPERIMENT_JSON,
                RUNS_CSV,
                VARIANTS_CSV,
                PAIRED_VS_BASELINE_CSV,
                SELECTED_ROLLOUTS_JSON,
            ):
                self.assertTrue(os.path.isfile(os.path.join(run_dir, filename)), msg=filename)

            self.assertFalse(os.path.exists(os.path.join(run_dir, "experiment_report.html")))
            self.assertFalse(os.path.exists(os.path.join(run_dir, "summary.csv")))
            with open(os.path.join(run_dir, EXPERIMENT_JSON), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["experiment_name"], "demo")

    def test_migrate_legacy_experiment_dir_writes_canonical_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "experiment_legacy")
            os.makedirs(os.path.join(run_dir, "traces", "variant_a", "rollout_0"), exist_ok=True)

            with open(os.path.join(run_dir, "experiment_resolved.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": "legacy_demo",
                        "baseline": "variant_a",
                        "rollouts": {"num_rollouts": 1},
                        "execution": {"mode": "serial"},
                        "terminal": {"mode": "compact"},
                        "reporting": {"output_root": "rollouts"},
                        "shared_plan": {"imports": ["task/demo.yaml"]},
                        "variants": [{"name": "variant_a", "imports": ["backend/demo.yaml"]}],
                    },
                    f,
                    indent=2,
                )
            with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "experiment_name": "legacy_demo",
                        "created_at": "2026-03-16T00:00:00",
                        "num_rollouts": 1,
                        "baseline_variant": "variant_a",
                        "summary": [],
                    },
                    f,
                    indent=2,
                )
            with open(os.path.join(run_dir, SELECTED_ROLLOUTS_JSON), "w", encoding="utf-8") as f:
                json.dump([{"rollout_id": "rollout_0", "rollout_index": 0}], f, indent=2)
            with open(os.path.join(run_dir, "per_rollout.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("variant_name,rollout_id,success\nvariant_a,rollout_0,1\n")
            with open(os.path.join(run_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("variant_name,n_rollouts,success_rate\nvariant_a,1,1.0\n")

            migrate_legacy_experiment_dir(run_dir)

            for filename in (
                EXPERIMENT_JSON,
                RUNS_CSV,
                VARIANTS_CSV,
                PAIRED_VS_BASELINE_CSV,
                SELECTED_ROLLOUTS_JSON,
            ):
                self.assertTrue(os.path.isfile(os.path.join(run_dir, filename)), msg=filename)

            data = load_experiment_review_data(run_dir)
            self.assertEqual(data.experiment_name, "legacy_demo")
            self.assertEqual(data.baseline_variant, "variant_a")
            self.assertEqual(len(data.run_rows), 1)
            self.assertEqual(len(data.variant_rows), 1)


if __name__ == "__main__":
    unittest.main()
