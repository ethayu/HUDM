from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class MWMLocalWorkflowTests(unittest.TestCase):
    def test_local_desktop_configs_are_small_and_cpu_safe(self) -> None:
        local_dir = ROOT / "configs" / "local"
        expected = {
            "collect_pusht_smoke.yaml",
            "eval_pusht_smoke.yaml",
            "benchmark_pusht_smoke.yaml",
            "train_pusht_cpu_smoke.yaml",
        }
        self.assertEqual({path.name for path in local_dir.glob("*.yaml")}, expected)

        train_cfg = yaml.safe_load((local_dir / "train_pusht_cpu_smoke.yaml").read_text(encoding="utf-8"))
        self.assertTrue(train_cfg["train"]["no_cuda"])
        self.assertEqual(train_cfg["train"]["cpu_devices"], 1)
        self.assertLessEqual(train_cfg["train"]["batch_size"], 2)
        self.assertLessEqual(train_cfg["schedule"]["max_epochs"], 1)
        self.assertLessEqual(float(train_cfg["train"]["limit_train_batches"]), 1.0)

        eval_cfg = yaml.safe_load((local_dir / "eval_pusht_smoke.yaml").read_text(encoding="utf-8"))
        self.assertEqual(eval_cfg["device"], "cpu")
        self.assertLessEqual(eval_cfg["eval"]["episodes"], 2)
        self.assertEqual(eval_cfg["eval"]["num_envs"], 1)
        self.assertFalse(eval_cfg["eval"]["save_video"])
        self.assertLessEqual(eval_cfg["planner"]["pop_size"], 16)
        self.assertLessEqual(eval_cfg["planner"]["n_iter"], 2)

        bench_cfg = yaml.safe_load((local_dir / "benchmark_pusht_smoke.yaml").read_text(encoding="utf-8"))
        self.assertEqual(bench_cfg["env_id"], "swm/PushT-v1")
        self.assertEqual(bench_cfg["eval_config"], "configs/local/eval_pusht_smoke.yaml")
        self.assertEqual(bench_cfg["manifest"]["group"], "local_pusht_smoke")
        self.assertEqual(len(bench_cfg["runs"]), 1)

    def test_local_scripts_are_not_slurm_gated_or_parcc_path_bound(self) -> None:
        scripts = [
            ROOT / "scripts" / "local_verify.sh",
            ROOT / "scripts" / "local_benchmark_smoke.sh",
            ROOT / "scripts" / "local_train_smoke.sh",
        ]
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            self.assertNotIn("SLURM_JOB_ID", text, script)
            self.assertNotIn("/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python", text, script)
            self.assertIn('${MWM_PYTHON:-python}', text, script)

        verify_text = (ROOT / "scripts" / "local_verify.sh").read_text(encoding="utf-8")
        self.assertIn("git ls-files", verify_text)


if __name__ == "__main__":
    unittest.main()
