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
            "collect_ogb_cube_smoke.yaml",
            "collect_reacher_smoke.yaml",
            "eval_pusht_smoke.yaml",
            "benchmark_pusht_smoke.yaml",
            "train_ogb_cube_cpu_smoke.yaml",
            "train_pusht_cpu_smoke.yaml",
            "train_reacher_cpu_smoke.yaml",
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

        reacher_collect_cfg = yaml.safe_load((local_dir / "collect_reacher_smoke.yaml").read_text(encoding="utf-8"))
        self.assertEqual(reacher_collect_cfg["env_id"], "swm/ReacherDMControl-v0")
        self.assertEqual(reacher_collect_cfg["env_kwargs"]["task"], "qpos_match")
        self.assertEqual(reacher_collect_cfg["restore"]["import_path"], "mwm.swm.restore.reacher_qpos_match_restore_spec")
        self.assertTrue(reacher_collect_cfg["eager_write"])
        self.assertEqual(reacher_collect_cfg["keys_to_save"], ["pixels", "action", "qpos", "qvel", "observation"])
        self.assertLessEqual(reacher_collect_cfg["episodes"], 4)

        reacher_train_cfg = yaml.safe_load((local_dir / "train_reacher_cpu_smoke.yaml").read_text(encoding="utf-8"))
        self.assertEqual(reacher_train_cfg["env_id"], "swm/ReacherDMControl-v0")
        self.assertEqual(reacher_train_cfg["data"]["frameskip"], 5)
        self.assertEqual(reacher_train_cfg["data"]["keys_to_load"], ["pixels", "action", "qpos", "qvel", "observation"])
        self.assertEqual(reacher_train_cfg["data"]["keys_to_cache"], ["action", "qpos", "qvel", "observation"])
        self.assertEqual(reacher_train_cfg["model"]["K"], [192])
        self.assertEqual(reacher_train_cfg["model"]["action_block"], 5)
        self.assertTrue(reacher_train_cfg["train"]["no_cuda"])
        self.assertEqual(reacher_train_cfg["train"]["run_name"], "local_reacher_cpu_smoke")

        cube_collect_cfg = yaml.safe_load((local_dir / "collect_ogb_cube_smoke.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cube_collect_cfg["env_id"], "swm/OGBCube-v0")
        self.assertEqual(cube_collect_cfg["env_kwargs"]["env_type"], "single")
        self.assertEqual(cube_collect_cfg["env_kwargs"]["ob_type"], "states")
        self.assertEqual(cube_collect_cfg["env_kwargs"]["width"], 224)
        self.assertEqual(cube_collect_cfg["env_kwargs"]["height"], 224)
        self.assertEqual(cube_collect_cfg["restore"]["import_path"], "mwm.ogbench.restore.ogbench_cube_restore_spec")
        self.assertTrue(cube_collect_cfg["eager_write"])
        self.assertGreaterEqual(cube_collect_cfg["max_episode_steps"], 20)
        self.assertEqual(
            cube_collect_cfg["keys_to_save"],
            ["pixels", "action", "qpos", "qvel", "observation", "privileged/block_0_pos", "privileged/block_0_quat"],
        )
        self.assertLessEqual(cube_collect_cfg["episodes"], 4)

        cube_train_cfg = yaml.safe_load((local_dir / "train_ogb_cube_cpu_smoke.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cube_train_cfg["env_id"], "swm/OGBCube-v0")
        self.assertEqual(cube_train_cfg["base"]["checkpoint"], "models--quentinll--lewm-cube")
        self.assertEqual(cube_train_cfg["data"]["frameskip"], 5)
        self.assertEqual(
            cube_train_cfg["data"]["keys_to_load"],
            ["pixels", "action", "qpos", "qvel", "observation", "privileged/block_0_pos", "privileged/block_0_quat"],
        )
        self.assertEqual(
            cube_train_cfg["data"]["keys_to_cache"],
            ["action", "qpos", "qvel", "observation", "privileged/block_0_pos", "privileged/block_0_quat"],
        )
        self.assertEqual(cube_train_cfg["model"]["K"], [192])
        self.assertEqual(cube_train_cfg["model"]["action_block"], 5)
        self.assertTrue(cube_train_cfg["train"]["no_cuda"])
        self.assertEqual(cube_train_cfg["train"]["run_name"], "local_ogb_cube_cpu_smoke")

    def test_local_scripts_are_not_slurm_gated_or_parcc_path_bound(self) -> None:
        scripts = [
            ROOT / "scripts" / "local" / "local_verify.sh",
            ROOT / "scripts" / "local" / "local_benchmark_smoke.sh",
            ROOT / "scripts" / "local" / "local_train_smoke.sh",
            ROOT / "scripts" / "local" / "local_reacher_train_smoke.sh",
            ROOT / "scripts" / "local" / "local_ogb_cube_train_smoke.sh",
        ]
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            self.assertNotIn("SLURM_JOB_ID", text, script)
            self.assertNotIn("/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python", text, script)
            self.assertIn('${MWM_PYTHON:-python}', text, script)

        verify_text = (ROOT / "scripts" / "local" / "local_verify.sh").read_text(encoding="utf-8")
        self.assertIn("git ls-files", verify_text)

        reacher_text = (ROOT / "scripts" / "local" / "local_reacher_train_smoke.sh").read_text(encoding="utf-8")
        self.assertIn('${MUJOCO_GL:-egl}', reacher_text)
        self.assertIn('${PYOPENGL_PLATFORM:-egl}', reacher_text)
        self.assertIn("-m mwm.data.collection configs/local/collect_reacher_smoke.yaml", reacher_text)
        self.assertIn("-m mwm.training.lewm configs/local/train_reacher_cpu_smoke.yaml", reacher_text)
        for name in ("config.json", "weights.pt", "world_metadata.json"):
            self.assertIn(name, reacher_text)

        cube_text = (ROOT / "scripts" / "local" / "local_ogb_cube_train_smoke.sh").read_text(encoding="utf-8")
        self.assertIn('${MUJOCO_GL:-egl}', cube_text)
        self.assertIn('${PYOPENGL_PLATFORM:-egl}', cube_text)
        self.assertIn("-m mwm.data.collection configs/local/collect_ogb_cube_smoke.yaml", cube_text)
        self.assertIn("-m mwm.training.lewm configs/local/train_ogb_cube_cpu_smoke.yaml", cube_text)
        for name in ("config.json", "weights.pt", "world_metadata.json"):
            self.assertIn(name, cube_text)


if __name__ == "__main__":
    unittest.main()
