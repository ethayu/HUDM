from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mwm.config_cli import load_config


class MWMConfigCLITests(unittest.TestCase):
    def test_load_config_merges_defaults_yaml_and_dotlist_overrides(self) -> None:
        defaults = {
            "env_id": "swm/PushT-v1",
            "seed": 0,
            "train": {"batch_size": 8, "devices": 1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text(
                "seed: 5\n"
                "train:\n"
                "  devices: 2\n",
                encoding="utf-8",
            )

            cfg = load_config(defaults, cfg_path, ["train.batch_size=16", "env_id=swm/TwoRoom-v1"])

        self.assertEqual(cfg.env_id, "swm/TwoRoom-v1")
        self.assertEqual(cfg.seed, 5)
        self.assertEqual(cfg.train.batch_size, 16)
        self.assertEqual(cfg.train.devices, 2)

    def test_load_config_accepts_empty_overrides(self) -> None:
        defaults = {"seed": 0, "train": {"batch_size": 8}}
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text("seed: 5\n", encoding="utf-8")

            cfg = load_config(defaults, cfg_path, [])

        self.assertEqual(cfg.seed, 5)
        self.assertEqual(cfg.train.batch_size, 8)

    def test_train_export_rejects_ignored_set_overrides(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mwm.training.stable_wm",
                "missing.yaml",
                "--export-from-lightning",
                "missing.ckpt",
                "--set",
                "train.run_name=ignored",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--set is only supported for training", result.stderr)

    def test_data_verify_module_help_lists_modes(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mwm.data.verify",
                "--help",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--paper-parity", result.stdout)
        self.assertIn("--all", result.stdout)


if __name__ == "__main__":
    unittest.main()
