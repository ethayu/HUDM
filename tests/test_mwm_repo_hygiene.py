from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from mwm.swm.restore import eval_callables_for_env


ROOT = Path(__file__).resolve().parents[1]


def _tracked_review_files() -> list[Path]:
    skip_dirs = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "checkpoints",
        "checkpoints_mwm",
        "data",
        "logs",
        "rollouts",
        "synthetic",
    }
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".sh", ".sbatch"}:
            files.append(path)
    return files


class MWMRepoHygieneTests(unittest.TestCase):
    def test_no_legacy_runtime_symbols_in_source_configs_or_docs(self) -> None:
        forbidden = [
            "h" + "udm",
            "H" + "UDM",
            "swm_" + "hd" + "f5",
            "Hier" + "WorldModel",
            "train_" + "world_swm",
            "swm_" + "latent_cem",
            "planner." + "fidelity",
            "baseline_" + "upstream_lewm",
        ]
        allowed = {
            ROOT / ".gitignore",
            Path(__file__).resolve(),
        }
        hits: list[str] = []
        for path in _tracked_review_files():
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)} contains {token}")
        self.assertEqual(hits, [])

    def test_configs_are_lance_only_and_use_scheduler_branch(self) -> None:
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "format" in cfg:
                self.assertEqual(cfg["format"], "lance", path)
            data = cfg.get("data")
            if isinstance(data, dict) and "format" in data:
                self.assertEqual(data["format"], "lance", path)
                self.assertTrue(str(data.get("path", "")).endswith(".lance"), path)
                if path.name.startswith("eval_"):
                    self.assertNotIn("frameskip", data, path)
            planner = cfg.get("planner")
            if isinstance(planner, dict):
                self.assertIn("scheduler", planner, path)
                self.assertNotIn("fidelity", planner, path)
            self.assertNotIn("baseline", cfg, path)

    def test_paper_parity_lewm_train_config_tracks_paper_protocol_not_repo_defaults(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs" / "train_mwm_lewm_pusht_upstream.yaml").read_text(encoding="utf-8"))

        self.assertEqual(cfg["seed"], 3072)
        self.assertEqual(cfg["data"]["path"], "data/upstream/pusht_expert_train.lance")
        self.assertEqual(cfg["data"]["format"], "lance")
        self.assertEqual(cfg["data"]["split_ratio"], 0.9)
        self.assertEqual(cfg["data"]["frameskip"], 5)
        self.assertEqual(cfg["data"]["keys_to_load"], ["pixels", "action", "proprio", "state"])
        self.assertEqual(cfg["data"]["keys_to_cache"], ["action", "proprio", "state"])

        self.assertEqual(cfg["model"]["D"], 192)
        self.assertEqual(cfg["model"]["K"], [192])
        self.assertEqual(cfg["model"]["history_size"], 3)
        self.assertEqual(cfg["model"]["num_preds"], 1)
        self.assertEqual(cfg["model"]["vit_image_size"], 224)
        self.assertEqual(cfg["model"]["predictor_depth"], 6)
        self.assertEqual(cfg["model"]["predictor_heads"], 16)
        self.assertEqual(cfg["model"]["predictor_dim_head"], 64)
        self.assertEqual(cfg["model"]["predictor_mlp_dim"], 2048)
        self.assertEqual(cfg["model"]["predictor_dropout"], 0.1)

        self.assertEqual(cfg["train"]["backend"], "stable_worldmodel_lewm")
        self.assertEqual(cfg["train"]["batch_size"], 128)
        self.assertEqual(cfg["train"]["num_workers"], 6)
        self.assertEqual(cfg["train"]["prefetch_factor"], 3)
        self.assertEqual(cfg["train"]["drop_last"], True)
        self.assertEqual(cfg["train"]["precision"], "bf16")
        self.assertEqual(cfg["train"]["gradient_clip_val"], 1.0)
        self.assertEqual(cfg["schedule"]["max_epochs"], 10)

        self.assertEqual(cfg["optim"]["lr"], 5e-5)
        self.assertEqual(cfg["optim"]["weight_decay"], 1e-3)
        self.assertEqual(cfg["loss"]["sigreg_weight"], 0.09)
        self.assertEqual(cfg["loss"]["sigreg_knots"], 17)
        self.assertEqual(cfg["loss"]["sigreg_num_proj"], 1024)

    def test_public_single_fidelity_configs_use_exact_lewm_backend(self) -> None:
        for name in ("train_mwm_lewm_pusht.yaml", "train_mwm_lewm_tworoom.yaml"):
            cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))

            self.assertIn("base", cfg, name)
            self.assertEqual(cfg["base"]["family"], "lewm", name)
            self.assertIn("checkpoint", cfg["base"], name)
            self.assertEqual(cfg["mwm"]["component_policy"]["shared"], ["latent_producer"], name)
            self.assertEqual(cfg["mwm"]["component_policy"]["per_level"], ["transition"], name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["regularizers"], "shared_latent", name)
            self.assertEqual(cfg["model"]["D"], 192, name)
            self.assertEqual(cfg["model"]["K"], [192], name)
            self.assertEqual(cfg["train"]["backend"], "stable_worldmodel_lewm", name)
            self.assertEqual(cfg["model"]["history_size"], 3, name)
            self.assertEqual(cfg["model"]["num_preds"], 1, name)
            self.assertEqual(cfg["loss"]["sigreg_weight"], 0.09, name)
            self.assertEqual(cfg["train"]["batch_size"], 128, name)
            self.assertEqual(cfg["schedule"]["max_epochs"], 10, name)

    def test_paper_parity_train_configs_use_base_adaptive_resolver(self) -> None:
        expected_checkpoints = {
            "train_mwm_lewm_pusht_upstream.yaml": "models--quentinll--lewm-pusht",
            "train_mwm_lewm_tworoom_upstream.yaml": "models--quentinll--lewm-tworooms",
        }
        for name, checkpoint in expected_checkpoints.items():
            cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))

            self.assertEqual(cfg["base"], {"family": "lewm", "checkpoint": checkpoint}, name)
            self.assertEqual(cfg["mwm"]["component_policy"]["shared"], ["latent_producer"], name)
            self.assertEqual(cfg["mwm"]["component_policy"]["per_level"], ["transition"], name)
            self.assertEqual(cfg["mwm"]["component_policy"]["reconstructor"], [], name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["regularizers"], "shared_latent", name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["reconstructor_detach_encoder"], True, name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["reconstructor_contributes_to_encoder_loss"], False, name)

    def test_scheduled_configs_use_lewm_base_adapter_training_recipe(self) -> None:
        for name in ("train_mwm_scheduled_pusht.yaml", "train_mwm_scheduled_tworoom.yaml"):
            cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))

            self.assertIn("base", cfg, name)
            self.assertEqual(cfg["base"]["family"], "lewm", name)
            self.assertIn("checkpoint", cfg["base"], name)
            self.assertEqual(cfg["mwm"]["component_policy"]["shared"], ["latent_producer"], name)
            self.assertEqual(cfg["mwm"]["component_policy"]["per_level"], ["transition"], name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["regularizers"], "shared_latent", name)
            self.assertEqual(cfg["model"]["D"], 192, name)
            self.assertEqual(cfg["model"]["K"], [48, 96, 144, 192], name)
            self.assertEqual(cfg["train"]["backend"], "stable_worldmodel_lewm", name)
            self.assertEqual(cfg["train"]["batch_size"], 128, name)
            self.assertEqual(cfg["train"]["num_workers"], 6, name)
            self.assertEqual(cfg["loss"]["sigreg_weight"], 0.09, name)
            self.assertEqual(cfg["schedule"]["max_epochs"], 10, name)

    def test_paper_parity_tworoom_configs_exist_and_use_paper_eval_profile(self) -> None:
        train_cfg = yaml.safe_load((ROOT / "configs" / "train_mwm_lewm_tworoom_upstream.yaml").read_text(encoding="utf-8"))
        eval_cfg = yaml.safe_load((ROOT / "configs" / "eval_mwm_paper_tworoom.yaml").read_text(encoding="utf-8"))
        bench_cfg = yaml.safe_load((ROOT / "configs" / "benchmark_mwm_paper_parity.yaml").read_text(encoding="utf-8"))

        self.assertEqual(train_cfg["data"]["path"], "data/upstream/tworoom.lance")
        self.assertEqual(train_cfg["data"]["keys_to_load"], ["pixels", "action", "proprio"])
        self.assertEqual(train_cfg["data"]["keys_to_cache"], ["action", "proprio"])
        self.assertEqual(train_cfg["train"]["backend"], "stable_worldmodel_lewm")
        self.assertEqual(train_cfg["model"]["K"], [192])
        self.assertEqual(eval_cfg["data"]["path"], "data/upstream/tworoom.lance")
        self.assertEqual(eval_cfg["data"]["format"], "lance")
        self.assertEqual(eval_cfg["data"]["keys_to_cache"], ["action", "proprio"])
        self.assertEqual(eval_cfg["eval"]["episodes"], 50)
        self.assertEqual(eval_cfg["eval"]["goal_offset"], 25)
        self.assertEqual(eval_cfg["eval"]["budget"], 50)
        self.assertEqual(eval_cfg["planner"]["batch_size"], 1)
        self.assertEqual(eval_cfg["planner"]["n_iter"], 30)
        self.assertEqual(eval_cfg["planner"]["topk"], 30)
        self.assertEqual(bench_cfg["gate"]["env_ids"], ["swm/PushT-v1", "swm/TwoRoom-v1"])

    def test_tworoom_official_schema_uses_future_proprio_as_eval_goal(self) -> None:
        spec_id, callables = eval_callables_for_env(
            "swm/TwoRoom-v1",
            {"pixels", "action", "proprio", "pos_agent", "pos_target"},
        )

        self.assertEqual(spec_id, "point_state_goal_state")
        self.assertEqual(callables[0]["args"]["state"]["value"], "proprio")
        self.assertEqual(callables[1]["args"]["goal_state"]["value"], "goal_proprio")

    def test_tworoom_pos_schema_uses_future_agent_position_as_eval_goal(self) -> None:
        spec_id, callables = eval_callables_for_env(
            "swm/TwoRoom-v1",
            {"pixels", "action", "pos_agent", "pos_target"},
        )

        self.assertEqual(spec_id, "point_state_goal_state")
        self.assertEqual(callables[0]["args"]["state"]["value"], "pos_agent")
        self.assertEqual(callables[1]["args"]["goal_state"]["value"], "goal_pos_agent")

    def test_local_tworoom_train_configs_match_available_lance_columns(self) -> None:
        for name in ("train_mwm_lewm_tworoom.yaml", "train_mwm_scheduled_tworoom.yaml"):
            cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))

            self.assertEqual(cfg["data"]["path"], "data/tworoom_swm.lance")
            self.assertEqual(cfg["data"]["keys_to_load"], ["pixels", "action", "proprio"])
            self.assertEqual(cfg["data"]["keys_to_cache"], ["action", "proprio"])

    def test_paper_parity_eval_config_tracks_upstream_eval_protocol(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs" / "eval_mwm_paper_pusht.yaml").read_text(encoding="utf-8"))

        self.assertEqual(cfg["data"]["path"], "data/upstream/pusht_expert_train.lance")
        self.assertEqual(cfg["data"]["format"], "lance")
        self.assertEqual(cfg["data"]["action_preprocessing"], "standard_scaler")
        self.assertEqual(cfg["eval"]["episodes"], 50)
        self.assertEqual(cfg["eval"]["goal_offset"], 25)
        self.assertEqual(cfg["eval"]["budget"], 50)
        self.assertEqual(cfg["eval"]["num_envs"], 50)
        self.assertEqual(cfg["eval"]["sampling"], "stable_worldmodel")

        self.assertEqual(cfg["planner"]["horizon"], 5)
        self.assertEqual(cfg["planner"]["receding_horizon"], 5)
        self.assertEqual(cfg["planner"]["action_block"], 5)
        self.assertEqual(cfg["planner"]["pop_size"], 300)
        self.assertEqual(cfg["planner"]["topk"], 30)
        self.assertEqual(cfg["planner"]["n_iter"], 30)
        self.assertEqual(cfg["planner"]["init_std"], 1.0)
        self.assertEqual(cfg["planner"]["batch_size"], 1)

    def test_gpu_runner_scripts_require_slurm_allocation(self) -> None:
        scripts = [
            ROOT / "scripts" / "run_mwm_single_level_match.sh",
            ROOT / "scripts" / "run_mwm_train_single_level_env.sh",
            ROOT / "scripts" / "run_mwm_single_level_benchmark.sh",
            ROOT / "scripts" / "run_mwm_train_v1_env.sh",
            ROOT / "scripts" / "run_mwm_v1_benchmark.sh",
            ROOT / "scripts" / "run_mwm_paper_reference.sh",
        ]
        for script in scripts:
            text = script.read_text(encoding="utf-8")

            self.assertIn("SLURM_JOB_ID", text, script)
            self.assertIn("must run inside a Slurm allocation", text, script)

    def test_lance_only_runtime_has_no_hdf5_support_paths(self) -> None:
        runtime_files = [
            ROOT / "eval_mwm.py",
            ROOT / "verify_mwm_data.py",
        ]
        hits: list[str] = []
        for path in runtime_files:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for token in ("hdf5", ".h5"):
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)} contains {token}")

        self.assertEqual(hits, [])

    def test_legacy_top_level_swm_and_reference_diagnostics_are_removed(self) -> None:
        removed = [
            "benchmark_swm.py",
            "collect_swm.py",
            "plan_swm.py",
            "train_world_swm.py",
            "datasets/swm_hdf5.py",
            "docs/SWM_FIRST.md",
            "scripts/lewm_reference_matrix.py",
            "scripts/slurm_lewm_reference_matrix.sbatch",
            "scripts/slurm_lewm_official_pusht_eval.sbatch",
            "docs/superpowers/paper-parity-investigation-2026-05-28.md",
        ]
        for rel in removed:
            self.assertFalse((ROOT / rel).exists(), rel)
        self.assertTrue((ROOT / "mwm" / "planning" / "scheduled_cem.py").is_file())

    def test_lewm_adapter_file_does_not_export_dead_generic_scaffolding(self) -> None:
        text = (ROOT / "mwm" / "adapters" / "lewm.py").read_text(encoding="utf-8")
        forbidden = [
            "class MWMComponents",
            "class MWMAdapter",
            "class MWMImporter",
            "class LeWMAdapter",
            "class HFViTCLSBackbone",
            "class StablePretrainingViTBackbone",
        ]
        for token in forbidden:
            self.assertNotIn(token, text)

    def test_unused_helper_modules_and_ogbench_restore_support_are_removed(self) -> None:
        removed = [
            "mwm/metrics.py",
            "mwm/swm/wrappers.py",
        ]
        for rel in removed:
            self.assertFalse((ROOT / rel).exists(), rel)

        training_text = (ROOT / "mwm" / "training.py").read_text(encoding="utf-8")
        for token in (
            "StablePretrainingMWMModule",
            "build_stable_pretraining_module",
            "build_stable_sigreg",
        ):
            self.assertNotIn(token, training_text)

        restore_text = (ROOT / "mwm" / "swm" / "restore.py").read_text(encoding="utf-8")
        self.assertNotIn("OGB", restore_text)
        self.assertNotIn("needs_restore_recorder=True", restore_text)


if __name__ == "__main__":
    unittest.main()
