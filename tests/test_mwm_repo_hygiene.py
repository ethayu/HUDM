from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from mwm.swm.restore import eval_callables_for_env


ROOT = Path(__file__).resolve().parents[1]


def _first_line_index(lines: list[str], tokens: tuple[str, ...]) -> int | None:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(token in line for token in tokens):
            return index
    return None


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
        for key in (
            "encoder",
            "freeze_encoder",
            "normalize_imagenet",
            "vit_size",
            "vit_patch_size",
            "vit_image_size",
            "vit_pretrained",
            "vit_use_mask_token",
            "dynamics",
            "predictor_depth",
            "predictor_heads",
            "predictor_dim_head",
            "predictor_mlp_dim",
            "predictor_dropout",
            "projector_hidden_dim",
        ):
            self.assertNotIn(key, cfg["model"])

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

    def test_public_single_fidelity_configs_use_lewm_base_adapter_backend(self) -> None:
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

    def test_train_configs_do_not_override_base_architecture_knobs(self) -> None:
        forbidden_model_keys = {
            "encoder",
            "freeze_encoder",
            "normalize_imagenet",
            "vit_model_name",
            "vit_size",
            "vit_patch_size",
            "vit_image_size",
            "vit_pretrained",
            "vit_use_mask_token",
            "dynamics",
            "predictor_depth",
            "predictor_heads",
            "predictor_dim_head",
            "predictor_mlp_scale",
            "predictor_mlp_dim",
            "predictor_dropout",
            "predictor_emb_dropout",
            "projector_hidden_dim",
        }

        for path in sorted((ROOT / "configs").glob("train_mwm*.yaml")):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            model = cfg.get("model", {})
            self.assertFalse(forbidden_model_keys & set(model), path)

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
        work_tokens = (
            '"$PY"',
            "train_mwm.py",
            "benchmark_mwm.py",
            "verify_mwm_",
            "prepare_upstream_",
        )
        scripts = sorted((ROOT / "scripts").glob("run_mwm*.sh"))
        self.assertGreater(len(scripts), 0)

        for script in scripts:
            text = script.read_text(encoding="utf-8")
            if not any(token in text for token in work_tokens):
                continue
            lines = text.splitlines()
            guard_line = _first_line_index(lines, ("SLURM_JOB_ID",))
            work_line = _first_line_index(lines, work_tokens)

            self.assertIn("SLURM_JOB_ID", text, script)
            self.assertIn("must run inside a Slurm allocation", text, script)
            self.assertIsNotNone(guard_line, script)
            self.assertIsNotNone(work_line, script)
            self.assertLess(guard_line, work_line, script)

    def test_slurm_mwm_scripts_refuse_direct_bash_before_gpu_or_work(self) -> None:
        risk_tokens = (
            "nvidia-smi",
            "torch.cuda",
            "exec scripts/run_mwm",
        )
        scripts = sorted((ROOT / "scripts").glob("slurm_mwm*.sbatch"))
        self.assertGreater(len(scripts), 0)

        for script in scripts:
            text = script.read_text(encoding="utf-8")
            if not any(token in text for token in risk_tokens):
                continue
            lines = text.splitlines()
            guard_line = _first_line_index(lines, ("SLURM_JOB_ID",))
            work_line = _first_line_index(lines, risk_tokens)

            self.assertIn("SLURM_JOB_ID", text, script)
            self.assertIn("must be submitted with sbatch", text, script)
            self.assertIsNotNone(guard_line, script)
            self.assertIsNotNone(work_line, script)
            self.assertLess(guard_line, work_line, script)

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

    def test_upstream_data_prep_is_lance_only(self) -> None:
        text = (ROOT / "prepare_upstream_lewm_data.py").read_text(encoding="utf-8", errors="ignore").lower()
        forbidden = [
            "hdf5",
            ".h5",
            "tar.zst",
            "zstd",
            "stable_worldmodel.data.convert",
            "source_format",
            "dest_format",
        ]
        for token in forbidden:
            self.assertNotIn(token, text)

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

    def test_lewm_adapter_keeps_generic_model_logic_in_world_model(self) -> None:
        adapter_dir = ROOT / "mwm" / "adapters"
        self.assertTrue((adapter_dir / "lewm_stable.py").is_file())
        self.assertFalse((adapter_dir / "lewm_common.py").exists())
        self.assertFalse((adapter_dir / "lewm_model.py").exists())
        self.assertFalse((adapter_dir / "lewm_import.py").exists())
        self.assertFalse((ROOT / "mwm" / "adapters" / "lewm_direct.py").exists())

        facade_lines = (ROOT / "mwm" / "adapters" / "lewm.py").read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(facade_lines), 80)
        facade_text = "\n".join(facade_lines)
        self.assertNotIn("lewm_direct", facade_text)
        self.assertNotIn("lewm_model", facade_text)
        self.assertNotIn("lewm_import", facade_text)
        self.assertNotIn("build_lewm_matryoshka_model", facade_text)
        self.assertNotIn("MWMLeWMAdapterConfig", facade_text)

        world_model_text = (ROOT / "mwm" / "models" / "world_model.py").read_text(encoding="utf-8")
        self.assertIn("class MatryoshkaWorldModel", world_model_text)
        self.assertIn("class TransitionPackage", world_model_text)
        for path in (adapter_dir / "lewm.py", adapter_dir / "lewm_stable.py", ROOT / "mwm" / "models" / "world_model.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("source_model", text, path)
            self.assertNotIn("delegated_source_cost", text, path)
            self.assertNotIn("ImportedLeWMMWMWorldModel", text, path)
            self.assertNotIn("build_mwm_lewm_from_object", text, path)
        prep_text = (ROOT / "prepare_upstream_lewm.py").read_text(encoding="utf-8")
        self.assertNotIn("LeWMObjectImporter", prep_text)
        self.assertNotIn("build_mwm_lewm_from_object", prep_text)

    def test_generic_world_model_fallbacks_and_raw_lewm_training_are_removed(self) -> None:
        world_model_text = (ROOT / "mwm" / "models" / "world_model.py").read_text(encoding="utf-8")
        for token in (
            "MWMActionSpec",
            "MWMComponentSpec",
            "_DefaultDynamics",
            "_DefaultImageDecoder",
            "def mwm_prediction_loss",
        ):
            self.assertNotIn(token, world_model_text)

        self.assertFalse((ROOT / "mwm" / "training.py").exists())

        train_entrypoint_text = (ROOT / "train_mwm.py").read_text(encoding="utf-8")
        for token in (
            "_build_exact_lewm_object",
            "_resolve_model_cfg",
            "_load_train_valid_datasets",
            "_run_stable_pretraining",
            "module.model.predict",
            'backend in {"stable_worldmodel_lewm", "exact_lewm"}',
        ):
            self.assertNotIn(token, train_entrypoint_text)

    def test_unused_helper_modules_and_ogbench_restore_support_are_removed(self) -> None:
        removed = [
            "mwm/metrics.py",
            "mwm/training.py",
            "mwm/swm/wrappers.py",
        ]
        for rel in removed:
            self.assertFalse((ROOT / rel).exists(), rel)

        restore_text = (ROOT / "mwm" / "swm" / "restore.py").read_text(encoding="utf-8")
        self.assertNotIn("OGB", restore_text)
        self.assertNotIn("needs_restore_recorder=True", restore_text)


if __name__ == "__main__":
    unittest.main()
