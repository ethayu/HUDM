from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys
import unittest

import yaml

from mwm.swm.restore import eval_callables_for_env, validate_restore_columns
from mwm.upstream.paper_parity import paper_parity_dataset_spec


ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLDMODEL_VERSION = "0.1.0"
OLD_FLAT_SCRIPT_REF_RE = re.compile(
    r"scripts/(?:local_|run_mwm_|submit_mwm_|poll_mwm_|slurm_mwm_|slurm_research_|research_)"
)
ROOT_CLI_COMMANDS = (
    "collect_" + "mwm_data.py",
    "train_" + "mwm.py",
    "eval_" + "mwm.py",
    "benchmark_" + "mwm.py",
    "prepare_" + "upstream_lewm.py",
    "prepare_" + "upstream_lewm_data.py",
    "verify_" + "mwm_data.py",
    "verify_" + "mwm_benchmark.py",
    "render_" + "benchmark_review.py",
)
PACKAGE_CLI_COMMANDS = (
    "-m mwm.data.collection",
    "-m mwm.upstream.lewm_checkpoints",
    "-m mwm.upstream.lewm_data",
    "-m mwm.training.stable_wm",
    "-m mwm.eval.runner",
    "-m mwm.benchmark.matrix",
    "-m mwm.data.verify",
    "-m mwm.benchmark.verify",
    "-m mwm.benchmark.render_review",
)
REQUIRED_TRACKED_PACKAGE_MODULES = (
    "mwm/benchmark/checkpoint_verify.py",
    "mwm/benchmark/matrix_identity.py",
    "mwm/benchmark/output_verify.py",
    "mwm/benchmark/paper_targets.py",
    "mwm/benchmark/plot_contract.py",
    "mwm/benchmark/render_review.py",
    "mwm/benchmark/static_verify.py",
    "mwm/data/collection.py",
    "mwm/upstream/__init__.py",
    "mwm/upstream/converters/__init__.py",
    "mwm/upstream/converters/ogb_cube.py",
    "mwm/upstream/converters/reacher.py",
    "mwm/upstream/lewm_checkpoints.py",
    "mwm/upstream/lewm_data.py",
)
OLD_ROOT_SCRIPT_PATH_RE = re.compile(
    r"(?:python(?:\S*)?\s+)?(?:\./)?scripts/"
    r"(?:local_|run_mwm_|submit_mwm_|poll_mwm_|slurm_mwm_|slurm_research_|research_)[\w/.-]*\.py"
    r"|(?:python(?:\S*)?\s+)(?:\./)?"
    r"(?:local_|run_mwm_|submit_mwm_|poll_mwm_|slurm_mwm_|slurm_research_|research_)[\w/.-]*\.py"
    r"|(?:python(?:\S*)?\s+)?(?:\./)?scripts/"
    r"(?:convert_reacher_h5_to_lance|convert_ogb_cube_hdf5_to_lance)\.py"
)


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
        "__pycache__",
        "checkpoints",
        "checkpoints_mwm",
        "data",
        "logs",
        "rollouts",
        "synthetic",
    }
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    files: list[Path] = []
    for rel in tracked:
        path = ROOT / rel
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in Path(rel).parts):
            continue
        if path.suffix in {".py", ".yaml", ".yml", ".md", ".sh", ".sbatch"}:
            files.append(path)
    return files


def _active_command_reference_files() -> list[Path]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "README.md", "REVIEW_GUIDE.md", "docs", "scripts"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    files: list[Path] = []
    for rel in tracked:
        path = ROOT / rel
        if rel.startswith("docs/superpowers/"):
            continue
        if path.is_file() and path.suffix in {".py", ".md", ".sh", ".sbatch", ".json"}:
            files.append(path)
    return files


class MWMRepoHygieneTests(unittest.TestCase):
    def test_stable_wm_adapter_refactor_public_surface(self) -> None:
        expected_files = {
            "mwm/diagnostics/__init__.py",
            "mwm/diagnostics/flops.py",
            "mwm/models/common.py",
            "mwm/models/lewm.py",
            "mwm/models/prejepa.py",
            "mwm/training/stable_wm.py",
            "mwm/training/stable_wm_callbacks.py",
            "mwm/training/stable_wm_config.py",
            "mwm/training/stable_wm_data.py",
            "mwm/training/stable_wm_export.py",
            "mwm/training/stable_wm_lightning.py",
            "mwm/training/stable_wm_model.py",
            "mwm/training/stable_wm_runtime.py",
            "mwm/training/stable_wm_transforms.py",
        }
        retired_files = {
            "mwm/models/base_adaptive.py",
            "mwm/models/flops.py",
            "mwm/training/lewm.py",
            "mwm/training/lewm_callbacks.py",
            "mwm/training/lewm_config.py",
            "mwm/training/lewm_data.py",
            "mwm/training/lewm_export.py",
            "mwm/training/lewm_lightning.py",
            "mwm/training/lewm_model.py",
            "mwm/training/lewm_runtime.py",
            "mwm/training/lewm_transforms.py",
        }
        for rel in expected_files:
            with self.subTest(expected=rel):
                self.assertTrue((ROOT / rel).is_file(), rel)
        for rel in retired_files:
            with self.subTest(retired=rel):
                self.assertFalse((ROOT / rel).exists(), rel)

    def test_no_generic_training_code_uses_lewm_namespace(self) -> None:
        forbidden = (
            "mwm.training." + "lewm",
            "lewm_" + "base_adapter",
            "validate_" + "lewm_loss_config",
            "prepare_" + "lewm_base_adapter_context",
            "build_trainable_" + "model_from_base",
            "run_" + "lewm_base_adapter_training",
            "export_" + "lewm_base_adapter_lightning_checkpoint",
        )
        paths = sorted((ROOT / "mwm" / "training").glob("*.py"))
        hits: list[str] = []
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)} contains {token}")
        self.assertEqual(hits, [])

    def test_flop_accounting_has_diagnostics_owner(self) -> None:
        self.assertTrue((ROOT / "mwm" / "diagnostics" / "flops.py").is_file())
        self.assertFalse((ROOT / "mwm" / "models" / "flops.py").exists())
        hits: list[str] = []
        for path in [*sorted((ROOT / "mwm").rglob("*.py")), *sorted((ROOT / "tests").glob("test_mwm*.py"))]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "mwm.models." + "flops" in text:
                hits.append(str(path.relative_to(ROOT)))
        self.assertEqual(hits, [])

    def test_requirements_pin_stable_worldmodel_to_verified_release(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        stable_worldmodel_lines = [
            line.strip()
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#") and line.startswith("stable-worldmodel")
        ]

        self.assertEqual(len(stable_worldmodel_lines), 1)
        self.assertEqual(stable_worldmodel_lines[0], f"stable-worldmodel[env]=={STABLE_WORLDMODEL_VERSION}")

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
        for path in sorted((ROOT / "configs").rglob("*.yaml")):
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

    def test_configs_are_grouped_by_type(self) -> None:
        root_yaml = sorted(path.name for path in (ROOT / "configs").glob("*.yaml"))
        self.assertEqual(root_yaml, [])
        for folder in ("train", "eval", "benchmark", "manifest"):
            self.assertTrue((ROOT / "configs" / folder).is_dir(), folder)

    def test_lewm_training_configs_follow_stable_wm_contract(self) -> None:
        expected_levels = {
            "mwm_lewm_pusht.yaml": [192],
            "mwm_lewm_reacher_upstream.yaml": [192],
            "mwm_lewm_ogb_cube_upstream.yaml": [192],
            "mwm_lewm_tworoom.yaml": [192],
            "mwm_lewm_pusht_upstream.yaml": [192],
            "mwm_lewm_tworoom_upstream.yaml": [192],
            "mwm_lewm_scheduled_pusht.yaml": [48, 96, 144],
            "mwm_lewm_scheduled_tworoom.yaml": [48, 96, 144],
            "mwm_lewm_dense_pusht.yaml": [6, 12, 48, 96, 144, 192],
            "mwm_lewm_dense_reacher.yaml": [6, 12, 48, 96, 144, 192],
            "mwm_lewm_dense_ogb_cube.yaml": [6, 12, 48, 96, 144, 192],
            "mwm_lewm_dense_tworoom.yaml": [6, 12, 48, 96, 144, 192],
        }
        for name, levels in expected_levels.items():
            cfg = yaml.safe_load((ROOT / "configs" / "train" / name).read_text(encoding="utf-8"))

            self.assertEqual(cfg["base"]["family"], "lewm", name)
            self.assertIn("checkpoint", cfg["base"], name)
            self.assertEqual(cfg["mwm"]["component_policy"], {
                "shared": ["latent_producer"],
                "per_level": ["transition"],
                "reconstructor": ["decoder"],
            }, name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["regularizers"], "shared_latent", name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["reconstructor_detach_encoder"], True, name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["reconstructor_contributes_to_encoder_loss"], False, name)
            self.assertIn("recon_latent_weight", cfg["loss"], name)
            self.assertNotIn("recon_weight", cfg["loss"], name)
            self.assertEqual(cfg["data"]["format"], "lance", name)
            self.assertTrue(str(cfg["data"]["path"]).endswith(".lance"), name)
            self.assertEqual(cfg["model"]["D"], 192, name)
            self.assertEqual(cfg["model"]["K"], levels, name)
            self.assertEqual(cfg["model"]["action_block"], 5, name)
            self.assertEqual(cfg["train"]["backend"], "stable_worldmodel_lewm", name)
            self.assertEqual(set(cfg["schedule"]), {"max_epochs"}, name)
            self.assertEqual(cfg["schedule"]["max_epochs"], 10, name)
            if name.startswith(("mwm_lewm_scheduled_", "mwm_lewm_dense_")) or name in {
                "mwm_lewm_reacher_upstream.yaml",
                "mwm_lewm_ogb_cube_upstream.yaml",
            }:
                self.assertTrue(str(cfg["data"]["path"]).startswith("data/upstream/"), name)

            if "reacher" in name:
                self.assertEqual(cfg["env_id"], "swm/ReacherDMControl-v0", name)
                self.assertEqual(cfg["base"]["checkpoint"], "models--quentinll--lewm-reacher", name)
                self.assertEqual(cfg["data"]["path"], "data/upstream/reacher.lance", name)
                self.assertEqual(cfg["data"]["keys_to_load"], ["pixels", "action", "qpos", "qvel", "observation"], name)
                self.assertEqual(cfg["data"]["keys_to_cache"], ["action", "qpos", "qvel", "observation"], name)
                self.assertEqual(cfg["restore"]["import_path"], "mwm.swm.restore.reacher_qpos_match_restore_spec", name)
                self.assertEqual(cfg["loss"]["sigreg_weight"], 0.09, name)
                self.assertEqual(cfg["loss"]["sigreg_knots"], 17, name)
                self.assertEqual(cfg["loss"]["sigreg_num_proj"], 1024, name)
            if "ogb_cube" in name:
                self.assertEqual(cfg["env_id"], "swm/OGBCube-v0", name)
                self.assertEqual(cfg["base"]["checkpoint"], "models--quentinll--lewm-cube", name)
                self.assertEqual(cfg["data"]["path"], "data/upstream/ogb_cube_single_expert.lance", name)
                self.assertEqual(cfg["data"]["keys_to_load"], ["pixels", "action", "observation"], name)
                self.assertEqual(cfg["data"]["keys_to_cache"], ["action", "observation"], name)
                self.assertEqual(cfg["restore"]["import_path"], "mwm.ogbench.restore.ogbench_cube_restore_spec", name)
                self.assertEqual(cfg["loss"]["sigreg_weight"], 0.09, name)
                self.assertEqual(cfg["loss"]["sigreg_knots"], 17, name)
                self.assertEqual(cfg["loss"]["sigreg_num_proj"], 1024, name)

    def test_all_lewm_training_configs_use_decoder_reconstruction_contract(self) -> None:
        paths = [
            *sorted((ROOT / "configs" / "train").glob("*.yaml")),
            *sorted((ROOT / "configs" / "local").glob("train_*.yaml")),
            *sorted((ROOT / "configs" / "research").glob("train_mwm*.yaml")),
        ]
        self.assertGreater(len(paths), 0)
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(cfg["mwm"]["component_policy"]["reconstructor"], ["decoder"])
                self.assertIn("recon_latent_weight", cfg["loss"])
                self.assertNotIn("recon_weight", cfg["loss"])

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

        for path in sorted((ROOT / "configs" / "train").glob("mwm*.yaml")):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            model = cfg.get("model", {})
            self.assertFalse(forbidden_model_keys & set(model), path)

    def test_paper_parity_eval_configs_follow_base_inference_contract(self) -> None:
        for name in ("paper_pusht.yaml", "paper_reacher.yaml", "paper_tworoom.yaml", "paper_ogb_cube.yaml"):
            cfg = yaml.safe_load((ROOT / "configs" / "eval" / name).read_text(encoding="utf-8"))

            self.assertEqual(cfg["data"]["format"], "lance", name)
            self.assertEqual(cfg["data"]["action_preprocessing"], "standard_scaler", name)
            self.assertEqual(cfg["eval"]["sampling"], "stable_worldmodel", name)
            self.assertEqual(cfg["eval"]["goal_offset"], 25, name)
            self.assertEqual(cfg["eval"]["episodes"], 50, name)
            self.assertEqual(cfg["eval"]["budget"], 50, name)
            self.assertEqual(cfg["planner"]["pop_size"], 300, name)
            self.assertEqual(cfg["planner"]["topk"], 30, name)
            self.assertEqual(cfg["planner"]["n_iter"], 30, name)
            self.assertEqual(cfg["planner"]["action_block"], 5, name)
            self.assertEqual(cfg["planner"]["scheduler"]["mpc"], {"mode": "fixed", "level": "finest"}, name)
            self.assertEqual(cfg["planner"]["scheduler"]["cem"], {"mode": "fixed", "level": "base"}, name)
            self.assertEqual(cfg["planner"]["scheduler"]["rollout"], {"mode": "fixed", "level": "base"}, name)

        bench_cfg = yaml.safe_load((ROOT / "configs" / "benchmark" / "paper_parity_pusht.yaml").read_text(encoding="utf-8"))
        self.assertEqual(bench_cfg["env_id"], "swm/PushT-v1")
        self.assertEqual([run["role"] for run in bench_cfg["runs"]], ["upstream_lewm_converted", "retrained_lewm_identity"])
        self.assertEqual(bench_cfg["paper_targets"]["tolerance_pp"], 1.0)

        reacher_eval = yaml.safe_load((ROOT / "configs" / "eval" / "paper_reacher.yaml").read_text(encoding="utf-8"))
        self.assertEqual(reacher_eval["env_id"], "swm/ReacherDMControl-v0")
        self.assertEqual(reacher_eval["data"]["path"], "data/upstream/reacher.lance")
        self.assertEqual(reacher_eval["data"]["keys_to_load"], ["pixels", "action", "qpos", "qvel", "observation"])
        self.assertEqual(reacher_eval["data"]["keys_to_cache"], ["action", "qpos", "qvel", "observation"])
        self.assertEqual(reacher_eval["env"]["kwargs"], {"task": "qpos_match"})
        self.assertEqual(reacher_eval["restore"]["import_path"], "mwm.swm.restore.reacher_qpos_match_restore_spec")

        cube_eval = yaml.safe_load((ROOT / "configs" / "eval" / "paper_ogb_cube.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cube_eval["env_id"], "swm/OGBCube-v0")
        self.assertEqual(cube_eval["checkpoint"]["run_dir"], "checkpoints_mwm/upstream_lewm_ogb_cube")
        self.assertEqual(cube_eval["data"]["path"], "data/upstream/ogb_cube_single_expert.lance")
        self.assertEqual(
            cube_eval["data"]["keys_to_load"],
            ["pixels", "action", "qpos", "qvel", "observation", "privileged/block_0_pos", "privileged/block_0_quat"],
        )
        self.assertEqual(
            cube_eval["data"]["keys_to_cache"],
            ["action", "qpos", "qvel", "observation", "privileged/block_0_pos", "privileged/block_0_quat"],
        )
        self.assertEqual(cube_eval["env"]["kwargs"]["env_type"], "single")
        self.assertEqual(cube_eval["env"]["kwargs"]["ob_type"], "states")
        self.assertEqual(cube_eval["env"]["kwargs"]["width"], 224)
        self.assertEqual(cube_eval["env"]["kwargs"]["height"], 224)
        self.assertEqual(cube_eval["restore"]["import_path"], "mwm.ogbench.restore.ogbench_cube_restore_spec")

    def test_eval_and_benchmark_configs_use_nested_scheduler_schema(self) -> None:
        legacy_keys = {"policy", "level", "base_level", "start_level", "end_level", "rollout_level", "rollout_levels"}
        for directory in ("benchmark", "eval", "local", "research"):
            for path in sorted((ROOT / "configs" / directory).glob("**/*.yaml")):
                cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                schedulers = []
                if isinstance(cfg.get("planner"), dict) and isinstance(cfg["planner"].get("scheduler"), dict):
                    schedulers.append(cfg["planner"]["scheduler"])
                for run in cfg.get("runs", []) or []:
                    planner = run.get("planner", {}) if isinstance(run, dict) else {}
                    if isinstance(planner.get("scheduler"), dict):
                        schedulers.append(planner["scheduler"])
                for scheduler in schedulers:
                    self.assertFalse(legacy_keys & set(scheduler), path)
                    self.assertEqual(set(scheduler), {"enabled", "mpc", "cem", "rollout"}, path)
                    self.assertIn(scheduler["mpc"]["mode"], {"fixed", "linear"}, path)
                    self.assertIn(scheduler["cem"]["mode"], {"fixed", "linear"}, path)
                    self.assertIn(scheduler["rollout"]["mode"], {"fixed", "linear"}, path)

        reacher_bench = yaml.safe_load((ROOT / "configs" / "benchmark" / "paper_parity_reacher.yaml").read_text(encoding="utf-8"))
        self.assertEqual(reacher_bench["env_id"], "swm/ReacherDMControl-v0")
        self.assertEqual(reacher_bench["eval_config"], "configs/eval/paper_reacher.yaml")
        self.assertEqual([run["role"] for run in reacher_bench["runs"]], ["upstream_lewm_converted", "retrained_lewm_identity"])
        self.assertEqual(reacher_bench["runs"][0]["checkpoint"], "checkpoints_mwm/upstream_lewm_reacher")
        self.assertEqual(reacher_bench["runs"][1]["checkpoint"], "checkpoints_mwm/retrained_lewm_identity_reacher_upstream")

        cube_bench = yaml.safe_load((ROOT / "configs" / "benchmark" / "paper_parity_ogb_cube.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cube_bench["env_id"], "swm/OGBCube-v0")
        self.assertEqual(cube_bench["eval_config"], "configs/eval/paper_ogb_cube.yaml")
        self.assertEqual(cube_bench["manifest"]["config"], "configs/manifest/ogb_cube_paper_seed42.yaml")
        self.assertEqual(cube_bench["paper_targets"]["success_rate"], {"swm/OGBCube-v0": 74.0})
        self.assertEqual([run["role"] for run in cube_bench["runs"]], ["upstream_lewm_converted", "retrained_lewm_identity"])
        self.assertEqual(cube_bench["runs"][0]["checkpoint"], "checkpoints_mwm/upstream_lewm_ogb_cube")
        self.assertEqual(cube_bench["runs"][1]["checkpoint"], "checkpoints_mwm/retrained_lewm_identity_ogb_cube_upstream")

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

    def test_reacher_qpos_match_restore_is_opt_in_and_uses_goal_qpos(self) -> None:
        import_path = "mwm.swm.restore.reacher_qpos_match_restore_spec"

        spec = validate_restore_columns(
            "swm/ReacherDMControl-v0",
            {"pixels", "action", "qpos", "qvel", "observation"},
            import_path=import_path,
        )
        spec_id, callables = eval_callables_for_env(
            "swm/ReacherDMControl-v0",
            {"pixels", "action", "qpos", "qvel", "observation"},
            import_path=import_path,
        )

        self.assertEqual(spec.spec_id, "reacher_qpos_match_qpos_qvel")
        self.assertEqual(spec_id, "reacher_qpos_match_qpos_qvel")
        self.assertEqual(callables[0]["method"], "set_state")
        self.assertEqual(callables[0]["args"]["qpos"]["value"], "qpos")
        self.assertEqual(callables[0]["args"]["qvel"]["value"], "qvel")
        self.assertEqual(callables[1]["method"], "set_target_qpos")
        self.assertEqual(callables[1]["args"]["target_qpos"]["value"], "goal_qpos")

        with self.assertRaisesRegex(ValueError, "qvel"):
            validate_restore_columns(
                "swm/ReacherDMControl-v0",
                {"pixels", "action", "qpos"},
                import_path=import_path,
            )

    def test_ogbench_cube_restore_is_opt_in_and_uses_goal_block_pose(self) -> None:
        import_path = "mwm.ogbench.restore.ogbench_cube_restore_spec"
        columns = {
            "pixels",
            "action",
            "qpos",
            "qvel",
            "observation",
            "privileged/block_0_pos",
            "privileged/block_0_quat",
        }

        spec = validate_restore_columns("swm/OGBCube-v0", columns, import_path=import_path)
        spec_id, callables = eval_callables_for_env("swm/OGBCube-v0", columns, import_path=import_path)

        self.assertEqual(spec.spec_id, "ogbench_cube_single_qpos_qvel_target_pose")
        self.assertEqual(spec_id, "ogbench_cube_single_qpos_qvel_target_pose")
        self.assertEqual(callables[0]["method"], "set_state")
        self.assertEqual(callables[0]["args"]["qpos"]["value"], "qpos")
        self.assertEqual(callables[0]["args"]["qvel"]["value"], "qvel")
        self.assertEqual(callables[1]["method"], "set_target_pos")
        self.assertEqual(callables[1]["args"]["cube_id"]["value"], 0)
        self.assertFalse(callables[1]["args"]["cube_id"]["in_dataset"])
        self.assertEqual(callables[1]["args"]["target_pos"]["value"], "goal_privileged/block_0_pos")
        self.assertEqual(callables[1]["args"]["target_quat"]["value"], "goal_privileged/block_0_quat")

        underscore_columns = {
            "pixels",
            "action",
            "qpos",
            "qvel",
            "observation",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
        }
        _, underscore_callables = eval_callables_for_env("swm/OGBCube-v0", underscore_columns, import_path=import_path)
        self.assertEqual(underscore_callables[1]["args"]["target_pos"]["value"], "goal_privileged_block_0_pos")
        self.assertEqual(underscore_callables[1]["args"]["target_quat"]["value"], "goal_privileged_block_0_quat")

        with self.assertRaisesRegex(ValueError, "privileged/block_0_quat"):
            validate_restore_columns(
                "swm/OGBCube-v0",
                columns - {"privileged/block_0_quat"},
                import_path=import_path,
            )

    def test_local_tworoom_train_configs_match_available_lance_columns(self) -> None:
        for name in ("mwm_lewm_tworoom.yaml",):
            cfg = yaml.safe_load((ROOT / "configs" / "train" / name).read_text(encoding="utf-8"))

            self.assertEqual(cfg["data"]["path"], "data/tworoom_swm.lance")
            self.assertEqual(cfg["data"]["keys_to_load"], ["pixels", "action", "proprio"])
            self.assertEqual(cfg["data"]["keys_to_cache"], ["action", "proprio"])

    def test_paper_scheduled_tworoom_train_config_matches_upstream_lance_columns(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs" / "train" / "mwm_lewm_scheduled_tworoom.yaml").read_text(encoding="utf-8"))

        self.assertEqual(cfg["data"]["path"], "data/upstream/tworoom.lance")
        self.assertEqual(cfg["data"]["keys_to_load"], ["pixels", "action", "proprio"])
        self.assertEqual(cfg["data"]["keys_to_cache"], ["action", "proprio"])

    def test_upstream_reacher_preparation_is_declared(self) -> None:
        source = (ROOT / "mwm" / "upstream" / "lewm_checkpoints.py").read_text(encoding="utf-8")
        self.assertIn('"name": "upstream_lewm_reacher"', source)
        self.assertIn('"repo": "quentinll/lewm-reacher"', source)
        self.assertIn('"restore_spec": "reacher_qpos_match_qpos_qvel"', source)
        self.assertIn('"name": "upstream_lewm_ogb_cube"', source)
        self.assertIn('"repo": "quentinll/lewm-cube"', source)
        self.assertIn('"restore_spec": "ogbench_cube_single_qpos_qvel_target_pose"', source)

        reacher = paper_parity_dataset_spec("reacher")
        self.assertEqual(reacher.lance_name, "reacher.lance")
        self.assertEqual(reacher.env_id, "swm/ReacherDMControl-v0")
        self.assertEqual(reacher.restore_spec, "reacher_qpos_match_qpos_qvel")
        ogb_cube = paper_parity_dataset_spec("ogb_cube")
        self.assertEqual(ogb_cube.lance_name, "ogb_cube_single_expert.lance")
        self.assertEqual(ogb_cube.env_id, "swm/OGBCube-v0")
        self.assertEqual(ogb_cube.restore_spec, "ogbench_cube_single_qpos_qvel_target_pose")
        self.assertTrue((ROOT / "mwm" / "upstream" / "converters" / "ogb_cube.py").is_file())

    def test_gpu_runner_scripts_require_slurm_allocation(self) -> None:
        work_tokens = (
            '"$PY" -m mwm.training.stable_wm',
            '"$PY" -m mwm.benchmark.matrix',
            '"$PY" -m mwm.benchmark.verify',
            '"$PY" -m mwm.data.verify',
            "mwm.upstream.lewm_checkpoints",
            "mwm.upstream.lewm_data",
        )
        scripts = sorted((ROOT / "scripts" / "slurm").glob("run_mwm*.sh"))
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

    def test_benchmark_comparison_scripts_finish_all_envs_before_reporting_failure(self) -> None:
        scripts = [
            ROOT / "scripts" / "slurm" / "run_mwm_identity_parity.sh",
            ROOT / "scripts" / "slurm" / "run_mwm_scheduled_comparison.sh",
            ROOT / "scripts" / "slurm" / "run_mwm_dense_comparison.sh",
        ]
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            self.assertIn("status=0", text, script)
            self.assertIn("run_step()", text, script)
            self.assertIn('exit "$status"', text, script)
            for line in text.splitlines():
                stripped = line.strip()
                if "-m mwm.benchmark.matrix" in stripped or "-m mwm.benchmark.verify" in stripped:
                    self.assertTrue(stripped.startswith("run_step "), f"{script}: {stripped}")

    def test_reacher_slurm_training_and_parity_commands_exist(self) -> None:
        identity_runner = (ROOT / "scripts" / "slurm" / "run_mwm_train_identity_env.sh").read_text(encoding="utf-8")
        self.assertIn("reacher)", identity_runner)
        self.assertIn("configs/train/mwm_lewm_reacher_upstream.yaml", identity_runner)
        self.assertIn("ogb_cube|cube)", identity_runner)
        self.assertIn("configs/train/mwm_lewm_ogb_cube_upstream.yaml", identity_runner)
        self.assertIn("{pusht|reacher|ogb_cube|tworoom}", identity_runner)

        dense_runner = (ROOT / "scripts" / "slurm" / "run_mwm_train_dense_env.sh").read_text(encoding="utf-8")
        self.assertIn("reacher)", dense_runner)
        self.assertIn("configs/train/mwm_lewm_dense_reacher.yaml", dense_runner)
        self.assertIn("ogb_cube|cube)", dense_runner)
        self.assertIn("configs/train/mwm_lewm_dense_ogb_cube.yaml", dense_runner)
        self.assertIn("{pusht|reacher|ogb_cube|tworoom}", dense_runner)

        for name in (
            "slurm_mwm_train_reacher_identity.sbatch",
            "slurm_mwm_train_reacher_dense.sbatch",
            "slurm_mwm_train_ogb_cube_identity.sbatch",
            "slurm_mwm_train_ogb_cube_dense.sbatch",
        ):
            text = (ROOT / "scripts" / "slurm" / name).read_text(encoding="utf-8")
            self.assertIn("run_mwm_train_", text, name)
            self.assertIn('torch.empty(1, device="cuda")', text, name)

        parity_runner = (ROOT / "scripts" / "slurm" / "run_mwm_identity_parity.sh").read_text(encoding="utf-8")
        self.assertIn("configs/benchmark/paper_parity_reacher.yaml", parity_runner)
        self.assertIn("configs/benchmark/paper_parity_ogb_cube.yaml", parity_runner)
        self.assertIn('export MUJOCO_GL="${MUJOCO_GL:-egl}"', parity_runner)
        self.assertIn('export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"', parity_runner)

        dense_comparison = (ROOT / "scripts" / "slurm" / "run_mwm_dense_comparison.sh").read_text(encoding="utf-8")
        self.assertIn("configs/benchmark/dense_reacher.yaml", dense_comparison)
        self.assertIn("configs/benchmark/dense_ogb_cube.yaml", dense_comparison)
        self.assertIn('export MUJOCO_GL="${MUJOCO_GL:-egl}"', dense_comparison)
        self.assertIn('export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"', dense_comparison)

    def test_reacher_h5_to_lance_conversion_tool_is_explicit_and_isolated(self) -> None:
        script = ROOT / "mwm" / "upstream" / "converters" / "reacher.py"
        text = script.read_text(encoding="utf-8")

        self.assertIn("h5py.File", text)
        self.assertIn("LanceWriter", text)
        self.assertIn("write_dataset_metadata", text)
        self.assertIn("validate_restore_columns", text)
        self.assertNotIn("prepare_reacher", text)
        self.assertIn("pixels", text)
        self.assertIn("qpos", text)
        self.assertIn("qvel", text)
        self.assertIn("observation", text)

    def test_reacher_h5_to_lance_conversion_tool_is_runnable_by_module(self) -> None:
        result = subprocess.run(
            [
                "/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python",
                "-m",
                "mwm.upstream.converters.reacher",
                "--help",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_upstream_preparation_tools_are_runnable_by_module(self) -> None:
        for module_name, expected in (
            ("mwm.upstream.lewm_checkpoints", "config"),
            ("mwm.upstream.lewm_data", "--source-h5"),
        ):
            with self.subTest(module=module_name):
                result = subprocess.run(
                    [
                        "/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python",
                        "-m",
                        module_name,
                        "--help",
                    ],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)

    def test_slurm_mwm_scripts_refuse_direct_bash_before_gpu_or_work(self) -> None:
        risk_tokens = (
            "nvidia-smi",
            "torch.cuda",
            "exec scripts/slurm/run_mwm",
        )
        scripts = sorted((ROOT / "scripts" / "slurm").glob("slurm_mwm*.sbatch"))
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
            ROOT / "mwm" / "eval" / "runner.py",
            ROOT / "mwm" / "data" / "verify.py",
        ]
        hits: list[str] = []
        for path in runtime_files:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for token in ("hdf5", ".h5"):
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)} contains {token}")

        self.assertEqual(hits, [])

    def test_desktop_entrypoints_accept_set_overrides(self) -> None:
        entrypoints = [
            ROOT / "mwm" / "data" / "collection.py",
            ROOT / "mwm" / "training" / "stable_wm.py",
            ROOT / "mwm" / "eval" / "runner.py",
            ROOT / "mwm" / "benchmark" / "matrix.py",
        ]
        for path in entrypoints:
            text = path.read_text(encoding="utf-8")
            self.assertIn("load_config", text, path)
            self.assertIn('"--set"', text, path)

    def test_mwm_clis_are_package_modules_without_root_wrappers(self) -> None:
        migrated = {
            "train_" + "mwm.py": "mwm.training.stable_wm",
            "eval_" + "mwm.py": "mwm.eval.runner",
            "benchmark_" + "mwm.py": "mwm.benchmark.matrix",
            "verify_" + "mwm_benchmark.py": "mwm.benchmark.verify",
            "verify_" + "mwm_data.py": "mwm.data.verify",
            "render_" + "benchmark_review.py": "mwm.benchmark.render_review",
        }
        for root_name, module in migrated.items():
            root_path = ROOT / root_name
            package_path = ROOT / Path(*module.split(".")).with_suffix(".py")
            with self.subTest(root=root_name):
                self.assertTrue(package_path.is_file(), package_path)
                self.assertFalse(root_path.exists(), root_name)
                package_text = package_path.read_text(encoding="utf-8")
                self.assertIn("def main", package_text, package_path)
                self.assertIn("if __name__ == \"__main__\"", package_text, package_path)

    def test_required_replacement_modules_are_tracked(self) -> None:
        for rel in REQUIRED_TRACKED_PACKAGE_MODULES:
            with self.subTest(path=rel):
                result = subprocess.run(
                    ["git", "ls-files", "--error-unmatch", rel],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_data_upstream_clis_are_package_modules_without_root_wrappers(self) -> None:
        package_modules = [
            ROOT / "mwm" / "data" / "collection.py",
            ROOT / "mwm" / "upstream" / "lewm_checkpoints.py",
            ROOT / "mwm" / "upstream" / "lewm_data.py",
            ROOT / "mwm" / "upstream" / "converters" / "reacher.py",
            ROOT / "mwm" / "upstream" / "converters" / "ogb_cube.py",
        ]
        retired = [
            ROOT / "collect_mwm_data.py",
            ROOT / "prepare_upstream_lewm.py",
            ROOT / "prepare_upstream_lewm_data.py",
            ROOT / "scripts" / "research" / "convert_reacher_h5_to_lance.py",
            ROOT / "scripts" / "research" / "convert_ogb_cube_hdf5_to_lance.py",
        ]
        for path in package_modules:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)
                self.assertIn("def main(", path.read_text(encoding="utf-8"), path)
                ignored = subprocess.run(
                    ["git", "check-ignore", "-q", str(path.relative_to(ROOT))],
                    cwd=ROOT,
                    check=False,
                )
                self.assertNotEqual(ignored.returncode, 0, path)
        for path in retired:
            with self.subTest(path=path):
                self.assertFalse(path.exists(), path)

    def test_retired_compatibility_facades_are_absent(self) -> None:
        retired = [
            ROOT / "mwm" / "benchmark" / "artifacts.py",
            ROOT / "mwm" / "data" / "stable_wm.py",
            ROOT / "mwm" / "checkpoints.py",
        ]
        split_modules = [
            ROOT / "mwm" / "io.py",
            ROOT / "mwm" / "checkpoint_io.py",
            ROOT / "mwm" / "checkpoint_contract.py",
            ROOT / "mwm" / "data" / "metadata.py",
            ROOT / "mwm" / "data" / "sampling.py",
            ROOT / "mwm" / "benchmark" / "io.py",
            ROOT / "mwm" / "benchmark" / "summary.py",
            ROOT / "mwm" / "benchmark" / "html.py",
            ROOT / "mwm" / "benchmark" / "plots.py",
        ]
        for path in retired:
            self.assertFalse(path.exists(), path)
        for path in split_modules:
            self.assertTrue(path.is_file(), path)

    def test_docs_describe_local_and_slurm_workflows_separately(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        review = (ROOT / "REVIEW_GUIDE.md").read_text(encoding="utf-8")
        for text in (readme, review):
            self.assertIn("Local Desktop Workflow", text)
            self.assertIn("Slurm", text)
            self.assertIn("scripts/local/local_verify.sh", text)
            self.assertIn("scripts/local/local_benchmark_smoke.sh", text)

    def test_scripts_are_grouped_by_workflow_type(self) -> None:
        scripts_root = ROOT / "scripts"
        self.assertFalse((ROOT / "run_cem_sweep.py").exists())
        self.assertTrue((scripts_root / "research" / "run_cem_sweep.py").is_file())
        self.assertEqual(
            sorted(path.name for path in scripts_root.iterdir() if path.is_file()),
            ["README.md"],
        )
        for folder in ("local", "slurm", "research"):
            self.assertTrue((scripts_root / folder).is_dir(), folder)

        self.assertEqual(
            sorted(path.name for path in (scripts_root / "local").glob("*.sh")),
            [
                "local_benchmark_smoke.sh",
                "local_ogb_cube_train_smoke.sh",
                "local_reacher_train_smoke.sh",
                "local_train_smoke.sh",
                "local_verify.sh",
            ],
        )
        self.assertGreater(len(list((scripts_root / "slurm").glob("slurm_mwm*.sbatch"))), 0)
        self.assertGreater(len(list((scripts_root / "slurm").glob("run_mwm*.sh"))), 0)
        self.assertGreater(len(list((scripts_root / "research").glob("research_*"))), 0)
        tracked_scripts = subprocess.check_output(
            ["git", "ls-files", "scripts"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        tracked_pycache = [path for path in tracked_scripts if "__pycache__" in Path(path).parts]
        self.assertEqual(tracked_pycache, [])

    def test_active_docs_scripts_tests_and_reports_use_package_module_commands(self) -> None:
        active_paths = [
            ROOT / "README.md",
            ROOT / "REVIEW_GUIDE.md",
            ROOT / "scripts" / "README.md",
        ]
        hits: list[str] = []
        for path in active_paths:
            text = path.read_text(encoding="utf-8")
            for match in OLD_FLAT_SCRIPT_REF_RE.finditer(text):
                hits.append(f"{path.relative_to(ROOT)} references {match.group(0)}")
        for path in _active_command_reference_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for command in ROOT_CLI_COMMANDS:
                if command in text:
                    hits.append(f"{path.relative_to(ROOT)} references {command}")
            for match in OLD_ROOT_SCRIPT_PATH_RE.finditer(text):
                reference = match.group(0).strip().strip("'\"`")
                hits.append(f"{path.relative_to(ROOT)} references old script path {reference}")
        self.assertEqual(hits, [])

        docs_text = "\n".join(
            (ROOT / rel).read_text(encoding="utf-8")
            for rel in ("README.md", "REVIEW_GUIDE.md", "scripts/README.md")
        )
        for command in PACKAGE_CLI_COMMANDS:
            self.assertIn(command, docs_text)

    def test_package_module_clis_show_help(self) -> None:
        modules = {
            "mwm.data.collection": ("config", "--set"),
            "mwm.upstream.lewm_checkpoints": ("config",),
            "mwm.upstream.lewm_data": ("--source-h5",),
            "mwm.training.stable_wm": ("config", "--set"),
            "mwm.eval.runner": ("config", "--set"),
            "mwm.benchmark.matrix": ("config", "--roles"),
            "mwm.data.verify": ("--paper-parity",),
            "mwm.benchmark.verify": ("--static-only", "--roles"),
            "mwm.benchmark.render_review": ("output_dir",),
        }
        for module, expected in modules.items():
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-S", "-m", module, "--help"],
                    cwd=ROOT,
                    env={"PYTHONDONTWRITEBYTECODE": "1"},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                    check=False,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                for token in expected:
                    self.assertIn(token, result.stdout)

    def test_active_file_review_does_not_document_deleted_root_clis(self) -> None:
        text = (ROOT / "LIBRARY_FILE_REVIEW.md").read_text(encoding="utf-8")
        hits = [command for command in ROOT_CLI_COMMANDS if command in text]
        self.assertEqual(hits, [])

    def test_slurm_poll_scripts_honor_configured_python(self) -> None:
        for script in sorted((ROOT / "scripts" / "slurm").glob("poll_mwm*_jobs.sh")):
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                self.assertIn("MWM_PYTHON", text)
                self.assertIn('PY="${MWM_PYTHON:-', text)

    def test_upstream_data_prep_keeps_runtime_lance_but_allows_cube_hdf5_conversion(self) -> None:
        text = (ROOT / "mwm" / "upstream" / "lewm_data.py").read_text(encoding="utf-8", errors="ignore").lower()
        self.assertIn("convert_ogb_cube_hdf5_to_lance", text)
        forbidden = ["tar.zst", "zstd", "stable_worldmodel.data.convert", "source_format", "dest_format"]
        for token in forbidden:
            self.assertNotIn(token, text)

    def test_removed_legacy_runtime_paths_stay_absent(self) -> None:
        removed = [
            "datasets/swm_hdf5.py",
            "mwm/adapters/lewm_common.py",
            "mwm/adapters/lewm_import.py",
            "mwm/adapters/lewm_model.py",
            "mwm/adapters/lewm_stable.py",
            "mwm/training.py",
        ]
        for rel in removed:
            self.assertFalse((ROOT / rel).exists(), rel)

        runtime_files = [
            ROOT / "mwm" / "adapters" / "lewm.py",
            ROOT / "mwm" / "upstream" / "lewm_checkpoints.py",
            ROOT / "mwm" / "training" / "stable_wm.py",
        ]
        self.assertFalse((ROOT / "mwm" / "models" / "world_model.py").exists())
        forbidden = (
            "source_model",
            "delegated_source_cost",
            "ImportedLeWMMWMWorldModel",
            "build_mwm_lewm_from_object",
            "constructor_identity_base_lewm",
        )
        for path in runtime_files:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, path)

    def test_lewm_adapter_does_not_export_base_specific_builder_facades(self) -> None:
        text = (ROOT / "mwm" / "adapters" / "lewm.py").read_text(encoding="utf-8")

        self.assertNotIn("def build_mwm_lewm_from_stable_config", text)
        self.assertNotIn("def build_mwm_lewm_from_upstream_object", text)
        self.assertEqual(text.count("encoder = _instantiate_module(source_config[\"encoder\"])"), 1)


if __name__ == "__main__":
    unittest.main()
