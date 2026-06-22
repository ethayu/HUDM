from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest

import yaml

from mwm.swm.restore import eval_callables_for_env


ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLDMODEL_VERSION = "0.1.0"
OLD_FLAT_SCRIPT_REF_RE = re.compile(
    r"scripts/(?:local_|run_mwm_|submit_mwm_|poll_mwm_|slurm_mwm_|slurm_research_|research_)"
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
        ".git",
        ".pytest_cache",
        ".worktrees",
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

    def test_lewm_training_configs_follow_base_adaptive_contract(self) -> None:
        expected_levels = {
            "mwm_lewm_pusht.yaml": [192],
            "mwm_lewm_tworoom.yaml": [192],
            "mwm_lewm_pusht_upstream.yaml": [192],
            "mwm_lewm_tworoom_upstream.yaml": [192],
            "mwm_scheduled_pusht.yaml": [48, 96, 144],
            "mwm_scheduled_tworoom.yaml": [48, 96, 144],
            "mwm_dense_pusht.yaml": [6, 12, 48, 96, 144, 192],
            "mwm_dense_tworoom.yaml": [6, 12, 48, 96, 144, 192],
        }
        for name, levels in expected_levels.items():
            cfg = yaml.safe_load((ROOT / "configs" / "train" / name).read_text(encoding="utf-8"))

            self.assertEqual(cfg["base"]["family"], "lewm", name)
            self.assertIn("checkpoint", cfg["base"], name)
            self.assertEqual(cfg["mwm"]["component_policy"], {
                "shared": ["latent_producer"],
                "per_level": ["transition"],
                "reconstructor": [],
            }, name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["regularizers"], "shared_latent", name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["reconstructor_detach_encoder"], True, name)
            self.assertEqual(cfg["mwm"]["loss_terms"]["reconstructor_contributes_to_encoder_loss"], False, name)
            self.assertEqual(cfg["data"]["format"], "lance", name)
            self.assertTrue(str(cfg["data"]["path"]).endswith(".lance"), name)
            self.assertEqual(cfg["model"]["D"], 192, name)
            self.assertEqual(cfg["model"]["K"], levels, name)
            self.assertEqual(cfg["model"]["action_block"], 5, name)
            self.assertEqual(cfg["train"]["backend"], "stable_worldmodel_lewm", name)
            self.assertEqual(set(cfg["schedule"]), {"max_epochs"}, name)
            self.assertEqual(cfg["schedule"]["max_epochs"], 10, name)
            if name.startswith(("mwm_scheduled_", "mwm_dense_")):
                self.assertTrue(str(cfg["data"]["path"]).startswith("data/upstream/"), name)

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
        for name in ("paper_pusht.yaml", "paper_tworoom.yaml"):
            cfg = yaml.safe_load((ROOT / "configs" / "eval" / name).read_text(encoding="utf-8"))

            self.assertEqual(cfg["data"]["format"], "lance", name)
            self.assertEqual(cfg["data"]["action_preprocessing"], "standard_scaler", name)
            self.assertEqual(cfg["eval"]["sampling"], "stable_worldmodel", name)
            self.assertEqual(cfg["eval"]["goal_offset"], 25, name)
            self.assertEqual(cfg["planner"]["action_block"], 5, name)
            self.assertEqual(cfg["planner"]["scheduler"]["policy"], "fixed", name)
            self.assertEqual(cfg["planner"]["scheduler"]["rollout_level"]["level"], "base", name)

        bench_cfg = yaml.safe_load((ROOT / "configs" / "benchmark" / "paper_parity_pusht.yaml").read_text(encoding="utf-8"))
        self.assertEqual(bench_cfg["env_id"], "swm/PushT-v1")
        self.assertEqual([run["role"] for run in bench_cfg["runs"]], ["upstream_lewm_converted", "retrained_lewm_identity"])
        self.assertEqual(bench_cfg["paper_targets"]["tolerance_pp"], 1.0)

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
        for name in ("mwm_lewm_tworoom.yaml",):
            cfg = yaml.safe_load((ROOT / "configs" / "train" / name).read_text(encoding="utf-8"))

            self.assertEqual(cfg["data"]["path"], "data/tworoom_swm.lance")
            self.assertEqual(cfg["data"]["keys_to_load"], ["pixels", "action", "proprio"])
            self.assertEqual(cfg["data"]["keys_to_cache"], ["action", "proprio"])

    def test_paper_scheduled_tworoom_train_config_matches_upstream_lance_columns(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs" / "train" / "mwm_scheduled_tworoom.yaml").read_text(encoding="utf-8"))

        self.assertEqual(cfg["data"]["path"], "data/upstream/tworoom.lance")
        self.assertEqual(cfg["data"]["keys_to_load"], ["pixels", "action", "proprio"])
        self.assertEqual(cfg["data"]["keys_to_cache"], ["action", "proprio"])

    def test_gpu_runner_scripts_require_slurm_allocation(self) -> None:
        work_tokens = (
            '"$PY"',
            "train_mwm.py",
            "benchmark_mwm.py",
            "verify_mwm_",
            "prepare_upstream_",
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
                if "benchmark_mwm.py" in stripped or "verify_mwm_benchmark.py" in stripped:
                    self.assertTrue(stripped.startswith("run_step "), f"{script}: {stripped}")

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

    def test_desktop_entrypoints_accept_set_overrides(self) -> None:
        entrypoints = [
            ROOT / "collect_mwm_data.py",
            ROOT / "train_mwm.py",
            ROOT / "eval_mwm.py",
            ROOT / "benchmark_mwm.py",
        ]
        for path in entrypoints:
            text = path.read_text(encoding="utf-8")
            self.assertIn("load_config", text, path)
            self.assertIn('"--set"', text, path)

    def test_top_level_mwm_clis_are_plain_entrypoints(self) -> None:
        migrated = {
            "train_mwm.py": "mwm.training.lewm",
            "eval_mwm.py": "mwm.eval.runner",
            "benchmark_mwm.py": "mwm.benchmark.matrix",
            "verify_mwm_benchmark.py": "mwm.benchmark.verify",
            "verify_mwm_data.py": "mwm.data.verify",
        }
        for root_name, module in migrated.items():
            root_path = ROOT / root_name
            package_path = ROOT / Path(*module.split(".")).with_suffix(".py")
            with self.subTest(root=root_name):
                self.assertTrue(package_path.is_file(), package_path)
                root_text = root_path.read_text(encoding="utf-8")
                self.assertNotIn("import *", root_text)
                self.assertNotIn("sys.modules", root_text)
                self.assertIn(f"from {module} import", root_text)
                self.assertIn("main", root_text)
                root_tree = ast.parse(root_text)
                root_defs = [
                    node.name
                    for node in root_tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ]
                self.assertEqual(root_defs, [], root_name)

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

    def test_active_docs_and_tests_do_not_reference_flat_script_paths(self) -> None:
        active_paths = [
            ROOT / "README.md",
            ROOT / "REVIEW_GUIDE.md",
            *sorted((ROOT / "tests").glob("test_mwm*.py")),
        ]
        hits: list[str] = []
        for path in active_paths:
            text = path.read_text(encoding="utf-8")
            for match in OLD_FLAT_SCRIPT_REF_RE.finditer(text):
                hits.append(f"{path.relative_to(ROOT)} references {match.group(0)}")
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
            ROOT / "mwm" / "models" / "world_model.py",
            ROOT / "prepare_upstream_lewm.py",
            ROOT / "train_mwm.py",
        ]
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
