from __future__ import annotations

import ast
import importlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MWMMigrationHygieneTests(unittest.TestCase):
    def test_canonical_utility_modules_exist(self) -> None:
        for rel in (
            "mwm/io.py",
            "mwm/imports.py",
            "mwm/data/paths.py",
            "mwm/adapters/constants.py",
        ):
            with self.subTest(path=rel):
                self.assertTrue((ROOT / rel).is_file(), rel)

    def test_model_and_training_code_is_split_by_responsibility(self) -> None:
        for rel in (
            "mwm/models/core.py",
            "mwm/models/transitions.py",
            "mwm/models/base_adaptive.py",
            "mwm/models/objectives.py",
            "mwm/models/planning_costs.py",
            "mwm/training/lewm_config.py",
            "mwm/training/lewm_data.py",
            "mwm/training/lewm_model.py",
            "mwm/training/lewm_runtime.py",
            "mwm/training/lewm_callbacks.py",
            "mwm/training/lewm_lightning.py",
            "mwm/training/lewm_export.py",
            "mwm/eval/validation.py",
            "mwm/eval/manifest.py",
            "mwm/eval/policy_builder.py",
            "mwm/eval/execution.py",
        ):
            with self.subTest(path=rel):
                self.assertTrue((ROOT / rel).is_file(), rel)
        self.assertLessEqual(len((ROOT / "mwm/models/world_model.py").read_text(encoding="utf-8").splitlines()), 40)

    def test_retired_facade_modules_are_absent(self) -> None:
        for rel in (
            "mwm/benchmark/artifacts.py",
            "mwm/data/stable_wm.py",
            "mwm/checkpoints.py",
        ):
            with self.subTest(path=rel):
                self.assertFalse((ROOT / rel).exists(), rel)

    def test_runtime_code_uses_canonical_imports(self) -> None:
        forbidden = (
            "mwm.benchmark.artifacts",
            "mwm.data.stable_wm",
            "mwm.checkpoints",
        )
        hits: list[str] = []
        for path in sorted((ROOT / "mwm").rglob("*.py")):
            if "benchmark/artifacts.py" in path.as_posix() or "data/stable_wm.py" in path.as_posix():
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)} imports {token}")
        for path in (
            ROOT / "collect_mwm_data.py",
            ROOT / "prepare_upstream_lewm.py",
            ROOT / "prepare_upstream_lewm_data.py",
            ROOT / "render_benchmark_review.py",
        ):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)} imports {token}")
        self.assertEqual(hits, [])

    def test_python_scripts_consume_public_library_boundaries(self) -> None:
        retired_modules = {
            "mwm.benchmark.artifacts",
            "mwm.data.stable_wm",
            "mwm.checkpoints",
        }
        root_cli_modules = {
            "benchmark_mwm",
            "collect_mwm_data",
            "eval_mwm",
            "prepare_upstream_lewm",
            "prepare_upstream_lewm_data",
            "render_benchmark_review",
            "train_mwm",
            "verify_mwm_benchmark",
            "verify_mwm_data",
        }
        hits: list[str] = []
        for path in sorted((ROOT / "scripts").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module in retired_modules:
                            hits.append(f"{path.relative_to(ROOT)} imports retired {module}")
                        if module.split(".")[0] in root_cli_modules:
                            hits.append(f"{path.relative_to(ROOT)} imports root CLI {module}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module in retired_modules:
                        hits.append(f"{path.relative_to(ROOT)} imports retired {module}")
                    if module.split(".")[0] in root_cli_modules:
                        hits.append(f"{path.relative_to(ROOT)} imports root CLI {module}")
                    if module == "mwm" or module.startswith("mwm."):
                        for alias in node.names:
                            if alias.name.startswith("_"):
                                hits.append(f"{path.relative_to(ROOT)} imports private {module}.{alias.name}")
        self.assertEqual(hits, [])

    def test_root_cli_scripts_are_entrypoints_not_import_facades(self) -> None:
        for script in (
            "train_mwm.py",
            "eval_mwm.py",
            "benchmark_mwm.py",
            "verify_mwm_benchmark.py",
            "verify_mwm_data.py",
        ):
            path = ROOT / script
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [
                node
                for node in tree.body
                if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
            ]
            self.assertEqual(imports, [], script)
            self.assertNotIn("sys.modules", path.read_text(encoding="utf-8"), script)

    def test_training_lewm_is_cli_orchestration_not_private_helper_barrel(self) -> None:
        source = (ROOT / "mwm/training/lewm.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        directly_imported = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        exported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, ast.List):
                        exported.update(item.value for item in node.value.elts if isinstance(item, ast.Constant))
        for symbol in (
            "_AllLevelPlateauEarlyStopping",
            "_ZScoreScaler",
            "_as_container",
            "_base_dataset",
            "_build_trainable_model_from_base",
            "_coerce_lightning_devices",
            "_column_normalizer",
            "_dataset_metadata",
            "_lewm_base_adapter_callbacks",
            "_lewm_base_adapter_checkpoint_callback",
            "_lewm_base_adapter_forward",
            "_load_lewm_base_adapter_lightning_state",
            "_load_lewm_base_adapter_train_valid_datasets",
            "_prepare_trainer_root",
            "_resolve_lewm_base_adapter_model_cfg",
            "_resolve_lewm_base_adapter_total_steps",
            "_resolve_lightning_trainer_runtime",
            "_select_lewm_base_adapter_export_checkpoint",
            "_stable_checkpoint_config_path",
        ):
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, directly_imported)
                self.assertNotIn(symbol, exported)
        module = importlib.import_module("mwm.training.lewm")
        for symbol in directly_imported:
            if symbol.startswith("_") and symbol not in {"_main"}:
                self.assertFalse(hasattr(module, symbol), symbol)
        private_module_calls = [
            f"{node.value.id}.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"lewm_data", "lewm_model", "lewm_lightning"}
            and node.attr.startswith("_")
        ]
        self.assertEqual(private_module_calls, [])

    def test_benchmark_analysis_helpers_have_a_public_owner(self) -> None:
        self.assertTrue((ROOT / "mwm/benchmark/analysis.py").is_file())
        for rel in ("mwm/benchmark/html.py", "mwm/benchmark/plots.py"):
            with self.subTest(path=rel):
                source = (ROOT / rel).read_text(encoding="utf-8")
                self.assertNotIn("from mwm.benchmark.summary import _", source)
        summary_source = (ROOT / "mwm/benchmark/summary.py").read_text(encoding="utf-8")
        for helper in ("_float", "_mean", "_role_label", "_env_label", "_paired_rows", "_outcome_rows"):
            with self.subTest(helper=helper):
                self.assertNotIn(f"def {helper}", summary_source)

    def test_data_loading_is_not_a_transform_module_concern(self) -> None:
        loading_source = (ROOT / "mwm/data/loading.py").read_text(encoding="utf-8")
        self.assertIn("def load_stable_wm_dataset_for_mwm", loading_source)
        transforms_source = (ROOT / "mwm/data/transforms.py").read_text(encoding="utf-8")
        self.assertNotIn("def load_stable_wm_dataset_for_mwm", transforms_source)
        self.assertNotIn("_ZScoreScaler", transforms_source)
        self.assertNotIn("_column_normalizer", transforms_source)

    def test_swm_envs_do_not_expose_noop_restore_wrapper_hook(self) -> None:
        source = (ROOT / "mwm/swm/envs.py").read_text(encoding="utf-8")
        self.assertNotIn("swm_extra_wrappers_for_env", source)

    def test_benchmark_config_spec_is_not_owned_by_runner(self) -> None:
        self.assertTrue((ROOT / "mwm/benchmark/config.py").is_file())
        matrix_source = (ROOT / "mwm/benchmark/matrix.py").read_text(encoding="utf-8")
        verify_source = (ROOT / "mwm/benchmark/verify.py").read_text(encoding="utf-8")
        for helper in (
            "_filter_resolved_by_roles",
            "_load_manifest_config",
            "_merged_run_config",
            "_role",
            "_validate_benchmark_matrix",
        ):
            with self.subTest(helper=helper):
                self.assertNotIn(f"def {helper}", matrix_source)
                self.assertNotIn(f"from mwm.benchmark.matrix import {helper}", verify_source)
        self.assertNotIn("from mwm.benchmark.matrix import (", verify_source)

    def test_eval_runner_delegates_validation_policy_and_execution(self) -> None:
        runner_source = (ROOT / "mwm/eval/runner.py").read_text(encoding="utf-8")
        for helper in (
            "_validate_dataset_metadata",
            "_validate_manifest",
            "_build_mwm_policy",
            "_run_batch",
            "_combine_swm_results",
            "_combine_mwm_diagnostics",
        ):
            with self.subTest(helper=helper):
                self.assertNotIn(f"def {helper}", runner_source)
        self.assertLessEqual(len(runner_source.splitlines()), 260)

    def test_artifact_tests_use_public_benchmark_verifier_api(self) -> None:
        source = (ROOT / "tests/test_mwm_artifacts.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        private_imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "mwm.benchmark.verify"
            for alias in node.names
            if alias.name.startswith("_")
        ]
        self.assertEqual(private_imports, [])

    def test_world_model_docs_describe_compatibility_facade(self) -> None:
        docs = (ROOT / "docs/mwm_adapter_contract.md").read_text(encoding="utf-8")
        self.assertIn("mwm.models.world_model", docs)
        self.assertIn("compatibility facade", docs)


if __name__ == "__main__":
    unittest.main()
