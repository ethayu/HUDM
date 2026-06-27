from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys
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

    def test_retired_facade_modules_are_absent(self) -> None:
        for rel in (
            "mwm/models/world_model.py",
            "mwm/benchmark/artifacts.py",
            "mwm/data/stable_wm.py",
            "mwm/checkpoints.py",
        ):
            with self.subTest(path=rel):
                self.assertFalse((ROOT / rel).exists(), rel)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("mwm.models.world_model")

    def test_runtime_code_uses_canonical_imports(self) -> None:
        forbidden = (
            "mwm.models.world_model",
            "mwm.benchmark.artifacts",
            "mwm.data.stable_wm",
            "mwm.checkpoints",
        )
        hits: list[str] = []
        paths = [*sorted((ROOT / "mwm").rglob("*.py")), *sorted(ROOT.glob("*.py")), *sorted((ROOT / "scripts").rglob("*.py"))]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    modules.append(module)
                    modules.extend(f"{module}.{alias.name}" for alias in node.names if module)
                for module in modules:
                    for token in forbidden:
                        if module == token or module.startswith(f"{token}."):
                            hits.append(f"{path.relative_to(ROOT)} imports {module}")
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)} mentions {token}")
        self.assertEqual(hits, [])

    def test_preprocessing_helpers_are_not_reexported_from_eval_policy(self) -> None:
        import mwm.eval.policy as policy

        self.assertNotIn("mwm_image_input_transform", policy.__all__)
        self.assertNotIn("imagenet_image_input_transform", policy.__all__)
        source = (ROOT / "mwm" / "eval" / "policy_builder.py").read_text(encoding="utf-8")
        self.assertIn("from mwm.preprocessing.images import mwm_image_input_transform", source)
        self.assertNotIn("from mwm.eval.policy import MWMWorldModelPolicy, mwm_image_input_transform", source)

    def test_docs_do_not_describe_removed_facades_as_public_api(self) -> None:
        docs = [
            ROOT / "README.md",
            ROOT / "REVIEW_GUIDE.md",
            ROOT / "LIBRARY_FILE_REVIEW.md",
            ROOT / "docs" / "mwm_adapter_contract.md",
        ]
        forbidden = ("mwm.models.world_model", "compatibility facade")
        hits = [
            f"{path.relative_to(ROOT)} mentions {token}"
            for path in docs
            for token in forbidden
            if token in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(hits, [])

    def test_model_package_exports_come_from_canonical_owner_modules(self) -> None:
        import mwm
        import mwm.models as models
        import mwm.models.losses as losses
        from mwm.models.base_adaptive import MatryoshkaWorldModel
        from mwm.models.core import MWMWorldModel
        from mwm.models.transitions import TransitionPackage
        from mwm.preprocessing.images import ImageNetPreprocess

        self.assertEqual(getattr(mwm, "__all__", None), [])
        self.assertEqual(getattr(models, "__all__", None), [])
        for module, name in (
            (mwm, "ImageNetPreprocess"),
            (mwm, "MatryoshkaWorldModel"),
            (mwm, "MWMWorldModel"),
            (mwm, "TransitionPackage"),
            (models, "ImageNetPreprocess"),
            (models, "MatryoshkaWorldModel"),
            (models, "MWMWorldModel"),
            (models, "TransitionPackage"),
        ):
            with self.subTest(module=module.__name__, name=name):
                self.assertFalse(hasattr(module, name), name)
        self.assertIs(ImageNetPreprocess, importlib.import_module("mwm.preprocessing.images").ImageNetPreprocess)
        self.assertIs(MatryoshkaWorldModel, importlib.import_module("mwm.models.base_adaptive").MatryoshkaWorldModel)
        self.assertIs(MWMWorldModel, importlib.import_module("mwm.models.core").MWMWorldModel)
        self.assertIs(TransitionPackage, importlib.import_module("mwm.models.transitions").TransitionPackage)
        for name in ("latent_regularizer_loss", "matryoshka_base_loss", "weighted_level_mean"):
            with self.subTest(loss=name):
                self.assertFalse(hasattr(mwm, name), name)
                self.assertFalse(hasattr(models, name), name)
                self.assertTrue(callable(getattr(losses, name)))

    def test_model_package_import_surface_is_light_and_explicit(self) -> None:
        code = (
            "import importlib.util, sys\n"
            "import mwm\n"
            "import mwm.models\n"
            "print(hasattr(mwm, 'MatryoshkaWorldModel'))\n"
            "print(hasattr(mwm.models, 'MatryoshkaWorldModel'))\n"
            "print(importlib.util.find_spec('mwm.models.world_model') is None)\n"
            "print('torch' in sys.modules)\n"
            "print('mwm.models.base_adaptive' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["False", "False", "True", "False", "False"])

    def test_model_child_imports_do_not_preload_adaptive_model(self) -> None:
        code = (
            "import sys\n"
            "import mwm.models.core\n"
            "import mwm.models.losses\n"
            "import mwm.models.transitions\n"
            "print('mwm.models.base_adaptive' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_python_scripts_consume_public_library_boundaries(self) -> None:
        retired_modules = {
            "mwm.benchmark.artifacts",
            "mwm.data.stable_wm",
            "mwm.checkpoints",
        }
        root_cli_modules = {
            "benchmark_" + "mwm",
            "collect_" + "mwm_data",
            "eval_" + "mwm",
            "prepare_" + "upstream_lewm",
            "prepare_" + "upstream_lewm_data",
            "render_" + "benchmark_review",
            "train_" + "mwm",
            "verify_" + "mwm_benchmark",
            "verify_" + "mwm_data",
        }
        retired_script_helper_modules = {
            "convert_ogb_cube_hdf5_to_lance",
            "convert_reacher_h5_to_lance",
            "research_identity_delta_audit",
            "research_identity_delta_collect",
            "research_reacher_identity_delta_audit",
            "run_cem_sweep",
        }
        hits: list[str] = []
        paths = [
            *sorted((ROOT / "mwm").rglob("*.py")),
            *sorted((ROOT / "scripts").rglob("*.py")),
            *sorted((ROOT / "tests").glob("test_mwm*.py")),
            *sorted(ROOT.glob("*.py")),
        ]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module in retired_modules:
                            hits.append(f"{path.relative_to(ROOT)} imports retired {module}")
                        if module.split(".")[0] in root_cli_modules:
                            hits.append(f"{path.relative_to(ROOT)} imports root CLI {module}")
                        if module.split(".")[0] in retired_script_helper_modules:
                            hits.append(f"{path.relative_to(ROOT)} imports retired script helper {module}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module in retired_modules:
                        hits.append(f"{path.relative_to(ROOT)} imports retired {module}")
                    if module.split(".")[0] in root_cli_modules:
                        hits.append(f"{path.relative_to(ROOT)} imports root CLI {module}")
                    if module.split(".")[0] in retired_script_helper_modules:
                        hits.append(f"{path.relative_to(ROOT)} imports retired script helper {module}")
                    if "scripts" in path.relative_to(ROOT).parts and (module == "mwm" or module.startswith("mwm.")):
                        for alias in node.names:
                            if alias.name.startswith("_"):
                                hits.append(f"{path.relative_to(ROOT)} imports private {module}.{alias.name}")
        self.assertEqual(hits, [])

    def test_root_cli_wrappers_are_absent(self) -> None:
        wrappers = (
            "train_" + "mwm.py",
            "eval_" + "mwm.py",
            "benchmark_" + "mwm.py",
            "verify_" + "mwm_benchmark.py",
            "verify_" + "mwm_data.py",
            "render_" + "benchmark_review.py",
        )
        package_modules = (
            "mwm/training/lewm.py",
            "mwm/eval/runner.py",
            "mwm/benchmark/matrix.py",
            "mwm/benchmark/verify.py",
            "mwm/data/verify.py",
            "mwm/benchmark/render_review.py",
        )
        for rel in wrappers:
            with self.subTest(wrapper=rel):
                self.assertFalse((ROOT / rel).exists(), rel)
        for rel in package_modules:
            with self.subTest(module=rel):
                source = (ROOT / rel).read_text(encoding="utf-8")
                self.assertIn("def main", source, rel)
                self.assertIn("if __name__ == \"__main__\"", source, rel)

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

    def test_eval_device_resolution_has_public_owner(self) -> None:
        module = importlib.import_module("mwm.eval.runner")
        source = (ROOT / "tests/test_mwm_artifacts.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        private_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "mwm.eval.runner"
            and any(alias.name == "_device" for alias in node.names)
        ]

        self.assertTrue(hasattr(module, "resolve_device"))
        self.assertIn("resolve_device", module.__all__)
        self.assertNotIn("_device", module.__all__)
        self.assertEqual(private_imports, [])

    def test_data_verify_public_surface_excludes_private_helpers(self) -> None:
        module = importlib.import_module("mwm.data.verify")
        self.assertEqual(
            set(module.__all__),
            {"DEFAULT_CONFIGS", "PAPER_PARITY_CONFIGS", "main", "verify_data_configs"},
        )
        self.assertFalse(any(name.startswith("_") for name in module.__all__))

    def test_artifact_tests_use_focused_benchmark_verifier_modules(self) -> None:
        source = (ROOT / "tests/test_mwm_artifacts.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        verify_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "mwm.benchmark.verify"
        ]
        focused_modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mwm.benchmark.")
        }
        self.assertEqual(verify_imports, [])
        self.assertIn("mwm.benchmark.plot_contract", focused_modules)
        self.assertIn("mwm.benchmark.paper_targets", focused_modules)
        self.assertIn("mwm.benchmark.checkpoint_verify", focused_modules)
        self.assertIn("mwm.benchmark.output_verify", focused_modules)
        self.assertIn("mwm.benchmark.static_verify", focused_modules)

    def test_benchmark_verify_is_cli_orchestrator_only(self) -> None:
        source = (ROOT / "mwm/benchmark/verify.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defs = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        self.assertLessEqual(defs, {"main"})
        exports = [
            elt.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, ast.List)
            for elt in node.value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        self.assertEqual(exports, ["main"])
        top_level_benchmark_imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mwm.benchmark.")
            for alias in node.names
        ]
        self.assertEqual(top_level_benchmark_imports, [])
        for helper in (
            "required_plots_for_benchmark",
            "validate_paper_targets",
            "load_checkpoint_metadata_for_benchmark",
            "validate_benchmark_role_checkpoint_contract",
            "verify_benchmark_output",
            "verify_benchmark_static",
        ):
            with self.subTest(helper=helper):
                self.assertNotIn(f"def {helper}", source)

    def test_benchmark_verify_import_is_lazy(self) -> None:
        code = (
            "import sys\n"
            "import mwm.benchmark.verify\n"
            "print('mwm.benchmark.static_verify' in sys.modules)\n"
            "print('mwm.benchmark.output_verify' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["False", "False"])

    def test_benchmark_verifier_modules_declare_public_surfaces(self) -> None:
        expected = {
            "mwm/benchmark/checkpoint_verify.py": {
                "load_checkpoint_metadata_for_benchmark",
                "validate_benchmark_role_checkpoint_contract",
            },
            "mwm/benchmark/matrix_identity.py": {
                "expected_cells_from_resolved",
                "load_expected_resolved",
                "metric_identity",
            },
            "mwm/benchmark/output_verify.py": {"verify_benchmark_output"},
            "mwm/benchmark/paper_targets.py": {
                "append_paper_target_errors",
                "normalize_paper_target_config",
                "validate_paper_targets",
            },
            "mwm/benchmark/plot_contract.py": {
                "BASE_REQUIRED_PLOTS",
                "EFFICIENCY_RATIOS_PLOT",
                "PAIRED_SUCCESS_DELTA_PLOT",
                "PAIR_REQUIRED_PLOTS",
                "REQUIRED_PLOTS",
                "SCHEDULE_LEVEL_USAGE_PLOT",
                "SCHEDULE_REQUIRED_PLOTS",
                "SCHEDULE_USAGE_BY_ROLE_PLOT",
                "SUCCESS_BY_ENV_ROLE_PLOT",
                "SUCCESS_VS_COMPUTE_PLOT",
                "SUCCESS_VS_WALL_TIME_PLOT",
                "required_plots_for_benchmark",
            },
            "mwm/benchmark/static_verify.py": {"verify_benchmark_static"},
        }
        for rel, names in expected.items():
            with self.subTest(module=rel):
                source = (ROOT / rel).read_text(encoding="utf-8")
                tree = ast.parse(source)
                exports = [
                    elt.value
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    for target in node.targets
                    if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, ast.List)
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                self.assertEqual(set(exports), names)

    def test_model_docs_describe_canonical_owner_modules(self) -> None:
        docs = (ROOT / "docs/mwm_adapter_contract.md").read_text(encoding="utf-8")
        self.assertNotIn("mwm.models.world_model", docs)
        self.assertNotIn("compatibility facade", docs)
        self.assertIn("mwm.models.base_adaptive.MatryoshkaWorldModel", docs)
        self.assertIn("mwm.models.core.MWMWorldModel", docs)


if __name__ == "__main__":
    unittest.main()
