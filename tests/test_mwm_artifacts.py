from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import benchmark_mwm
from omegaconf import OmegaConf

from benchmark_mwm import DEFAULTS, _merged_run_config, _validate_gate_matrix
from mwm.adapters.lewm import build_mwm_lewm_from_stable_config
from mwm.benchmark.artifacts import eval_summary_row, write_default_plots, write_review_html
from mwm.checkpoints import (
    CHECKPOINT_FORMAT,
    CONFIG_FILENAME,
    LEWM_BASE_ADAPTER_ARCH,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    file_sha256,
    load_world_metadata,
    save_world_checkpoint,
    validate_checkpoint_contract,
)
from mwm.data.manifest import generate_manifest, load_manifest, manifest_file_sha256
from mwm.data.stable_wm import StartGoalPair, write_dataset_metadata
from verify_mwm_benchmark import (
    _validate_checkpoint_metadata,
    _validate_paper_targets,
    _validate_role_checkpoint_contract,
    validate_paper_targets,
    validate_single_level_matches,
    verify_benchmark_static,
)
from verify_mwm_data import verify_data_configs


def _payload(role: str, env_id: str, seed: int, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_id": env_id,
        "checkpoint_epoch": 0,
        "checkpoint_run_dir": "checkpoints_mwm/example",
        "config": {"sha256": f"cfg-{role}-{seed}"},
        "manifest": {"manifest_sha256": f"manifest-{env_id}-{seed}", "sha256": f"file-{role}-{seed}"},
        "episodes": 6,
        "goal_offset": 25,
        "swm_results": {"success_rate": float(seed * 10)},
        "planning_diagnostics": {
            "plans": 2,
            "steps": 10,
            "bits_used_total": 1000 + seed,
            "plan_time_total_sec": 0.5,
            "schedule_level_counts": {"0": 1, "1": 1},
        },
        "schedule": {"policy": "linear_cem"},
        "role": role,
        "seed": seed,
        "wall_time_sec": 1.0 + seed,
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _lewm_stable_config() -> dict:
    return {
        "_target_": "stable_worldmodel.wm.lewm.LeWM",
        "encoder": {"_target_": "tests.test_mwm_core.FakeLeWMEncoder", "out_dim": 4},
        "predictor": {
            "_target_": "tests.test_mwm_core.FakeLeWMPredictor",
            "input_dim": 4,
            "hidden_dim": 4,
            "output_dim": 4,
        },
        "action_encoder": {"_target_": "tests.test_mwm_core.FakeLeWMActionEncoder", "action_dim": 2, "out_dim": 4},
        "projector": {"_target_": "torch.nn.Identity"},
        "pred_proj": {"_target_": "torch.nn.Identity"},
    }


class MWMArtifactTests(unittest.TestCase):
    def test_base_adaptive_checkpoint_metadata_persisted_from_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = build_mwm_lewm_from_stable_config(
                source_config=_lewm_stable_config(),
                source_config_sha256="abc",
                training_recipe={"history_size": 2, "num_preds": 1, "loss_scope": {"regularizers": "shared_latent"}},
                K=(4,),
                action_dim=2,
                action_block=1,
                image_shape=(8, 8),
                normalize_imagenet=False,
            )
            out_dir = Path(tmp) / "checkpoint"

            save_world_checkpoint(model, out_dir, metadata={"env_id": "swm/PushT-v1"})

            metadata = load_world_metadata(out_dir)
            self.assertEqual(metadata["adapter_family"], "lewm")
            self.assertEqual(metadata["source_config_sha256"], "abc")
            self.assertTrue(metadata["fresh_init"])
            self.assertEqual(metadata["component_policy"]["shared"], ["latent_producer"])
            self.assertEqual(metadata["loss_scope"]["regularizers"], "shared_latent")
            self.assertIn("training_recipe", metadata)

    def test_base_adaptive_checkpoint_contract_requires_metadata(self) -> None:
        config = {
            "target": "mwm.adapters.lewm.build_mwm_lewm_from_stable_config",
            "kwargs": {
                "source_config_sha256": "abc",
                "component_policy": {
                    "shared": ["latent_producer"],
                    "per_level": ["transition"],
                    "reconstructor": [],
                },
            },
        }
        valid_metadata = {
            "adapter_family": "lewm",
            "source_config_sha256": "abc",
            "fresh_init": True,
            "component_policy": {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": []},
        }

        invalid_cases = [
            ("adapter_family", {**valid_metadata, "adapter_family": "other"}),
            ("fresh_init", {**valid_metadata, "fresh_init": False}),
            ("source_config_sha256", {k: v for k, v in valid_metadata.items() if k != "source_config_sha256"}),
            ("source_config_sha256", {**valid_metadata, "source_config_sha256": "wrong"}),
            ("component_policy", {k: v for k, v in valid_metadata.items() if k != "component_policy"}),
            ("component_policy", {**valid_metadata, "component_policy": "not-a-policy"}),
            ("missing=.*per_level", {**valid_metadata, "component_policy": {"shared": ["latent_producer"], "reconstructor": []}}),
            (
                "unknown=.*extra",
                {
                    **valid_metadata,
                    "component_policy": {
                        "shared": ["latent_producer"],
                        "per_level": ["transition"],
                        "reconstructor": [],
                        "extra": [],
                    },
                },
            ),
            (
                "shared latent producer",
                {**valid_metadata, "component_policy": {"shared": [], "per_level": ["transition"], "reconstructor": []}},
            ),
        ]
        for expected, metadata in invalid_cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    validate_checkpoint_contract(config, metadata)

    def test_manifest_logical_hash_is_separate_from_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_manifest(
                env_id="swm/PushT-v1",
                dataset_path="data/pusht_swm.lance",
                pairs=[StartGoalPair(episode=0, start_step=1, goal_step=3, start_row=1, goal_row=3)],
                goal_offset=2,
                eval_budget=50,
                seed=0,
                restore_spec="pusht_state_goal_state",
                dataset_metadata={"format": "swm_lance"},
                dependency_shas={"local_repo": {"sha256": "abc"}},
            )
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            loaded = load_manifest(path)
            self.assertEqual(loaded["manifest_sha256"], manifest["manifest_sha256"])
            self.assertNotEqual(manifest_file_sha256(path), manifest["manifest_sha256"])

    def test_duplicate_benchmark_cell_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "eval.yaml"
            cfg_path.write_text(
                """
env_id: swm/PushT-v1
checkpoint: {run_dir: checkpoints_mwm/example, epoch: null}
data: {path: data/pusht_swm.lance, format: lance}
eval: {seed: 0}
planner: {scheduler: {policy: fixed, level: finest, rollout_level: base}}
""",
                encoding="utf-8",
            )
            cfg = OmegaConf.merge(
                DEFAULTS,
                {
                    "gate": {"enabled": True, "env_ids": ["swm/PushT-v1"], "seeds": [0], "roles": ["mwm_scheduled"]},
                    "runs": [
                        {"name": "a", "role": "mwm_scheduled", "config": str(cfg_path)},
                        {"name": "b", "role": "mwm_scheduled", "config": str(cfg_path)},
                    ],
                },
            )
            resolved = [(run, _merged_run_config(run)[1]) for run in cfg.runs]
            with self.assertRaisesRegex(ValueError, "duplicate cells"):
                _validate_gate_matrix(cfg, resolved)

    def test_benchmark_verifier_static_only_accepts_paper_parity_config(self) -> None:
        report = verify_benchmark_static("configs/benchmark_mwm_paper_parity.yaml")

        self.assertEqual(report["runs"], 4)
        self.assertEqual(report["paper_targets"]["tolerance_pp"], 1.0)
        self.assertEqual(report["paper_targets"]["single_level_tolerance_pp"], 5.0)

        reference_report = verify_benchmark_static("configs/benchmark_mwm_paper_reference.yaml")
        self.assertEqual(reference_report["runs"], 4)
        self.assertIn(("swm/PushT-v1", 42, "stable_wm_reference"), reference_report["expected_cells"])

    def test_benchmark_role_filter_runs_upstream_gate_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_cfg = root / "eval.yaml"
            eval_cfg.write_text(
                """
env_id: swm/PushT-v1
checkpoint: {run_dir: checkpoints_mwm/example, epoch: null}
data: {path: data/pusht_swm.lance, format: lance}
eval: {seed: 0}
planner: {scheduler: {policy: fixed, level: finest, rollout_level: base}}
""",
                encoding="utf-8",
            )
            bench_cfg = root / "benchmark.yaml"
            bench_cfg.write_text(
                f"""
output_dir: {root / "out"}
require_shared_manifests: false
gate: {{enabled: true, env_ids: [swm/PushT-v1], seeds: [0], roles: [upstream_lewm_converted, retrained_lewm_single]}}
runs:
  - name: upstream
    role: upstream_lewm_converted
    config: {eval_cfg}
    overrides:
      checkpoint:
        run_dir: checkpoints_mwm/upstream
  - name: retrained
    role: retrained_lewm_single
    config: {eval_cfg}
    overrides:
      checkpoint:
        run_dir: checkpoints_mwm/retrained
""",
                encoding="utf-8",
            )

            calls: list[str] = []
            old_run_eval = benchmark_mwm.run_eval_mwm

            def _fake_run(cfg_path: str) -> None:
                cfg = OmegaConf.load(cfg_path)
                calls.append(str(cfg.checkpoint.run_dir))
                _payload("stub", str(cfg.env_id), int(cfg.eval.seed), Path(str(cfg.eval.output_path)))

            benchmark_mwm.run_eval_mwm = _fake_run
            try:
                benchmark_mwm.main(str(bench_cfg), roles=["upstream_lewm_converted"])
            finally:
                benchmark_mwm.run_eval_mwm = old_run_eval

            summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(calls, ["checkpoints_mwm/upstream"])
            self.assertEqual([row["role"] for row in summary["runs"]], ["upstream_lewm_converted"])
            static_report = verify_benchmark_static(str(bench_cfg), roles=["upstream_lewm_converted"])
            self.assertEqual(static_report["expected_cells"], [("swm/PushT-v1", 0, "upstream_lewm_converted")])

    def test_benchmark_failure_writes_traceback_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_cfg = root / "eval.yaml"
            eval_cfg.write_text(
                """
env_id: swm/PushT-v1
checkpoint: {run_dir: checkpoints_mwm/example, epoch: null}
data: {path: data/pusht_swm.lance, format: lance}
eval: {seed: 0}
planner: {scheduler: {policy: fixed, level: finest, rollout_level: base}}
""",
                encoding="utf-8",
            )
            bench_cfg = root / "benchmark.yaml"
            bench_cfg.write_text(
                f"""
output_dir: {root / "out"}
require_shared_manifests: false
gate: {{enabled: false}}
runs:
  - {{name: broken, role: mwm_scheduled, config: {eval_cfg}}}
""",
                encoding="utf-8",
            )

            old_run_eval = benchmark_mwm.run_eval_mwm

            def _fail(_: str) -> None:
                raise ValueError("synthetic benchmark failure")

            benchmark_mwm.run_eval_mwm = _fail
            try:
                with self.assertRaisesRegex(RuntimeError, "run.log"):
                    benchmark_mwm.main(str(bench_cfg))
            finally:
                benchmark_mwm.run_eval_mwm = old_run_eval

            log_text = (root / "out" / "000_broken" / "run.log").read_text(encoding="utf-8")
            self.assertIn("Traceback", log_text)
            self.assertIn("ValueError: synthetic benchmark failure", log_text)

    def test_checkpoint_verifier_requires_action_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / CONFIG_FILENAME
            weights_path = root / WEIGHTS_FILENAME
            metadata_path = root / METADATA_FILENAME
            cfg_path.write_text(
                json.dumps(
                    {
                        "target": "mwm.adapters.lewm.build_mwm_lewm_from_stable_config",
                        "kwargs": {"action_dim": 2, "action_block": 1, "K": [2]},
                    }
                ),
                encoding="utf-8",
            )
            weights_path.write_bytes(b"weights")
            metadata_path.write_text(
                json.dumps(
                    {
                        "format": CHECKPOINT_FORMAT,
                        "action_dim": 2,
                        "action_block": 1,
                        "levels": [2],
                        "artifacts": {
                            "config": {"path": CONFIG_FILENAME, "sha256": file_sha256(cfg_path)},
                            "weights": {"path": WEIGHTS_FILENAME, "sha256": file_sha256(weights_path)},
                        },
                    }
                ),
                encoding="utf-8",
            )

            errors: list[str] = []
            _validate_checkpoint_metadata(root, errors)
            self.assertTrue(any("missing action_spec" in error for error in errors), errors)

    def test_paper_target_verifier_checks_upstream_and_retrained_match(self) -> None:
        cfg = OmegaConf.create(
            {
                "paper_targets": {
                    "enabled": True,
                    "upstream_tolerance_pp": 5.0,
                    "retrained_match_tolerance_pp": 5.0,
                    "success_rate": {"swm/PushT-v1": 96.0},
                }
            }
        )
        rows = [
            {"env_id": "swm/PushT-v1", "role": "upstream_lewm_converted", "success_rate": 96.0},
            {"env_id": "swm/PushT-v1", "role": "retrained_lewm_single", "success_rate": 94.0},
        ]

        errors: list[str] = []
        _validate_paper_targets(cfg, rows, errors)
        self.assertEqual(errors, [])

        rows[1]["success_rate"] = 70.0
        _validate_paper_targets(cfg, rows, errors)
        self.assertTrue(any("single-level match check failed" in error for error in errors), errors)

    def test_paper_target_upstream_only_gate_skips_retrained_match(self) -> None:
        cfg = OmegaConf.create(
            {
                "gate": {"roles": ["upstream_lewm_converted"]},
                "paper_targets": {
                    "enabled": True,
                    "tolerance_pp": 1.0,
                    "single_level_tolerance_pp": 5.0,
                    "success_rate": {"swm/PushT-v1": 96.0},
                },
            }
        )
        rows = [{"env_id": "swm/PushT-v1", "role": "upstream_lewm_converted", "success_rate": 96.0}]

        errors: list[str] = []
        _validate_paper_targets(cfg, rows, errors)

        self.assertEqual(errors, [])

    def test_paper_target_gate_requires_reference_when_mwm_misses_by_more_than_one_point(self) -> None:
        rows = [
            {"env_id": "swm/PushT-v1", "role": "upstream_lewm_converted", "success_rate": 94.0},
            {"env_id": "swm/PushT-v1", "role": "stable_wm_reference", "success_rate": 96.0},
            {"env_id": "swm/PushT-v1", "role": "retrained_lewm_single", "success_rate": 95.0},
        ]
        cfg = {
            "paper_targets": {
                "enabled": True,
                "tolerance_pp": 1.0,
                "single_level_tolerance_pp": 5.0,
                "success_rate": {"swm/PushT-v1": 96.0},
            }
        }

        errors = validate_paper_targets(rows, cfg)

        self.assertTrue(any("MWM evaluator" in error for error in errors), errors)

    def test_single_level_match_gate_is_independent_from_paper_target(self) -> None:
        cfg = {
            "paper_targets": {
                "enabled": True,
                "tolerance_pp": 1.0,
                "single_level_tolerance_pp": 5.0,
                "success_rate": {"swm/PushT-v1": 96.0, "swm/TwoRoom-v1": 87.0},
            }
        }
        rows = [
            {"env_id": "swm/PushT-v1", "role": "upstream_lewm_converted", "success_rate": 92.0},
            {"env_id": "swm/PushT-v1", "role": "retrained_lewm_single", "success_rate": 96.0},
            {"env_id": "swm/TwoRoom-v1", "role": "upstream_lewm_converted", "success_rate": 86.0},
            {"env_id": "swm/TwoRoom-v1", "role": "retrained_lewm_single", "success_rate": 84.0},
        ]

        self.assertEqual(validate_single_level_matches(rows, cfg), [])

        rows[1]["success_rate"] = 80.0
        errors = validate_single_level_matches(rows, cfg)
        self.assertTrue(any("single-level match check failed" in error for error in errors), errors)

    def test_single_level_match_gate_requires_both_roles(self) -> None:
        cfg = {
            "paper_targets": {
                "enabled": True,
                "single_level_tolerance_pp": 5.0,
                "success_rate": {"swm/PushT-v1": 96.0},
            }
        }
        rows = [{"env_id": "swm/PushT-v1", "role": "upstream_lewm_converted", "success_rate": 92.0}]

        errors = validate_single_level_matches(rows, cfg)

        self.assertTrue(any("missing retrained_lewm_single rows" in error for error in errors), errors)

    def test_role_checkpoint_contract_rejects_direct_target_and_wrong_backend(self) -> None:
        row = {"role": "retrained_lewm_single", "checkpoint_run_dir": "checkpoints_mwm/retrained"}
        metadata = {
            "levels": [192],
            "training_backend": "stable_pretraining",
            "model": {"target": "mwm.adapters.lewm.build_mwm_lewm"},
        }

        errors: list[str] = []
        _validate_role_checkpoint_contract(row, metadata, errors)

        self.assertTrue(any("Le-WM base-adapter backend" in error for error in errors), errors)
        self.assertTrue(any("base-adapter target" in error for error in errors), errors)
        self.assertTrue(any("corrected architecture version" in error for error in errors), errors)

    def test_role_checkpoint_contract_accepts_base_adaptive_lewm_target(self) -> None:
        rows = [
            {"role": "upstream_lewm_converted", "checkpoint_run_dir": "checkpoints_mwm/upstream"},
            {"role": "retrained_lewm_single", "checkpoint_run_dir": "checkpoints_mwm/retrained"},
            {"role": "mwm_scheduled", "checkpoint_run_dir": "checkpoints_mwm/scheduled"},
        ]
        metadatas = [
            {
                "role": "upstream_lewm_converted",
                "levels": [192],
                "architecture_version": LEWM_BASE_ADAPTER_ARCH,
                "model": {"target": "mwm.adapters.lewm.build_mwm_lewm_from_upstream_object"},
            },
            {
                "levels": [192],
                "training_backend": "stable_worldmodel_lewm",
                "architecture_version": LEWM_BASE_ADAPTER_ARCH,
                "model": {"target": "mwm.adapters.lewm.build_mwm_lewm_from_stable_config"},
            },
            {
                "levels": [48, 96, 144],
                "training_backend": "stable_worldmodel_lewm",
                "architecture_version": LEWM_BASE_ADAPTER_ARCH,
                "model": {"target": "mwm.adapters.lewm.build_mwm_lewm_from_stable_config"},
            },
        ]

        errors: list[str] = []
        for row, metadata in zip(rows, metadatas):
            _validate_role_checkpoint_contract(row, metadata, errors)

        self.assertEqual(errors, [])

    def test_lance_dataset_verifier_rejects_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "train.yaml"
            cfg_path.write_text("data: {path: missing.lance, format: lance}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing Lance dataset"):
                verify_data_configs([cfg_path])

    def test_lance_dataset_verifier_accepts_sidecar_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "tiny.lance"
            dataset.mkdir()
            write_dataset_metadata(
                dataset,
                {
                    "format": "swm_lance",
                    "env_id": "swm/PushT-v1",
                    "restore_spec": "pusht_state_goal_state",
                    "image_shape": [8, 8],
                    "action_dim": 2,
                    "action_low": [-1.0, -1.0],
                    "action_high": [1.0, 1.0],
                    "dataset": {"pixels_key": "pixels", "action_key": "action"},
                },
            )
            cfg_path = root / "train.yaml"
            cfg_path.write_text(f"data: {{path: {dataset}, format: lance}}\n", encoding="utf-8")
            report = verify_data_configs([cfg_path])
            self.assertEqual(report["count"], 1)

    def test_dataset_verifier_rejects_hdf5_runtime_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "tiny.h5"
            dataset.write_bytes(b"placeholder")
            cfg_path = root / "eval.yaml"
            cfg_path.write_text(f"data: {{path: {dataset}, format: hdf5}}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires format lance"):
                verify_data_configs([cfg_path])

    def test_review_html_has_required_plots_and_drilldowns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roles = ["upstream_lewm_converted", "retrained_lewm_single", "mwm_scheduled"]
            rows = []
            outputs = []
            for seed in range(3):
                for role in roles:
                    output_path = root / f"run_{seed}_{role}" / "eval.json"
                    payload = _payload(role, "swm/PushT-v1", seed, output_path)
                    rows.append(eval_summary_row(f"run_{seed}_{role}", output_path, payload))
                    outputs.append(payload)

            plots = write_default_plots(root / "plots", rows)
            names = {Path(plot).name for plot in plots}
            self.assertTrue(
                {
                    "efficiency_ratios.png",
                    "paired_success_delta.png",
                    "schedule_usage_by_role.png",
                    "success_vs_compute.png",
                    "success_by_env_role.png",
                    "success_vs_wall_time.png",
                    "schedule_level_usage.png",
                }.issubset(names)
            )

            html_path = root / "review.html"
            write_review_html(html_path, "MWM Review", rows, outputs, plots=plots, expected_cells=9)
            text = html_path.read_text(encoding="utf-8")
            for token in ("Gate Status", "Plots", "Paired Seed Comparison", "Run Drilldown"):
                self.assertIn(token, text)
            for row in rows:
                self.assertIn(Path(row["output_json"]).parent.name, text)


if __name__ == "__main__":
    unittest.main()
