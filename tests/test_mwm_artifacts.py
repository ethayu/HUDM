from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from mwm.benchmark import matrix as benchmark_mwm
from mwm.benchmark.config import DEFAULTS, merged_run_config, validate_benchmark_matrix
from mwm.adapters.builder import STABLE_CONFIG_TARGET, build_mwm_from_stable_config
from mwm.adapters.constants import LEWM_BASE_ADAPTER_ARCH
from mwm.checkpoint_contract import validate_checkpoint_contract
from mwm.checkpoint_io import (
    CHECKPOINT_FORMAT,
    CONFIG_FILENAME,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    file_sha256,
    load_world_metadata,
    save_world_checkpoint,
    validate_checkpoint_directory,
)
from mwm.data.manifest import generate_manifest, load_manifest, manifest_file_sha256
from mwm.data.metadata import write_dataset_metadata
from mwm.data.sampling import StartGoalPair, sample_start_goal_pairs
from mwm.benchmark.verify import (
    append_paper_target_errors,
    load_checkpoint_metadata_for_benchmark,
    required_plots_for_benchmark,
    validate_benchmark_role_checkpoint_contract,
    validate_paper_targets,
    verify_benchmark_static,
)
from mwm.data.verify import verify_data_configs


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


class FakeLeWMEncoder(nn.Module):
    def __init__(self, out_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(3, out_dim)

    def forward(self, x: torch.Tensor, interpolate_pos_encoding: bool = False) -> SimpleNamespace:
        del interpolate_pos_encoding
        pooled = x.mean(dim=(-2, -1))
        return SimpleNamespace(last_hidden_state=self.proj(pooled).unsqueeze(1))


class FakeLeWMActionEncoder(nn.Module):
    def __init__(self, action_dim: int = 2, out_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(action_dim, out_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.proj(action)


class FakeLeWMPredictor(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        return z + action_emb


def _lewm_source_config() -> dict:
    return {
        "_target_": "stable_worldmodel.wm.lewm.LeWM",
        "encoder": {"_target_": "tests.test_mwm_artifacts.FakeLeWMEncoder", "out_dim": 4},
        "predictor": {
            "_target_": "tests.test_mwm_artifacts.FakeLeWMPredictor",
            "input_dim": 4,
            "hidden_dim": 4,
            "output_dim": 4,
        },
        "action_encoder": {"_target_": "tests.test_mwm_artifacts.FakeLeWMActionEncoder", "action_dim": 2, "out_dim": 4},
        "projector": {"_target_": "torch.nn.Identity"},
        "pred_proj": {"_target_": "torch.nn.Identity"},
    }


class MWMArtifactTests(unittest.TestCase):
    def test_stable_worldmodel_sampling_includes_last_valid_start(self) -> None:
        class TinyDataset:
            lengths = np.asarray([5], dtype=np.int64)
            offsets = np.asarray([0], dtype=np.int64)

        pairs = sample_start_goal_pairs(
            TinyDataset(),
            count=3,
            goal_offset_steps=2,
            seed=0,
            mode="stable_worldmodel",
        )

        self.assertEqual([pair.start_step for pair in pairs], [0, 1, 2])
        self.assertEqual([pair.goal_step for pair in pairs], [2, 3, 4])

    def test_base_adaptive_checkpoint_metadata_persisted_from_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = build_mwm_from_stable_config(
                family="lewm",
                source_config=_lewm_source_config(),
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
        policy = {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": []}
        config = {
            "target": STABLE_CONFIG_TARGET,
            "kwargs": {
                "family": "lewm",
                "source_config": _lewm_source_config(),
                "source_config_sha256": "abc",
                "training_recipe": {},
                "K": [4],
                "component_policy": policy,
            },
        }
        valid_metadata = {
            "adapter_family": "lewm",
            "source_config_sha256": "abc",
            "fresh_init": True,
            "component_policy": policy,
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
                    "env_id": "swm/PushT-v1",
                    "seed": 0,
                    "eval_config": str(cfg_path),
                    "manifest": {"group": "test_manifest", "path": str(Path(tmp) / "manifest.json")},
                    "runs": [
                        {"name": "a", "role": "mwm_scheduled", "checkpoint": "checkpoints_mwm/a"},
                        {"name": "b", "role": "mwm_scheduled", "checkpoint": "checkpoints_mwm/b"},
                    ],
                },
            )
            resolved = [(run, merged_run_config(cfg, run)[1]) for run in cfg.runs]
            with self.assertRaisesRegex(ValueError, "duplicate cells"):
                validate_benchmark_matrix(cfg, resolved)

    def test_benchmark_run_config_merges_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "eval.yaml"
            cfg_path.write_text(
                """
env_id: swm/PushT-v1
checkpoint: {run_dir: checkpoints_mwm/example, epoch: null}
data: {path: data/pusht_swm.lance, format: lance}
eval: {seed: 0}
env:
  max_episode_steps: 100
  goal_conditioned: true
  kwargs:
    difficulty: base
planner: {scheduler: {policy: fixed, level: finest, rollout_level: base}}
""",
                encoding="utf-8",
            )
            cfg = OmegaConf.merge(
                DEFAULTS,
                {
                    "env_id": "swm/PushT-v1",
                    "seed": 0,
                    "eval_config": str(cfg_path),
                    "manifest": {"group": "test_manifest", "path": str(Path(tmp) / "manifest.json")},
                    "runs": [
                        {
                            "name": "offset_probe",
                            "role": "mwm_scheduled",
                            "checkpoint": "checkpoints_mwm/example",
                            "env": {
                                "max_episode_steps": 200,
                                "kwargs": {"difficulty": "hard", "variant": "offset100"},
                            },
                        }
                    ],
                },
            )

            _, run_cfg = merged_run_config(cfg, cfg.runs[0])

            self.assertEqual(run_cfg.env.max_episode_steps, 200)
            self.assertTrue(run_cfg.env.goal_conditioned)
            self.assertEqual(run_cfg.env.kwargs.difficulty, "hard")
            self.assertEqual(run_cfg.env.kwargs.variant, "offset100")

    def test_benchmark_verifier_static_only_accepts_paper_parity_config(self) -> None:
        report = verify_benchmark_static("configs/benchmark/paper_parity_pusht.yaml", check_checkpoints=False)

        self.assertEqual(report["runs"], 2)
        self.assertEqual(report["env_id"], "swm/PushT-v1")
        self.assertEqual(report["paper_targets"]["tolerance_pp"], 1.0)
        self.assertEqual(report["paper_targets"]["retrained_match_tolerance_pp"], 5.0)

    def test_scheduled_pusht_benchmark_contract_is_single_env_shared_manifest(self) -> None:
        report = verify_benchmark_static("configs/benchmark/scheduled_pusht.yaml", check_checkpoints=False)

        self.assertEqual(report["runs"], 2)
        self.assertEqual(report["output_dir"], "rollouts/mwm_scheduled_pusht")
        self.assertEqual(report["env_id"], "swm/PushT-v1")
        self.assertEqual(report["manifest"]["group"], "pusht_paper_seed42")
        cells = {tuple(cell) for cell in report["expected_cells"]}
        self.assertEqual(
            cells,
            {
                ("swm/PushT-v1", 42, "upstream_lewm_converted"),
                ("swm/PushT-v1", 42, "mwm_scheduled"),
            },
        )

    def test_dense_pusht_benchmark_contract_is_single_env_shared_manifest(self) -> None:
        report = verify_benchmark_static("configs/benchmark/dense_pusht.yaml", check_checkpoints=False)

        self.assertEqual(report["runs"], 2)
        self.assertEqual(report["output_dir"], "rollouts/mwm_dense_pusht")
        self.assertEqual(report["env_id"], "swm/PushT-v1")
        self.assertEqual(report["manifest"]["group"], "pusht_paper_seed42")
        cells = {tuple(cell) for cell in report["expected_cells"]}
        self.assertEqual(
            cells,
            {
                ("swm/PushT-v1", 42, "upstream_lewm_converted"),
                ("swm/PushT-v1", 42, "mwm_dense"),
            },
        )

    def test_dense_benchmark_requires_scheduler_plots(self) -> None:
        cfg = OmegaConf.create({"runs": [{"role": "upstream_lewm_converted"}, {"role": "mwm_dense"}]})

        required = required_plots_for_benchmark(cfg)

        self.assertIn("schedule_level_usage.png", required)
        self.assertIn("schedule_usage_by_role.png", required)

    def test_benchmark_static_rejects_legacy_gate_and_run_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eval_cfg = Path(tmp) / "eval.yaml"
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
            bench_cfg = Path(tmp) / "benchmark.yaml"
            bench_cfg.write_text(
                f"""
output_dir: {Path(tmp) / "out"}
env_id: swm/PushT-v1
seed: 0
eval_config: {eval_cfg}
manifest: {{group: test_manifest, path: {Path(tmp) / "manifest.json"}}}
gate: {{enabled: true}}
runs:
  - name: stale
    role: mwm_scheduled
    checkpoint: checkpoints_mwm/example
    manifest_group: stale
    overrides:
      checkpoint:
        run_dir: checkpoints_mwm/example
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "legacy benchmark field"):
                verify_benchmark_static(str(bench_cfg), check_checkpoints=False)

    def test_benchmark_static_verifier_rejects_stale_scheduled_checkpoint_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config_path = checkpoint / CONFIG_FILENAME
            weights_path = checkpoint / WEIGHTS_FILENAME
            metadata_path = checkpoint / METADATA_FILENAME
            policy = {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": []}
            config_path.write_text(
                json.dumps(
                    {
                        "target": STABLE_CONFIG_TARGET,
                        "kwargs": {
                            "family": "lewm",
                            "action_dim": 10,
                            "action_block": 5,
                            "K": [48, 96, 144, 192],
                            "source_config": _lewm_source_config(),
                            "source_config_sha256": "abc",
                            "training_recipe": {},
                            "component_policy": policy,
                        },
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
                        "action_block": 5,
                        "action_spec": {"dim": 10, "base_dim": 2, "block": 5},
                        "adapter_family": "lewm",
                        "fresh_init": True,
                        "source_config_sha256": "abc",
                        "component_policy": policy,
                        "training_backend": "stable_worldmodel_lewm",
                        "architecture_version": LEWM_BASE_ADAPTER_ARCH,
                        "levels": [48, 96, 144, 192],
                        "model": {"target": STABLE_CONFIG_TARGET},
                        "artifacts": {
                            "config": {"path": CONFIG_FILENAME, "sha256": file_sha256(config_path)},
                            "weights": {"path": WEIGHTS_FILENAME, "sha256": file_sha256(weights_path)},
                        },
                    }
                ),
                encoding="utf-8",
            )
            eval_cfg = root / "eval.yaml"
            eval_cfg.write_text(
                f"""
env_id: swm/PushT-v1
checkpoint: {{run_dir: {checkpoint}, epoch: null}}
data: {{path: data/pusht_swm.lance, format: lance}}
eval: {{seed: 0}}
planner: {{scheduler: {{policy: fixed, level: finest, rollout_level: base}}}}
""",
                encoding="utf-8",
            )
            bench_cfg = root / "benchmark.yaml"
            bench_cfg.write_text(
                f"""
output_dir: {root / "out"}
env_id: swm/PushT-v1
seed: 0
eval_config: {eval_cfg}
manifest: {{group: scheduled_stale, path: {root / "manifest.json"}}}
runs:
  - name: scheduled
    role: mwm_scheduled
    checkpoint: {checkpoint}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scheduled MWM checkpoint must be K=\\[48,96,144\\]"):
                verify_benchmark_static(str(bench_cfg))

    def test_benchmark_static_verifier_rejects_stale_dense_checkpoint_levels(self) -> None:
        row = {"role": "mwm_dense", "checkpoint_run_dir": "checkpoints_mwm/dense"}
        metadata = {
            "D": 192,
            "levels": [6, 12, 48, 96, 144],
            "training_backend": "stable_worldmodel_lewm",
            "architecture_version": LEWM_BASE_ADAPTER_ARCH,
            "model": {"target": STABLE_CONFIG_TARGET},
        }

        errors: list[str] = []
        validate_benchmark_role_checkpoint_contract(row, metadata, errors)

        self.assertTrue(any("dense MWM checkpoint must be K=[6,12,48,96,144,192]" in error for error in errors), errors)

    def test_benchmark_static_cli_can_skip_checkpoint_contracts(self) -> None:
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_cfg = root / "eval.yaml"
            eval_cfg.write_text(
                """
env_id: swm/PushT-v1
checkpoint: {run_dir: missing_checkpoint, epoch: null}
data: {path: data/upstream/pusht_expert_train.lance, format: lance}
eval: {seed: 0}
planner: {scheduler: {policy: fixed, level: finest, rollout_level: base}}
""",
                encoding="utf-8",
            )
            bench_cfg = root / "benchmark.yaml"
            bench_cfg.write_text(
                f"""
output_dir: {root / "out"}
env_id: swm/PushT-v1
seed: 0
eval_config: {eval_cfg}
manifest: {{group: local_static, path: {root / "manifest.json"}}}
runs:
  - name: missing_checkpoint_run
    role: upstream_lewm_converted
    checkpoint: {root / "missing_checkpoint"}
""",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "verify_mwm_benchmark.py",
                    str(bench_cfg),
                    "--static-only",
                    "--no-checkpoints",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"check_checkpoints": false', result.stdout.lower())

    def test_benchmark_role_filter_runs_upstream_first(self) -> None:
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
env_id: swm/PushT-v1
seed: 0
eval_config: {eval_cfg}
manifest: {{group: filter_test, path: {root / "manifest.json"}}}
runs:
  - name: upstream
    role: upstream_lewm_converted
    checkpoint: checkpoints_mwm/upstream
  - name: retrained
    role: retrained_lewm_identity
    checkpoint: checkpoints_mwm/retrained
""",
                encoding="utf-8",
            )

            calls: list[str] = []
            eval_manifest_keys: list[tuple[Any, Any]] = []
            old_run_eval = benchmark_mwm.run_eval_mwm

            def _fake_run(cfg_path: str) -> None:
                cfg = OmegaConf.load(cfg_path)
                calls.append(str(cfg.checkpoint.run_dir))
                eval_manifest_keys.append(
                    (
                        cfg.eval.get("write_manifest_path", None),
                        cfg.eval.get("writemanifest_path", None),
                    )
                )
                _payload("stub", str(cfg.env_id), int(cfg.eval.seed), Path(str(cfg.eval.output_path)))

            benchmark_mwm.run_eval_mwm = _fake_run
            try:
                benchmark_mwm.main(str(bench_cfg), roles=["upstream_lewm_converted"])
            finally:
                benchmark_mwm.run_eval_mwm = old_run_eval

            summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(calls, ["checkpoints_mwm/upstream"])
            self.assertEqual(eval_manifest_keys, [(str(root / "manifest.json"), None)])
            self.assertEqual([row["role"] for row in summary["runs"]], ["upstream_lewm_converted"])
            static_report = verify_benchmark_static(
                str(bench_cfg),
                roles=["upstream_lewm_converted"],
                check_checkpoints=False,
            )
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
env_id: swm/PushT-v1
seed: 0
eval_config: {eval_cfg}
manifest: {{group: failure_test, path: {root / "manifest.json"}}}
runs:
  - {{name: broken, role: mwm_scheduled, checkpoint: checkpoints_mwm/example}}
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
            policy = {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": []}
            cfg_path.write_text(
                json.dumps(
                    {
                        "target": STABLE_CONFIG_TARGET,
                        "kwargs": {
                            "family": "lewm",
                            "source_config": _lewm_source_config(),
                            "source_config_sha256": "abc",
                            "training_recipe": {},
                            "action_dim": 2,
                            "action_block": 1,
                            "K": [2],
                            "component_policy": policy,
                        },
                    }
                ),
                encoding="utf-8",
            )
            weights_path.write_bytes(b"weights")
            metadata_path.write_text(
                json.dumps(
                    {
                        "format": CHECKPOINT_FORMAT,
                        "adapter_family": "lewm",
                        "fresh_init": True,
                        "source_config_sha256": "abc",
                        "component_policy": policy,
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
            load_checkpoint_metadata_for_benchmark(root, errors)
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
            {"env_id": "swm/PushT-v1", "role": "retrained_lewm_identity", "success_rate": 94.0},
        ]

        errors: list[str] = []
        append_paper_target_errors(cfg, rows, errors)
        self.assertEqual(errors, [])

        rows[1]["success_rate"] = 70.0
        append_paper_target_errors(cfg, rows, errors)
        self.assertTrue(any("retrained match check failed" in error for error in errors), errors)

    def test_paper_target_upstream_only_benchmark_skips_retrained_match(self) -> None:
        cfg = OmegaConf.create(
            {
                "runs": [{"role": "upstream_lewm_converted"}],
                "paper_targets": {
                    "enabled": True,
                    "tolerance_pp": 1.0,
                    "retrained_match_tolerance_pp": 5.0,
                    "success_rate": {"swm/PushT-v1": 96.0},
                },
            }
        )
        rows = [{"env_id": "swm/PushT-v1", "role": "upstream_lewm_converted", "success_rate": 96.0}]

        errors: list[str] = []
        append_paper_target_errors(cfg, rows, errors)

        self.assertEqual(errors, [])

    def test_paper_target_check_fails_when_upstream_misses_by_more_than_one_point(self) -> None:
        rows = [
            {"env_id": "swm/PushT-v1", "role": "upstream_lewm_converted", "success_rate": 94.0},
            {"env_id": "swm/PushT-v1", "role": "retrained_lewm_identity", "success_rate": 95.0},
        ]
        cfg = {
            "paper_targets": {
                "enabled": True,
                "tolerance_pp": 1.0,
                "retrained_match_tolerance_pp": 5.0,
                "success_rate": {"swm/PushT-v1": 96.0},
            }
        }

        errors = validate_paper_targets(rows, cfg)

        self.assertTrue(any("paper target check failed" in error for error in errors), errors)

    def test_role_checkpoint_contract_rejects_direct_target_and_wrong_backend(self) -> None:
        row = {"role": "retrained_lewm_identity", "checkpoint_run_dir": "checkpoints_mwm/retrained"}
        metadata = {
            "D": 192,
            "levels": [192],
            "training_backend": "stable_pretraining",
            "model": {"target": "mwm.adapters.lewm.build_mwm_lewm"},
        }

        errors: list[str] = []
        validate_benchmark_role_checkpoint_contract(row, metadata, errors)

        self.assertTrue(any("Le-WM base-adapter backend" in error for error in errors), errors)
        self.assertTrue(any("generic base-adaptive target" in error for error in errors), errors)
        self.assertTrue(any("corrected architecture version" in error for error in errors), errors)

    def test_role_checkpoint_contract_accepts_base_adaptive_lewm_target(self) -> None:
        rows = [
            {"role": "upstream_lewm_converted", "checkpoint_run_dir": "checkpoints_mwm/upstream"},
            {"role": "retrained_lewm_identity", "checkpoint_run_dir": "checkpoints_mwm/retrained"},
            {"role": "mwm_scheduled", "checkpoint_run_dir": "checkpoints_mwm/scheduled"},
            {"role": "mwm_dense", "checkpoint_run_dir": "checkpoints_mwm/dense"},
        ]
        metadatas = [
            {
                "role": "upstream_lewm_converted",
                "levels": [128],
                "architecture_version": LEWM_BASE_ADAPTER_ARCH,
                "model": {"D": 128, "target": STABLE_CONFIG_TARGET},
            },
            {
                "D": 128,
                "levels": [128],
                "training_backend": "stable_worldmodel_lewm",
                "architecture_version": LEWM_BASE_ADAPTER_ARCH,
                "model": {"target": STABLE_CONFIG_TARGET},
            },
            {
                "D": 192,
                "levels": [48, 96, 144],
                "training_backend": "stable_worldmodel_lewm",
                "architecture_version": LEWM_BASE_ADAPTER_ARCH,
                "model": {"target": STABLE_CONFIG_TARGET},
            },
            {
                "D": 192,
                "levels": [6, 12, 48, 96, 144, 192],
                "training_backend": "stable_worldmodel_lewm",
                "architecture_version": LEWM_BASE_ADAPTER_ARCH,
                "model": {"target": STABLE_CONFIG_TARGET},
            },
        ]

        errors: list[str] = []
        for row, metadata in zip(rows, metadatas):
            validate_benchmark_role_checkpoint_contract(row, metadata, errors)

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


if __name__ == "__main__":
    unittest.main()
