from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import mwm.checkpoint_io as checkpoint_io
import mwm.checkpoint_keymaps as checkpoint_keymaps
from mwm.benchmark import matrix as benchmark_mwm
from mwm.benchmark import output_verify as output_verify_module
from mwm.benchmark.config import DEFAULTS, merged_run_config, validate_benchmark_matrix
from mwm.benchmark.html import write_review_html
from mwm.benchmark.summary import eval_summary_row, write_per_env_table, write_summary_csv
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
from mwm.data.manifest import generate_manifest, load_manifest, manifest_file_sha256, write_manifest
from mwm.data.collection import record_dataset_to_path
from mwm.data.metadata import write_dataset_metadata
from mwm.data.sampling import StartGoalPair, sample_start_goal_pairs
from mwm.eval.runner import resolve_device as eval_device
from mwm.benchmark.checkpoint_verify import (
    load_checkpoint_metadata_for_benchmark,
    validate_benchmark_role_checkpoint_contract,
)
from mwm.benchmark.paper_targets import append_paper_target_errors, validate_paper_targets
from mwm.benchmark.plot_contract import BASE_REQUIRED_PLOTS, required_plots_for_benchmark
from mwm.benchmark.output_verify import verify_benchmark_output
from mwm.benchmark.static_verify import verify_benchmark_static
from mwm.io import file_sha256 as io_file_sha256, write_json, write_metrics_jsonl
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
            "dynamics_flops_total": 2000 + seed,
            "plan_time_total_sec": 0.5,
            "schedule_level_counts": {"0": 1, "1": 1},
        },
        "schedule": {"enabled": True, "mpc": {"mode": "fixed", "level": "finest"}, "cem": {"mode": "linear", "start_level": "coarsest", "end_level": "finest"}, "rollout": {"mode": "fixed", "level": "base"}},
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
    def test_eval_auto_device_uses_cuda_when_probe_succeeds(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch, "empty", return_value=torch.empty(1)) as empty,
        ):
            device = eval_device("auto")

        self.assertEqual(device.type, "cuda")
        empty.assert_called_once_with(1, device="cuda")

    def test_eval_auto_device_falls_back_to_cpu_when_cuda_probe_fails(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch, "empty", side_effect=RuntimeError("device unavailable")),
            self.assertWarnsRegex(RuntimeWarning, "falling back to CPU"),
        ):
            device = eval_device("auto")

        self.assertEqual(device.type, "cpu")

    def test_record_dataset_eager_write_uses_world_writer_path(self) -> None:
        class FakeWorld:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def collect(
                self,
                path=None,
                episodes: int = 0,
                seed: int | None = None,
                format: str = "lance",
                writer=None,
                progress: bool = True,
            ) -> None:
                self.calls.append(
                    {
                        "path": path,
                        "episodes": episodes,
                        "seed": seed,
                        "format": format,
                        "writer": writer,
                        "progress": progress,
                    }
                )
                self.assert_main_thread_writer_call()
                with writer as w:
                    w.write_episodes(
                        [
                            {
                                "pixels": [
                                    np.zeros((2, 2, 3), dtype=np.uint8),
                                    np.ones((2, 2, 3), dtype=np.uint8),
                                ],
                                "action": [np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)],
                                "qpos": [np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
                                "qvel": [np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
                                "success": [np.nan, np.nan],
                            }
                        ]
                    )

            def assert_main_thread_writer_call(self) -> None:
                call = self.calls[-1]
                if call["path"] is not None:
                    raise AssertionError("eager collection must pass writer= instead of path=")
                if call["writer"] is None:
                    raise AssertionError("eager collection must provide a writer")

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out.lance"
            world = FakeWorld()

            record_dataset_to_path(
                world,
                output_path,
                episodes=1,
                seed=3,
                format="lance",
                eager_write=True,
                keys_to_save=["pixels", "action", "qpos", "qvel"],
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(world.calls), 1)
            self.assertEqual(world.calls[0]["episodes"], 1)
            self.assertEqual(world.calls[0]["seed"], 3)

    def test_hf_vit_encoder_keys_remap_to_custom_vit_layers(self) -> None:
        self.assertTrue(hasattr(checkpoint_io, "remap_hf_vit_encoder_keys"))
        self.assertIs(checkpoint_io.remap_hf_vit_encoder_keys, checkpoint_keymaps.remap_hf_vit_encoder_keys)
        state = {
            "encoder.encoder.layer.2.attention.attention.query.weight": torch.ones(1),
            "encoder.encoder.layer.2.attention.attention.key.bias": torch.ones(1) * 2,
            "encoder.encoder.layer.2.attention.attention.value.weight": torch.ones(1) * 3,
            "encoder.encoder.layer.2.attention.output.dense.bias": torch.ones(1) * 4,
            "encoder.encoder.layer.2.intermediate.dense.weight": torch.ones(1) * 5,
            "encoder.encoder.layer.2.output.dense.bias": torch.ones(1) * 6,
            "decoder.weight": torch.ones(1) * 7,
        }

        remapped = checkpoint_keymaps.remap_hf_vit_encoder_keys(state)

        self.assertEqual(
            set(remapped),
            {
                "encoder.layers.2.attention.q_proj.weight",
                "encoder.layers.2.attention.k_proj.bias",
                "encoder.layers.2.attention.v_proj.weight",
                "encoder.layers.2.attention.o_proj.bias",
                "encoder.layers.2.mlp.fc1.weight",
                "encoder.layers.2.mlp.fc2.bias",
                "decoder.weight",
            },
        )
        self.assertIs(remapped["encoder.layers.2.attention.q_proj.weight"], state["encoder.encoder.layer.2.attention.attention.query.weight"])
        self.assertIs(remapped["decoder.weight"], state["decoder.weight"])

    def test_hf_vit_encoder_key_remap_leaves_native_keys_unchanged(self) -> None:
        self.assertTrue(hasattr(checkpoint_io, "remap_hf_vit_encoder_keys"))
        state = {"encoder.layers.0.attention.q_proj.weight": torch.ones(1)}

        self.assertIs(checkpoint_keymaps.remap_hf_vit_encoder_keys(state), state)

    def test_custom_vit_encoder_keys_remap_to_hf_vit_layers(self) -> None:
        self.assertTrue(hasattr(checkpoint_io, "remap_custom_vit_encoder_keys_to_hf"))
        self.assertIs(checkpoint_io.remap_custom_vit_encoder_keys_to_hf, checkpoint_keymaps.remap_custom_vit_encoder_keys_to_hf)
        state = {
            "encoder.layers.2.attention.q_proj.weight": torch.ones(1),
            "encoder.layers.2.attention.k_proj.bias": torch.ones(1) * 2,
            "encoder.layers.2.attention.v_proj.weight": torch.ones(1) * 3,
            "encoder.layers.2.attention.o_proj.bias": torch.ones(1) * 4,
            "encoder.layers.2.mlp.fc1.weight": torch.ones(1) * 5,
            "encoder.layers.2.mlp.fc2.bias": torch.ones(1) * 6,
            "decoder.weight": torch.ones(1) * 7,
        }

        remapped = checkpoint_keymaps.remap_custom_vit_encoder_keys_to_hf(state)

        self.assertEqual(
            set(remapped),
            {
                "encoder.encoder.layer.2.attention.attention.query.weight",
                "encoder.encoder.layer.2.attention.attention.key.bias",
                "encoder.encoder.layer.2.attention.attention.value.weight",
                "encoder.encoder.layer.2.attention.output.dense.bias",
                "encoder.encoder.layer.2.intermediate.dense.weight",
                "encoder.encoder.layer.2.output.dense.bias",
                "decoder.weight",
            },
        )
        self.assertIs(remapped["encoder.encoder.layer.2.attention.attention.query.weight"], state["encoder.layers.2.attention.q_proj.weight"])
        self.assertIs(remapped["decoder.weight"], state["decoder.weight"])

    def test_target_aware_vit_key_remap_keeps_matching_hf_keys(self) -> None:
        self.assertTrue(hasattr(checkpoint_io, "remap_vit_encoder_keys_for_model"))
        self.assertIs(checkpoint_io.remap_vit_encoder_keys_for_model, checkpoint_keymaps.remap_vit_encoder_keys_for_model)
        state = {
            "encoder.encoder.layer.0.attention.attention.query.weight": torch.ones(1),
            "decoder.weight": torch.ones(1) * 2,
        }
        model = SimpleNamespace(state_dict=lambda: dict(state))

        self.assertIs(checkpoint_keymaps.remap_vit_encoder_keys_for_model(state, model), state)

    def test_target_aware_vit_key_remap_maps_custom_keys_to_hf_model(self) -> None:
        state = {
            "encoder.layers.0.attention.q_proj.weight": torch.ones(1),
            "decoder.weight": torch.ones(1) * 2,
        }
        model = SimpleNamespace(
            state_dict=lambda: {
                "encoder.encoder.layer.0.attention.attention.query.weight": torch.empty(1),
                "decoder.weight": torch.empty(1),
            }
        )

        remapped = checkpoint_keymaps.remap_vit_encoder_keys_for_model(state, model)

        self.assertIn("encoder.encoder.layer.0.attention.attention.query.weight", remapped)
        self.assertNotIn("encoder.layers.0.attention.q_proj.weight", remapped)
        self.assertIs(remapped["encoder.encoder.layer.0.attention.attention.query.weight"], state["encoder.layers.0.attention.q_proj.weight"])

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

    def test_stable_wm_checkpoint_metadata_persisted_from_model(self) -> None:
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

    def test_legacy_lewm_checkpoint_without_decoder_policy_loads_for_eval(self) -> None:
        legacy_policy = {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": []}
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

            config_path = out_dir / CONFIG_FILENAME
            weights_path = out_dir / WEIGHTS_FILENAME
            metadata_path = out_dir / METADATA_FILENAME
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            config["kwargs"]["component_policy"] = legacy_policy
            metadata["component_policy"] = legacy_policy
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = {key: value for key, value in state.items() if not key.startswith("decoders.")}
            torch.save(state, weights_path)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            metadata["artifacts"]["config"]["sha256"] = file_sha256(config_path)
            metadata["artifacts"]["weights"]["sha256"] = file_sha256(weights_path)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            loaded, loaded_metadata, epoch = checkpoint_io.load_world_model_from_checkpoint(
                out_dir,
                None,
                torch.device("cpu"),
            )

        self.assertEqual(epoch, 0)
        self.assertEqual(loaded_metadata["component_policy"], legacy_policy)
        self.assertEqual(loaded.metadata["component_policy"], legacy_policy)
        self.assertEqual(len(loaded.decoders), 1)

    def test_modern_lewm_checkpoint_requires_decoder_weights(self) -> None:
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

            weights_path = out_dir / WEIGHTS_FILENAME
            metadata_path = out_dir / METADATA_FILENAME
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = {key: value for key, value in state.items() if not key.startswith("decoders.")}
            torch.save(state, weights_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["artifacts"]["weights"]["sha256"] = file_sha256(weights_path)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Missing key.*decoders"):
                checkpoint_io.load_world_model_from_checkpoint(out_dir, None, torch.device("cpu"))

    def test_stable_wm_checkpoint_contract_requires_metadata(self) -> None:
        policy = {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": ["decoder"]}
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
            (
                "missing=.*per_level",
                {**valid_metadata, "component_policy": {"shared": ["latent_producer"], "reconstructor": ["decoder"]}},
            ),
            (
                "unknown=.*extra",
                {
                    **valid_metadata,
                    "component_policy": {
                        "shared": ["latent_producer"],
                        "per_level": ["transition"],
                        "reconstructor": ["decoder"],
                        "extra": [],
                    },
                },
            ),
            (
                "shared latent producer",
                {**valid_metadata, "component_policy": {"shared": [], "per_level": ["transition"], "reconstructor": ["decoder"]}},
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
planner: {scheduler: {enabled: true, mpc: {mode: fixed, level: finest}, cem: {mode: fixed, level: base}, rollout: {mode: fixed, level: base}}}
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

    def test_benchmark_output_verifier_has_public_owner(self) -> None:
        self.assertTrue(callable(verify_benchmark_output))

    def test_benchmark_output_verifier_checks_minimal_output_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            run_dir = output_dir / "000_upstream"
            run_dir.mkdir(parents=True)
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True)
            eval_cfg = root / "eval.yaml"
            eval_cfg.write_text(
                """
env_id: swm/PushT-v1
checkpoint: {run_dir: checkpoints_mwm/example, epoch: null}
data: {path: data/pusht_swm.lance, format: lance}
eval: {seed: 0}
planner: {scheduler: {enabled: true, mpc: {mode: fixed, level: finest}, cem: {mode: fixed, level: base}, rollout: {mode: fixed, level: base}}}
""",
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            bench_cfg = root / "benchmark.yaml"
            bench_cfg.write_text(
                f"""
title: Minimal verifier fixture
output_dir: {output_dir}
env_id: swm/PushT-v1
seed: 0
eval_config: {eval_cfg}
manifest: {{group: verifier_fixture, path: {manifest_path}}}
runs:
  - name: upstream
    role: upstream_lewm_converted
    checkpoint: checkpoints_mwm/example
""",
                encoding="utf-8",
            )
            deps = {
                "stable-worldmodel": {"sha256": "stable-worldmodel"},
                "stable-pretraining": {"sha256": "stable-pretraining"},
                "torch": {"sha256": "torch"},
                "local_repo": {"commit_id": "local"},
            }
            manifest = generate_manifest(
                env_id="swm/PushT-v1",
                dataset_path="data/pusht_swm.lance",
                pairs=[StartGoalPair(episode=0, start_step=0, goal_step=1, start_row=0, goal_row=1)],
                goal_offset=1,
                eval_budget=1,
                seed=0,
                restore_spec="pusht_state_goal_state",
                dependency_shas=deps,
            )
            write_manifest(manifest_path, manifest)
            for plot_name in BASE_REQUIRED_PLOTS:
                (plots_dir / plot_name).write_bytes(b"png")

            resolved_cfg = run_dir / "resolved_config.yaml"
            resolved_cfg.write_text("env_id: swm/PushT-v1\n", encoding="utf-8")
            eval_path = run_dir / "eval.json"
            payload = {
                "env_id": "swm/PushT-v1",
                "checkpoint_epoch": 0,
                "checkpoint_run_dir": "checkpoints_mwm/example",
                "config": {"sha256": io_file_sha256(resolved_cfg)},
                "manifest": {
                    "path": str(manifest_path),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "sha256": manifest_file_sha256(manifest_path),
                },
                "episodes": 1,
                "goal_offset": 1,
                "swm_results": {"success_rate": 100.0},
                "planning_diagnostics": {
                    "plans": 1,
                    "steps": 1,
                    "bits_used_total": 10,
                    "dynamics_flops_total": 20,
                    "plan_time_total_sec": 0.1,
                    "summary": {"cem_cost_calls": 1, "candidate_action_values": 1},
                },
                "dependencies": deps,
                "schedule": "fixed",
                "role": "upstream_lewm_converted",
                "seed": 0,
                "wall_time_sec": 0.1,
            }
            write_json(eval_path, payload)
            row = eval_summary_row("upstream", eval_path, payload)
            self.assertEqual(row["dynamics_flops_total"], 20)
            write_json(run_dir / "summary.json", {"run": row})
            write_json(run_dir / "dependencies.json", deps)
            write_json(run_dir / "planning_diagnostics.json", payload["planning_diagnostics"])
            write_metrics_jsonl(run_dir / "metrics.jsonl", [row])
            write_metrics_jsonl(run_dir / "episode_traces.jsonl", [{"episode_index": 0, "success": True}])
            write_summary_csv(output_dir / "summary.csv", [row])
            write_metrics_jsonl(output_dir / "metrics.jsonl", [row])
            write_per_env_table(output_dir / "per_env_summary.csv", [row])
            plots = [str(plots_dir / name) for name in sorted(BASE_REQUIRED_PLOTS)]
            write_review_html(output_dir / "review.html", "Minimal verifier fixture", [row], [payload], plots=plots, expected_cells=1)
            write_json(
                output_dir / "summary.json",
                {
                    "title": "Minimal verifier fixture",
                    "output_dir": str(output_dir),
                    "runs": [row],
                    "manifest": {"group": "verifier_fixture", "path": str(manifest_path), "seed": 0},
                    "plots": plots,
                },
            )

            with mock.patch.object(output_verify_module, "load_checkpoint_metadata_for_benchmark", return_value={}):
                report = verify_benchmark_output(str(bench_cfg))
                self.assertEqual(report["runs"], 1)

                (plots_dir / next(iter(BASE_REQUIRED_PLOTS))).unlink()
                with self.assertRaisesRegex(ValueError, "missing file"):
                    verify_benchmark_output(str(bench_cfg))

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
planner: {scheduler: {enabled: true, mpc: {mode: fixed, level: finest}, cem: {mode: fixed, level: base}, rollout: {mode: fixed, level: base}}}
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

    def test_ogb_cube_parity_and_dense_benchmarks_are_single_env_shared_manifest(self) -> None:
        parity = verify_benchmark_static("configs/benchmark/paper_parity_ogb_cube.yaml", check_checkpoints=False)
        dense = verify_benchmark_static("configs/benchmark/dense_ogb_cube.yaml", check_checkpoints=False)

        self.assertEqual(parity["env_id"], "swm/OGBCube-v0")
        self.assertEqual(parity["paper_targets"]["success_rate"], {"swm/OGBCube-v0": 74.0})
        self.assertEqual(parity["manifest"]["group"], "ogb_cube_paper_seed42")
        self.assertEqual(
            {tuple(cell) for cell in parity["expected_cells"]},
            {
                ("swm/OGBCube-v0", 42, "upstream_lewm_converted"),
                ("swm/OGBCube-v0", 42, "retrained_lewm_identity"),
            },
        )
        self.assertEqual(dense["env_id"], "swm/OGBCube-v0")
        self.assertEqual(dense["manifest"]["group"], "ogb_cube_paper_seed42")
        self.assertEqual(
            {tuple(cell) for cell in dense["expected_cells"]},
            {
                ("swm/OGBCube-v0", 42, "upstream_lewm_converted"),
                ("swm/OGBCube-v0", 42, "mwm_dense"),
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
planner: {scheduler: {enabled: true, mpc: {mode: fixed, level: finest}, cem: {mode: fixed, level: base}, rollout: {mode: fixed, level: base}}}
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
            policy = {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": ["decoder"]}
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
planner: {{scheduler: {{enabled: true, mpc: {{mode: fixed, level: finest}}, cem: {{mode: fixed, level: base}}, rollout: {{mode: fixed, level: base}}}}}}
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
planner: {scheduler: {enabled: true, mpc: {mode: fixed, level: finest}, cem: {mode: fixed, level: base}, rollout: {mode: fixed, level: base}}}
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
                    "-m",
                    "mwm.benchmark.verify",
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
planner: {scheduler: {enabled: true, mpc: {mode: fixed, level: finest}, cem: {mode: fixed, level: base}, rollout: {mode: fixed, level: base}}}
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
planner: {scheduler: {enabled: true, mpc: {mode: fixed, level: finest}, cem: {mode: fixed, level: base}, rollout: {mode: fixed, level: base}}}
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
            output_dir = root / "out"
            output_dir.mkdir()
            (output_dir / "summary.json").write_text('{"stale": true}', encoding="utf-8")
            (output_dir / "summary.csv").write_text("stale\n", encoding="utf-8")
            (output_dir / "plots").mkdir()
            (output_dir / "plots" / "success_by_env_role.png").write_bytes(b"stale")

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
            self.assertFalse((root / "out" / "summary.json").exists())
            self.assertFalse((root / "out" / "summary.csv").exists())
            self.assertFalse((root / "out" / "plots").exists())

    def test_checkpoint_verifier_requires_action_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / CONFIG_FILENAME
            weights_path = root / WEIGHTS_FILENAME
            metadata_path = root / METADATA_FILENAME
            policy = {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": ["decoder"]}
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

    def test_paper_target_tolerance_respects_episode_granularity(self) -> None:
        rows = [
            {
                "env_id": "swm/PushT-v1",
                "role": "upstream_lewm_converted",
                "success_rate": 98.0,
                "episodes": 50,
            },
            {
                "env_id": "swm/PushT-v1",
                "role": "retrained_lewm_identity",
                "success_rate": 92.0,
                "episodes": 50,
            },
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

        self.assertEqual(errors, [])

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
        self.assertTrue(any("generic Stable-WM builder target" in error for error in errors), errors)
        self.assertTrue(any("corrected architecture version" in error for error in errors), errors)

    def test_role_checkpoint_contract_accepts_lewm_matryoshka_model_target(self) -> None:
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

    def test_reacher_h5_conversion_writes_lance_dataset_and_metadata(self) -> None:
        import h5py
        import lance

        from mwm.data.metadata import load_dataset_metadata
        from mwm.upstream.converters.reacher import convert_reacher_h5_to_lance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reacher.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("ep_offset", data=np.asarray([0, 2], dtype=np.int64))
                handle.create_dataset("ep_len", data=np.asarray([2, 2], dtype=np.int32))
                handle.create_dataset("pixels", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
                handle.create_dataset("action", data=np.zeros((4, 2), dtype=np.float64))
                handle.create_dataset("qpos", data=np.ones((4, 2), dtype=np.float64))
                handle.create_dataset("qvel", data=np.full((4, 2), 2.0, dtype=np.float64))
                handle.create_dataset("observation", data=np.zeros((4, 6), dtype=np.float32))

            output = convert_reacher_h5_to_lance(source, root / "reacher.lance", progress_every=0)
            dataset = lance.dataset(output)

            self.assertEqual(dataset.count_rows(), 4)
            self.assertEqual(dataset.schema.names, ["episode_idx", "step_idx", "pixels", "action", "qpos", "qvel", "observation"])
            metadata = load_dataset_metadata(output)
            self.assertEqual(metadata["restore_spec"], "reacher_qpos_match_qpos_qvel")
            self.assertEqual(metadata["source"]["format"], "hdf5")
            self.assertEqual(metadata["source"]["path"], str(source))
            self.assertEqual(metadata["source"]["hf_dataset"], "quentinll/lewm-reacher")

    def test_ogb_cube_hdf5_conversion_writes_lance_dataset_and_metadata(self) -> None:
        import h5py
        import lance

        from mwm.data.metadata import load_dataset_metadata
        from mwm.swm.restore import validate_restore_columns
        from mwm.upstream.converters.ogb_cube import convert_ogb_cube_hdf5_to_lance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "cube_single_expert.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("ep_offset", data=np.asarray([0, 2], dtype=np.int64))
                handle.create_dataset("ep_len", data=np.asarray([2], dtype=np.int32))
                handle.create_dataset("pixels", data=np.zeros((2, 8, 8, 3), dtype=np.uint8))
                handle.create_dataset("action", data=np.zeros((2, 5), dtype=np.float32))
                handle.create_dataset("qpos", data=np.ones((2, 21), dtype=np.float64))
                handle.create_dataset("qvel", data=np.full((2, 20), 2.0, dtype=np.float64))
                handle.create_dataset("observation", data=np.zeros((2, 28), dtype=np.float32))
                handle.create_dataset("privileged/block_0_pos", data=np.full((2, 3), 0.5, dtype=np.float64))
                handle.create_dataset("privileged/block_0_quat", data=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)))

            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                output = convert_ogb_cube_hdf5_to_lance(source, "ogb_cube_single_expert.lance", progress_every=0)
            finally:
                os.chdir(old_cwd)
            self.assertEqual(output, Path("ogb_cube_single_expert.lance"))
            dataset_path = root / output
            dataset = lance.dataset(dataset_path)
            metadata = load_dataset_metadata(dataset_path)

            self.assertEqual(dataset.count_rows(), 2)
            self.assertEqual(
                dataset.schema.names,
                [
                    "episode_idx",
                    "step_idx",
                    "pixels",
                    "action",
                    "observation",
                    "qpos",
                    "qvel",
                    "privileged/block_0_pos",
                    "privileged/block_0_quat",
                ],
            )
            self.assertEqual(metadata["env_id"], "swm/OGBCube-v0")
            self.assertEqual(metadata["restore_spec"], "ogbench_cube_single_qpos_qvel_target_pose")
            self.assertEqual(metadata["action_dim"], 5)
            self.assertEqual(metadata["action_low"], [-1.0] * 5)
            self.assertEqual(metadata["action_high"], [1.0] * 5)
            validate_restore_columns(
                "swm/OGBCube-v0",
                dataset.schema.names,
                import_path="mwm.ogbench.restore.ogbench_cube_restore_spec",
            )

    def test_ogb_cube_hdf5_conversion_accepts_upstream_underscore_privileged_columns(self) -> None:
        import h5py
        import lance

        from mwm.upstream.converters.ogb_cube import convert_ogb_cube_hdf5_to_lance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "cube_single_expert.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("ep_offset", data=np.asarray([0], dtype=np.int64))
                handle.create_dataset("ep_len", data=np.asarray([2], dtype=np.int32))
                handle.create_dataset("pixels", data=np.zeros((2, 8, 8, 3), dtype=np.uint8))
                handle.create_dataset("action", data=np.zeros((2, 5), dtype=np.float32))
                handle.create_dataset("qpos", data=np.ones((2, 21), dtype=np.float64))
                handle.create_dataset("qvel", data=np.full((2, 20), 2.0, dtype=np.float64))
                handle.create_dataset("observation", data=np.zeros((2, 28), dtype=np.float32))
                handle.create_dataset("privileged_block_0_pos", data=np.full((2, 3), 0.5, dtype=np.float64))
                handle.create_dataset("privileged_block_0_quat", data=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)))

            output = convert_ogb_cube_hdf5_to_lance(source, root / "ogb_cube_single_expert.lance", progress_every=0)
            dataset = lance.dataset(output)

            self.assertEqual(
                dataset.schema.names,
                [
                    "episode_idx",
                    "step_idx",
                    "pixels",
                    "action",
                    "observation",
                    "qpos",
                    "qvel",
                    "privileged/block_0_pos",
                    "privileged/block_0_quat",
                ],
            )

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
