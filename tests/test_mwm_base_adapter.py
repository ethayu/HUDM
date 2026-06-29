from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from mwm.adapters.base import ComponentGroup, ComponentPolicy, validate_component_policy
from mwm.adapters.builder import build_mwm_from_stable_config
from mwm.adapters.lewm import LeWMStableWMAdapter
from mwm.adapters.registry import family_for_target
from mwm.adapters.stable_config import load_stable_wm_config, root_target, stable_config_sha256
from mwm.checkpoint_io import load_world_model_from_checkpoint, save_world_checkpoint, validate_checkpoint_directory
from mwm.models.base_adaptive import MatryoshkaWorldModel


class FakeDINOBackbone(nn.Module):
    def __init__(self, hidden_size: int = 6, num_patches: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=int(hidden_size))
        self.hidden_size = int(hidden_size)
        self.num_patches = int(num_patches)
        self.scale = nn.Parameter(torch.arange(1, int(hidden_size) + 1, dtype=torch.float32))

    def forward(self, pixels: torch.Tensor, interpolate_pos_encoding: bool = True) -> SimpleNamespace:
        del interpolate_pos_encoding
        pooled = pixels.reshape(pixels.shape[0], -1).mean(dim=1, keepdim=True)
        patches = pooled[:, None, :] * self.scale.view(1, 1, -1)
        patches = patches.expand(-1, self.num_patches, -1)
        cls = torch.zeros(pixels.shape[0], 1, self.hidden_size, dtype=patches.dtype, device=patches.device)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, patches], dim=1))


class FakePreJEPAPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches: int,
        num_frames: int,
        dim: int,
        depth: int = 1,
        heads: int = 1,
        mlp_dim: int = 4,
        dim_head: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del num_patches, num_frames, depth, heads, mlp_dim, dim_head, dropout, emb_dropout
        self.dim = int(dim)
        self.proj = nn.Linear(int(dim), int(dim), bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AdapterPolicyTests(unittest.TestCase):
    def test_policy_requires_shared_latent_producer(self) -> None:
        groups = {
            "latent_producer": ComponentGroup(name="latent_producer", components=("encoder", "projector"), latent_producer=True),
            "transition": ComponentGroup(name="transition", components=("action_encoder", "predictor", "pred_proj")),
        }
        policy = ComponentPolicy(shared=(), per_level=("transition",), reconstructor=())

        with self.assertRaisesRegex(ValueError, "shared latent producer"):
            validate_component_policy(groups, policy)

    def test_policy_rejects_group_in_shared_and_per_level(self) -> None:
        groups = {
            "latent_producer": ComponentGroup(name="latent_producer", components=("encoder",), latent_producer=True),
        }
        policy = ComponentPolicy(shared=("latent_producer",), per_level=("latent_producer",), reconstructor=())

        with self.assertRaisesRegex(ValueError, "both shared and per-level"):
            validate_component_policy(groups, policy)

    def test_policy_rejects_unknown_group(self) -> None:
        groups = {
            "latent_producer": ComponentGroup(name="latent_producer", components=("encoder",), latent_producer=True),
        }
        policy = ComponentPolicy(shared=("latent_producer",), per_level=("missing",), reconstructor=())

        with self.assertRaisesRegex(ValueError, "Unknown component group"):
            validate_component_policy(groups, policy)

    def test_component_policy_from_mapping_and_as_dict(self) -> None:
        self.assertEqual(
            ComponentPolicy.from_mapping(None).as_dict(),
            {"shared": ["latent_producer"], "per_level": ["transition"], "reconstructor": []},
        )

        policy = ComponentPolicy.from_mapping(
            {
                "shared": ["latent_producer"],
                "per_level": ("transition", "reward_head"),
                "reconstructor": ["decoder"],
            }
        )

        self.assertEqual(policy.shared, ("latent_producer",))
        self.assertEqual(policy.per_level, ("transition", "reward_head"))
        self.assertEqual(policy.reconstructor, ("decoder",))
        self.assertEqual(
            policy.as_dict(),
            {
                "shared": ["latent_producer"],
                "per_level": ["transition", "reward_head"],
                "reconstructor": ["decoder"],
            },
        )

    def test_adapter_package_exports_public_builder_api(self) -> None:
        import importlib

        adapters = importlib.import_module("mwm.adapters")

        self.assertEqual(adapters.ComponentPolicy().as_dict()["shared"], ["latent_producer"])
        self.assertIs(adapters.build_mwm_from_stable_config, build_mwm_from_stable_config)
        self.assertIs(adapters.LeWMStableWMAdapter, LeWMStableWMAdapter)


class ConfigResolverTests(unittest.TestCase):
    def test_load_stable_wm_config_from_directory(self) -> None:
        with self.subTest("directory config"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload = {"_target_": "stable_worldmodel.wm.lewm.LeWM", "predictor": {"input_dim": 4}}
                (root / "config.json").write_text(json.dumps(payload), encoding="utf-8")

                loaded, path = load_stable_wm_config(root)

                self.assertEqual(loaded, payload)
                self.assertEqual(path, root / "config.json")
                self.assertEqual(stable_config_sha256(path), hashlib.sha256((root / "config.json").read_bytes()).hexdigest())

    def test_load_stable_wm_config_rejects_non_object_json_and_directories(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_stable_wm_config(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").mkdir()

            with self.assertRaises(FileNotFoundError):
                load_stable_wm_config(root)

    def test_root_target_reads_root_target(self) -> None:
        self.assertEqual(root_target({"_target_": "stable_worldmodel.wm.lewm.LeWM"}), "stable_worldmodel.wm.lewm.LeWM")
        with self.assertRaisesRegex(ValueError, "root _target_"):
            root_target({"_target_": ""})

    def test_family_detection_from_target(self) -> None:
        self.assertEqual(family_for_target("stable_worldmodel.wm.lewm.LeWM"), "lewm")
        self.assertEqual(family_for_target("stable_worldmodel.wm.prejepa.PreJEPA"), "prejepa")
        self.assertEqual(family_for_target("stable_worldmodel.wm.pldm.PLDM"), "pldm")
        with self.assertRaisesRegex(ValueError, "Unsupported Stable-WM target"):
            family_for_target("example.Unknown")


class LeWMStableConfigTests(unittest.TestCase):
    def _lewm_config(self) -> dict:
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
            "pred_proj": {
                "_target_": "stable_worldmodel.wm.lewm.module.MLP",
                "input_dim": 4,
                "output_dim": 4,
                "hidden_dim": 16,
                "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
            },
        }

    def test_generic_builder_dispatches_lewm_adapter_and_exports_generic_target(self) -> None:
        model = build_mwm_from_stable_config(
            family="lewm",
            source_config=self._lewm_config(),
            source_config_sha256="abc",
            training_recipe={"history_size": 2, "num_preds": 1, "loss": {"sigreg_weight": 0.0}},
            K=(4,),
            action_dim=2,
            action_block=1,
            image_shape=(8, 8),
            normalize_imagenet=False,
        )

        self.assertIsInstance(model, MatryoshkaWorldModel)
        self.assertEqual(model.metadata["adapter_family"], "lewm")
        self.assertTrue(model.metadata["fresh_init"])
        self.assertEqual(model.metadata["component_policy"]["shared"], ["latent_producer"])
        self.assertEqual(model.mwm_config["target"], "mwm.adapters.builder.build_mwm_from_stable_config")
        self.assertEqual(model.mwm_config["kwargs"]["family"], "lewm")
        self.assertEqual(model.mwm_config["kwargs"]["K"], [4])
        self.assertIs(model.encoder, model.encoder)
        self.assertEqual(len(model.transitions), 1)
        self.assertEqual(len(model.decoders), 1)
        self.assertEqual(model.metadata["component_policy"]["reconstructor"], ["decoder"])
        decoded = model.decode(0, torch.randn(2, 4))
        self.assertEqual(tuple(decoded.shape), (2, 3, 8, 8))
        out = model.training_loss({"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)})
        self.assertIn("recon_loss_l0", out)
        self.assertIn("loss", out)

    def test_resolve_spec_requires_shared_latent_producer(self) -> None:
        adapter = LeWMStableWMAdapter()

        with self.assertRaisesRegex(ValueError, "shared latent producer"):
            adapter.resolve_spec(
                source_config=self._lewm_config(),
                source_config_sha256="abc",
                training_recipe={},
                levels=(4,),
                component_policy=ComponentPolicy(shared=(), per_level=("transition",), reconstructor=()),
            )

    def test_resolve_spec_rejects_unsupported_non_default_policy(self) -> None:
        adapter = LeWMStableWMAdapter()

        with self.assertRaisesRegex(ValueError, "only supports"):
            adapter.resolve_spec(
                source_config=self._lewm_config(),
                source_config_sha256="abc",
                training_recipe={},
                levels=(2, 4),
                component_policy=ComponentPolicy(shared=("latent_producer", "transition"), per_level=(), reconstructor=()),
            )

    def test_resolve_spec_requires_base_latent_dimension_not_level_fallback(self) -> None:
        adapter = LeWMStableWMAdapter()
        source_config = self._lewm_config()
        source_config["predictor"].pop("input_dim")
        source_config["predictor"].pop("output_dim")

        with self.assertRaisesRegex(ValueError, "base latent dimension D"):
            adapter.resolve_spec(
                source_config=source_config,
                source_config_sha256="abc",
                training_recipe={},
                levels=(2,),
                component_policy=None,
            )

    def test_build_rejects_runtime_action_dim_mismatch(self) -> None:
        bad_config = self._lewm_config()
        bad_config["action_encoder"] = {
            "_target_": "tests.test_mwm_core.FakeLeWMActionEncoder",
            "action_dim": 3,
            "out_dim": 4,
        }

        with self.assertRaisesRegex(ValueError, "action_dim"):
            build_mwm_from_stable_config(
                family="lewm",
                source_config=bad_config,
                source_config_sha256="abc",
                training_recipe={},
                K=(4,),
                action_dim=2,
                action_block=1,
                image_shape=(8, 8),
                normalize_imagenet=False,
            )

    def test_config_driven_transition_widths_scale_per_level(self) -> None:
        model = build_mwm_from_stable_config(
            family="lewm",
            source_config=self._lewm_config(),
            source_config_sha256="abc",
            training_recipe={},
            K=(2, 4),
            action_dim=2,
            action_block=1,
            image_shape=(8, 8),
            normalize_imagenet=False,
        )

        self.assertEqual(model.K, [2, 4])
        self.assertEqual(model.D, 4)
        self.assertEqual(model.transitions[0].action_encoder.proj.out_features, 2)
        self.assertEqual(model.transitions[1].action_encoder.proj.out_features, 4)
        self.assertEqual([h["predictor_input_dim"] for h in model.metadata["head_architectures"]], [2, 4])
        self.assertEqual([h["pred_proj_hidden_dim"] for h in model.metadata["head_architectures"]], [8, 16])
        self.assertEqual(model.transitions[0].pred_proj.net[0].out_features, 8)
        self.assertEqual(model.transitions[1].pred_proj.net[0].out_features, 16)

    def test_k_equals_d_preserves_base_internal_widths_and_top_level_recipe_shape(self) -> None:
        model = build_mwm_from_stable_config(
            family="lewm",
            source_config=self._lewm_config(),
            source_config_sha256="abc",
            training_recipe={"history_size": 5, "num_preds": 2},
            K=(4,),
            action_dim=2,
            action_block=1,
            image_shape=(8, 8),
            normalize_imagenet=False,
        )

        self.assertEqual(model.history_size, 5)
        self.assertEqual(model.num_preds, 2)
        self.assertEqual(model.metadata["head_architectures"][0]["pred_proj_hidden_dim"], 16)
        self.assertEqual(model.transitions[0].pred_proj.net[0].out_features, 16)


class PreJEPAStableConfigTests(unittest.TestCase):
    def _prejepa_config(self) -> dict:
        return {
            "_target_": "stable_worldmodel.wm.prejepa.PreJEPA",
            "history_size": 2,
            "num_pred": 1,
            "interpolate_pos_encoding": True,
            "encoder": {
                "_target_": "tests.test_mwm_base_adapter.FakeDINOBackbone",
                "hidden_size": 6,
                "num_patches": 4,
            },
            "predictor": {
                "_target_": "tests.test_mwm_base_adapter.FakePreJEPAPredictor",
                "num_patches": 4,
                "num_frames": 2,
                "dim": 10,
                "depth": 1,
                "heads": 2,
                "mlp_dim": 12,
                "dim_head": 2,
                "dropout": 0.0,
                "emb_dropout": 0.0,
            },
            "extra_encoders": {
                "_target_": "torch.nn.ModuleDict",
                "modules": {
                    "proprio": {
                        "_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                        "in_chans": 3,
                        "emb_dim": 2,
                    },
                    "action": {
                        "_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                        "in_chans": 2,
                        "emb_dim": 2,
                    },
                },
            },
        }

    def test_dino_alias_builds_canonical_prejepa_adapter_with_fixed_extras(self) -> None:
        model = build_mwm_from_stable_config(
            family="dino",
            source_config=self._prejepa_config(),
            source_config_sha256="prejepa-sha",
            training_recipe={"history_size": 2, "num_preds": 1, "backbone": {"name": "dinov2_small"}},
            K=(3, 6),
            action_dim=2,
            action_block=1,
            image_shape=(4, 4),
            normalize_imagenet=False,
        )

        self.assertIsInstance(model, MatryoshkaWorldModel)
        self.assertEqual(model.metadata["adapter_family"], "prejepa")
        self.assertEqual(model.mwm_config["kwargs"]["family"], "prejepa")
        self.assertEqual(model.D, 6)
        self.assertEqual(model.metadata["D_visual"], 6)
        self.assertEqual(model.metadata["extra_dims"], {"proprio": 2, "action": 2})
        self.assertEqual(model.metadata["extra_order"], ["proprio", "action"])
        self.assertEqual(model.metadata["level_dims"], [7, 10])
        self.assertEqual([head["predictor_dim"] for head in model.metadata["head_architectures"]], [7, 10])
        self.assertEqual([transition.dim for transition in model.transitions], [7, 10])

    def test_prejepa_loss_excludes_action_slice_but_logs_pixel_and_proprio_losses(self) -> None:
        model = build_mwm_from_stable_config(
            family="prejepa",
            source_config=self._prejepa_config(),
            source_config_sha256="prejepa-sha",
            training_recipe={"history_size": 2, "num_preds": 1},
            K=(6,),
            action_dim=2,
            action_block=1,
            image_shape=(4, 4),
            normalize_imagenet=False,
        )
        batch = {
            "pixels": torch.zeros(1, 3, 3, 4, 4),
            "proprio": torch.ones(1, 3, 3),
            "action": torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]]),
        }

        out = model.training_loss(batch)

        self.assertTrue(torch.allclose(out["loss"], torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(out["pred_loss_l0"], torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(out["pixels_loss_l0"], torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(out["proprio_loss_l0"], torch.tensor(0.0), atol=1e-6))
        self.assertNotIn("action_loss_l0", out)

    def test_prejepa_rollout_cost_uses_patch_pixels_and_non_action_goal_extras(self) -> None:
        model = build_mwm_from_stable_config(
            family="prejepa",
            source_config=self._prejepa_config(),
            source_config_sha256="prejepa-sha",
            training_recipe={"history_size": 2, "num_preds": 1},
            K=(3,),
            action_dim=2,
            action_block=1,
            image_shape=(4, 4),
            normalize_imagenet=False,
        )
        infos = {
            "pixels": torch.zeros(1, 2, 2, 3, 4, 4),
            "goal": torch.zeros(1, 2, 2, 3, 4, 4),
            "proprio": torch.zeros(1, 2, 2, 3),
            "goal_proprio": torch.zeros(1, 2, 2, 3),
        }
        candidates = torch.zeros(1, 2, 3, 2)
        decision = SimpleNamespace(base_level_idx=0, rollout_level_indices=[0, 0, 0])

        cost = model.get_cost_with_fidelity(infos, candidates, decision)

        self.assertEqual(tuple(cost.shape), (1, 2))
        self.assertTrue(torch.isfinite(cost).all())
        self.assertEqual(model._last_cost_diagnostics["terminal_k"], 3)
        self.assertEqual(model._last_cost_diagnostics["num_patches"], 4)

    def test_prejepa_dynamic_rollout_keeps_fixed_extras_and_scores_terminal_level(self) -> None:
        model = build_mwm_from_stable_config(
            family="prejepa",
            source_config=self._prejepa_config(),
            source_config_sha256="prejepa-sha",
            training_recipe={"history_size": 2, "num_preds": 1},
            K=(3, 6),
            action_dim=2,
            action_block=1,
            image_shape=(4, 4),
            normalize_imagenet=False,
        )
        infos = {
            "pixels": torch.zeros(1, 2, 2, 3, 4, 4),
            "goal": torch.zeros(1, 2, 2, 3, 4, 4),
            "proprio": torch.zeros(1, 2, 2, 3),
            "goal_proprio": torch.zeros(1, 2, 2, 3),
        }
        candidates = torch.zeros(1, 2, 4, 2)
        decision = SimpleNamespace(base_level_idx=1, rollout_level_indices=[1, 0, 0, 0])

        cost = model.get_cost_with_fidelity(infos, candidates, decision)

        self.assertEqual(tuple(cost.shape), (1, 2))
        self.assertTrue(torch.isfinite(cost).all())
        self.assertEqual(model._last_cost_diagnostics["base_level_idx"], 1)
        self.assertEqual(model._last_cost_diagnostics["terminal_level_idx"], 0)
        self.assertEqual(model._last_cost_diagnostics["terminal_k"], 3)
        self.assertEqual(model._last_cost_diagnostics["level_dim"], 7)
        self.assertEqual(tuple(infos["predicted_pixels_emb"].shape), (1, 2, 5, 4, 3))
        self.assertEqual(tuple(infos["predicted_proprio_emb"].shape), (1, 2, 5, 4, 2))
        self.assertEqual(tuple(infos["predicted_action_emb"].shape), (1, 2, 5, 4, 2))

    def test_prejepa_dynamics_flop_audit_profiles_active_predictor(self) -> None:
        model = build_mwm_from_stable_config(
            family="prejepa",
            source_config=self._prejepa_config(),
            source_config_sha256="prejepa-sha",
            training_recipe={"history_size": 2, "num_preds": 1},
            K=(3, 6),
            action_dim=2,
            action_block=1,
            image_shape=(4, 4),
            normalize_imagenet=False,
        )
        infos = {
            "pixels": torch.zeros(1, 1, 2, 3, 4, 4),
            "goal": torch.zeros(1, 1, 2, 3, 4, 4),
            "proprio": torch.zeros(1, 1, 2, 3),
            "goal_proprio": torch.zeros(1, 1, 2, 3),
        }
        candidates = torch.zeros(1, 1, 3, 2)
        decision = SimpleNamespace(
            base_level_idx=1,
            rollout_level_indices=[1, 0, 0],
            metadata={"flop_accounting": "dynamics_audit"},
        )

        model.get_cost_with_fidelity(infos, candidates, decision)

        self.assertGreater(model._last_cost_diagnostics["dynamics_flops"], 0)
        self.assertEqual(model._last_cost_diagnostics["flop_accounting"], "dynamics_audit")

    def test_prejepa_checkpoint_round_trips_through_generic_builder(self) -> None:
        import tempfile

        model = build_mwm_from_stable_config(
            family="dinowm",
            source_config=self._prejepa_config(),
            source_config_sha256="prejepa-sha",
            training_recipe={"history_size": 2, "num_preds": 1, "backbone": {"name": "dinov2_small"}},
            K=(3, 6),
            action_dim=2,
            action_block=1,
            image_shape=(4, 4),
            normalize_imagenet=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_world_checkpoint(model, tmp)
            config, metadata = validate_checkpoint_directory(tmp, strict_metadata=True)
            loaded, loaded_metadata, epoch = load_world_model_from_checkpoint(tmp, None, torch.device("cpu"))

        self.assertEqual(epoch, 0)
        self.assertEqual(config["kwargs"]["family"], "prejepa")
        self.assertEqual(metadata["adapter_family"], "prejepa")
        self.assertEqual(metadata["D"], 6)
        self.assertEqual(metadata["D_visual"], 6)
        self.assertEqual(metadata["level_dims"], [7, 10])
        self.assertEqual(loaded_metadata["extra_dims"], {"proprio": 2, "action": 2})
        self.assertEqual(loaded.metadata["adapter_family"], "prejepa")
        self.assertEqual([transition.dim for transition in loaded.transitions], [7, 10])

    def test_prejepa_rejects_missing_action_encoder(self) -> None:
        config = self._prejepa_config()
        config["extra_encoders"]["modules"].pop("action")

        with self.assertRaisesRegex(ValueError, "requires an action extra encoder"):
            build_mwm_from_stable_config(
                family="prejepa",
                source_config=config,
                source_config_sha256="prejepa-sha",
                training_recipe={},
                K=(6,),
                action_dim=2,
                image_shape=(4, 4),
                normalize_imagenet=False,
            )

    def test_prejepa_rejects_cnn_fallback_encoder_paths(self) -> None:
        config = self._prejepa_config()
        config["encoder"] = {
            "_target_": "stable_worldmodel.wm.prejepa.module.create_backbone",
            "name": "microsoft/resnet-18",
        }

        with self.assertRaisesRegex(ValueError, "not CNN fallbacks"):
            build_mwm_from_stable_config(
                family="prejepa",
                source_config=config,
                source_config_sha256="prejepa-sha",
                training_recipe={},
                K=(6,),
                action_dim=2,
                image_shape=(4, 4),
                normalize_imagenet=False,
            )


class UnsupportedAdapterTests(unittest.TestCase):
    def test_generic_builder_dispatches_unsupported_adapters_without_lewm_special_case(self) -> None:
        cases = (
            ("pldm", "stable_worldmodel.wm.pldm.PLDM"),
        )
        for family, target in cases:
            with self.subTest(family=family):
                with self.assertRaisesRegex(ValueError, "Unsupported Stable-WM target family"):
                    build_mwm_from_stable_config(
                        family=family,
                        source_config={"_target_": target},
                        source_config_sha256="abc",
                        training_recipe={},
                        K=(4,),
                        action_dim=2,
                        image_shape=(8, 8),
                        normalize_imagenet=False,
                    )
