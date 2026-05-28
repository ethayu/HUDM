from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy
from mwm.adapters.lewm import LeWMStableWMAdapter, build_mwm_lewm_from_stable_config
import mwm.adapters.registry as adapter_registry
from mwm.adapters.registry import adapter_for_family, adapter_for_target, family_for_target, register_adapter
from mwm.adapters.stable_config import load_stable_wm_config, root_target, stable_config_sha256
from mwm.models.world_model import MatryoshkaWorldModel


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

    def test_base_spec_stores_fresh_init_and_loss_scope(self) -> None:
        groups = {
            "latent_producer": ComponentGroup(name="latent_producer", components=("encoder",), latent_producer=True),
            "transition": ComponentGroup(name="transition", components=("predictor",)),
        }
        policy = ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=())
        spec = StableWMBaseSpec(
            family="lewm",
            source_config={"_target_": "stable_worldmodel.wm.lewm.LeWM"},
            source_config_sha256="abc123",
            training_recipe={"loss": {"sigreg_weight": 0.09}},
            component_groups=groups,
            component_policy=policy,
            levels=(4,),
            D=4,
            fresh_init=True,
            loss_scope={"regularizers": "shared_latent"},
        )

        self.assertTrue(spec.fresh_init)
        self.assertEqual(spec.component_policy.shared, ("latent_producer",))
        self.assertEqual(spec.loss_scope["regularizers"], "shared_latent")
        self.assertEqual(
            spec.metadata(),
            {
                "adapter_family": "lewm",
                "source_config_sha256": "abc123",
                "component_policy": {
                    "shared": ["latent_producer"],
                    "per_level": ["transition"],
                    "reconstructor": [],
                },
                "levels": [4],
                "D": 4,
                "fresh_init": True,
                "loss_scope": {"regularizers": "shared_latent"},
            },
        )

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

    def test_adapter_package_exports_base_api_without_lewm_module(self) -> None:
        import importlib
        import sys

        class BlockLeWMImport:
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "mwm.adapters.lewm":
                    raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
                return None

        original_modules = dict(sys.modules)
        sys.modules.pop("mwm.adapters", None)
        sys.modules.pop("mwm.adapters.lewm", None)
        try:
            with mock.patch.object(sys, "meta_path", [BlockLeWMImport(), *sys.meta_path]):
                adapters = importlib.import_module("mwm.adapters")
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

        self.assertEqual(adapters.ComponentPolicy().as_dict()["shared"], ["latent_producer"])
        self.assertIn("ComponentPolicy", adapters.__all__)

    def test_adapter_package_reexports_available_lewm_public_api(self) -> None:
        import importlib
        import sys

        sys.modules.pop("mwm.adapters", None)
        adapters = importlib.import_module("mwm.adapters")
        lewm = importlib.import_module("mwm.adapters.lewm")

        for name in lewm.__all__:
            self.assertIn(name, adapters.__all__)
            self.assertIs(getattr(adapters, name), getattr(lewm, name))


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

    def test_registry_returns_registered_adapter(self) -> None:
        class DummyAdapter:
            family = "dummy"

        original_adapters = dict(adapter_registry._ADAPTERS)
        try:
            register_adapter(DummyAdapter())
            self.assertEqual(adapter_for_family("dummy").family, "dummy")
            self.assertEqual(adapter_for_target("dummy").family, "dummy")
        finally:
            adapter_registry._ADAPTERS.clear()
            adapter_registry._ADAPTERS.update(original_adapters)


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

    def test_lewm_adapter_declares_groups(self) -> None:
        adapter = LeWMStableWMAdapter()
        groups = adapter.component_groups()
        self.assertEqual(groups["latent_producer"].components, ("encoder", "projector"))
        self.assertTrue(groups["latent_producer"].latent_producer)
        self.assertEqual(groups["transition"].components, ("action_encoder", "predictor", "pred_proj"))

    def test_build_from_stable_config_fresh_initializes_without_weights(self) -> None:
        model = build_mwm_lewm_from_stable_config(
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
        out = model.training_loss({"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)})
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

    def test_build_rejects_runtime_action_dim_mismatch(self) -> None:
        bad_config = self._lewm_config()
        bad_config["action_encoder"] = {
            "_target_": "tests.test_mwm_core.FakeLeWMActionEncoder",
            "action_dim": 3,
            "out_dim": 4,
        }

        with self.assertRaisesRegex(ValueError, "action_dim"):
            build_mwm_lewm_from_stable_config(
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
        model = build_mwm_lewm_from_stable_config(
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
        model = build_mwm_lewm_from_stable_config(
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


class UnsupportedAdapterTests(unittest.TestCase):
    def test_prejepa_adapter_declares_groups_but_requires_recipe(self) -> None:
        from mwm.adapters.prejepa import PreJEPAStableWMAdapter

        adapter = PreJEPAStableWMAdapter()
        groups = adapter.component_groups()
        self.assertEqual(groups["latent_producer"].components, ("backbone",))
        self.assertEqual(groups["transition"].components, ("predictor", "extra_encoders"))
        with self.assertRaisesRegex(NotImplementedError, "training recipe"):
            adapter.resolve_spec(
                source_config={"_target_": "stable_worldmodel.wm.prejepa.PreJEPA"},
                source_config_sha256="abc",
                training_recipe={},
                levels=(4,),
                component_policy=None,
            )

    def test_pldm_adapter_declares_groups_but_requires_recipe(self) -> None:
        from mwm.adapters.pldm import PLDMStableWMAdapter

        adapter = PLDMStableWMAdapter()
        groups = adapter.component_groups()
        self.assertEqual(groups["latent_producer"].components, ("encoder", "projector"))
        self.assertEqual(groups["transition"].components, ("action_encoder", "predictor", "pred_proj"))
        with self.assertRaisesRegex(NotImplementedError, "training recipe"):
            adapter.resolve_spec(
                source_config={"_target_": "stable_worldmodel.wm.pldm.PLDM"},
                source_config_sha256="abc",
                training_recipe={},
                levels=(4,),
                component_policy=None,
            )
