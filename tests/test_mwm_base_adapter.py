from __future__ import annotations

import unittest
from unittest import mock

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy


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
