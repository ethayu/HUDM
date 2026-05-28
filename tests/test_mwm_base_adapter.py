from __future__ import annotations

import unittest

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
