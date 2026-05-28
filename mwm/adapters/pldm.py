from __future__ import annotations

from typing import Any

import torch.nn as nn

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy
from mwm.adapters.registry import register_adapter


class PLDMStableWMAdapter:
    family = "pldm"

    def component_groups(self) -> dict[str, ComponentGroup]:
        return {
            "latent_producer": ComponentGroup(
                name="latent_producer",
                components=("encoder", "projector"),
                latent_producer=True,
            ),
            "transition": ComponentGroup(name="transition", components=("action_encoder", "predictor", "pred_proj")),
            "reconstructor": ComponentGroup(name="reconstructor", components=()),
        }

    def default_policy(self) -> ComponentPolicy:
        return ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=())

    def resolve_spec(
        self,
        *,
        source_config: dict[str, Any],
        source_config_sha256: str,
        training_recipe: dict[str, Any],
        levels: tuple[int, ...],
        component_policy: ComponentPolicy | None,
    ) -> StableWMBaseSpec:
        del source_config, source_config_sha256, training_recipe, levels
        validate_component_policy(self.component_groups(), component_policy or self.default_policy())
        raise NotImplementedError("PLDM MWM support requires an explicit Stable-WM training recipe artifact.")

    def build_model(self, spec: StableWMBaseSpec) -> nn.Module:
        del spec
        raise NotImplementedError("PLDM MWM support requires an explicit Stable-WM training recipe artifact.")


register_adapter(PLDMStableWMAdapter())


__all__ = ["PLDMStableWMAdapter"]
