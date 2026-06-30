from __future__ import annotations

import copy
from typing import Any, Sequence

import torch.nn as nn

from mwm.adapters.base import ComponentPolicy
from mwm.adapters.registry import adapter_for_family, canonical_family, family_for_target
from mwm.adapters.stable_config import root_target


STABLE_CONFIG_TARGET = "mwm.adapters.builder.build_mwm_from_stable_config"


def _family_from_source_config(family: str | None, source_config: dict[str, Any]) -> str:
    detected = family_for_target(root_target(source_config))
    if family is None:
        return detected
    requested = canonical_family(str(family))
    if requested != detected:
        raise ValueError(f"Configured Stable-WM base family {family!r} does not match config target family {detected!r}.")
    return requested


def _set_model_config(model: Any, target: str, kwargs: dict[str, Any]) -> None:
    model.mwm_config = {"target": target, "kwargs": copy.deepcopy(kwargs)}


def build_mwm_from_stable_config(
    *,
    source_config: dict[str, Any],
    source_config_sha256: str,
    training_recipe: dict[str, Any],
    K: Sequence[int],
    action_dim: int,
    family: str | None = None,
    expected_D: int | None = None,
    action_block: int = 1,
    image_shape: Sequence[int] = (224, 224),
    normalize_imagenet: bool = True,
    component_policy: ComponentPolicy | dict[str, Any] | None = None,
) -> nn.Module:
    resolved_family = _family_from_source_config(family, source_config)
    adapter = adapter_for_family(resolved_family)
    if isinstance(component_policy, ComponentPolicy):
        policy = component_policy
    elif component_policy is None:
        policy = None
    else:
        policy = ComponentPolicy.from_mapping(component_policy)
    spec = adapter.resolve_spec(
        source_config=source_config,
        source_config_sha256=source_config_sha256,
        training_recipe=training_recipe,
        levels=tuple(int(k) for k in K),
        component_policy=policy,
    )
    if expected_D is not None and int(expected_D) != int(spec.D):
        raise ValueError(
            f"configured D={int(expected_D)} does not match base latent dimension D={int(spec.D)} "
            "from the Stable-WM source config."
        )
    model = adapter.build_model(
        spec,
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
    )
    kwargs = {
        "family": resolved_family,
        "source_config": copy.deepcopy(spec.source_config),
        "source_config_sha256": spec.source_config_sha256,
        "training_recipe": copy.deepcopy(spec.training_recipe),
        "K": [int(k) for k in spec.levels],
        "action_dim": int(action_dim),
        "action_block": int(action_block),
        "image_shape": [int(x) for x in image_shape],
        "normalize_imagenet": bool(normalize_imagenet),
        "component_policy": spec.component_policy.as_dict(),
    }
    if expected_D is not None:
        kwargs["expected_D"] = int(expected_D)
    _set_model_config(model, STABLE_CONFIG_TARGET, kwargs)
    return model


__all__ = [
    "STABLE_CONFIG_TARGET",
    "build_mwm_from_stable_config",
]
