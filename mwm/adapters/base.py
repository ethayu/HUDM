from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch.nn as nn


@dataclass(frozen=True)
class ComponentGroup:
    name: str
    components: tuple[str, ...]
    latent_producer: bool = False


@dataclass(frozen=True)
class ComponentPolicy:
    shared: tuple[str, ...] = ("latent_producer",)
    per_level: tuple[str, ...] = ("transition",)
    reconstructor: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ComponentPolicy:
        if value is None:
            return cls()
        return cls(
            shared=tuple(str(x) for x in value.get("shared", cls.shared)),
            per_level=tuple(str(x) for x in value.get("per_level", cls.per_level)),
            reconstructor=tuple(str(x) for x in value.get("reconstructor", cls.reconstructor)),
        )

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "shared": list(self.shared),
            "per_level": list(self.per_level),
            "reconstructor": list(self.reconstructor),
        }


@dataclass(frozen=True)
class StableWMBaseSpec:
    family: str
    source_config: dict[str, Any]
    source_config_sha256: str
    training_recipe: dict[str, Any]
    component_groups: Mapping[str, ComponentGroup]
    component_policy: ComponentPolicy
    levels: tuple[int, ...]
    D: int
    fresh_init: bool = True
    loss_scope: dict[str, Any] = field(default_factory=lambda: {"regularizers": "shared_latent"})

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter_family": self.family,
            "source_config_sha256": self.source_config_sha256,
            "component_policy": self.component_policy.as_dict(),
            "levels": [int(level) for level in self.levels],
            "D": int(self.D),
            "fresh_init": bool(self.fresh_init),
            "loss_scope": dict(self.loss_scope),
        }


class StableWMBaseAdapter(Protocol):
    family: str

    def component_groups(self) -> Mapping[str, ComponentGroup]:
        raise NotImplementedError

    def default_policy(self) -> ComponentPolicy:
        raise NotImplementedError

    def resolve_spec(
        self,
        *,
        source_config: dict[str, Any],
        source_config_sha256: str,
        training_recipe: dict[str, Any],
        levels: tuple[int, ...],
        component_policy: ComponentPolicy | None,
    ) -> StableWMBaseSpec:
        raise NotImplementedError

    def build_model(self, spec: StableWMBaseSpec, **runtime: Any) -> nn.Module:
        raise NotImplementedError


def validate_component_policy(groups: Mapping[str, ComponentGroup], policy: ComponentPolicy) -> None:
    known = set(groups)
    selected = set(policy.shared) | set(policy.per_level) | set(policy.reconstructor)
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(f"Unknown component group(s): {unknown}")

    overlap = sorted(set(policy.shared) & set(policy.per_level))
    if overlap:
        raise ValueError(f"Component group(s) cannot be both shared and per-level: {overlap}")

    shared_latent = [name for name in policy.shared if groups[name].latent_producer]
    if not shared_latent:
        raise ValueError("Component policy must include at least one shared latent producer group.")


__all__ = [
    "ComponentGroup",
    "ComponentPolicy",
    "StableWMBaseAdapter",
    "StableWMBaseSpec",
    "validate_component_policy",
]
