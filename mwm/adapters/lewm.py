from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

from hydra.utils import instantiate
import torch.nn as nn

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy
from mwm.adapters.registry import register_adapter
from mwm.models.world_model import MatryoshkaWorldModel, TransitionPackage


LEWM_BASE_ADAPTER_ARCH = "lewm_base_adapter_v1"


@dataclass(frozen=True)
class _TransitionConfigBundle:
    predictor_config: dict[str, Any]
    action_encoder_config: dict[str, Any]
    pred_proj_config: dict[str, Any]
    arch: dict[str, Any]


class LeWMStableWMAdapter:
    family = "lewm"

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

    def _validate_supported_policy(self, policy: ComponentPolicy) -> None:
        expected = self.default_policy()
        if policy != expected:
            raise ValueError(
                "Le-WM Stable-WM adapter only supports shared latent_producer and per-level transition policies."
            )

    def _base_latent_dim(self, source_config: dict[str, Any]) -> int:
        predictor_cfg = source_config.get("predictor", {})
        if not isinstance(predictor_cfg, dict):
            raise ValueError("Le-WM base latent dimension D requires a predictor config mapping.")
        dims = {
            key: int(predictor_cfg[key])
            for key in ("input_dim", "output_dim")
            if predictor_cfg.get(key) is not None
        }
        if not dims:
            raise ValueError(
                "Le-WM base latent dimension D must come from the base predictor input_dim/output_dim; "
                "it cannot be inferred from configured fidelity levels K."
            )
        if len(set(dims.values())) != 1:
            raise ValueError(f"Le-WM base predictor input/output dimensions disagree: {dims}.")
        d = next(iter(dims.values()))
        if d <= 0:
            raise ValueError(f"Le-WM base latent dimension D must be positive, got {d}.")
        return d

    def resolve_spec(
        self,
        *,
        source_config: dict[str, Any],
        source_config_sha256: str,
        training_recipe: dict[str, Any],
        levels: tuple[int, ...],
        component_policy: ComponentPolicy | None,
    ) -> StableWMBaseSpec:
        policy = component_policy or self.default_policy()
        groups = self.component_groups()
        validate_component_policy(groups, policy)
        self._validate_supported_policy(policy)
        source_copy = copy.deepcopy(source_config)
        recipe_copy = copy.deepcopy(training_recipe)
        d = self._base_latent_dim(source_copy)
        return StableWMBaseSpec(
            family=self.family,
            source_config=source_copy,
            source_config_sha256=str(source_config_sha256),
            training_recipe=recipe_copy,
            component_groups=groups,
            component_policy=policy,
            levels=tuple(int(level) for level in levels),
            D=d,
            fresh_init=True,
            loss_scope=copy.deepcopy(recipe_copy.get("loss_scope", {"regularizers": "shared_latent"})),
        )

    def build_model(self, spec: StableWMBaseSpec, **runtime: Any) -> MatryoshkaWorldModel:
        return _model_from_base_spec(spec, **runtime)


def _instantiate_module(config: dict[str, Any]) -> nn.Module:
    return instantiate(copy.deepcopy(config))


def _scale_positive_int(value: Any, k: int, D: int, minimum: int = 1) -> int:
    if int(D) <= 0:
        raise ValueError(f"D must be positive to scale Le-WM widths, got {D}.")
    return max(int(minimum), int(round(float(value) * float(k) / float(D))))


def _set_if_present(config: dict[str, Any], key: str, value: Any) -> bool:
    if key not in config:
        return False
    config[key] = value
    return True


def _level_config(
    config: dict[str, Any],
    k: int,
    D: int,
    width_keys: Sequence[str],
    scaled_keys: Sequence[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    level = copy.deepcopy(config)
    applied: dict[str, int] = {}
    for key in width_keys:
        if _set_if_present(level, key, int(k)):
            applied[key] = int(k)
    for key in scaled_keys:
        if key in level:
            value = _scale_positive_int(level[key], int(k), int(D))
            level[key] = value
            applied[key] = value
    return level, applied


def _transition_config_bundle_from_stable_config(
    k: int,
    D: int,
    source_config: dict[str, Any],
) -> _TransitionConfigBundle:
    predictor_config, predictor_widths = _level_config(
        source_config.get("predictor", {}),
        int(k),
        int(D),
        width_keys=("input_dim", "hidden_dim", "output_dim"),
        scaled_keys=("heads", "dim_head", "mlp_dim"),
    )
    action_encoder_config, action_encoder_widths = _level_config(
        source_config.get("action_encoder", {}),
        int(k),
        int(D),
        width_keys=("emb_dim", "out_dim"),
        scaled_keys=("hidden_dim",),
    )
    pred_proj_config, pred_proj_widths = _level_config(
        source_config.get("pred_proj", {"_target_": "torch.nn.Identity"}),
        int(k),
        int(D),
        width_keys=("input_dim", "output_dim"),
        scaled_keys=("hidden_dim",),
    )

    arch = {
        "K": int(k),
        "predictor_input_dim": int(predictor_widths.get("input_dim", k)),
        "predictor_hidden_dim": int(predictor_widths.get("hidden_dim", k)),
        "predictor_output_dim": int(predictor_widths.get("output_dim", k)),
        "predictor_heads": predictor_widths.get("heads"),
        "predictor_dim_head": predictor_widths.get("dim_head"),
        "predictor_mlp_dim": predictor_widths.get("mlp_dim"),
        "action_encoder_emb_dim": action_encoder_widths.get("emb_dim"),
        "action_encoder_out_dim": action_encoder_widths.get("out_dim"),
        "action_encoder_hidden_dim": action_encoder_widths.get("hidden_dim"),
        "pred_proj_input_dim": pred_proj_widths.get("input_dim"),
        "pred_proj_output_dim": pred_proj_widths.get("output_dim"),
        "pred_proj_hidden_dim": pred_proj_widths.get("hidden_dim"),
    }
    return _TransitionConfigBundle(
        predictor_config=predictor_config,
        action_encoder_config=action_encoder_config,
        pred_proj_config=pred_proj_config,
        arch=arch,
    )


def _source_component_order(source_config: dict[str, Any]) -> list[str]:
    default_order = ("encoder", "predictor", "action_encoder", "projector", "pred_proj")
    ordered: list[str] = []
    for key in source_config:
        if key in default_order and key not in ordered:
            ordered.append(key)
    for key in default_order:
        if key not in ordered:
            ordered.append(key)
    return ordered


def _instantiate_components_in_source_order(
    source_config: dict[str, Any],
    bundles: Sequence[_TransitionConfigBundle],
) -> tuple[nn.Module, nn.Module, list[TransitionPackage]]:
    encoder: nn.Module | None = None
    projector: nn.Module | None = None
    predictors: list[nn.Module | None] = [None for _ in bundles]
    action_encoders: list[nn.Module | None] = [None for _ in bundles]
    pred_projs: list[nn.Module | None] = [None for _ in bundles]

    for key in _source_component_order(source_config):
        if key == "encoder":
            encoder = _instantiate_module(source_config["encoder"])
        elif key == "projector":
            projector = _instantiate_module(source_config.get("projector", {"_target_": "torch.nn.Identity"}))
        elif key == "predictor":
            for idx, bundle in enumerate(bundles):
                predictors[idx] = _instantiate_module(bundle.predictor_config)
        elif key == "action_encoder":
            for idx, bundle in enumerate(bundles):
                action_encoders[idx] = _instantiate_module(bundle.action_encoder_config)
        elif key == "pred_proj":
            for idx, bundle in enumerate(bundles):
                pred_projs[idx] = _instantiate_module(bundle.pred_proj_config)

    if encoder is None:
        raise ValueError("Le-WM source_config must define an encoder.")
    if projector is None:
        projector = nn.Identity()

    transitions: list[TransitionPackage] = []
    for idx, (predictor, action_encoder, pred_proj) in enumerate(zip(predictors, action_encoders, pred_projs)):
        if predictor is None or action_encoder is None or pred_proj is None:
            raise ValueError(f"Le-WM source_config did not instantiate a complete transition package for level {idx}.")
        transitions.append(
            TransitionPackage(action_encoder=action_encoder, predictor=predictor, pred_proj=pred_proj)
        )
    return encoder, projector, transitions


def _validate_action_dim_from_source_config(source_config: dict[str, Any], action_dim: int) -> None:
    action_cfg = source_config.get("action_encoder", {})
    if not isinstance(action_cfg, dict):
        return
    for key in ("input_dim", "action_dim"):
        if key not in action_cfg:
            continue
        expected = int(action_cfg[key])
        if expected != int(action_dim):
            raise ValueError(
                f"Stable-WM action_encoder {key}={expected} does not match runtime action_dim={int(action_dim)}."
            )


def _model_from_base_spec(
    spec: StableWMBaseSpec,
    *,
    action_dim: int,
    action_block: int,
    image_shape: Sequence[int],
    normalize_imagenet: bool,
) -> MatryoshkaWorldModel:
    source_config = copy.deepcopy(spec.source_config)
    _validate_action_dim_from_source_config(source_config, int(action_dim))
    bundles = [
        _transition_config_bundle_from_stable_config(int(k), int(spec.D), source_config)
        for k in spec.levels
    ]
    encoder, projector, transitions = _instantiate_components_in_source_order(source_config, bundles)
    head_architectures = [copy.deepcopy(bundle.arch) for bundle in bundles]
    loss_recipe = spec.training_recipe.get("loss", {}) if isinstance(spec.training_recipe.get("loss", {}), dict) else {}
    predictor_config = source_config.get("predictor", {}) if isinstance(source_config.get("predictor", {}), dict) else {}
    history_size = int(
        spec.training_recipe.get(
            "history_size",
            loss_recipe.get("history_size", predictor_config.get("num_frames", source_config.get("history_size", 3))),
        )
    )
    num_preds = int(spec.training_recipe.get("num_preds", loss_recipe.get("num_preds", source_config.get("num_preds", 1))))

    metadata = {
        "adapter": "lewm",
        "adapter_family": "lewm",
        "architecture_version": LEWM_BASE_ADAPTER_ARCH,
        **spec.metadata(),
        "source_config": copy.deepcopy(spec.source_config),
        "training_recipe": copy.deepcopy(spec.training_recipe),
        "head_architectures": head_architectures,
        "action_preprocessing": "standard_scaler",
        "preprocessing_spec": {
            "image": "imagenet" if bool(normalize_imagenet) else "identity",
            "layout": "BCHW",
            "image_shape": [int(x) for x in image_shape],
        },
        "action_spec": {
            "dim": int(action_dim),
            "base_dim": int(action_dim) // max(1, int(action_block)),
            "block": int(action_block),
        },
    }
    model = MatryoshkaWorldModel(
        encoder=encoder,
        projector=projector,
        transitions=transitions,
        K=tuple(int(k) for k in spec.levels),
        D=int(spec.D),
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
        history_size=history_size,
        num_preds=num_preds,
        head_architectures=head_architectures,
        metadata=metadata,
        architecture_version=LEWM_BASE_ADAPTER_ARCH,
    )
    return model


register_adapter(LeWMStableWMAdapter())


__all__ = [
    "LEWM_BASE_ADAPTER_ARCH",
    "LeWMStableWMAdapter",
]
