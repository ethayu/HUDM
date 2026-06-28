from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

from hydra.utils import instantiate
import torch.nn as nn

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy
from mwm.adapters.constants import PREJEPA_DINO_ADAPTER_ARCH
from mwm.adapters.registry import register_adapter
from mwm.models.base_adaptive import MatryoshkaWorldModel
from mwm.models.prejepa import PreJEPALevelPredictor, PreJEPARuntimeStrategy


@dataclass(frozen=True)
class _PreJEPAShapeSpec:
    predictor_dim: int
    D_visual: int
    extra_dims: dict[str, int]
    extra_input_dims: dict[str, int]
    extra_order: list[str]
    num_patches: int

    @property
    def extra_total_dim(self) -> int:
        return sum(int(value) for value in self.extra_dims.values())


class PreJEPAStableWMAdapter:
    family = "prejepa"

    def component_groups(self) -> dict[str, ComponentGroup]:
        return {
            "latent_producer": ComponentGroup(
                name="latent_producer",
                components=("encoder",),
                latent_producer=True,
            ),
            "transition": ComponentGroup(name="transition", components=("predictor", "extra_encoders")),
            "reconstructor": ComponentGroup(name="reconstructor", components=()),
        }

    def default_policy(self) -> ComponentPolicy:
        return ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=())

    def _validate_supported_policy(self, policy: ComponentPolicy) -> None:
        expected = self.default_policy()
        if policy != expected:
            raise ValueError(
                "PreJEPA/DINO-WM adapter only supports a shared image latent producer and per-level "
                "predictors with shared fixed extra encoders."
            )

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
        shape = _prejepa_shape_spec(source_copy)
        bad_levels = [int(level) for level in levels if int(level) <= 0 or int(level) > shape.D_visual]
        if bad_levels:
            raise ValueError(f"PreJEPA K levels must be in [1, D_visual={shape.D_visual}], got {bad_levels}.")
        return StableWMBaseSpec(
            family=self.family,
            source_config=source_copy,
            source_config_sha256=str(source_config_sha256),
            training_recipe=recipe_copy,
            component_groups=groups,
            component_policy=policy,
            levels=tuple(int(level) for level in levels),
            D=int(shape.D_visual),
            fresh_init=True,
            loss_scope=copy.deepcopy(recipe_copy.get("loss_scope", {"regularizers": "unsupported"})),
        )

    def build_model(self, spec: StableWMBaseSpec, **runtime: Any) -> MatryoshkaWorldModel:
        return _model_from_prejepa_spec(spec, **runtime)


def _target(config: dict[str, Any], *, label: str) -> str:
    target = str(config.get("_target_", ""))
    if not target:
        raise ValueError(f"PreJEPA source_config.{label} must define _target_.")
    return target


def _reject_video_or_cnn_paths(source_config: dict[str, Any]) -> None:
    if bool(source_config.get("is_video_encoder", False)) or bool(source_config.get("video_encoder", False)):
        raise ValueError("PreJEPA/DINO-WM adapter v1 supports image encoders only; video encoders are unsupported.")
    encoder_cfg = source_config.get("encoder", {})
    if not isinstance(encoder_cfg, dict):
        raise ValueError("PreJEPA source_config must define an encoder config mapping.")
    target = _target(encoder_cfg, label="encoder").lower()
    name = str(encoder_cfg.get("name", "")).lower()
    if "cnn" in target or "resnet" in target or "cnn" in name or "resnet" in name:
        raise ValueError("PreJEPA/DINO-WM adapter v1 requires transformer image patch backbones, not CNN fallbacks.")


def _extra_modules(source_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extra_cfg = source_config.get("extra_encoders")
    if not isinstance(extra_cfg, dict):
        raise ValueError("PreJEPA source_config must define extra_encoders as a mapping.")
    modules = extra_cfg.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("PreJEPA source_config.extra_encoders.modules must be a nonempty mapping.")
    normalized: dict[str, dict[str, Any]] = {}
    for key, config in modules.items():
        if not isinstance(config, dict):
            raise ValueError(f"PreJEPA extra encoder {key!r} must be a config mapping.")
        target = _target(config, label=f"extra_encoders.modules.{key}")
        if not target.endswith(".Embedder"):
            raise ValueError(f"PreJEPA/DINO-WM v1 only supports Embedder extras, got {target!r} for {key!r}.")
        if "in_chans" not in config or "emb_dim" not in config:
            raise ValueError(f"PreJEPA extra encoder {key!r} must define in_chans and emb_dim.")
        normalized[str(key)] = copy.deepcopy(config)
    if "action" not in normalized:
        raise ValueError("PreJEPA/DINO-WM v1 requires an action extra encoder.")
    return normalized


def _prejepa_shape_spec(source_config: dict[str, Any]) -> _PreJEPAShapeSpec:
    root_target = _target(source_config, label="<root>")
    if not root_target.lower().endswith(".prejepa"):
        raise ValueError(f"PreJEPA adapter requires a PreJEPA source model target, got {root_target!r}.")
    _reject_video_or_cnn_paths(source_config)
    predictor_cfg = source_config.get("predictor")
    if not isinstance(predictor_cfg, dict):
        raise ValueError("PreJEPA source_config must define a predictor config mapping.")
    target = _target(predictor_cfg, label="predictor")
    if not target.endswith(".CausalPredictor") and not target.endswith(".FakePreJEPAPredictor"):
        raise ValueError(f"PreJEPA/DINO-WM v1 requires CausalPredictor-compatible predictors, got {target!r}.")
    if "dim" not in predictor_cfg:
        raise ValueError("PreJEPA predictor config must define dim.")
    if "num_patches" not in predictor_cfg:
        raise ValueError("PreJEPA predictor config must define num_patches.")
    modules = _extra_modules(source_config)
    extra_order = list(modules)
    extra_dims = {key: int(config["emb_dim"]) for key, config in modules.items()}
    extra_input_dims = {key: int(config["in_chans"]) for key, config in modules.items()}
    predictor_dim = int(predictor_cfg["dim"])
    extra_total = sum(extra_dims.values())
    D_visual = predictor_dim - extra_total
    if D_visual <= 0:
        raise ValueError(
            f"PreJEPA predictor dim={predictor_dim} must exceed extra embedding total dim={extra_total}."
        )
    num_patches = int(predictor_cfg["num_patches"])
    if num_patches <= 0:
        raise ValueError(f"PreJEPA predictor num_patches must be positive, got {num_patches}.")
    return _PreJEPAShapeSpec(
        predictor_dim=predictor_dim,
        D_visual=D_visual,
        extra_dims=extra_dims,
        extra_input_dims=extra_input_dims,
        extra_order=extra_order,
        num_patches=num_patches,
    )


def _instantiate_module(config: dict[str, Any]) -> nn.Module:
    return instantiate(copy.deepcopy(config))


def _instantiate_extra_encoders(source_config: dict[str, Any]) -> nn.ModuleDict:
    modules = _extra_modules(source_config)
    return nn.ModuleDict({key: _instantiate_module(config) for key, config in modules.items()})


def _scale_positive_int(value: Any, level_dim: int, base_dim: int, minimum: int = 1) -> int:
    if int(base_dim) <= 0:
        raise ValueError(f"base_dim must be positive to scale PreJEPA widths, got {base_dim}.")
    return max(int(minimum), int(round(float(value) * float(level_dim) / float(base_dim))))


def _level_predictor_config(
    predictor_config: dict[str, Any],
    *,
    level_dim: int,
    base_dim: int,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    level = copy.deepcopy(predictor_config)
    level["dim"] = int(level_dim)
    applied: dict[str, int | None] = {
        "predictor_dim": int(level_dim),
        "predictor_heads": None,
        "predictor_dim_head": None,
        "predictor_mlp_dim": None,
    }
    for key, arch_key in (("heads", "predictor_heads"), ("dim_head", "predictor_dim_head"), ("mlp_dim", "predictor_mlp_dim")):
        if key in level:
            if int(level_dim) == int(base_dim):
                value = int(level[key])
            else:
                value = _scale_positive_int(level[key], int(level_dim), int(base_dim))
            level[key] = value
            applied[arch_key] = value
    return level, applied


def _encoder_visual_dim(encoder: nn.Module) -> int | None:
    config = getattr(encoder, "config", None)
    for obj in (config, encoder):
        for key in ("hidden_size", "embed_dim", "dim"):
            value = getattr(obj, key, None)
            if value is not None:
                return int(value)
    return None


def _backbone_name(source_config: dict[str, Any], training_recipe: dict[str, Any]) -> str | None:
    encoder_cfg = source_config.get("encoder", {}) if isinstance(source_config.get("encoder", {}), dict) else {}
    if encoder_cfg.get("name") is not None:
        return str(encoder_cfg["name"])
    backbone = training_recipe.get("backbone", {}) if isinstance(training_recipe.get("backbone", {}), dict) else {}
    if backbone.get("name") is not None:
        return str(backbone["name"])
    return None


def _validate_action_dim(source_config: dict[str, Any], action_dim: int) -> None:
    modules = _extra_modules(source_config)
    action_dim_cfg = int(modules["action"]["in_chans"])
    if action_dim_cfg != int(action_dim):
        raise ValueError(
            f"PreJEPA action extra encoder in_chans={action_dim_cfg} does not match runtime action_dim={int(action_dim)}."
        )


def _history_size(source_config: dict[str, Any], training_recipe: dict[str, Any]) -> int:
    predictor_cfg = source_config.get("predictor", {}) if isinstance(source_config.get("predictor", {}), dict) else {}
    loss_cfg = training_recipe.get("loss", {}) if isinstance(training_recipe.get("loss", {}), dict) else {}
    return int(
        training_recipe.get(
            "history_size",
            loss_cfg.get("history_size", predictor_cfg.get("num_frames", source_config.get("history_size", 3))),
        )
    )


def _num_preds(source_config: dict[str, Any], training_recipe: dict[str, Any]) -> int:
    loss_cfg = training_recipe.get("loss", {}) if isinstance(training_recipe.get("loss", {}), dict) else {}
    return int(training_recipe.get("num_preds", loss_cfg.get("num_preds", source_config.get("num_pred", source_config.get("num_preds", 1)))))


def _model_from_prejepa_spec(
    spec: StableWMBaseSpec,
    *,
    action_dim: int,
    action_block: int,
    image_shape: Sequence[int],
    normalize_imagenet: bool,
) -> MatryoshkaWorldModel:
    source_config = copy.deepcopy(spec.source_config)
    shape = _prejepa_shape_spec(source_config)
    _validate_action_dim(source_config, int(action_dim))
    encoder = _instantiate_module(source_config["encoder"])
    encoder_dim = _encoder_visual_dim(encoder)
    if encoder_dim is not None and int(encoder_dim) != int(shape.D_visual):
        raise ValueError(
            f"PreJEPA encoder visual dim={int(encoder_dim)} does not match predictor-derived D_visual={shape.D_visual}."
        )
    predictor_cfg = source_config["predictor"]
    base_dim = int(shape.predictor_dim)
    predictors: list[PreJEPALevelPredictor] = []
    head_architectures: list[dict[str, Any]] = []
    level_dims: list[int] = []
    for k in spec.levels:
        level_dim = int(k) + int(shape.extra_total_dim)
        level_dims.append(level_dim)
        level_cfg, arch = _level_predictor_config(
            predictor_cfg,
            level_dim=level_dim,
            base_dim=base_dim,
        )
        predictor = _instantiate_module(level_cfg)
        predictors.append(
            PreJEPALevelPredictor(
                predictor,
                dim=level_dim,
                num_patches=shape.num_patches,
            )
        )
        head_architectures.append(
            {
                "K_visual": int(k),
                "level_dim": int(level_dim),
                "predictor_dim": int(arch["predictor_dim"]),
                "predictor_heads": arch["predictor_heads"],
                "predictor_dim_head": arch["predictor_dim_head"],
                "predictor_mlp_dim": arch["predictor_mlp_dim"],
                "num_patches": int(shape.num_patches),
                "fixed_extra_dims": copy.deepcopy(shape.extra_dims),
            }
        )

    extra_encoders = _instantiate_extra_encoders(source_config)
    strategy = PreJEPARuntimeStrategy(
        extra_encoders=extra_encoders,
        extra_order=shape.extra_order,
        extra_dims=shape.extra_dims,
        extra_input_dims=shape.extra_input_dims,
        visual_dim=shape.D_visual,
        num_patches=shape.num_patches,
        action_key="action",
        interpolate_pos_encoding=bool(source_config.get("interpolate_pos_encoding", True)),
    )
    history_size = _history_size(source_config, spec.training_recipe)
    num_preds = _num_preds(source_config, spec.training_recipe)
    backbone_name = _backbone_name(source_config, spec.training_recipe)
    decoder_architectures = [
        {
            "K_visual": int(k),
            "decoder": None,
            "latent_type": "prejepa_patch_latent",
            "num_patches": int(shape.num_patches),
            "level_dim": int(level_dim),
        }
        for k, level_dim in zip(spec.levels, level_dims)
    ]
    metadata = {
        "adapter": "prejepa",
        "adapter_family": "prejepa",
        "architecture_version": PREJEPA_DINO_ADAPTER_ARCH,
        **spec.metadata(),
        "source_config": copy.deepcopy(spec.source_config),
        "training_recipe": copy.deepcopy(spec.training_recipe),
        "source_model_target": str(source_config.get("_target_")),
        "source_predictor_target": str(predictor_cfg.get("_target_")),
        "source_backbone_target": str(source_config.get("encoder", {}).get("_target_")),
        "backbone_name": backbone_name,
        "D_visual": int(shape.D_visual),
        "full_dim": int(shape.predictor_dim),
        "extra_dims": copy.deepcopy(shape.extra_dims),
        "extra_input_dims": copy.deepcopy(shape.extra_input_dims),
        "extra_order": list(shape.extra_order),
        "level_dims": [int(value) for value in level_dims],
        "num_patches": int(shape.num_patches),
        "patch_size": int(spec.training_recipe.get("patch_size", 14)),
        "fixed_extra_policy": "visual_prefix_only_full_extras_every_level",
        "head_architectures": head_architectures,
        "decoder_architectures": decoder_architectures,
        "action_preprocessing": "standard_scaler",
        "preprocessing_spec": {
            "image": "imagenet" if bool(normalize_imagenet) else "identity",
            "layout": "BCHW",
            "image_shape": [int(x) for x in image_shape],
            "recipe": "dinov2_image" if backbone_name and "dino" in backbone_name.lower() else "prejepa_image",
        },
        "action_spec": {
            "dim": int(action_dim),
            "base_dim": int(action_dim) // max(1, int(action_block)),
            "block": int(action_block),
            "extra_embedding_dim": int(shape.extra_dims["action"]),
        },
    }
    return MatryoshkaWorldModel(
        encoder=encoder,
        projector=nn.Identity(),
        transitions=predictors,
        decoders=[nn.Identity() for _ in spec.levels],
        K=tuple(int(k) for k in spec.levels),
        D=int(shape.D_visual),
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
        history_size=history_size,
        num_preds=num_preds,
        head_architectures=head_architectures,
        decoder_architectures=decoder_architectures,
        metadata=metadata,
        architecture_version=PREJEPA_DINO_ADAPTER_ARCH,
        runtime_strategy=strategy,
    )


register_adapter(PreJEPAStableWMAdapter())


__all__ = [
    "PREJEPA_DINO_ADAPTER_ARCH",
    "PreJEPAStableWMAdapter",
]
