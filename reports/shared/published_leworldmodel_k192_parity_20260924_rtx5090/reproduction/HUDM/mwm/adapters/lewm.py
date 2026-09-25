from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

from hydra.utils import instantiate
import torch.nn as nn

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy
from mwm.adapters.constants import LEWM_BASE_ADAPTER_ARCH, LEWM_SHARED_SLIMMABLE_ARCH
from mwm.adapters.registry import register_adapter
from mwm.models.lewm import LeWMMatryoshkaWorldModel
from mwm.models.decoders import ConvImageDecoder
from mwm.models.transitions import TransitionPackage
from mwm.models.slimmable import SharedSlimmableTransition


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
            "reconstructor": ComponentGroup(name="reconstructor", components=("decoder",)),
        }

    def default_policy(self) -> ComponentPolicy:
        return ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=("decoder",))

    def _validate_supported_policy(self, policy: ComponentPolicy, training_recipe: dict[str, Any]) -> None:
        expected = self.default_policy()
        legacy_without_decoder = ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=())
        shared_policy = ComponentPolicy(
            shared=("latent_producer", "transition"),
            per_level=(),
            reconstructor=("decoder",),
        )
        shared_dynamics = training_recipe.get("shared_dynamics")
        if policy == shared_policy:
            if not isinstance(shared_dynamics, dict):
                raise ValueError("Shared Le-WM transition policy requires mwm.shared_dynamics configuration.")
            _validate_shared_dynamics_config(shared_dynamics)
            return
        if shared_dynamics is not None:
            raise ValueError(
                "mwm.shared_dynamics requires shared=[latent_producer, transition], per_level=[], "
                "reconstructor=[decoder]."
            )
        if policy not in {expected, legacy_without_decoder}:
            raise ValueError(
                "Le-WM adapter only supports either the legacy per-level transition policy or the opt-in "
                "shared slimmable transition policy."
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
        source_copy = copy.deepcopy(source_config)
        recipe_copy = copy.deepcopy(training_recipe)
        self._validate_supported_policy(policy, recipe_copy)
        if isinstance(recipe_copy.get("shared_dynamics"), dict):
            recipe_copy["shared_dynamics"] = _validate_shared_dynamics_config(
                recipe_copy["shared_dynamics"]
            )
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

    def build_model(self, spec: StableWMBaseSpec, **runtime: Any) -> LeWMMatryoshkaWorldModel:
        return _model_from_base_spec(spec, **runtime)


def _instantiate_module(config: dict[str, Any]) -> nn.Module:
    return instantiate(copy.deepcopy(config))


def _validate_shared_dynamics_config(value: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(value)
    unknown = set(config) - {"architecture", "min_k", "prefix_sampling"}
    if unknown:
        raise ValueError(f"Unknown mwm.shared_dynamics keys: {sorted(unknown)}")
    architecture = str(config.get("architecture", ""))
    if architecture != "slimmable_transformer_v1":
        raise ValueError(
            f"Unsupported shared dynamics architecture {architecture!r}; expected 'slimmable_transformer_v1'."
        )
    min_k = int(config.get("min_k", 1))
    if min_k <= 0:
        raise ValueError(f"mwm.shared_dynamics.min_k must be positive, got {min_k}.")
    sampling = copy.deepcopy(config.get("prefix_sampling", {}))
    if not isinstance(sampling, dict):
        raise ValueError("mwm.shared_dynamics.prefix_sampling must be a mapping.")
    sampling_unknown = set(sampling) - {"mode", "samples_per_batch"}
    if sampling_unknown:
        raise ValueError(f"Unknown mwm.shared_dynamics.prefix_sampling keys: {sorted(sampling_unknown)}")
    mode = str(sampling.get("mode", "discrete_log_uniform_non_anchor"))
    if mode != "discrete_log_uniform_non_anchor":
        raise ValueError(f"Unsupported shared dynamics prefix sampling mode {mode!r}.")
    samples = int(sampling.get("samples_per_batch", 1))
    if samples != 1:
        raise ValueError("slimmable_transformer_v1 requires prefix_sampling.samples_per_batch=1.")
    return {
        "architecture": architecture,
        "min_k": min_k,
        "prefix_sampling": {"mode": mode, "samples_per_batch": samples},
    }


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


def _slimmable_norm_kind(config: Any) -> str:
    if config is None:
        return "identity"
    if not isinstance(config, dict):
        raise ValueError("Shared Le-WM pred_proj norm_fn must be a Hydra config mapping or null.")
    target = str(config.get("_target_", ""))
    if target.endswith("BatchNorm1d"):
        return "batch_norm"
    if target.endswith("LayerNorm"):
        return "layer_norm"
    if target.endswith("Identity"):
        return "identity"
    raise ValueError(f"Unsupported shared Le-WM pred_proj norm target {target!r}.")


def _build_shared_slimmable_transition(
    source_config: dict[str, Any],
    *,
    D: int,
    action_dim: int,
) -> SharedSlimmableTransition:
    predictor = source_config.get("predictor")
    action_encoder = source_config.get("action_encoder")
    pred_proj = source_config.get("pred_proj")
    if not isinstance(predictor, dict) or not isinstance(action_encoder, dict) or not isinstance(pred_proj, dict):
        raise ValueError("Shared Le-WM dynamics requires predictor, action_encoder, and pred_proj config mappings.")
    predictor_dims = {
        "input_dim": int(predictor.get("input_dim", D)),
        "hidden_dim": int(predictor.get("hidden_dim", D)),
        "output_dim": int(predictor.get("output_dim", predictor.get("input_dim", D))),
    }
    if set(predictor_dims.values()) != {int(D)}:
        raise ValueError(
            "Shared slimmable Le-WM requires predictor input_dim=hidden_dim=output_dim=D; "
            f"got {predictor_dims} with D={D}."
        )
    if int(action_encoder.get("input_dim", action_dim)) != int(action_dim):
        raise ValueError(
            f"Shared Le-WM action encoder input_dim={action_encoder.get('input_dim')} does not match {action_dim}."
        )
    if int(action_encoder.get("emb_dim", D)) != int(D):
        raise ValueError("Shared slimmable Le-WM requires the full-width action emb_dim to equal D.")
    for key in ("input_dim", "output_dim"):
        if int(pred_proj.get(key, D)) != int(D):
            raise ValueError(f"Shared slimmable Le-WM requires pred_proj.{key}=D={D}.")
    required_predictor = ("num_frames", "depth", "heads", "dim_head", "mlp_dim")
    missing = [key for key in required_predictor if predictor.get(key) is None]
    if missing:
        raise ValueError(f"Shared Le-WM predictor config is missing required values: {missing}.")
    return SharedSlimmableTransition(
        D=int(D),
        action_dim=int(action_dim),
        num_frames=int(predictor["num_frames"]),
        depth=int(predictor["depth"]),
        max_heads=int(predictor["heads"]),
        max_dim_head=int(predictor["dim_head"]),
        max_mlp_dim=int(predictor["mlp_dim"]),
        predictor_dropout=float(predictor.get("dropout", 0.0)),
        predictor_emb_dropout=float(predictor.get("emb_dropout", 0.0)),
        action_smoothed_dim=int(action_encoder.get("smoothed_dim", 10)),
        action_mlp_scale=int(action_encoder.get("mlp_scale", 4)),
        pred_proj_hidden_dim=int(pred_proj.get("hidden_dim", D)),
        pred_proj_norm=_slimmable_norm_kind(pred_proj.get("norm_fn")),
    )


def _model_from_base_spec(
    spec: StableWMBaseSpec,
    *,
    action_dim: int,
    action_block: int,
    image_shape: Sequence[int],
    normalize_imagenet: bool,
) -> LeWMMatryoshkaWorldModel:
    source_config = copy.deepcopy(spec.source_config)
    _validate_action_dim_from_source_config(source_config, int(action_dim))
    shared_dynamics_raw = spec.training_recipe.get("shared_dynamics")
    shared_dynamics = (
        _validate_shared_dynamics_config(shared_dynamics_raw)
        if isinstance(shared_dynamics_raw, dict)
        else None
    )
    bundles = [
        _transition_config_bundle_from_stable_config(int(k), int(spec.D), source_config)
        for k in spec.levels
    ]
    if shared_dynamics is None:
        encoder, projector, transitions = _instantiate_components_in_source_order(source_config, bundles)
        shared_transition = None
        architecture_version = LEWM_BASE_ADAPTER_ARCH
    else:
        if tuple(spec.levels) != tuple(sorted(spec.levels)):
            raise ValueError(f"Shared Le-WM anchors must be sorted, got {list(spec.levels)}.")
        if not spec.levels or int(spec.levels[-1]) != int(spec.D):
            raise ValueError(f"Shared Le-WM anchors must include D={spec.D}, got {list(spec.levels)}.")
        if int(shared_dynamics["min_k"]) > int(spec.D):
            raise ValueError(f"shared_dynamics.min_k cannot exceed D={spec.D}.")
        encoder, projector, transitions = _instantiate_components_in_source_order(source_config, ())
        shared_transition = _build_shared_slimmable_transition(
            source_config,
            D=int(spec.D),
            action_dim=int(action_dim),
        )
        architecture_version = LEWM_SHARED_SLIMMABLE_ARCH
    head_architectures = [copy.deepcopy(bundle.arch) for bundle in bundles]
    decoder_architectures = [
        {
            "K": int(k),
            "decoder": "ConvImageDecoder",
            "latent_dim": int(k),
            "image_shape": [int(x) for x in image_shape],
        }
        for k in spec.levels
    ]
    decoders = [
        ConvImageDecoder(latent_dim=int(k), image_shape=tuple(int(x) for x in image_shape))
        for k in spec.levels
    ]
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
        "architecture_version": architecture_version,
        **spec.metadata(),
        "source_config": copy.deepcopy(spec.source_config),
        "training_recipe": copy.deepcopy(spec.training_recipe),
        "head_architectures": head_architectures,
        "decoder_architectures": decoder_architectures,
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
    if shared_transition is not None and shared_dynamics is not None:
        metadata.update(
            {
                "dynamics_architecture": "slimmable_transformer_v1",
                "shared_dynamics": copy.deepcopy(shared_dynamics),
                "supported_k": {"min": int(shared_dynamics["min_k"]), "max": int(spec.D), "arbitrary": True},
                "prefix_sampling": copy.deepcopy(shared_dynamics["prefix_sampling"]),
                "shared_transition_architecture": shared_transition.architecture(),
            }
        )
    model = LeWMMatryoshkaWorldModel(
        encoder=encoder,
        projector=projector,
        transitions=transitions,
        decoders=decoders,
        K=tuple(int(k) for k in spec.levels),
        D=int(spec.D),
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
        history_size=history_size,
        num_preds=num_preds,
        head_architectures=head_architectures,
        decoder_architectures=decoder_architectures,
        metadata=metadata,
        architecture_version=architecture_version,
        shared_transition=shared_transition,
        shared_dynamics=shared_dynamics,
    )
    return model


register_adapter(LeWMStableWMAdapter())


__all__ = [
    "LEWM_BASE_ADAPTER_ARCH",
    "LEWM_SHARED_SLIMMABLE_ARCH",
    "LeWMStableWMAdapter",
]
