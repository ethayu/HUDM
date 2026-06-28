from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mwm.training.lewm_config import as_container, validate_lewm_loss_config


def resolve_lewm_base_adapter_model_cfg(cfg: Any, dataset: Any) -> dict[str, Any]:
    frameskip = int(cfg.data.get("frameskip", 1))
    action_dim = int(dataset.get_dim(str(cfg.data.action_key))) * frameskip
    image_shape_cfg = cfg.model.get("image_shape", "auto")
    image_shape = (
        (int(cfg.model.get("image_size", 224)), int(cfg.model.get("image_size", 224)))
        if str(image_shape_cfg).lower() == "auto"
        else tuple(int(x) for x in image_shape_cfg)
    )
    return {
        "D": int(cfg.model.D),
        "K": tuple(int(k) for k in cfg.model.K),
        "action_dim": action_dim,
        "image_shape": tuple(int(x) for x in image_shape),
        "action_block": int(cfg.model.get("action_block", frameskip)),
    }


def stable_checkpoint_config_path(checkpoint: str) -> Path:
    from stable_worldmodel.data import get_cache_dir

    root = Path(str(checkpoint)).expanduser()
    if root.exists():
        if root.is_dir():
            return root / "config.json"
        if root.name == "config.json":
            return root
        return root.parent / "config.json"
    return Path(get_cache_dir(None, sub_folder="checkpoints")) / str(checkpoint) / "config.json"


def build_trainable_model_from_base(cfg: Any, model_cfg: dict[str, Any]) -> torch.nn.Module:
    from mwm.adapters.base import ComponentPolicy
    from mwm.adapters.builder import build_mwm_from_stable_config
    from mwm.adapters.registry import family_for_target
    from mwm.adapters.stable_config import load_stable_wm_config, root_target, stable_config_sha256

    base = cfg.get("base", {}) if hasattr(cfg, "get") else {}
    base = as_container(base)
    if not base:
        raise ValueError("Trainable Le-WM MWM requires a Stable-WM base checkpoint config.")

    config_path = stable_checkpoint_config_path(str(base["checkpoint"]))
    source_config, loaded_path = load_stable_wm_config(config_path)
    detected_family = family_for_target(root_target(source_config))
    configured_family = str(base.get("family", detected_family))
    if configured_family != detected_family:
        raise ValueError(
            f"Configured Stable-WM base family {configured_family!r} does not match config target family {detected_family!r}."
        )
    if configured_family != "lewm":
        raise ValueError(f"Unsupported trainable Stable-WM base family {configured_family!r}.")

    mwm_cfg = as_container(cfg.get("mwm", {}) if hasattr(cfg, "get") else {})
    loss_cfg = as_container(cfg.get("loss", {}) if hasattr(cfg, "get") else {})
    validate_lewm_loss_config(loss_cfg)
    model_section = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    recipe = {
        **dict(loss_cfg),
        "history_size": int(model_section.get("history_size", loss_cfg.get("history_size", 3))),
        "num_preds": int(model_section.get("num_preds", loss_cfg.get("num_preds", 1))),
        "action_preprocessing": "standard_scaler",
        "loss_scope": dict(mwm_cfg.get("loss_terms", {"regularizers": "shared_latent"})),
    }
    return build_mwm_from_stable_config(
        family=configured_family,
        source_config=source_config,
        source_config_sha256=stable_config_sha256(loaded_path),
        training_recipe=recipe,
        K=tuple(int(k) for k in model_cfg["K"]),
        action_dim=int(model_cfg["action_dim"]),
        expected_D=int(model_cfg["D"]) if "D" in model_cfg else None,
        action_block=int(model_cfg.get("action_block", 1)),
        image_shape=tuple(int(x) for x in model_cfg["image_shape"]),
        normalize_imagenet=bool(model_cfg.get("normalize_imagenet", True)),
        component_policy=ComponentPolicy.from_mapping(mwm_cfg["component_policy"])
        if "component_policy" in mwm_cfg
        else None,
    )


def metadata_for_model(metadata: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    merged = {**metadata, **dict(getattr(model, "metadata", {}) or {})}
    mwm_config = getattr(model, "mwm_config", None)
    if isinstance(mwm_config, dict):
        merged["model"] = as_container(mwm_config)
    return merged


__all__ = [
    "build_trainable_model_from_base",
    "metadata_for_model",
    "resolve_lewm_base_adapter_model_cfg",
    "stable_checkpoint_config_path",
]
