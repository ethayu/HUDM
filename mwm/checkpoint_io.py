from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mwm.checkpoint_contract import validate_checkpoint_contract
from mwm.checkpoint_keymaps import (
    remap_custom_vit_encoder_keys_to_hf,
    remap_hf_vit_encoder_keys,
    remap_vit_encoder_keys_for_model,
)
from mwm.imports import import_object
from mwm.io import file_sha256, load_json, write_json


METADATA_FILENAME = "world_metadata.json"
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pt"
CHECKPOINT_FORMAT = "mwm_world_v1"
CANONICAL_FILES = {CONFIG_FILENAME, WEIGHTS_FILENAME, METADATA_FILENAME}


def _json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _json_load(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def model_config(model: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    cfg = getattr(model, "mwm_config", None)
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(model, "config_dict"):
        cfg = model.config_dict()
        if isinstance(cfg, dict):
            return dict(cfg)
    raise ValueError("MWM checkpoints require an explicit importable model config.")


def save_world_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> None:
    _json_dump(Path(run_dir) / METADATA_FILENAME, metadata)


def load_world_metadata(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing MWM metadata: {path}")
    return _json_load(path)


def save_world_checkpoint(
    model: Any,
    run_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    extras = sorted(path.name for path in root.iterdir() if path.name not in CANONICAL_FILES)
    if extras:
        raise ValueError(f"Canonical MWM checkpoint directory {root} contains non-checkpoint files: {extras}")
    cfg = model_config(model, config=config)
    cfg_path = root / CONFIG_FILENAME
    weights_path = root / WEIGHTS_FILENAME
    _json_dump(cfg_path, cfg)
    torch.save(model.state_dict(), weights_path)
    meta = dict(metadata or {})
    model_metadata = getattr(model, "metadata", {})
    if isinstance(model_metadata, dict):
        for key in (
            "action_spec",
            "action_dim",
            "action_block",
            "preprocessing_spec",
            "architecture_version",
            "head_architectures",
            "action_preprocessing",
            "adapter_family",
            "source_config_sha256",
            "component_policy",
            "fresh_init",
            "loss_scope",
            "training_recipe",
            "D",
            "D_visual",
            "full_dim",
            "extra_dims",
            "extra_input_dims",
            "extra_order",
            "level_dims",
            "num_patches",
            "patch_size",
            "backbone_name",
            "fixed_extra_policy",
        ):
            if key in model_metadata and key not in meta:
                meta[key] = model_metadata[key]
    meta["format"] = CHECKPOINT_FORMAT
    meta["weights"] = WEIGHTS_FILENAME
    meta["artifacts"] = {
        **dict(meta.get("artifacts", {})),
        "config": {"path": CONFIG_FILENAME, "sha256": file_sha256(cfg_path)},
        "weights": {"path": WEIGHTS_FILENAME, "sha256": file_sha256(weights_path)},
    }
    save_world_metadata(root, meta)


def instantiate_from_config(config: dict[str, Any]) -> Any:
    target = config.get("target")
    if not target:
        raise ValueError("MWM config.json must include `target`.")
    kwargs = dict(config.get("kwargs", {}))
    return import_object(str(target))(**kwargs)


def load_checkpoint_config_and_metadata(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(run_dir)
    return _json_load(root / CONFIG_FILENAME), _json_load(root / METADATA_FILENAME)


def validate_checkpoint_directory(
    run_dir: str | Path,
    *,
    strict_artifacts: bool = True,
    strict_metadata: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(run_dir)
    for name in (CONFIG_FILENAME, WEIGHTS_FILENAME, METADATA_FILENAME):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical MWM checkpoint file: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Empty canonical MWM checkpoint file: {path}")
    extras = sorted(path.name for path in root.iterdir() if path.name not in CANONICAL_FILES)
    if extras:
        raise ValueError(f"Canonical MWM checkpoint directory {root} contains non-checkpoint files: {extras}")
    config, metadata = load_checkpoint_config_and_metadata(root)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format {metadata.get('format')!r}; expected {CHECKPOINT_FORMAT!r}.")
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("MWM checkpoint artifacts are not a mapping.")
    for label, filename in (("config", CONFIG_FILENAME), ("weights", WEIGHTS_FILENAME)):
        artifact = artifacts.get(label, {})
        if not isinstance(artifact, dict):
            raise ValueError(f"MWM checkpoint artifact {label!r} is not a mapping.")
        if artifact.get("path") != filename:
            raise ValueError(f"MWM checkpoint artifact {label!r} path mismatch.")
        sha = artifact.get("sha256")
        if strict_artifacts and not sha:
            raise ValueError(f"MWM checkpoint missing {label} sha256.")
        if sha and file_sha256(root / filename) != sha:
            raise ValueError(f"MWM checkpoint {label} sha256 mismatch.")
    if not config.get("target"):
        raise ValueError("MWM checkpoint config missing import target.")
    validate_checkpoint_contract(config, metadata, strict_metadata=strict_metadata)
    return config, metadata


def load_world_model_from_checkpoint(
    run_dir: str | Path,
    epoch: int | None,
    device: torch.device,
) -> tuple[Any, dict[str, Any], int]:
    if epoch is not None:
        raise ValueError("MWM v1 checkpoints are single-export checkpoints; epoch selection is not supported.")
    root = Path(run_dir)
    cfg_path = root / CONFIG_FILENAME
    weights_path = root / WEIGHTS_FILENAME
    config, metadata = validate_checkpoint_directory(root, strict_artifacts=False, strict_metadata=False)
    model = instantiate_from_config(config).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    remapped = remap_vit_encoder_keys_for_model(state_dict, model)
    incompatible = model.load_state_dict(remapped, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        if not _allow_legacy_lewm_missing_decoders(config, metadata, missing, unexpected):
            details = []
            if missing:
                details.append(f"Missing key(s): {missing}")
            if unexpected:
                details.append(f"Unexpected key(s): {unexpected}")
            raise RuntimeError("Error(s) in loading state_dict for MWM checkpoint: " + "; ".join(details))
    model.eval()
    return model, metadata, 0


def _allow_legacy_lewm_missing_decoders(
    config: dict[str, Any],
    metadata: dict[str, Any],
    missing: list[str],
    unexpected: list[str],
) -> bool:
    if unexpected or not missing:
        return False
    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, dict):
        return False
    family = str(kwargs.get("family", metadata.get("adapter_family", ""))).lower()
    if family != "lewm":
        return False
    policy = kwargs.get("component_policy", metadata.get("component_policy", {}))
    if not isinstance(policy, dict):
        return False
    reconstructor = policy.get("reconstructor", None)
    if reconstructor != []:
        return False
    return all(key.startswith("decoders.") for key in missing)


__all__ = [
    "CHECKPOINT_FORMAT",
    "CANONICAL_FILES",
    "CONFIG_FILENAME",
    "METADATA_FILENAME",
    "WEIGHTS_FILENAME",
    "file_sha256",
    "instantiate_from_config",
    "load_checkpoint_config_and_metadata",
    "load_world_metadata",
    "load_world_model_from_checkpoint",
    "remap_custom_vit_encoder_keys_to_hf",
    "remap_hf_vit_encoder_keys",
    "remap_vit_encoder_keys_for_model",
    "save_world_checkpoint",
    "save_world_metadata",
    "validate_checkpoint_directory",
]
