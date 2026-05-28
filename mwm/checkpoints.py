from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch


METADATA_FILENAME = "world_metadata.json"
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pt"
CHECKPOINT_FORMAT = "mwm_world_v1"
LEWM_BASE_ADAPTER_ARCH = "lewm_base_adapter_v1"
CANONICAL_FILES = {CONFIG_FILENAME, WEIGHTS_FILENAME, METADATA_FILENAME}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_load(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _import_object(target: str) -> Any:
    module_name, _, attr = str(target).partition(":")
    if not attr:
        module_name, _, attr = str(target).rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import target {target!r}")
    return getattr(importlib.import_module(module_name), attr)


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


def _checkpoint_model_target(config: dict[str, Any], metadata: dict[str, Any]) -> str:
    target = str(config.get("target", ""))
    if target:
        return target
    model_meta = metadata.get("model", {})
    return str(model_meta.get("target", "")) if isinstance(model_meta, dict) else ""


def validate_checkpoint_contract(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Reject stale trainable MWM checkpoints before model instantiation."""

    target = _checkpoint_model_target(config, metadata)
    if target.endswith("build_mwm_lewm_from_stable_config"):
        from mwm.adapters.base import ComponentPolicy, validate_component_policy
        from mwm.adapters.lewm import LeWMStableWMAdapter

        if metadata.get("adapter_family") != "lewm":
            raise ValueError("Base-adaptive Le-WM checkpoints require metadata adapter_family='lewm'.")
        if metadata.get("fresh_init") is not True:
            raise ValueError("Base-adaptive Le-WM checkpoints require metadata fresh_init=True.")
        if not metadata.get("source_config_sha256"):
            raise ValueError("Base-adaptive Le-WM checkpoints require nonempty metadata source_config_sha256.")
        component_policy = metadata.get("component_policy")
        if not component_policy:
            raise ValueError("Base-adaptive Le-WM checkpoints require metadata component_policy.")
        if not isinstance(component_policy, dict):
            raise ValueError("Base-adaptive Le-WM metadata component_policy must be a mapping.")
        adapter = LeWMStableWMAdapter()
        policy = ComponentPolicy.from_mapping(component_policy)
        validate_component_policy(adapter.component_groups(), policy)
        if policy != adapter.default_policy():
            raise ValueError("Base-adaptive Le-WM metadata component_policy is not supported.")

        kwargs = config.get("kwargs", {})
        if isinstance(kwargs, dict):
            expected_sha = kwargs.get("source_config_sha256")
            if expected_sha and str(metadata.get("source_config_sha256")) != str(expected_sha):
                raise ValueError("Base-adaptive Le-WM metadata source_config_sha256 does not match config.")
            expected_policy = kwargs.get("component_policy")
            if isinstance(expected_policy, dict) and policy != ComponentPolicy.from_mapping(expected_policy):
                raise ValueError("Base-adaptive Le-WM metadata component_policy does not match config.")
        return
    if target.endswith("build_mwm_lewm_from_object"):
        if len(metadata.get("levels", [])) != 1:
            raise ValueError("Trusted upstream Le-WM object imports must remain single-fidelity eval-only checkpoints.")
        return
    if target.endswith("build_mwm_lewm"):
        arch = str(metadata.get("architecture_version", ""))
        if arch != LEWM_BASE_ADAPTER_ARCH:
            raise ValueError(
                "Old generic MWM trainable checkpoint architecture detected; "
                "retrain required with architecture_version='lewm_base_adapter_v1'."
            )
        return


def instantiate_from_config(config: dict[str, Any]) -> Any:
    target = config.get("target")
    if not target:
        raise ValueError("MWM config.json must include `target`.")
    kwargs = dict(config.get("kwargs", {}))
    return _import_object(str(target))(**kwargs)


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
    meta_path = root / METADATA_FILENAME
    for path in (cfg_path, weights_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical MWM checkpoint file: {path}")
    extras = sorted(path.name for path in root.iterdir() if path.name not in CANONICAL_FILES)
    if extras:
        raise ValueError(f"Canonical MWM checkpoint directory {root} contains non-checkpoint files: {extras}")
    metadata = _json_load(meta_path)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format {metadata.get('format')!r}; expected {CHECKPOINT_FORMAT!r}.")
    artifacts = metadata.get("artifacts", {})
    config_artifact = artifacts.get("config", {}) if isinstance(artifacts, dict) else {}
    weights_artifact = artifacts.get("weights", {}) if isinstance(artifacts, dict) else {}
    if config_artifact.get("sha256") and config_artifact["sha256"] != file_sha256(cfg_path):
        raise ValueError(f"Config hash mismatch for {cfg_path}")
    if weights_artifact.get("sha256") and weights_artifact["sha256"] != file_sha256(weights_path):
        raise ValueError(f"Weights hash mismatch for {weights_path}")
    config = _json_load(cfg_path)
    validate_checkpoint_contract(config, metadata)
    model = instantiate_from_config(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, metadata, 0


__all__ = [
    "CHECKPOINT_FORMAT",
    "CANONICAL_FILES",
    "CONFIG_FILENAME",
    "LEWM_BASE_ADAPTER_ARCH",
    "METADATA_FILENAME",
    "WEIGHTS_FILENAME",
    "file_sha256",
    "instantiate_from_config",
    "load_world_metadata",
    "load_world_model_from_checkpoint",
    "save_world_checkpoint",
    "save_world_metadata",
    "validate_checkpoint_contract",
]
