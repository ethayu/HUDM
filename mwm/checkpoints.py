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


def _canonical_component_policy(value: Any, *, label: str) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"Base-adaptive MWM {label} component_policy must be a mapping.")
    expected_keys = {"shared", "per_level", "reconstructor"}
    actual_keys = set(value)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing or unknown:
        raise ValueError(
            f"Base-adaptive MWM {label} component_policy must define exactly "
            f"{sorted(expected_keys)}; missing={missing}, unknown={unknown}."
        )
    normalized: dict[str, list[str]] = {}
    for key in sorted(expected_keys):
        raw = value[key]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
            raise ValueError(f"Base-adaptive MWM {label} component_policy.{key} must be a list.")
        normalized[key] = [str(item) for item in raw]
    return normalized


def _validate_stable_config_checkpoint(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    from mwm.adapters.base import ComponentPolicy, validate_component_policy
    from mwm.adapters.registry import adapter_for_family, family_for_target
    from mwm.adapters.stable_config import root_target

    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("Base-adaptive MWM checkpoints require config kwargs.")
    source_config = kwargs.get("source_config")
    if not isinstance(source_config, dict):
        raise ValueError("Base-adaptive MWM checkpoints require config kwargs.source_config.")
    family = str(
        kwargs.get("family")
        or metadata.get("adapter_family")
        or family_for_target(root_target(source_config))
    )
    if metadata.get("adapter_family") != family:
        raise ValueError(f"Base-adaptive MWM checkpoints require metadata adapter_family={family!r}.")
    role = str(metadata.get("role", ""))
    expected_fresh = False if role == "upstream_lewm_converted" else True
    if metadata.get("fresh_init") is not expected_fresh:
        raise ValueError(f"Base-adaptive MWM checkpoints require metadata fresh_init={expected_fresh}.")
    if not metadata.get("source_config_sha256"):
        raise ValueError("Base-adaptive MWM checkpoints require nonempty metadata source_config_sha256.")
    component_policy = metadata.get("component_policy")
    if not component_policy:
        raise ValueError("Base-adaptive MWM checkpoints require metadata component_policy.")
    canonical_policy = _canonical_component_policy(component_policy, label="metadata")
    adapter = adapter_for_family(family)
    policy = ComponentPolicy.from_mapping(canonical_policy)
    validate_component_policy(adapter.component_groups(), policy)

    expected_sha = kwargs.get("source_config_sha256")
    if expected_sha and str(metadata.get("source_config_sha256")) != str(expected_sha):
        raise ValueError("Base-adaptive MWM metadata source_config_sha256 does not match config.")
    expected_policy = kwargs.get("component_policy")
    if expected_policy is not None:
        expected_canonical = _canonical_component_policy(expected_policy, label="config")
        if canonical_policy != expected_canonical:
            raise ValueError("Base-adaptive MWM metadata component_policy does not match config.")
    adapter.resolve_spec(
        source_config=source_config,
        source_config_sha256=str(metadata.get("source_config_sha256")),
        training_recipe=dict(kwargs.get("training_recipe", {})),
        levels=tuple(int(k) for k in kwargs.get("K", metadata.get("levels", []))),
        component_policy=policy,
    )

def validate_checkpoint_contract(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Reject stale trainable MWM checkpoints before model instantiation."""

    from mwm.adapters.builder import STABLE_CONFIG_TARGET

    target = _checkpoint_model_target(config, metadata)
    if target == STABLE_CONFIG_TARGET:
        _validate_stable_config_checkpoint(config, metadata)
        return
    raise ValueError(f"Unsupported MWM checkpoint target {target!r}; expected {STABLE_CONFIG_TARGET!r}.")


def instantiate_from_config(config: dict[str, Any]) -> Any:
    target = config.get("target")
    if not target:
        raise ValueError("MWM config.json must include `target`.")
    kwargs = dict(config.get("kwargs", {}))
    return _import_object(str(target))(**kwargs)


def _remap_hf_vit_encoder_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Translate HF ViT encoder keys (encoder.encoder.layer.X.*) to custom ViT keys (encoder.layers.X.*)."""
    import re
    _HF_ATTN_MAP = {
        "attention.attention.query": "attention.q_proj",
        "attention.attention.key": "attention.k_proj",
        "attention.attention.value": "attention.v_proj",
        "attention.output.dense": "attention.o_proj",
        "intermediate.dense": "mlp.fc1",
        "output.dense": "mlp.fc2",
    }
    _hf_layer_re = re.compile(r"^(encoder\.encoder\.layer\.)(\d+)\.(.*)")

    def _remap(key: str) -> str:
        m = _hf_layer_re.match(key)
        if not m:
            return key
        idx, rest = m.group(2), m.group(3)
        for hf_pat, custom_pat in _HF_ATTN_MAP.items():
            if rest.startswith(hf_pat + ".") or rest == hf_pat:
                rest = custom_pat + rest[len(hf_pat):]
                break
        return f"encoder.layers.{idx}.{rest}"

    needs_remap = any(_hf_layer_re.match(k) for k in state_dict)
    if not needs_remap:
        return state_dict
    return {_remap(k): v for k, v in state_dict.items()}


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
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = _remap_hf_vit_encoder_keys(state_dict)
    model.load_state_dict(state_dict)
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
