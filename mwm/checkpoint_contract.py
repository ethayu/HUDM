from __future__ import annotations

from typing import Any


def _checkpoint_model_target(config: dict[str, Any], metadata: dict[str, Any]) -> str:
    target = str(config.get("target", ""))
    if target:
        return target
    model_meta = metadata.get("model", {})
    return str(model_meta.get("target", "")) if isinstance(model_meta, dict) else ""


def _coerce_int(value: Any, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc


def checkpoint_full_latent_dim(metadata: dict[str, Any]) -> int:
    model_meta = metadata.get("model", {})
    model_kwargs = model_meta.get("kwargs", {}) if isinstance(model_meta, dict) else {}
    candidates = [
        metadata.get("D"),
        model_meta.get("D") if isinstance(model_meta, dict) else None,
        model_kwargs.get("D") if isinstance(model_kwargs, dict) else None,
        model_kwargs.get("expected_D") if isinstance(model_kwargs, dict) else None,
    ]
    dims = [_coerce_int(value, label=f"D candidate {idx}") for idx, value in enumerate(candidates) if value is not None]
    if not dims:
        raise ValueError("MWM checkpoint missing full latent dimension D.")
    if len(set(dims)) != 1:
        raise ValueError(f"MWM checkpoint has inconsistent full latent dimension D values {dims}.")
    return dims[0]


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


def _validate_strict_metadata_contract(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("MWM checkpoint config kwargs are not a mapping.")
    action_spec = metadata.get("action_spec")
    if not isinstance(action_spec, dict):
        raise ValueError("MWM checkpoint missing action_spec mapping.")
    dim = _coerce_int(action_spec.get("dim"), label="action_spec.dim")
    base_dim = _coerce_int(action_spec.get("base_dim"), label="action_spec.base_dim")
    block = _coerce_int(action_spec.get("block"), label="action_spec.block")
    cfg_dim = _coerce_int(kwargs.get("action_dim"), label="config action_dim")
    cfg_block = _coerce_int(kwargs.get("action_block", metadata.get("action_block", 1)), label="config action_block")
    meta_base_dim = _coerce_int(metadata.get("action_dim"), label="metadata action_dim")
    meta_block = _coerce_int(metadata.get("action_block", 1), label="metadata action_block")
    if dim != cfg_dim:
        raise ValueError("MWM checkpoint action_spec.dim does not match config action_dim.")
    if block != cfg_block:
        raise ValueError("MWM checkpoint action_spec.block does not match config action_block.")
    if block != meta_block:
        raise ValueError("MWM checkpoint action_spec.block does not match metadata action_block.")
    if base_dim != meta_base_dim:
        raise ValueError("MWM checkpoint action_spec.base_dim does not match metadata action_dim.")
    if base_dim * block != dim:
        raise ValueError("MWM checkpoint action spec is internally inconsistent.")
    cfg_levels = [int(k) for k in kwargs.get("K", [])] if "K" in kwargs else []
    meta_levels = [int(k) for k in metadata.get("levels", [])] if "levels" in metadata else []
    if cfg_levels and meta_levels and cfg_levels != meta_levels:
        raise ValueError("MWM checkpoint levels do not match config K.")
    checkpoint_full_latent_dim(metadata)


def validate_checkpoint_contract(
    config: dict[str, Any],
    metadata: dict[str, Any],
    *,
    strict_metadata: bool = False,
) -> None:
    """Reject stale trainable MWM checkpoints before model instantiation."""

    from mwm.adapters.builder import STABLE_CONFIG_TARGET

    target = _checkpoint_model_target(config, metadata)
    if target == STABLE_CONFIG_TARGET:
        _validate_stable_config_checkpoint(config, metadata)
        if strict_metadata:
            _validate_strict_metadata_contract(config, metadata)
        return
    raise ValueError(f"Unsupported MWM checkpoint target {target!r}; expected {STABLE_CONFIG_TARGET!r}.")


__all__ = ["checkpoint_full_latent_dim", "validate_checkpoint_contract"]
