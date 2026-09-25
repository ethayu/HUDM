from __future__ import annotations

from importlib import import_module
from typing import Any


_ADAPTERS: dict[str, Any] = {}
_FAMILY_ALIASES = {
    "dino": "prejepa",
    "dinowm": "prejepa",
    "dino-wm": "prejepa",
}
_BUILTIN_ADAPTER_MODULES = (
    "mwm.adapters.lewm",
    "mwm.adapters.prejepa",
)


def canonical_family(family: str) -> str:
    normalized = str(family).lower()
    return _FAMILY_ALIASES.get(normalized, normalized)


def _ensure_builtin_adapters_loaded() -> None:
    for module_name in _BUILTIN_ADAPTER_MODULES:
        import_module(module_name)


def family_for_target(target: str) -> str:
    canonical = canonical_family(target)
    if canonical in _ADAPTERS:
        return canonical

    target_lower = target.lower()
    if ".lewm." in target_lower or target_lower.endswith(".lewm"):
        return "lewm"
    if ".prejepa." in target_lower or target_lower.endswith(".prejepa"):
        return "prejepa"
    if ".pldm." in target_lower or target_lower.endswith(".pldm"):
        return "pldm"

    raise ValueError(f"Unsupported Stable-WM target: {target}")


def register_adapter(adapter: Any) -> None:
    _ADAPTERS[canonical_family(adapter.family)] = adapter


def adapter_for_family(family: str) -> Any:
    canonical = canonical_family(family)
    _ensure_builtin_adapters_loaded()
    try:
        return _ADAPTERS[canonical]
    except KeyError as exc:
        raise ValueError(f"Unsupported Stable-WM target family: {family}") from exc


def adapter_for_target(target: str) -> Any:
    return adapter_for_family(family_for_target(target))


__all__ = [
    "adapter_for_family",
    "adapter_for_target",
    "canonical_family",
    "family_for_target",
    "register_adapter",
]
