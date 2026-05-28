from __future__ import annotations

from typing import Any


_ADAPTERS: dict[str, Any] = {}


def family_for_target(target: str) -> str:
    if target in _ADAPTERS:
        return target

    target_lower = target.lower()
    if ".lewm." in target_lower or target_lower.endswith(".lewm"):
        return "lewm"
    if ".prejepa." in target_lower or target_lower.endswith(".prejepa"):
        return "prejepa"
    if ".pldm." in target_lower or target_lower.endswith(".pldm"):
        return "pldm"

    raise ValueError(f"Unsupported Stable-WM target: {target}")


def register_adapter(adapter: Any) -> None:
    _ADAPTERS[adapter.family] = adapter


def adapter_for_family(family: str) -> Any:
    try:
        return _ADAPTERS[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported Stable-WM target family: {family}") from exc


def adapter_for_target(target: str) -> Any:
    return adapter_for_family(family_for_target(target))


__all__ = [
    "adapter_for_family",
    "adapter_for_target",
    "family_for_target",
    "register_adapter",
]
