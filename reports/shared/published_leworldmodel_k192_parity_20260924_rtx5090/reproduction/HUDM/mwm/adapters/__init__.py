from importlib import import_module

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseAdapter, StableWMBaseSpec, validate_component_policy
from mwm.adapters.builder import (
    STABLE_CONFIG_TARGET,
    build_mwm_from_stable_config,
)
from mwm.adapters.registry import adapter_for_family, adapter_for_target, family_for_target, register_adapter
from mwm.adapters.stable_config import load_stable_wm_config, root_target, stable_config_sha256

__all__ = [
    "ComponentGroup",
    "ComponentPolicy",
    "STABLE_CONFIG_TARGET",
    "StableWMBaseAdapter",
    "StableWMBaseSpec",
    "adapter_for_family",
    "adapter_for_target",
    "build_mwm_from_stable_config",
    "family_for_target",
    "load_stable_wm_config",
    "register_adapter",
    "root_target",
    "stable_config_sha256",
    "validate_component_policy",
]

try:
    _lewm = import_module("mwm.adapters.lewm")
except ModuleNotFoundError as exc:
    if exc.name != "mwm.adapters.lewm":
        raise
else:
    for _name in getattr(_lewm, "__all__", ()):
        globals()[_name] = getattr(_lewm, _name)
        if _name not in __all__:
            __all__.append(_name)
    del _lewm

try:
    _prejepa = import_module("mwm.adapters.prejepa")
except ModuleNotFoundError as exc:
    if exc.name != "mwm.adapters.prejepa":
        raise
else:
    for _name in getattr(_prejepa, "__all__", ()):
        globals()[_name] = getattr(_prejepa, _name)
        if _name not in __all__:
            __all__.append(_name)
    del _prejepa
