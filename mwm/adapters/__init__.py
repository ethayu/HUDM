from importlib import import_module

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseAdapter, StableWMBaseSpec, validate_component_policy

__all__ = [
    "ComponentGroup",
    "ComponentPolicy",
    "StableWMBaseAdapter",
    "StableWMBaseSpec",
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
    del _lewm, _name
