from __future__ import annotations

from mwm.adapters.lewm_stable import (
    LeWMStableWMAdapter,
    build_mwm_lewm_from_stable_config,
    build_mwm_lewm_from_upstream_object,
)
from mwm.adapters.registry import register_adapter

register_adapter(LeWMStableWMAdapter())

__all__ = [
    "LeWMStableWMAdapter",
    "build_mwm_lewm_from_stable_config",
    "build_mwm_lewm_from_upstream_object",
]
