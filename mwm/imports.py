from __future__ import annotations

import importlib
from typing import Any


def import_object(path: str) -> Any:
    module_name, sep, attr = str(path).partition(":")
    if not sep:
        module_name, sep, attr = str(path).rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import path must be 'module:attr' or 'module.attr', got {path!r}")
    return getattr(importlib.import_module(module_name), attr)


__all__ = ["import_object"]
