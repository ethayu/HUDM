from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_config_sha256(path: str | Path) -> str:
    config_path = Path(path)
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def load_stable_wm_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path)
    if config_path.name != "config.json":
        config_path = config_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Stable-WM config.json must contain a JSON object: {config_path}")
    return config, config_path


def root_target(config: dict[str, Any]) -> str:
    target = config.get("_target_")
    if not isinstance(target, str) or not target:
        raise ValueError("Stable-WM config must define a nonempty root _target_")
    return target


__all__ = [
    "load_stable_wm_config",
    "root_target",
    "stable_config_sha256",
]
