from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


def load_config(defaults: Any, cfg_path: str | Path, overrides: Iterable[str] = ()) -> Any:
    cfg = OmegaConf.merge(defaults, OmegaConf.load(str(cfg_path)))
    overrides = list(overrides)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
