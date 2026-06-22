from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dataset_metadata_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_suffix(p.suffix + ".metadata.json") if p.suffix else p / "metadata.json"


def load_dataset_metadata(path: str | Path, *, required: bool = False) -> dict[str, Any]:
    meta_path = dataset_metadata_path(path)
    if not meta_path.is_file():
        if required:
            raise FileNotFoundError(f"Missing dataset metadata: {meta_path}")
        return {}
    return dict(json.loads(meta_path.read_text(encoding="utf-8")))


def write_dataset_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    meta_path = dataset_metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["dataset_metadata_path", "load_dataset_metadata", "write_dataset_metadata"]
