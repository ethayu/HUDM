from __future__ import annotations

from pathlib import Path


def local_path(path: str | Path) -> str:
    candidate = Path(str(path))
    return str(candidate.resolve()) if candidate.exists() else str(path)


__all__ = ["local_path"]
