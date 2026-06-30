from __future__ import annotations

from pathlib import Path
from typing import Any

from mwm.io import file_sha256, jsonable, load_json, write_json, write_metrics_jsonl


def write_run_sidecars(run_dir: str | Path, row: dict[str, Any], payload: dict[str, Any]) -> None:
    root = Path(run_dir)
    write_json(root / "summary.json", {"run": row})
    write_json(root / "dependencies.json", dict(payload.get("dependencies", {})))
    write_json(root / "planning_diagnostics.json", dict(payload.get("planning_diagnostics", {})))


__all__ = ["file_sha256", "jsonable", "load_json", "write_json", "write_metrics_jsonl", "write_run_sidecars"]
