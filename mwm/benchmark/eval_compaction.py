from __future__ import annotations

import errno
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


COMPLETION_FILES = (
    "eval.json",
    "resolved_config.yaml",
    "metrics.jsonl",
    "summary.json",
    "planning_diagnostics.json",
    "episode_traces.jsonl",
)


class EvalCompactionError(RuntimeError):
    """Raised when removing policy diagnostics would not be lossless."""


def _atomic_write_json_once(path: Path, payload: dict[str, Any], *, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    mode = path.stat().st_mode
    retryable = {errno.EAGAIN, errno.EWOULDBLOCK}
    for attempt, delay_seconds in enumerate((0.0, 0.05, 0.25, 1.0), start=1):
        if delay_seconds:
            time.sleep(delay_seconds)
        try:
            _atomic_write_json_once(path, payload, mode=mode)
            return
        except OSError as exc:
            if exc.errno not in retryable or attempt == 4:
                raise


def compact_completed_eval(run_dir: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Remove an exact duplicate diagnostic field from one completed benchmark cell.

    Partial cells are skipped. A completed cell is changed only when its
    ``policy_diagnostics`` value exactly equals ``planning_diagnostics``.
    """

    root = Path(run_dir)
    missing = [name for name in COMPLETION_FILES if not (root / name).is_file()]
    if missing:
        return {"run_dir": str(root), "status": "partial", "missing": missing, "reclaimed_bytes": 0}

    eval_path = root / "eval.json"
    before = eval_path.stat().st_size
    with eval_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise EvalCompactionError(f"{eval_path}: expected a JSON object")
    if "planning_diagnostics" not in payload:
        raise EvalCompactionError(f"{eval_path}: completed cell is missing planning_diagnostics")
    if "policy_diagnostics" not in payload:
        return {"run_dir": str(root), "status": "already_compacted", "reclaimed_bytes": 0}
    if payload["policy_diagnostics"] != payload["planning_diagnostics"]:
        raise EvalCompactionError(
            f"{eval_path}: policy_diagnostics differs from planning_diagnostics; refusing to compact"
        )

    del payload["policy_diagnostics"]
    if dry_run:
        return {"run_dir": str(root), "status": "would_compact", "reclaimed_bytes": 0}

    _atomic_write_json(eval_path, payload)
    after = eval_path.stat().st_size
    return {
        "run_dir": str(root),
        "status": "compacted",
        "reclaimed_bytes": max(0, before - after),
    }


__all__ = ["COMPLETION_FILES", "EvalCompactionError", "compact_completed_eval"]
