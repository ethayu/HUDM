from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from mwm.io import file_sha256, jsonable

MANIFEST_SCHEMA_VERSION = "mwm_swm_eval_manifest_v1"


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_sha256(payload: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def manifest_file_sha256(path: str | Path) -> str:
    return file_sha256(path)


def _pair_row(pair: Any) -> dict[str, int]:
    if isinstance(pair, dict):
        return {
            "episode": int(pair["episode"]),
            "start_step": int(pair["start_step"]),
            "goal_step": int(pair["goal_step"]),
            "start_row": int(pair["start_row"]),
            "goal_row": int(pair["goal_row"]),
        }
    return {
        "episode": int(pair.episode),
        "start_step": int(pair.start_step),
        "goal_step": int(pair.goal_step),
        "start_row": int(pair.start_row),
        "goal_row": int(pair.goal_row),
    }


def generate_manifest(
    *,
    env_id: str,
    dataset_path: str,
    pairs: Iterable[Any],
    goal_offset: int,
    goal_indexing: str = "exact",
    effective_goal_offset: int | None = None,
    eval_budget: int,
    seed: int,
    restore_spec: str,
    dataset_metadata: dict[str, Any] | None = None,
    dependency_shas: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "env_id": str(env_id),
        "dataset_path": str(dataset_path),
        "goal_offset": int(goal_offset),
        "goal_indexing": str(goal_indexing),
        "effective_goal_offset": int(
            goal_offset if effective_goal_offset is None else effective_goal_offset
        ),
        "eval_budget": int(eval_budget),
        "seed": int(seed),
        "restore_spec": str(restore_spec),
        "dataset_metadata": jsonable(dataset_metadata or {}),
        "dependency_shas": dict(dependency_shas or {}),
        "pairs": [_pair_row(pair) for pair in pairs],
    }
    payload["manifest_sha256"] = manifest_sha256({k: v for k, v in payload.items() if k != "manifest_sha256"})
    return payload


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.get("manifest_sha256")
    if expected:
        actual = manifest_sha256({k: v for k, v in payload.items() if k != "manifest_sha256"})
        if str(expected) != str(actual):
            raise ValueError(f"Manifest hash mismatch for {path}: expected {expected}, got {actual}")
    return dict(payload)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "generate_manifest",
    "load_manifest",
    "manifest_file_sha256",
    "manifest_sha256",
    "write_manifest",
]
