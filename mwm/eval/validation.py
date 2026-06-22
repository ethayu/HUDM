from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mwm.checkpoint_io import CHECKPOINT_FORMAT
from mwm.data.metadata import load_dataset_metadata
from mwm.eval.action_preprocessing import (
    stat_keys_for_action_process,
    uses_standardized_action_space,
)


def close_dataset(dataset: Any) -> None:
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def eval_keys_to_load(cfg: Any, model: Any, metadata: dict[str, Any]) -> list[str] | None:
    raw = list(cfg.data.get("keys_to_load", []))
    if not raw:
        return None
    keys = [str(k) for k in raw]
    if uses_standardized_action_space(model, metadata, cfg):
        for key in stat_keys_for_action_process(cfg):
            if key not in keys:
                keys.append(key)
    return keys


def dataset_path(dataset: Any, cfg: Any) -> str:
    return str(cfg.data.path or getattr(dataset, "path", getattr(dataset, "uri", "")))


def validate_dataset_metadata(dataset: Any, checkpoint_metadata: dict[str, Any], cfg: Any) -> None:
    path = dataset_path(dataset, cfg)
    dataset_meta = load_dataset_metadata(path, required=False)
    if not dataset_meta:
        return
    ckpt_format = str(checkpoint_metadata.get("format"))
    if ckpt_format != CHECKPOINT_FORMAT:
        raise ValueError(f"Checkpoint format must be {CHECKPOINT_FORMAT!r}, got {checkpoint_metadata.get('format')!r}.")
    if str(dataset_meta.get("format", "")) != "swm_lance":
        raise ValueError(f"Dataset format must be swm_lance, got {dataset_meta.get('format')!r}.")
    checks = (
        ("env_id", str),
        ("restore_spec", str),
        ("action_dim", int),
    )
    for key, caster in checks:
        if key not in dataset_meta:
            raise ValueError(f"Dataset metadata {path} is missing required key {key!r}.")
        if key not in checkpoint_metadata:
            continue
        if caster(dataset_meta[key]) != caster(checkpoint_metadata[key]):
            raise ValueError(
                f"Dataset metadata {key}={dataset_meta[key]!r} does not match checkpoint {key}={checkpoint_metadata[key]!r}."
            )

    dataset_shape = tuple(int(x) for x in dataset_meta.get("image_shape", ()))
    checkpoint_shape = tuple(int(x) for x in checkpoint_metadata.get("image_shape", ()))
    if checkpoint_shape and dataset_shape != checkpoint_shape:
        raise ValueError(
            f"Dataset image_shape={dataset_meta.get('image_shape')!r} does not match checkpoint image_shape={checkpoint_metadata['image_shape']!r}."
        )
    for key in ("action_low", "action_high"):
        if key not in dataset_meta:
            raise ValueError(f"Dataset metadata {path} is missing required key {key!r}.")
        if key not in checkpoint_metadata:
            continue
        ds_bound = np.asarray(dataset_meta[key], dtype=np.float32).reshape(-1)
        ckpt_bound = np.asarray(checkpoint_metadata[key], dtype=np.float32).reshape(-1)
        if ds_bound.shape != ckpt_bound.shape or not np.allclose(ds_bound, ckpt_bound):
            raise ValueError(f"Dataset {key}={ds_bound.tolist()} does not match checkpoint {key}={ckpt_bound.tolist()}.")
    ckpt_dataset = checkpoint_metadata.get("dataset")
    if not isinstance(ckpt_dataset, dict):
        raise ValueError("Checkpoint metadata is missing required dataset key mapping.")
    dataset_key_meta = dataset_meta.get("dataset", {})
    for meta_key, actual in (
        ("pixels_key", getattr(dataset, "pixels_key", "pixels")),
        ("action_key", getattr(dataset, "action_key", "action")),
    ):
        if meta_key not in ckpt_dataset:
            raise ValueError(f"Checkpoint metadata dataset mapping is missing {meta_key!r}.")
        if str(ckpt_dataset[meta_key]) != str(actual):
            raise ValueError(f"Checkpoint dataset {meta_key}={ckpt_dataset[meta_key]!r} does not match configured {actual!r}.")
        if isinstance(dataset_key_meta, dict) and meta_key in dataset_key_meta and str(dataset_key_meta[meta_key]) != str(actual):
            raise ValueError(f"Dataset metadata {meta_key}={dataset_key_meta[meta_key]!r} does not match configured {actual!r}.")


def same_dataset_ref(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_path = Path(left)
    right_path = Path(right)
    if left_path.exists() and right_path.exists():
        return left_path.resolve() == right_path.resolve()
    return str(left) == str(right)


def validate_manifest(
    manifest: dict[str, Any],
    *,
    path: str,
    dataset: Any,
    cfg: Any,
    env_id: str,
    restore_spec_id: str,
) -> None:
    expected = {
        "env_id": str(env_id),
        "restore_spec": str(restore_spec_id),
        "seed": int(cfg.eval.seed),
        "goal_offset": int(cfg.eval.goal_offset),
        "eval_budget": int(cfg.eval.budget),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest {path} has {key}={manifest.get(key)!r}, expected {value!r}.")
    path_for_dataset = dataset_path(dataset, cfg)
    if not same_dataset_ref(str(manifest.get("dataset_path", "")), path_for_dataset):
        raise ValueError(f"Manifest {path} was generated for dataset {manifest.get('dataset_path')!r}, not {path_for_dataset!r}.")
    if len(manifest.get("pairs", [])) != int(cfg.eval.episodes):
        raise ValueError(f"Manifest {path} has {len(manifest.get('pairs', []))} pairs, expected {int(cfg.eval.episodes)}.")


def dataset_runtime_metadata(dataset: Any, cfg: Any) -> dict[str, Any]:
    sidecar = load_dataset_metadata(dataset_path(dataset, cfg), required=False)
    return {
        "path": dataset_path(dataset, cfg),
        "split_ratio": float(getattr(dataset, "split_ratio", 1.0)),
        "pixels_key": str(getattr(dataset, "pixels_key", "pixels")),
        "action_key": str(getattr(dataset, "action_key", "action")),
        "column_names": [str(x) for x in getattr(dataset, "column_names", [])],
        "sidecar": sidecar,
    }


__all__ = [
    "close_dataset",
    "dataset_path",
    "dataset_runtime_metadata",
    "eval_keys_to_load",
    "same_dataset_ref",
    "validate_dataset_metadata",
    "validate_manifest",
]
