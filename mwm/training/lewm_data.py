from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mwm.adapters.constants import LEWM_BASE_ADAPTER_ARCH
from mwm.data.metadata import load_dataset_metadata
from mwm.data.paths import local_path
from mwm.data.transforms import build_lewm_base_adapter_dataset_transform
from mwm.dependency_refs import dependency_refs
from mwm.swm.restore import validate_restore_columns
from mwm.training.lewm_model import resolve_lewm_base_adapter_model_cfg


def dataset_metadata(path: str | Path) -> dict[str, Any]:
    return load_dataset_metadata(path, required=False)


def base_dataset(dataset: Any) -> Any:
    return getattr(dataset, "dataset", dataset)


def dataset_available_columns(dataset: Any) -> list[str]:
    base = base_dataset(dataset)
    schema_names = getattr(base, "_schema_names", None)
    if schema_names:
        return [str(col) for col in schema_names if str(col) not in {"episode_idx", "step_idx"}]
    return [str(col) for col in getattr(base, "column_names", [])]


def close_dataset_handles(*datasets: Any) -> None:
    seen: set[int] = set()
    for dataset in datasets:
        base = base_dataset(dataset)
        if id(base) in seen:
            continue
        seen.add(id(base))
        close = getattr(base, "close", None)
        if callable(close):
            close()


def load_lewm_base_adapter_train_valid_datasets(cfg: Any) -> tuple[Any, Any, Any]:
    import stable_pretraining as spt
    from stable_worldmodel.data import load_dataset

    data_format = str(cfg.data.get("format", "lance"))
    if data_format != "lance":
        raise ValueError(f"Exact Le-WM parity training only supports Lance datasets, got format={data_format!r}.")
    history_size = int(cfg.model.get("history_size", cfg.loss.get("history_size", 3)))
    num_preds = int(cfg.model.get("num_preds", cfg.loss.get("num_preds", 1)))
    keys_to_load = list(cfg.data.get("keys_to_load", ["pixels", "action", "proprio", "state"]))
    dataset = load_dataset(
        local_path(cfg.data.path),
        transform=None,
        format=data_format,
        frameskip=int(cfg.data.get("frameskip", 1)),
        num_steps=history_size + num_preds,
        keys_to_load=keys_to_load,
        keys_to_cache=list(cfg.data.get("keys_to_cache", ["action", "proprio", "state"])),
    )
    pixels_key = str(cfg.data.pixels_key)
    img_size = int(cfg.model.get("image_size", 224))
    dataset.transform = build_lewm_base_adapter_dataset_transform(
        dataset,
        pixels_key=pixels_key,
        image_size=img_size,
        keys_to_load=keys_to_load,
    )

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[float(cfg.data.split_ratio), 1.0 - float(cfg.data.split_ratio)],
        generator=rnd_gen,
    )
    return train_set, val_set, dataset


def prepare_lewm_base_adapter_context(cfg: Any) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    from mwm.adapters.builder import STABLE_CONFIG_TARGET

    tr_ds, va_ds, base_ds = load_lewm_base_adapter_train_valid_datasets(cfg)
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec = validate_restore_columns(str(cfg.env_id), dataset_available_columns(base_ds), import_path=restore_import_path)
    model_cfg = resolve_lewm_base_adapter_model_cfg(cfg, base_dataset(base_ds))
    dataset_meta = dataset_metadata(str(cfg.data.path))
    base_action_dim = int(dataset_meta.get("action_dim", base_dataset(base_ds).get_dim(str(cfg.data.action_key))))
    metadata = {
        "env_id": str(cfg.env_id),
        "restore_spec": restore_spec.spec_id,
        "image_shape": [int(x) for x in model_cfg["image_shape"]],
        "action_dim": base_action_dim,
        "action_block": int(model_cfg.get("action_block", 1)),
        "action_preprocessing": "standard_scaler",
        "levels": [int(k) for k in model_cfg["K"]],
        "architecture_version": LEWM_BASE_ADAPTER_ARCH,
        "action_spec": {
            "dim": int(model_cfg["action_dim"]),
            "base_dim": base_action_dim,
            "block": int(model_cfg.get("action_block", 1)),
        },
        "training_backend": str(cfg.train.backend),
        "dependencies": dependency_refs(Path(__file__).resolve().parents[2]),
        "dataset": {
            "path": str(cfg.data.path),
            "pixels_key": str(cfg.data.pixels_key),
            "action_key": str(cfg.data.action_key),
            "split": "stable_pretraining_random_split",
            "normalized_columns": list(cfg.data.get("keys_to_load", ["pixels", "action", "proprio", "state"])),
        },
        "model": {"target": STABLE_CONFIG_TARGET},
    }
    for key in ("action_low", "action_high"):
        if key in dataset_meta:
            metadata[key] = dataset_meta[key]
    return tr_ds, va_ds, base_ds, model_cfg, metadata


__all__ = [
    "base_dataset",
    "close_dataset_handles",
    "dataset_available_columns",
    "dataset_metadata",
    "load_lewm_base_adapter_train_valid_datasets",
    "prepare_lewm_base_adapter_context",
]
