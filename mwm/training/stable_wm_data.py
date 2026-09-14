from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from mwm.data.metadata import load_dataset_metadata
from mwm.data.paths import local_path
from mwm.dependency_refs import dependency_refs
from mwm.swm.restore import validate_restore_columns
from mwm.training.stable_wm_model import resolve_stable_wm_adapter_model_cfg
from mwm.training.stable_wm_transforms import build_stable_wm_adapter_dataset_transform


def stable_wm_training_data_source(cfg: Any) -> tuple[str, str]:
    """Resolve an optional source-only training artifact.

    ``data.path``/``data.format`` remain the Lance runtime contract embedded
    in exported checkpoints. A training source lets parity runs consume an
    authoritative upstream artifact without changing runtime/evaluation I/O.
    """
    source = cfg.data.get("training_source", {})
    path = source.get("path", cfg.data.path)
    data_format = source.get("format", cfg.data.get("format", "lance"))
    return str(path), str(data_format)


def upstream_split_with_generator(
    dataset: Any,
    *,
    split_ratio: float,
    seed: int,
) -> tuple[Any, Any, torch.Generator]:
    """Match LeWM's shared split-and-shuffle generator state."""
    import stable_pretraining as spt

    rnd_gen = torch.Generator().manual_seed(int(seed))
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[float(split_ratio), 1.0 - float(split_ratio)],
        generator=rnd_gen,
    )
    return train_set, val_set, rnd_gen


def dataset_metadata(path: str | Path) -> dict[str, Any]:
    return load_dataset_metadata(path, required=False)


def base_dataset(dataset: Any) -> Any:
    return getattr(dataset, "dataset", dataset)


def dataset_available_columns(dataset: Any) -> list[str]:
    base = base_dataset(dataset)
    schema_names = getattr(base, "_schema_names", None)
    if schema_names:
        return [str(col) for col in schema_names if str(col) not in {"episode_idx", "step_idx"}]
    open_source = getattr(base, "_open_h5", None)
    if callable(open_source):
        with open_source() as handle:
            return [str(col) for col in handle.keys() if str(col) not in {"ep_len", "ep_offset"}]
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
            continue
        source_handle = getattr(base, "h5_file", None)
        if source_handle is not None:
            source_handle.close()
            base.h5_file = None


def load_stable_wm_adapter_train_valid_datasets(cfg: Any) -> tuple[Any, Any, Any, torch.Generator]:
    from stable_worldmodel.data import load_dataset

    data_path, data_format = stable_wm_training_data_source(cfg)
    history_size = int(cfg.model.get("history_size", cfg.loss.get("history_size", 3)))
    num_preds = int(cfg.model.get("num_preds", cfg.loss.get("num_preds", 1)))
    keys_to_load = list(cfg.data.get("keys_to_load", ["pixels", "action", "proprio", "state"]))
    dataset = load_dataset(
        local_path(data_path),
        transform=None,
        format=data_format,
        frameskip=int(cfg.data.get("frameskip", 1)),
        num_steps=history_size + num_preds,
        keys_to_load=keys_to_load,
        keys_to_cache=list(cfg.data.get("keys_to_cache", ["action", "proprio", "state"])),
    )
    pixels_key = str(cfg.data.pixels_key)
    img_size = int(cfg.model.get("image_size", 224))
    dataset.transform = build_stable_wm_adapter_dataset_transform(
        dataset,
        pixels_key=pixels_key,
        image_size=img_size,
        keys_to_load=keys_to_load,
    )

    train_set, val_set, rnd_gen = upstream_split_with_generator(
        dataset,
        split_ratio=float(cfg.data.split_ratio),
        seed=int(cfg.seed),
    )
    # Upstream LeWM intentionally reuses this same, now-advanced generator for
    # the shuffled training DataLoader. Returning it preserves that exact sample
    # order instead of silently restarting the generator from cfg.seed.
    return train_set, val_set, dataset, rnd_gen


def prepare_stable_wm_adapter_context(
    cfg: Any,
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any], torch.Generator]:
    from mwm.adapters.builder import STABLE_CONFIG_TARGET

    tr_ds, va_ds, base_ds, train_generator = load_stable_wm_adapter_train_valid_datasets(cfg)
    training_data_path, training_data_format = stable_wm_training_data_source(cfg)
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec = validate_restore_columns(str(cfg.env_id), dataset_available_columns(base_ds), import_path=restore_import_path)
    model_cfg = resolve_stable_wm_adapter_model_cfg(cfg, base_dataset(base_ds))
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
        "action_spec": {
            "dim": int(model_cfg["action_dim"]),
            "base_dim": base_action_dim,
            "block": int(model_cfg.get("action_block", 1)),
        },
        "training_backend": str(cfg.train.backend),
        "dependencies": dependency_refs(Path(__file__).resolve().parents[2]),
        "dataset": {
            "path": training_data_path,
            "format": training_data_format,
            "runtime_path": str(cfg.data.path),
            "runtime_format": str(cfg.data.get("format", "lance")),
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
    return tr_ds, va_ds, base_ds, model_cfg, metadata, train_generator


__all__ = [
    "base_dataset",
    "close_dataset_handles",
    "dataset_available_columns",
    "dataset_metadata",
    "load_stable_wm_adapter_train_valid_datasets",
    "prepare_stable_wm_adapter_context",
    "stable_wm_training_data_source",
    "upstream_split_with_generator",
]
