from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from mwm.checkpoint_io import save_world_checkpoint
from mwm.training.lewm_config import DEFAULTS, make_run_dir
from mwm.training.lewm_data import close_dataset_handles, prepare_lewm_base_adapter_context
from mwm.training.lewm_model import build_trainable_model_from_base, metadata_for_model


def load_lewm_base_adapter_lightning_state(lewm: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {k.removeprefix("model."): v for k, v in state_dict.items() if str(k).startswith("model.")}
    if not model_state:
        raise ValueError(f"Lightning checkpoint {checkpoint_path} does not contain any 'model.' parameters.")
    missing, unexpected = lewm.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Could not load Le-WM base-adapter state from {checkpoint_path}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return checkpoint if isinstance(checkpoint, dict) else {}


def export_lewm_base_adapter_lightning_checkpoint(
    cfg_path: str,
    checkpoint_path: str,
    output_dir: str | None = None,
) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    if str(cfg.train.backend).lower() != "stable_worldmodel_lewm":
        raise ValueError("Lightning export is only supported for the Le-WM base-adapter training backend.")
    torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))
    torch.manual_seed(int(cfg.seed))
    run_dir = output_dir or make_run_dir(
        str(cfg.train.checkpoint_dir),
        str(cfg.train.run_name),
        timestamp=bool(cfg.train.get("timestamp_run_dir", False)),
    )
    tr_ds, va_ds, base_ds, model_cfg, metadata = prepare_lewm_base_adapter_context(cfg)
    del tr_ds, va_ds
    try:
        model = build_trainable_model_from_base(cfg, model_cfg)
        checkpoint = load_lewm_base_adapter_lightning_state(model, checkpoint_path)
        train_info = {
            "epoch": int(checkpoint.get("epoch", 0)),
            "last_checkpoint": str(checkpoint_path),
            "exported_from_lightning_checkpoint": True,
        }
        save_world_checkpoint(model, run_dir, metadata={**metadata_for_model(metadata, model), **train_info})
    finally:
        close_dataset_handles(base_ds)
    print(f"Exported Le-WM base-adapter Lightning checkpoint to canonical MWM checkpoint: {run_dir}")


__all__ = ["load_lewm_base_adapter_lightning_state", "export_lewm_base_adapter_lightning_checkpoint"]
