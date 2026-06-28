from __future__ import annotations

import datetime
import os
from typing import Any

from omegaconf import OmegaConf


DEFAULTS = {
    "seed": 42,
    "env_id": "swm/PushT-v1",
    "data": {
        "path": "data/swm_dataset.lance",
        "format": "lance",
        "split_ratio": 0.8,
        "pixels_key": "pixels",
        "action_key": "action",
        "frameskip": 1,
    },
    "model": {
        "D": 256,
        "K": [32, 64, 128, 256],
        "action_dim": "auto",
        "image_shape": "auto",
    },
    "restore": {"import_path": None},
    "train": {
        "batch_size": 8,
        "horizon": 16,
        "num_workers": 0,
        "no_cuda": False,
        "devices": 1,
        "cpu_devices": 1,
        "strategy": "auto",
        "num_nodes": 1,
        "sync_batchnorm": False,
        "use_distributed_sampler": True,
        "checkpoint_dir": "checkpoints_mwm",
        "run_name": "mwm_lewm",
        "backend": "stable_worldmodel_lewm",
        "timestamp_run_dir": False,
        "clean_trainer_root": True,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "matmul_precision": "high",
        "prefetch_factor": 2,
        "checkpoint_every_n_train_steps": 0,
        "checkpoint_monitor": None,
        "checkpoint_mode": "min",
        "save_top_k": 0,
        "export_checkpoint": "last",
        "slurm_auto_requeue": False,
    },
    "optim": {"lr": 3e-4},
    "loss": {"rollout_weight": 1.0, "recon_latent_weight": 0.0, "sigreg_weight": 0.0},
    "schedule": {"max_epochs": 30, "lr_max_epochs": None},
}


def make_run_dir(root: str, tag: str, *, timestamp: bool = False) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{tag}_{ts}" if timestamp else tag)
    os.makedirs(path, exist_ok=True)
    return path


def as_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {k: as_container(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {k: as_container(v) for k, v in vars(value).items()}
    return value


def validate_lewm_loss_config(loss_cfg: Any) -> None:
    loss = as_container(loss_cfg)
    if isinstance(loss, dict) and "recon_weight" in loss:
        raise ValueError("loss.recon_weight has been removed; use loss.recon_latent_weight instead.")


_as_container = as_container


__all__ = ["DEFAULTS", "_as_container", "as_container", "make_run_dir", "validate_lewm_loss_config"]
