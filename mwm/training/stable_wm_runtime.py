from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch


def coerce_lightning_devices(value: Any) -> int | str | list[int]:
    if value is None:
        return 1
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "auto":
            return "auto"
        if "," in stripped:
            devices = [int(part.strip()) for part in stripped.split(",") if part.strip()]
            if not devices:
                raise ValueError("train.devices must specify at least one device.")
            return devices
        value = int(stripped)
    if isinstance(value, (list, tuple)):
        devices = [int(device) for device in value]
        if not devices:
            raise ValueError("train.devices must specify at least one device.")
        return devices
    devices_int = int(value)
    if devices_int < 1:
        raise ValueError(f"train.devices must be positive, got {devices_int}.")
    return devices_int


def resolve_lightning_trainer_runtime(cfg: Any) -> dict[str, Any]:
    use_cuda = not bool(cfg.train.no_cuda) and torch.cuda.is_available()
    accelerator = "gpu" if use_cuda else "cpu"
    devices_key = "devices" if use_cuda else "cpu_devices"
    devices = coerce_lightning_devices(cfg.train.get(devices_key, 1))
    num_nodes = int(cfg.train.get("num_nodes", 1))
    if num_nodes < 1:
        raise ValueError(f"train.num_nodes must be positive, got {num_nodes}.")
    return {
        "accelerator": accelerator,
        "devices": devices,
        "strategy": str(cfg.train.get("strategy", "auto") or "auto"),
        "num_nodes": num_nodes,
        "sync_batchnorm": bool(cfg.train.get("sync_batchnorm", False)),
        "use_distributed_sampler": bool(cfg.train.get("use_distributed_sampler", True)),
    }


def prepare_trainer_root(run_dir: str | Path, cfg: Any, *, logs_root: str | Path = "logs") -> Path:
    trainer_root = Path(logs_root) / "mwm_training" / Path(run_dir).name
    if bool(cfg.train.get("clean_trainer_root", True)) and trainer_root.exists():
        shutil.rmtree(trainer_root)
    trainer_root.mkdir(parents=True, exist_ok=True)
    return trainer_root


def resolve_stable_wm_adapter_total_steps(cfg: Any, train_loader: Any) -> int:
    lr_epochs = cfg.schedule.get("lr_max_epochs", None)
    if lr_epochs is None:
        lr_epochs = cfg.schedule.max_epochs
    lr_epochs_int = int(lr_epochs)
    if lr_epochs_int < 1:
        raise ValueError(f"schedule.lr_max_epochs must be positive when set, got {lr_epochs_int}.")
    return lr_epochs_int * len(train_loader)


__all__ = [
    "coerce_lightning_devices",
    "prepare_trainer_root",
    "resolve_stable_wm_adapter_total_steps",
    "resolve_lightning_trainer_runtime",
]
