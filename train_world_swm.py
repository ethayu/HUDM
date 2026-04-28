from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

from datasets.swm_hdf5 import SWMHDF5Episodes, load_swm_dataset_metadata
from hudm.swm_envs import infer_swm_action_space, parse_env_kwargs, parse_image_shape
from hudm.swm_restore import validate_restore_columns
from hudm.world_io import save_world_checkpoint
from hudm.world_training import run_epoch
from models.world.model import HierWorldModel


DEFAULTS = {
    "seed": 42,
    "env_id": "swm/PushT-v1",
    "image_shape": "auto",
    "max_episode_steps": 100,
    "env_kwargs": {},
    "data": {
        "path": "data/swm_dataset.h5",
        "split_ratio": 0.8,
        "pixels_key": "pixels",
        "action_key": "action",
    },
    "model": {
        "input": "images",
        "D": 256,
        "K": [32, 64, 128, 256],
        "decoder_mode": "per_level",
        "dynamics_mode": "per_level",
    },
    "restore": {"import_path": None},
    "train": {
        "batch_size": 8,
        "horizon": 16,
        "num_workers": 0,
        "no_cuda": False,
        "checkpoint_dir": "checkpoints_swm",
        "run_name": "swm_world",
    },
    "optim": {"lr": 3e-4},
    "loss": {
        "recon_weight": 1.0,
        "teacher_weight": 1.0,
        "rollout_weight": 1.0,
    },
    "schedule": {
        "max_epochs": 30,
        "patience": 5,
        "min_delta": 1e-3,
    },
    "wandb": {
        "enable": False,
        "project": "hudm-world",
        "run_name": "swm-world",
    },
}


def make_run_dir(root: str, tag: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{tag}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def _load_dataset_metadata(path: str | Path) -> dict[str, Any]:
    return load_swm_dataset_metadata(path, required=False)


def _validate_dataset_metadata(cfg, dataset: SWMHDF5Episodes, restore_spec_id: str) -> None:
    meta = _load_dataset_metadata(dataset.path)
    if not meta:
        return
    if str(meta.get("env_id")) != str(cfg.env_id):
        raise ValueError(f"Dataset env_id={meta.get('env_id')!r} does not match training env_id={str(cfg.env_id)!r}.")
    if str(meta.get("restore_spec")) != str(restore_spec_id):
        raise ValueError(
            f"Dataset restore_spec={meta.get('restore_spec')!r} does not match expected {restore_spec_id!r}."
        )
    if "image_shape" in meta and tuple(int(x) for x in meta["image_shape"]) != tuple(dataset.image_shape):
        raise ValueError(f"Dataset metadata image_shape={meta['image_shape']} does not match pixels {dataset.image_shape}.")
    if "action_dim" in meta and int(meta["action_dim"]) != int(dataset.action_dim):
        raise ValueError(f"Dataset metadata action_dim={meta['action_dim']} does not match actions {dataset.action_dim}.")
    dataset_key_meta = meta.get("dataset", {})
    if isinstance(dataset_key_meta, dict):
        if "pixels_key" in dataset_key_meta and str(dataset_key_meta["pixels_key"]) != str(cfg.data.pixels_key):
            raise ValueError(
                f"Dataset metadata pixels_key={dataset_key_meta['pixels_key']!r} does not match configured {str(cfg.data.pixels_key)!r}."
            )
        if "action_key" in dataset_key_meta and str(dataset_key_meta["action_key"]) != str(cfg.data.action_key):
            raise ValueError(
                f"Dataset metadata action_key={dataset_key_meta['action_key']!r} does not match configured {str(cfg.data.action_key)!r}."
            )


def _action_metadata(cfg, dataset: SWMHDF5Episodes) -> tuple[int, list[float], list[float]]:
    meta = _load_dataset_metadata(dataset.path)
    if "action_low" in meta and "action_high" in meta:
        return int(meta["action_dim"]), list(meta["action_low"]), list(meta["action_high"])
    env_kwargs = parse_env_kwargs(OmegaConf.to_container(cfg.get("env_kwargs", {}), resolve=True))
    action_dim, low, high = infer_swm_action_space(
        str(cfg.env_id),
        image_shape=dataset.image_shape,
        max_episode_steps=int(cfg.max_episode_steps),
        env_kwargs=env_kwargs,
    )
    return action_dim, low.tolist(), high.tolist()


def main(cfg_path: str) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    torch.manual_seed(int(cfg.seed))
    if str(cfg.model.input) != "images":
        raise ValueError(f"HUDM v1 is RGB-only and requires model.input='images', got {cfg.model.input!r}.")

    tr_ds = SWMHDF5Episodes(
        cfg.data.path,
        horizon=int(cfg.train.horizon),
        split="train",
        split_ratio=float(cfg.data.split_ratio),
        seed=int(cfg.seed),
        pixels_key=str(cfg.data.pixels_key),
        action_key=str(cfg.data.action_key),
    )
    va_ds = SWMHDF5Episodes(
        cfg.data.path,
        horizon=int(cfg.train.horizon),
        split="valid",
        split_ratio=float(cfg.data.split_ratio),
        seed=int(cfg.seed),
        pixels_key=str(cfg.data.pixels_key),
        action_key=str(cfg.data.action_key),
    )
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec = validate_restore_columns(str(cfg.env_id), tr_ds.column_names, import_path=restore_import_path)
    _validate_dataset_metadata(cfg, tr_ds, restore_spec.spec_id)

    image_shape = tr_ds.image_shape if str(cfg.image_shape).lower() == "auto" else parse_image_shape(cfg.image_shape)
    if image_shape != tr_ds.image_shape:
        raise ValueError(f"Configured image_shape {image_shape} does not match dataset pixels {tr_ds.image_shape}.")
    action_dim, action_low, action_high = _action_metadata(cfg, tr_ds)
    if int(action_dim) != int(tr_ds.action_dim):
        raise ValueError(f"Dataset action_dim={tr_ds.action_dim} does not match env action_dim={action_dim}.")

    device = torch.device("cuda" if torch.cuda.is_available() and not bool(cfg.train.no_cuda) else "cpu")
    model = HierWorldModel(
        K=[int(k) for k in cfg.model.K],
        D=int(cfg.model.D),
        action_dim=int(action_dim),
        input=str(cfg.model.input),
        decoder_mode=str(cfg.model.decoder_mode),
        dynamics_mode=str(cfg.model.dynamics_mode),
        image_shape=image_shape,
    ).to(device)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )

    run_dir = make_run_dir(str(cfg.train.checkpoint_dir), str(cfg.train.run_name))
    with open(os.path.join(run_dir, "world.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    metadata = {
        "format": "swm_hdf5",
        "env_id": str(cfg.env_id),
        "restore_spec": restore_spec.spec_id,
        "image_shape": list(image_shape),
        "action_dim": int(action_dim),
        "action_low": [float(x) for x in action_low],
        "action_high": [float(x) for x in action_high],
        "dataset": {
            "path": str(tr_ds.path),
            "pixels_key": str(cfg.data.pixels_key),
            "action_key": str(cfg.data.action_key),
        },
        "model": {
            "input": "images",
            "D": int(cfg.model.D),
            "K": [int(k) for k in cfg.model.K],
            "decoder_mode": str(cfg.model.decoder_mode),
            "dynamics_mode": str(cfg.model.dynamics_mode),
        },
    }

    if bool(cfg.wandb.enable):
        if wandb is None:
            raise RuntimeError("wandb.enable=true but wandb is not installed.")
        wandb.init(
            project=str(cfg.wandb.project),
            name=str(cfg.wandb.run_name),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=run_dir,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optim.lr))
    best_val = float("inf")
    no_improve = 0
    max_epochs = int(cfg.schedule.max_epochs)
    patience = int(cfg.schedule.patience)
    min_delta = float(cfg.schedule.min_delta)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_logs = run_epoch(model, tr_loader, cfg, device, optimizer=optimizer, train=True)
        with torch.no_grad():
            val_loss, val_logs = run_epoch(model, va_loader, cfg, device, train=False)
        print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}")
        if bool(cfg.wandb.enable) and wandb is not None:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                **{f"train/{k}": v for k, v in train_logs.items()},
                **{f"val/{k}": v for k, v in val_logs.items()},
                "epoch": epoch,
            })
        if val_loss + min_delta < best_val:
            best_val = val_loss
            no_improve = 0
            save_world_checkpoint(model, run_dir, epoch=epoch, metadata={**metadata, "best_val": best_val})
        else:
            no_improve += 1
            if no_improve >= patience:
                print("converged (patience reached)")
                break

    print(f"Training complete. Checkpoints: {run_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python train_world_swm.py configs/world_swm.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
