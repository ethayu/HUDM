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
        "resume_checkpoint": None,
        "slurm_auto_requeue": False,
        "gradient_clip_val": 1.0,
        # Seed applied by SPT immediately before Trainer.fit. None preserves
        # the historical MWM behavior of reusing the top-level construction
        # seed; upstream LeWM's Manager(seed=None) resolves this value to 0.
        "fit_seed": None,
        # Keep model initialization independently controllable from the
        # top-level data split/shuffle seed. None preserves historical MWM
        # behavior by using the top-level seed.
        "model_init_seed": None,
    },
    "optim": {"lr": 3e-4},
    "decoder_training": {
        "enabled": True,
        "mode": "separate_optimizer",
        "gradient_clip_val": None,
        "lr": None,
        "weight_decay": None,
    },
    "loss": {"rollout_weight": 1.0, "sigreg_weight": 0.0},
    "schedule": {"max_epochs": 30, "lr_max_epochs": None},
}


def stable_wm_model_init_seed(cfg: Any) -> int:
    """Resolve the model-construction seed without changing the data seed."""
    configured = cfg.train.get("model_init_seed", None)
    return int(cfg.seed) if configured is None else int(configured)


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


def validate_stable_wm_loss_config(loss_cfg: Any, decoder_training_cfg: Any | None = None) -> None:
    loss = as_container(loss_cfg)
    if isinstance(loss, dict) and "recon_weight" in loss:
        raise ValueError(
            "loss.recon_weight is not supported with optimizer-isolated decoder training; "
            "use decoder_training.enabled to turn decoder training on or off."
        )
    if isinstance(loss, dict) and float(loss.get("recon_latent_weight", 0.0)) != 0.0:
        raise ValueError(
            "loss.recon_latent_weight must be 0 with optimizer-isolated decoder training; "
            "decoder reconstruction cannot contribute gradients to the encoder."
        )
    if decoder_training_cfg is None:
        return
    decoder_training = as_container(decoder_training_cfg)
    if not isinstance(decoder_training, dict):
        raise ValueError("decoder_training must be a mapping.")
    unknown = set(decoder_training) - {
        "enabled",
        "mode",
        "gradient_clip_val",
        "lr",
        "weight_decay",
        "activation_checkpointing",
    }
    if unknown:
        raise ValueError(f"Unknown decoder_training keys: {sorted(unknown)}")
    activation_checkpointing = decoder_training.get("activation_checkpointing", False)
    if not isinstance(activation_checkpointing, bool):
        if not isinstance(activation_checkpointing, (list, tuple)):
            raise ValueError("decoder_training.activation_checkpointing must be a boolean or a list of level indices.")
        if any(not isinstance(level_idx, int) or level_idx < 0 for level_idx in activation_checkpointing):
            raise ValueError(
                "decoder_training.activation_checkpointing level indices must be non-negative integers."
            )
        if len(set(activation_checkpointing)) != len(activation_checkpointing):
            raise ValueError("decoder_training.activation_checkpointing level indices must be unique.")
    mode = str(decoder_training.get("mode", "separate_optimizer"))
    if mode != "separate_optimizer":
        raise ValueError(
            f"Unsupported decoder_training.mode {mode!r}; expected 'separate_optimizer'."
        )
    for key in ("gradient_clip_val", "lr", "weight_decay"):
        value = decoder_training.get(key)
        if value is not None and float(value) < 0:
            raise ValueError(f"decoder_training.{key} must be non-negative, got {value}.")


__all__ = [
    "DEFAULTS",
    "as_container",
    "make_run_dir",
    "stable_wm_model_init_seed",
    "validate_stable_wm_loss_config",
]
