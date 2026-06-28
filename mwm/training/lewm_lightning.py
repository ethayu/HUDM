from __future__ import annotations

import os
from typing import Any

import lightning as pl
import torch
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from mwm.data.module import PrebuiltLoaderDataModule
from mwm.models.base_adaptive import MatryoshkaWorldModel
from mwm.training.lewm_callbacks import lewm_base_adapter_callbacks, select_lewm_base_adapter_export_checkpoint
from mwm.training.lewm_config import as_container, validate_lewm_loss_config
from mwm.training.lewm_runtime import (
    prepare_trainer_root,
    resolve_lewm_base_adapter_total_steps,
    resolve_lightning_trainer_runtime,
)


def lewm_base_adapter_forward(module: Any, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    cfg = module.lewm_base_adapter_cfg
    if not isinstance(module.model, MatryoshkaWorldModel):
        raise RuntimeError("Le-WM training requires the MWM base-adapter model, not a raw Stable-WM object.")
    validate_lewm_loss_config(cfg.loss)
    output = module.model.training_loss(
        batch,
        level_weights=cfg.loss.get("level_weights", None),
        rollout_weight=float(cfg.loss.get("rollout_weight", 1.0)),
        recon_latent_weight=float(cfg.loss.get("recon_latent_weight", 0.0)),
        sigreg=module.sigreg,
        sigreg_weight=float(cfg.loss.get("sigreg_weight", cfg.loss.get("sigreg", {}).get("weight", 0.0))),
        sigreg_scope=str(cfg.loss.get("sigreg_scope", "shared_latent")),
    )
    if hasattr(module, "log_dict"):
        module.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}, on_step=True, sync_dist=True)
    return output


def run_lewm_base_adapter_training(
    lewm: torch.nn.Module,
    train_set: Any,
    val_set: Any,
    cfg: Any,
    run_dir: str,
) -> dict[str, Any]:
    import stable_pretraining as spt
    from stable_worldmodel.wm.loss import SIGReg

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    num_workers = int(cfg.train.num_workers)
    loader_kwargs = {
        "batch_size": int(cfg.train.batch_size),
        "num_workers": num_workers,
        "drop_last": bool(cfg.train.get("drop_last", True)),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": int(cfg.train.get("prefetch_factor", 3)) if num_workers > 0 else None,
        "pin_memory": bool(cfg.train.get("pin_memory", torch.cuda.is_available() and not bool(cfg.train.no_cuda))),
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=rnd_gen, **{k: v for k, v in loader_kwargs.items() if v is not None})
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **{k: v for k, v in {**loader_kwargs, "drop_last": False}.items() if v is not None and k != "generator"},
    )
    total_steps = resolve_lewm_base_adapter_total_steps(cfg, train_loader)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {
                "type": "AdamW",
                "lr": float(cfg.optim.lr),
                "weight_decay": float(cfg.optim.get("weight_decay", 0.0)),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(0.01 * total_steps)),
                "max_steps": total_steps,
            },
            "interval": "epoch",
        },
    }
    trainer_root = prepare_trainer_root(run_dir, cfg)
    callbacks = lewm_base_adapter_callbacks(cfg)
    checkpoint_cb = next(callback for callback in callbacks if isinstance(callback, ModelCheckpoint))
    trainer_runtime = resolve_lightning_trainer_runtime(cfg)
    trainer = pl.Trainer(
        **trainer_runtime,
        precision=cfg.train.get("precision", "bf16"),
        max_epochs=int(cfg.schedule.max_epochs),
        gradient_clip_val=float(cfg.train.get("gradient_clip_val", 1.0)),
        default_root_dir=str(trainer_root),
        limit_train_batches=cfg.train.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        plugins=[SLURMEnvironment(auto_requeue=bool(cfg.train.get("slurm_auto_requeue", False)))]
        if os.environ.get("SLURM_JOB_ID")
        else None,
    )
    module = spt.Module(
        model=lewm,
        sigreg=SIGReg(knots=int(cfg.loss.get("sigreg_knots", 17)), num_proj=int(cfg.loss.get("sigreg_num_proj", 1024))),
        forward=lewm_base_adapter_forward,
        optim=optimizers,
        hparams=as_container(cfg),
    )
    module.lewm_base_adapter_cfg = cfg
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=PrebuiltLoaderDataModule(train_loader, val_loader),
        seed=int(cfg.seed),
    )
    manager()
    selected_checkpoint = select_lewm_base_adapter_export_checkpoint(checkpoint_cb, cfg)
    selected_checkpoint_state: dict[str, Any] = {}
    if selected_checkpoint:
        from mwm.training.lewm_export import load_lewm_base_adapter_lightning_state

        selected_checkpoint_state = load_lewm_base_adapter_lightning_state(lewm, selected_checkpoint)
    return {
        "epoch": int(selected_checkpoint_state.get("epoch", getattr(trainer, "current_epoch", int(cfg.schedule.max_epochs)))),
        "last_checkpoint": str(checkpoint_cb.last_model_path or "") or None,
        "best_checkpoint": str(checkpoint_cb.best_model_path or "") or None,
        "best_model_score": None
        if checkpoint_cb.best_model_score is None
        else float(checkpoint_cb.best_model_score.detach().cpu().item()),
        "checkpoint_monitor": str(checkpoint_cb.monitor or "") or None,
        "export_checkpoint": str(cfg.train.get("export_checkpoint", "last")),
        "selected_lightning_checkpoint": str(selected_checkpoint or "") or None,
    }


__all__ = ["lewm_base_adapter_forward", "run_lewm_base_adapter_training"]
