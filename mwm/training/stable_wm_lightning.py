from __future__ import annotations

import os
from typing import Any

import lightning as pl
import stable_pretraining as spt
import torch
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from mwm.data.module import PrebuiltLoaderDataModule
from mwm.models.common import MatryoshkaRuntimeModel
from mwm.training.stable_wm_callbacks import stable_wm_adapter_callbacks, select_stable_wm_adapter_export_checkpoint
from mwm.training.stable_wm_config import as_container, validate_stable_wm_loss_config
from mwm.training.stable_wm_runtime import (
    prepare_trainer_root,
    resolve_stable_wm_adapter_total_steps,
    resolve_lightning_trainer_runtime,
)


def stable_wm_parameter_partitions(
    model: torch.nn.Module,
) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    """Return trainable world-model and decoder parameters with no overlap."""

    decoders = getattr(model, "decoders", None)
    decoder_ids = {id(param) for param in decoders.parameters()} if decoders is not None else set()
    world_parameters = tuple(
        param for param in model.parameters() if param.requires_grad and id(param) not in decoder_ids
    )
    decoder_parameters = tuple(
        param for param in model.parameters() if param.requires_grad and id(param) in decoder_ids
    )
    world_ids = {id(param) for param in world_parameters}
    partitioned_ids = world_ids | {id(param) for param in decoder_parameters}
    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    if world_ids & decoder_ids:
        raise RuntimeError("World-model and decoder optimizer parameter sets overlap.")
    if partitioned_ids != trainable_ids:
        raise RuntimeError("World-model and decoder optimizer parameter sets do not cover all trainable parameters.")
    if not world_parameters:
        raise RuntimeError("World-model optimizer has no trainable parameters.")
    return world_parameters, decoder_parameters


def stable_wm_upstream_lewm_parameter_order(
    model: torch.nn.Module,
    world_parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.nn.Parameter, ...]:
    """Match raw single-level LeWM's cross-tensor registration order.

    Gradient-norm clipping reduces across tensors in optimizer-group order.
    That reduction is not bitwise permutation-invariant under BF16 training,
    even when every individual gradient is identical. Raw LeWM registers its
    modules as encoder, predictor, action encoder, projector, and prediction
    projector. MWM groups transition modules under ``transitions.0`` and has a
    different natural module order, so strict single-level parity must map the
    optimizer group back to the raw order.

    Multi-level models have no single raw-LeWM parameter order and retain their
    native ordering.
    """

    transitions = getattr(model, "transitions", None)
    if transitions is None or len(transitions) != 1:
        return world_parameters
    prefixes = (
        "encoder.",
        "transitions.0.predictor.",
        "transitions.0.action_encoder.",
        "projector.",
        "transitions.0.pred_proj.",
    )
    world_ids = {id(value) for value in world_parameters}
    named_world = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) in world_ids
    ]
    if not all(any(name.startswith(prefix) for name, _ in named_world) for prefix in prefixes):
        return world_parameters
    ordered = tuple(
        parameter
        for prefix in prefixes
        for name, parameter in named_world
        if name.startswith(prefix)
    )
    expected_ids = {id(parameter) for parameter in world_parameters}
    ordered_ids = {id(parameter) for parameter in ordered}
    if len(ordered) != len(world_parameters) or ordered_ids != expected_ids:
        unmatched = [
            name for name, _ in named_world if not any(name.startswith(prefix) for prefix in prefixes)
        ]
        raise RuntimeError(
            "Raw LeWM optimizer ordering does not cover the strict single-level world partition; "
            f"unmatched={unmatched[:8]}"
        )
    return ordered


def _decoder_training_settings(cfg: Any) -> dict[str, Any]:
    def enabled_clip(value: Any) -> float | None:
        if value is None:
            return None
        clip = float(value)
        return clip if clip > 0 else None

    raw = cfg.get("decoder_training", {})
    enabled = bool(raw.get("enabled", True))
    world_lr = float(cfg.optim.lr)
    world_weight_decay = float(cfg.optim.get("weight_decay", 0.0))
    world_clip = cfg.train.get("gradient_clip_val", 1.0)
    decoder_lr = raw.get("lr", None)
    decoder_weight_decay = raw.get("weight_decay", None)
    decoder_clip = raw.get("gradient_clip_val", None)
    return {
        "enabled": enabled,
        "world_lr": world_lr,
        "world_weight_decay": world_weight_decay,
        "world_gradient_clip_val": enabled_clip(world_clip),
        "decoder_lr": world_lr if decoder_lr is None else float(decoder_lr),
        "decoder_weight_decay": world_weight_decay
        if decoder_weight_decay is None
        else float(decoder_weight_decay),
        "decoder_gradient_clip_val": enabled_clip(world_clip if decoder_clip is None else decoder_clip),
    }


def _gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
    gradients = [param.grad.detach().float().norm(2) for param in parameters if param.grad is not None]
    if gradients:
        return torch.stack(gradients).norm(2)
    device = parameters[0].device if parameters else torch.device("cpu")
    return torch.zeros((), device=device, dtype=torch.float32)


def stable_wm_fit_seed(cfg: Any) -> int:
    """Resolve the RNG seed SPT applies immediately before Trainer.fit."""

    configured = cfg.train.get("fit_seed", None)
    return int(cfg.seed) if configured is None else int(configured)


def stable_wm_adapter_forward(module: Any, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    cfg = module.stable_wm_adapter_cfg
    if not isinstance(module.model, MatryoshkaRuntimeModel):
        raise RuntimeError("Stable-WM adapter training requires an MWM runtime model, not a raw Stable-WM object.")
    validate_stable_wm_loss_config(cfg.loss, cfg.get("decoder_training", {}))
    loss_kwargs = {
        "level_weights": cfg.loss.get("level_weights", None),
        "rollout_weight": float(cfg.loss.get("rollout_weight", 1.0)),
        "recon_latent_weight": float(cfg.loss.get("recon_latent_weight", 0.0)),
        "sigreg": module.sigreg,
        "sigreg_weight": float(cfg.loss.get("sigreg_weight", cfg.loss.get("sigreg", {}).get("weight", 0.0))),
        "sigreg_scope": str(cfg.loss.get("sigreg_scope", "shared_latent")),
    }
    if hasattr(module.model, "decoders"):
        loss_kwargs.update(
            {
                "decoder_training_enabled": bool(cfg.get("decoder_training", {}).get("enabled", True)),
                "random_prefix_weight": float(cfg.loss.get("random_prefix_weight", 0.0)),
                "sample_random_prefixes": str(stage).strip().lower() in {"fit", "train"},
            }
        )
    output = module.model.training_loss(batch, **loss_kwargs)
    world_loss = output["loss"]
    if hasattr(module, "log_dict"):
        module.log_dict(
            {
                f"{stage}/{k}": v.detach().float() if k == "sampled_k" else v.detach()
                for k, v in output.items()
                if "loss" in k or k == "sampled_k"
            },
            on_step=True,
            sync_dist=True,
        )
    if (
        str(stage).strip().lower() in {"fit", "train"}
        and bool(cfg.get("decoder_training", {}).get("enabled", True))
        and hasattr(module.model, "decoders")
    ):
        decoder_loss = output.get("decoder_loss")
        if decoder_loss is None:
            raise RuntimeError("Decoder training is enabled but the objective did not return decoder_loss.")
        # spt.Module performs one backward from state["loss"]. Decoder inputs are
        # detached, so this joint scalar retains disjoint world/decoder graphs.
        output["loss"] = world_loss + decoder_loss
    return output


class OptimizerIsolatedSPTModule(spt.Module):
    """Fill SPT 0.1.6's optimizer metadata gap without replacing its train loop."""

    def __init__(
        self,
        *,
        world_parameters: tuple[torch.nn.Parameter, ...],
        decoder_parameters: tuple[torch.nn.Parameter, ...],
        world_gradient_clip_val: float | None,
        decoder_gradient_clip_val: float | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stable_wm_optimizer_parameters = {
            "world_opt": world_parameters,
            "decoder_opt": decoder_parameters,
        }
        self._stable_wm_optimizer_clip_vals = {
            "world_opt": world_gradient_clip_val,
            "decoder_opt": decoder_gradient_clip_val,
        }

    def configure_optimizers(self):
        optimizers, schedulers = super().configure_optimizers()
        if len(optimizers) != 2 or len(schedulers) != 2:
            raise RuntimeError("SPT did not configure the expected world and decoder optimizers.")
        # Regex priority requires decoder_opt first in self.optim, but the custom
        # predecessor serialized world optimizer/scheduler state first.
        decoder_optimizer, world_optimizer = optimizers
        decoder_scheduler, world_scheduler = schedulers
        world_scheduler["name"] = "world_lr"
        decoder_scheduler["name"] = "decoder_lr"
        return [world_optimizer, decoder_optimizer], [world_scheduler, decoder_scheduler]

    def on_train_start(self) -> None:
        optimizer_names = tuple(self._stable_wm_optimizer_parameters)
        self._optimizer_index_to_name.update(enumerate(optimizer_names))
        self._optimizer_frequencies.update({name: 1 for name in optimizer_names})
        self._optimizer_gradient_clip_val.update(self._stable_wm_optimizer_clip_vals)
        self._optimizer_gradient_clip_algorithm.update({name: "norm" for name in optimizer_names})
        super().on_train_start()

        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        if len(optimizers) != len(optimizer_names):
            raise RuntimeError("SPT did not configure the expected world and decoder optimizers.")
        for name, optimizer in zip(optimizer_names, optimizers):
            raw_optimizer = getattr(optimizer, "optimizer", optimizer)
            actual_ids = {id(param) for group in raw_optimizer.param_groups for param in group["params"]}
            expected_ids = {id(param) for param in self._stable_wm_optimizer_parameters[name]}
            if actual_ids != expected_ids:
                raise RuntimeError(f"{name} parameter assignment does not match the asserted partition.")

    def after_manual_backward(self) -> None:
        for name, parameters in self._stable_wm_optimizer_parameters.items():
            self.log(
                f"fit/{name.removesuffix('_opt')}_grad_norm",
                _gradient_norm(parameters),
                on_step=True,
                sync_dist=True,
            )


class WorldOnlySPTModule(spt.Module):
    """Assert that strict world-only training registers no decoder parameters."""

    def __init__(
        self,
        *,
        world_parameters: tuple[torch.nn.Parameter, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stable_wm_world_parameters = world_parameters

    def configure_optimizers(self):
        optimizers, schedulers = super().configure_optimizers()
        if len(optimizers) != 1 or len(schedulers) != 1:
            raise RuntimeError("Strict world-only training requires one optimizer and one scheduler.")
        raw_optimizer = getattr(optimizers[0], "optimizer", optimizers[0])
        if len(raw_optimizer.param_groups) != 1:
            raise RuntimeError("Strict world-only AdamW must contain exactly one parameter group.")
        group = raw_optimizer.param_groups[0]
        actual_ids = {id(parameter) for parameter in group["params"]}
        expected_ids = {id(parameter) for parameter in self._stable_wm_world_parameters}
        if actual_ids != expected_ids:
            raise RuntimeError("Strict world-only optimizer parameters do not match the asserted partition.")
        # Reorder before optimizer state exists. Lightning/SPT clipping and
        # AdamW checkpoint serialization now follow raw LeWM's tensor order.
        group["params"] = list(self._stable_wm_world_parameters)
        return optimizers, schedulers

    def on_train_start(self) -> None:
        super().on_train_start()
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        if len(optimizers) != 1:
            raise RuntimeError("Strict world-only training requires exactly one optimizer.")
        raw_optimizer = getattr(optimizers[0], "optimizer", optimizers[0])
        actual_order = tuple(
            id(param) for group in raw_optimizer.param_groups for param in group["params"]
        )
        expected_order = tuple(id(param) for param in self._stable_wm_world_parameters)
        if actual_order != expected_order:
            raise RuntimeError(
                "Strict world-only optimizer parameter order does not match raw LeWM."
            )


def run_stable_wm_adapter_training(
    model: torch.nn.Module,
    train_set: Any,
    val_set: Any,
    cfg: Any,
    run_dir: str,
    *,
    train_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    from stable_worldmodel.wm.loss import SIGReg

    rnd_gen = train_generator
    if rnd_gen is None:
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
    total_steps = resolve_stable_wm_adapter_total_steps(cfg, train_loader)
    settings = _decoder_training_settings(cfg)
    world_parameters, decoder_parameters = stable_wm_parameter_partitions(model)
    if settings["enabled"] and not hasattr(model, "decoders"):
        # Decoder training is a LeWM-only facility. Other Stable-WM adapter
        # families retain the historical world-only SPT path.
        settings["enabled"] = False
    elif settings["enabled"] and not decoder_parameters:
        raise ValueError("decoder_training.enabled=true requires model.decoders with trainable parameters.")

    scheduler = {
        "type": "LinearWarmupCosineAnnealingLR",
        "warmup_steps": max(1, int(0.01 * total_steps)),
        "max_steps": total_steps,
    }
    if settings["enabled"]:
        # stable-pretraining 0.1.6 uses ordered, first-match regex assignment.
        # Keep the specific decoder expression before the broad model expression.
        optimizers = {
            "decoder_opt": {
                "modules": r"^model\.decoders(?:\.|$)",
                "optimizer": {
                    "type": "AdamW",
                    "lr": settings["decoder_lr"],
                    "weight_decay": settings["decoder_weight_decay"],
                },
                "scheduler": scheduler,
                "interval": "epoch",
            },
            "world_opt": {
                "modules": r"^model(?:\.|$)",
                "optimizer": {
                    "type": "AdamW",
                    "lr": settings["world_lr"],
                    "weight_decay": settings["world_weight_decay"],
                },
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    elif decoder_parameters:
        # Upstream LeWM registers no decoder at all. A negative lookahead keeps
        # MWM's exported decoder weights in the model while excluding them and
        # all decoder descendants from the sole optimizer parameter group.
        optimizers = {
            "world_opt": {
                "modules": r"^model\.(?!decoders(?:\.|$))",
                "optimizer": {
                    "type": "AdamW",
                    "lr": settings["world_lr"],
                    "weight_decay": settings["world_weight_decay"],
                },
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
        world_parameters = stable_wm_upstream_lewm_parameter_order(
            model, world_parameters
        )
    else:
        # Other Stable-WM families have no decoder partition and retain the
        # historical broad-model optimizer path.
        optimizers = {
            "model_opt": {
                "modules": "model",
                "optimizer": {
                    "type": "AdamW",
                    "lr": settings["world_lr"],
                    "weight_decay": settings["world_weight_decay"],
                },
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    trainer_root = prepare_trainer_root(run_dir, cfg)
    callbacks = stable_wm_adapter_callbacks(cfg)
    checkpoint_cb = next(callback for callback in callbacks if isinstance(callback, ModelCheckpoint))
    trainer_runtime = resolve_lightning_trainer_runtime(cfg)
    trainer = pl.Trainer(
        **trainer_runtime,
        precision=cfg.train.get("precision", "bf16"),
        max_epochs=int(cfg.schedule.max_epochs),
        gradient_clip_val=None if settings["enabled"] else settings["world_gradient_clip_val"],
        default_root_dir=str(trainer_root),
        limit_train_batches=cfg.train.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=CSVLogger(save_dir=str(trainer_root), name="csv_logs"),
        enable_checkpointing=True,
        enable_progress_bar=True,
        plugins=[SLURMEnvironment(auto_requeue=bool(cfg.train.get("slurm_auto_requeue", False)))]
        if os.environ.get("SLURM_JOB_ID")
        else None,
    )
    module_kwargs = {
        "model": model,
        "sigreg": SIGReg(
            knots=int(cfg.loss.get("sigreg_knots", 17)),
            num_proj=int(cfg.loss.get("sigreg_num_proj", 1024)),
        ),
        "forward": stable_wm_adapter_forward,
        "optim": optimizers,
        "hparams": as_container(cfg),
    }
    if settings["enabled"]:
        module = OptimizerIsolatedSPTModule(
            world_parameters=world_parameters,
            decoder_parameters=decoder_parameters,
            world_gradient_clip_val=settings["world_gradient_clip_val"],
            decoder_gradient_clip_val=settings["decoder_gradient_clip_val"],
            **module_kwargs,
        )
    elif decoder_parameters:
        module = WorldOnlySPTModule(
            world_parameters=world_parameters,
            **module_kwargs,
        )
    else:
        module = spt.Module(**module_kwargs)
    module.stable_wm_adapter_cfg = cfg
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=PrebuiltLoaderDataModule(train_loader, val_loader),
        seed=stable_wm_fit_seed(cfg),
        ckpt_path=cfg.train.get("resume_checkpoint", None),
    )
    manager()
    selected_checkpoint = select_stable_wm_adapter_export_checkpoint(checkpoint_cb, cfg)
    selected_checkpoint_state: dict[str, Any] = {}
    if selected_checkpoint:
        from mwm.training.stable_wm_export import load_stable_wm_adapter_lightning_state

        selected_checkpoint_state = load_stable_wm_adapter_lightning_state(model, selected_checkpoint)
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


__all__ = [
    "OptimizerIsolatedSPTModule",
    "WorldOnlySPTModule",
    "run_stable_wm_adapter_training",
    "stable_wm_adapter_forward",
    "stable_wm_fit_seed",
    "stable_wm_parameter_partitions",
    "stable_wm_upstream_lewm_parameter_order",
]
