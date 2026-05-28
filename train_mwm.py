from __future__ import annotations

import datetime
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import lightning as pl
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from mwm.dependency_refs import dependency_refs
from mwm.training import mwm_spt_forward
from mwm.adapters.lewm import LeWMMatryoshkaWorldModel, build_mwm_lewm, mwm_from_lewm_object
from mwm.data.stable_wm import MWMTrainSampleTransform, load_dataset_metadata, load_stable_wm_dataset_for_mwm
from mwm.swm.restore import validate_restore_columns
from mwm.checkpoints import save_world_checkpoint


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
        "encoder": "cnn",
        "D": 256,
        "K": [32, 64, 128, 256],
        "action_dim": "auto",
        "image_shape": "auto",
        "freeze_encoder": False,
    },
    "restore": {"import_path": None},
    "train": {
        "batch_size": 8,
        "horizon": 16,
        "num_workers": 0,
        "no_cuda": False,
        "checkpoint_dir": "checkpoints_mwm",
        "run_name": "mwm_lewm",
        "backend": "stable_pretraining",
        "timestamp_run_dir": False,
        "clean_trainer_root": True,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "matmul_precision": "high",
        "prefetch_factor": 2,
        "checkpoint_every_n_train_steps": 0,
        "slurm_auto_requeue": False,
    },
    "optim": {"lr": 3e-4},
    "loss": {"rollout_weight": 1.0, "recon_weight": 0.0, "sigreg_weight": 0.0},
    "schedule": {"max_epochs": 30, "patience": 5, "min_delta": 1e-3},
}


def make_run_dir(root: str, tag: str, *, timestamp: bool = False) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{tag}_{ts}" if timestamp else tag)
    os.makedirs(path, exist_ok=True)
    return path


def _dataset_metadata(path: str | Path) -> dict[str, Any]:
    return load_dataset_metadata(path, required=False)


def _local_path(path: str | Path) -> str:
    p = Path(str(path))
    return str(p.resolve()) if p.exists() else str(path)


def _base_dataset(dataset: Any) -> Any:
    return getattr(dataset, "dataset", dataset)


def _close_dataset_handles(*datasets: Any) -> None:
    seen: set[int] = set()
    for dataset in datasets:
        base = _base_dataset(dataset)
        if id(base) in seen:
            continue
        seen.add(id(base))
        close = getattr(base, "close", None)
        if callable(close):
            close()


def _dataset_path(dataset: Any, cfg: Any) -> str:
    del dataset
    return str(cfg.data.path)


def _as_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {k: _as_container(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {k: _as_container(v) for k, v in vars(value).items()}
    return value


def _infer_image_shape_action_dim(dataset: Any) -> tuple[tuple[int, int], int]:
    sample = dataset[0]
    x = sample["x"]
    a = sample["a"]
    if x.ndim != 4:
        raise ValueError(f"Expected sample x as (T,C,H,W), got {tuple(x.shape)}")
    return (int(x.shape[-2]), int(x.shape[-1])), int(a.reshape(a.shape[0], -1).shape[-1])


def _resolve_model_cfg(cfg: Any, dataset: Any) -> dict[str, Any]:
    meta = _dataset_metadata(_dataset_path(dataset, cfg))
    inferred_image_shape, inferred_action_dim = _infer_image_shape_action_dim(dataset)
    image_shape = inferred_image_shape if str(cfg.model.image_shape).lower() == "auto" else tuple(cfg.model.image_shape)
    action_dim = inferred_action_dim if str(cfg.model.action_dim).lower() == "auto" else int(cfg.model.action_dim)
    if "action_dim" in meta and int(meta["action_dim"]) != int(action_dim):
        frameskip = int(getattr(_base_dataset(dataset), "frameskip", cfg.data.get("frameskip", 1)))
        if int(meta["action_dim"]) * frameskip != int(action_dim):
            raise ValueError(f"Dataset metadata action_dim={meta['action_dim']} does not match configured {action_dim}.")
    if "image_shape" in meta and tuple(int(x) for x in meta["image_shape"]) != tuple(int(x) for x in image_shape):
        raise ValueError(f"Dataset metadata image_shape={meta['image_shape']} does not match configured {image_shape}.")
    model_cfg = {
        "encoder": str(cfg.model.encoder),
        "D": int(cfg.model.D),
        "K": tuple(int(k) for k in cfg.model.K),
        "action_dim": int(action_dim),
        "image_shape": tuple(int(x) for x in image_shape),
        "freeze_encoder": bool(cfg.model.freeze_encoder),
        "action_block": int(cfg.model.get("action_block", getattr(_base_dataset(dataset), "frameskip", 1))),
    }
    passthrough = (
        "dynamics",
        "normalize_imagenet",
        "vit_model_name",
        "vit_size",
        "vit_patch_size",
        "vit_image_size",
        "vit_pretrained",
        "vit_use_mask_token",
        "action_block",
        "predictor_depth",
        "predictor_heads",
        "predictor_dim_head",
        "predictor_mlp_scale",
        "predictor_mlp_dim",
        "predictor_dropout",
    )
    for key in passthrough:
        if key in cfg.model:
            model_cfg[key] = cfg.model[key]
    return model_cfg


def _load_train_valid_datasets(cfg: Any) -> tuple[Any, Any, Any]:
    data_format = str(cfg.data.get("format", "lance"))
    if data_format != "lance":
        raise ValueError(f"MWM v1 training only supports Lance datasets, got format={data_format!r}.")

    base = load_stable_wm_dataset_for_mwm(
        _local_path(cfg.data.path),
        format=data_format,
        frameskip=int(cfg.data.get("frameskip", 1)),
        num_steps=int(cfg.train.horizon),
        transform=MWMTrainSampleTransform(
            pixels_key=str(cfg.data.pixels_key),
            action_key=str(cfg.data.action_key),
        ),
    )
    episodes = list(range(len(getattr(base, "lengths", []))))
    if len(episodes) < 2:
        raise ValueError("Stable-WM MWM training needs at least two episodes for an episode-level train/valid split.")
    rng = torch.Generator().manual_seed(int(cfg.seed))
    order = torch.randperm(len(episodes), generator=rng).tolist()
    n_train_eps = min(len(episodes) - 1, max(1, int(round(len(episodes) * float(cfg.data.split_ratio)))))
    train_eps = set(order[:n_train_eps])
    train_idx = [i for i, (ep, _) in enumerate(base.clip_indices) if int(ep) in train_eps]
    valid_idx = [i for i, (ep, _) in enumerate(base.clip_indices) if int(ep) not in train_eps]
    if not train_idx or not valid_idx:
        raise ValueError("Episode-level split produced an empty train or validation set.")
    return Subset(base, train_idx), Subset(base, valid_idx), base


def _run_epoch(model, loader, cfg, device: torch.device, optimizer=None) -> tuple[float, dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    rollout_losses: list[float] = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = mwm_spt_forward(model, batch, loss_cfg=OmegaConf.to_container(cfg.loss, resolve=True))
        loss = out["loss"]
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        rollout_losses.append(float(out["rollout_loss"].detach().cpu().item()))
    return (
        float(sum(losses) / max(1, len(losses))),
        {"rollout_loss": float(sum(rollout_losses) / max(1, len(rollout_losses)))},
    )


def _limit_batches(loader: DataLoader, limit: Any):
    if isinstance(limit, float) and 0 < float(limit) <= 1:
        max_batches = max(1, int(round(len(loader) * float(limit))))
    else:
        max_batches = int(limit)
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        yield batch


def _build_exact_lewm_object(model_cfg: dict[str, Any], cfg: Any) -> torch.nn.Module:
    from stable_pretraining.backbone.utils import vit_hf
    from stable_worldmodel.wm.lewm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor

    d = int(model_cfg["D"])
    history_size = int(cfg.model.get("history_size", cfg.loss.get("history_size", 3)))
    encoder = vit_hf(
        size=str(model_cfg.get("vit_size", "tiny")),
        patch_size=int(model_cfg.get("vit_patch_size", 14)),
        image_size=int(model_cfg.get("vit_image_size", 224)),
        pretrained=bool(model_cfg.get("vit_pretrained", False)),
        use_mask_token=bool(model_cfg.get("vit_use_mask_token", False)),
    )
    return LeWM(
        encoder=encoder,
        predictor=Predictor(
            num_frames=history_size,
            input_dim=d,
            hidden_dim=d,
            output_dim=d,
            depth=int(model_cfg.get("predictor_depth", 6)),
            heads=int(model_cfg.get("predictor_heads", 16)),
            mlp_dim=int(model_cfg.get("predictor_mlp_dim", 2048)),
            dim_head=int(model_cfg.get("predictor_dim_head", 64)),
            dropout=float(model_cfg.get("predictor_dropout", 0.1)),
            emb_dropout=float(model_cfg.get("predictor_emb_dropout", 0.0)),
        ),
        action_encoder=Embedder(input_dim=int(model_cfg["action_dim"]), emb_dim=d),
        projector=MLP(input_dim=d, output_dim=d, hidden_dim=int(cfg.model.get("projector_hidden_dim", 2048)), norm_fn=torch.nn.BatchNorm1d),
        pred_proj=MLP(input_dim=d, output_dim=d, hidden_dim=int(cfg.model.get("projector_hidden_dim", 2048)), norm_fn=torch.nn.BatchNorm1d),
    )


def _resolve_exact_model_cfg(cfg: Any, dataset: Any) -> dict[str, Any]:
    frameskip = int(cfg.data.get("frameskip", 1))
    action_dim = int(dataset.get_dim(str(cfg.data.action_key))) * frameskip
    img_size = int(cfg.model.get("vit_image_size", cfg.model.get("image_size", 224)))
    model_cfg = {
        "encoder": str(cfg.model.encoder),
        "D": int(cfg.model.D),
        "K": tuple(int(k) for k in cfg.model.K),
        "action_dim": action_dim,
        "image_shape": (img_size, img_size),
        "freeze_encoder": bool(cfg.model.freeze_encoder),
        "action_block": int(cfg.model.get("action_block", frameskip)),
    }
    for key in (
        "normalize_imagenet",
        "vit_size",
        "vit_patch_size",
        "vit_image_size",
        "vit_pretrained",
        "vit_use_mask_token",
        "history_size",
        "num_preds",
        "predictor_depth",
        "predictor_heads",
        "predictor_dim_head",
        "predictor_mlp_dim",
        "predictor_dropout",
        "predictor_emb_dropout",
        "projector_hidden_dim",
    ):
        if key in cfg.model:
            model_cfg[key] = cfg.model[key]
    return model_cfg


class _ZScoreScaler:
    def __init__(self, eps: float = 1e-8) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.eps = float(eps)

    def fit(self, values: Any) -> "_ZScoreScaler":
        arr = np.asarray(values).reshape(-1, np.asarray(values).shape[-1])
        arr = arr[~np.isnan(arr).any(axis=1)]
        self.mean = arr.mean(axis=0, keepdims=True)
        self.std = arr.std(axis=0, keepdims=True)
        return self

    def __call__(self, values: Any) -> Any:
        if self.mean is None or self.std is None:
            raise RuntimeError("_ZScoreScaler must be fitted before use.")
        if torch.is_tensor(values):
            mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
            std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
            return ((values - mean) / std.clamp(min=self.eps)).float()
        return (values - self.mean) / np.maximum(self.std, self.eps)


def _column_normalizer(dataset: Any, source: str, target: str) -> Any:
    from stable_pretraining.data.transforms import WrapTorchTransform

    scaler = _ZScoreScaler().fit(np.asarray(dataset.get_col_data(source)))
    return WrapTorchTransform(scaler, source=source, target=target)


def _load_exact_lewm_train_valid_datasets(cfg: Any) -> tuple[Any, Any, Any]:
    import stable_pretraining as spt
    from stable_pretraining import data as dt
    from stable_worldmodel.data import load_dataset

    data_format = str(cfg.data.get("format", "lance"))
    if data_format != "lance":
        raise ValueError(f"Exact Le-WM parity training only supports Lance datasets, got format={data_format!r}.")
    history_size = int(cfg.model.get("history_size", cfg.loss.get("history_size", 3)))
    num_preds = int(cfg.model.get("num_preds", cfg.loss.get("num_preds", 1)))
    keys_to_load = list(cfg.data.get("keys_to_load", ["pixels", "action", "proprio", "state"]))
    dataset = load_dataset(
        _local_path(cfg.data.path),
        transform=None,
        format=data_format,
        frameskip=int(cfg.data.get("frameskip", 1)),
        num_steps=history_size + num_preds,
        keys_to_load=keys_to_load,
        keys_to_cache=list(cfg.data.get("keys_to_cache", ["action", "proprio", "state"])),
    )
    pixels_key = str(cfg.data.pixels_key)
    img_size = int(cfg.model.get("vit_image_size", cfg.model.get("image_size", 224)))
    imagenet_stats = dt.dataset_stats.ImageNet
    transforms = [
        dt.transforms.ToImage(**imagenet_stats, source=pixels_key, target=pixels_key),
        dt.transforms.Resize(img_size, source=pixels_key, target=pixels_key),
    ]
    for col in keys_to_load:
        if str(col).startswith("pixels"):
            continue
        if col in getattr(dataset, "column_names", []):
            transforms.append(_column_normalizer(dataset, str(col), str(col)))
    dataset.transform = dt.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[float(cfg.data.split_ratio), 1.0 - float(cfg.data.split_ratio)],
        generator=rnd_gen,
    )
    return train_set, val_set, dataset


def _exact_lewm_forward(module: Any, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    cfg = module.exact_cfg
    if isinstance(module.model, LeWMMatryoshkaWorldModel):
        output = module.model.training_loss(
            batch,
            level_weights=cfg.loss.get("level_weights", None),
            rollout_weight=float(cfg.loss.get("rollout_weight", 1.0)),
            sigreg=module.sigreg,
            sigreg_weight=float(cfg.loss.get("sigreg_weight", cfg.loss.get("sigreg", {}).get("weight", 0.0))),
            sigreg_scope=str(cfg.loss.get("sigreg_scope", "shared_latent")),
        )
        if hasattr(module, "log_dict"):
            module.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}, on_step=True, sync_dist=True)
        return output
    history_size = int(cfg.model.get("history_size", cfg.loss.get("history_size", 3)))
    num_preds = int(cfg.model.get("num_preds", cfg.loss.get("num_preds", 1)))
    sigreg_weight = float(cfg.loss.get("sigreg_weight", cfg.loss.get("sigreg", {}).get("weight", 0.0)))
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = module.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    pred_emb = module.model.predict(emb[:, :history_size], act_emb[:, :history_size])
    tgt_emb = emb[:, num_preds:]
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = module.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
    if hasattr(module, "log_dict"):
        module.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}, on_step=True, sync_dist=True)
    return output


def _run_exact_lewm_training(
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
    total_steps = int(cfg.schedule.max_epochs) * len(train_loader)
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
    trainer_root = _prepare_trainer_root(run_dir, cfg)
    checkpoint_cb = _exact_lewm_checkpoint_callback(cfg)
    trainer = pl.Trainer(
        accelerator="cpu" if bool(cfg.train.no_cuda) or not torch.cuda.is_available() else "gpu",
        devices=1,
        precision=cfg.train.get("precision", "bf16"),
        max_epochs=int(cfg.schedule.max_epochs),
        gradient_clip_val=float(cfg.train.get("gradient_clip_val", 1.0)),
        default_root_dir=str(trainer_root),
        limit_train_batches=cfg.train.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),
        callbacks=[checkpoint_cb],
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
        forward=_exact_lewm_forward,
        optim=optimizers,
        hparams=_as_container(cfg),
    )
    module.exact_cfg = cfg
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=_PrebuiltLoaderDataModule(train_loader, val_loader),
        seed=int(cfg.seed),
    )
    manager()
    return {
        "epoch": int(getattr(trainer, "current_epoch", int(cfg.schedule.max_epochs))),
        "last_checkpoint": str(checkpoint_cb.last_model_path or "") or None,
    }


class _PrebuiltLoaderDataModule(pl.LightningDataModule):
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader


class _MWMDataModule(pl.LightningDataModule):
    def __init__(self, train_ds: Any, valid_ds: Any, cfg: Any) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.cfg = cfg

    def _loader_kwargs(self, *, shuffle: bool) -> dict[str, Any]:
        num_workers = int(self.cfg.train.num_workers)
        kwargs: dict[str, Any] = {
            "batch_size": int(self.cfg.train.batch_size),
            "shuffle": bool(shuffle),
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available() and not bool(self.cfg.train.no_cuda),
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = int(self.cfg.train.get("prefetch_factor", 2))
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_ds, **self._loader_kwargs(shuffle=False))


def _spt_forward(module: Any, batch: dict[str, torch.Tensor], stage: str = "fit") -> dict[str, torch.Tensor]:
    out = mwm_spt_forward(module.model, batch, loss_cfg=module.loss_cfg)
    metric = "train_loss" if stage == "fit" else "val_loss" if stage == "validate" else f"{stage}_loss"
    if hasattr(module, "log"):
        module.log(metric, out["loss"], on_step=False, on_epoch=True, prog_bar=False)
    return out


def _prepare_trainer_root(run_dir: str | Path, cfg: Any, *, logs_root: str | Path = "logs") -> Path:
    trainer_root = Path(logs_root) / "mwm_training" / Path(run_dir).name
    if bool(cfg.train.get("clean_trainer_root", True)) and trainer_root.exists():
        shutil.rmtree(trainer_root)
    trainer_root.mkdir(parents=True, exist_ok=True)
    return trainer_root


def _exact_lewm_checkpoint_callback(cfg: Any) -> ModelCheckpoint:
    checkpoint_steps = int(cfg.train.get("checkpoint_every_n_train_steps", 0) or 0)
    checkpoint_kwargs: dict[str, Any] = {"save_last": True, "save_top_k": 0}
    if checkpoint_steps > 0:
        checkpoint_kwargs.update({"every_n_train_steps": checkpoint_steps, "every_n_epochs": 0})
    return ModelCheckpoint(**checkpoint_kwargs)


def _run_stable_pretraining(model: torch.nn.Module, train_ds: Any, valid_ds: Any, cfg: Any, run_dir: str) -> dict[str, Any]:
    import stable_pretraining as spt

    trainer_root = _prepare_trainer_root(run_dir, cfg)
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True)
    callbacks = [
        checkpoint_cb,
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(cfg.schedule.patience),
            min_delta=float(cfg.schedule.min_delta),
        ),
    ]
    trainer = pl.Trainer(
        accelerator="cpu" if bool(cfg.train.no_cuda) or not torch.cuda.is_available() else "gpu",
        devices=1,
        max_epochs=int(cfg.schedule.max_epochs),
        default_root_dir=str(trainer_root),
        limit_train_batches=cfg.train.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=callbacks,
        plugins=[SLURMEnvironment(auto_requeue=bool(cfg.train.get("slurm_auto_requeue", False)))]
        if os.environ.get("SLURM_JOB_ID")
        else None,
    )
    module = spt.Module(
        forward=_spt_forward,
        model=model,
        loss_cfg=_as_container(cfg.loss),
        optim={"optimizer": {"type": "Adam", "lr": float(cfg.optim.lr)}},
        hparams=_as_container(cfg),
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=_MWMDataModule(train_ds, valid_ds, cfg),
        seed=int(cfg.seed),
    )
    manager()
    best_path = str(checkpoint_cb.best_model_path or "")
    if best_path:
        state = torch.load(best_path, map_location="cpu")
        model_state = {k.removeprefix("model."): v for k, v in state.get("state_dict", {}).items() if k.startswith("model.")}
        if model_state:
            model.load_state_dict(model_state)
    return {
        "best_checkpoint": best_path or None,
        "best_val": float(checkpoint_cb.best_model_score.item()) if checkpoint_cb.best_model_score is not None else None,
        "epoch": int(getattr(trainer, "current_epoch", int(cfg.schedule.max_epochs))),
    }


def _prepare_exact_lewm_context(cfg: Any) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    tr_ds, va_ds, base_ds = _load_exact_lewm_train_valid_datasets(cfg)
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec = validate_restore_columns(str(cfg.env_id), _base_dataset(base_ds).column_names, import_path=restore_import_path)
    model_cfg = _resolve_exact_model_cfg(cfg, _base_dataset(base_ds))
    dataset_meta = _dataset_metadata(str(cfg.data.path))
    base_action_dim = int(dataset_meta.get("action_dim", _base_dataset(base_ds).get_dim(str(cfg.data.action_key))))
    metadata = {
        "env_id": str(cfg.env_id),
        "restore_spec": restore_spec.spec_id,
        "image_shape": [int(x) for x in model_cfg["image_shape"]],
        "action_dim": base_action_dim,
        "action_block": int(model_cfg.get("action_block", 1)),
        "action_preprocessing": "standard_scaler",
        "levels": [int(k) for k in model_cfg["K"]],
        "architecture_version": LeWMMatryoshkaWorldModel.architecture_version,
        "action_spec": {
            "dim": int(model_cfg["action_dim"]),
            "base_dim": base_action_dim,
            "block": int(model_cfg.get("action_block", 1)),
        },
        "training_backend": str(cfg.train.backend),
        "dependencies": dependency_refs(Path(__file__).resolve().parent),
        "dataset": {
            "path": str(cfg.data.path),
            "pixels_key": str(cfg.data.pixels_key),
            "action_key": str(cfg.data.action_key),
            "split": "stable_pretraining_random_split",
            "normalized_columns": list(cfg.data.get("keys_to_load", ["pixels", "action", "proprio", "state"])),
        },
        "model": {"target": "mwm.adapters.lewm.build_mwm_lewm", **model_cfg},
    }
    for key in ("action_low", "action_high"):
        if key in dataset_meta:
            metadata[key] = dataset_meta[key]
    return tr_ds, va_ds, base_ds, model_cfg, metadata


def _load_exact_lewm_lightning_state(lewm: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {k.removeprefix("model."): v for k, v in state_dict.items() if str(k).startswith("model.")}
    if not model_state:
        raise ValueError(f"Lightning checkpoint {checkpoint_path} does not contain any 'model.' parameters.")
    missing, unexpected = lewm.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Could not load exact Le-WM state from {checkpoint_path}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return checkpoint if isinstance(checkpoint, dict) else {}


def export_exact_lewm_lightning_checkpoint(
    cfg_path: str,
    checkpoint_path: str,
    output_dir: str | None = None,
) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    if str(cfg.train.backend).lower() not in {"stable_worldmodel_lewm", "exact_lewm"}:
        raise ValueError("Lightning export is only supported for the exact Le-WM training backend.")
    torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))
    torch.manual_seed(int(cfg.seed))
    run_dir = output_dir or make_run_dir(
        str(cfg.train.checkpoint_dir),
        str(cfg.train.run_name),
        timestamp=bool(cfg.train.get("timestamp_run_dir", False)),
    )
    tr_ds, va_ds, base_ds, model_cfg, metadata = _prepare_exact_lewm_context(cfg)
    del tr_ds, va_ds
    try:
        model = build_mwm_lewm(model_cfg)
        checkpoint = _load_exact_lewm_lightning_state(model, checkpoint_path)
        train_info = {
            "epoch": int(checkpoint.get("epoch", 0)),
            "last_checkpoint": str(checkpoint_path),
            "exported_from_lightning_checkpoint": True,
        }
        save_world_checkpoint(model, run_dir, metadata={**metadata, **train_info})
    finally:
        _close_dataset_handles(base_ds)
    print(f"Exported exact Le-WM Lightning checkpoint to canonical MWM checkpoint: {run_dir}")


def main(cfg_path: str) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))
    torch.manual_seed(int(cfg.seed))
    backend = str(cfg.train.backend).lower()
    model_levels = [int(k) for k in cfg.model.K]
    is_single_level_lewm = (
        len(model_levels) == 1
        and int(model_levels[0]) == int(cfg.model.D)
        and str(cfg.model.get("dynamics", "lewm")).lower() in {"lewm", "stable_wm_lewm"}
    )
    if backend == "stable_pretraining" and str(cfg.model.get("dynamics", "lewm")).lower() in {"lewm", "stable_wm_lewm"}:
        raise ValueError(
            "Trainable Le-WM MWM must use train.backend=stable_worldmodel_lewm "
            "so both K=[D] and scheduled K use the adapter-owned Le-WM base path."
        )
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(cfg.train.no_cuda) else "cpu")
    run_dir = make_run_dir(
        str(cfg.train.checkpoint_dir),
        str(cfg.train.run_name),
        timestamp=bool(cfg.train.get("timestamp_run_dir", False)),
    )

    if backend in {"stable_worldmodel_lewm", "exact_lewm"}:
        tr_ds, va_ds, base_ds, model_cfg, metadata = _prepare_exact_lewm_context(cfg)
        model = build_mwm_lewm(model_cfg)
        train_info = _run_exact_lewm_training(model, tr_ds, va_ds, cfg, run_dir)
        save_world_checkpoint(model, run_dir, metadata={**metadata, **train_info})
        _close_dataset_handles(base_ds)
        print(f"Exact Le-WM training complete. Checkpoints: {run_dir}")
        return

    tr_ds, va_ds, base_ds = _load_train_valid_datasets(cfg)
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec = validate_restore_columns(str(cfg.env_id), _base_dataset(base_ds).column_names, import_path=restore_import_path)
    model_cfg = _resolve_model_cfg(cfg, tr_ds)
    dataset_meta = _dataset_metadata(_dataset_path(base_ds, cfg))
    base_action_dim = int(
        dataset_meta.get("action_dim", model_cfg["action_dim"] // max(1, int(model_cfg.get("action_block", 1))))
    )
    metadata = {
        "env_id": str(cfg.env_id),
        "restore_spec": restore_spec.spec_id,
        "image_shape": [int(x) for x in model_cfg["image_shape"]],
        "action_dim": base_action_dim,
        "action_block": int(model_cfg.get("action_block", 1)),
        "action_preprocessing": "identity",
        "levels": [int(k) for k in model_cfg["K"]],
        "action_spec": {
            "dim": int(model_cfg["action_dim"]),
            "base_dim": base_action_dim,
            "block": int(model_cfg.get("action_block", 1)),
        },
        "training_backend": str(cfg.train.backend),
        "dependencies": dependency_refs(Path(__file__).resolve().parent),
        "dataset": {
            "path": _dataset_path(base_ds, cfg),
            "pixels_key": str(cfg.data.pixels_key),
            "action_key": str(cfg.data.action_key),
        },
        "model": {"target": "mwm.adapters.lewm.build_mwm_lewm", **model_cfg},
    }
    for key in ("action_low", "action_high"):
        if key in dataset_meta:
            metadata[key] = dataset_meta[key]

    model = build_mwm_lewm(model_cfg)
    if backend == "stable_pretraining":
        train_info = _run_stable_pretraining(model, tr_ds, va_ds, cfg, run_dir)
        save_world_checkpoint(model, run_dir, metadata={**metadata, **train_info})
        print(f"Training complete. Checkpoints: {run_dir}")
        return

    model.to(device)
    tr_loader = DataLoader(tr_ds, batch_size=int(cfg.train.batch_size), shuffle=True, num_workers=int(cfg.train.num_workers))
    va_loader = DataLoader(va_ds, batch_size=int(cfg.train.batch_size), shuffle=False, num_workers=int(cfg.train.num_workers))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optim.lr))
    best_val = float("inf")
    no_improve = 0
    for epoch in range(1, int(cfg.schedule.max_epochs) + 1):
        train_loss, train_logs = _run_epoch(model, tr_loader, cfg, device, optimizer=optimizer)
        with torch.no_grad():
            val_loss, val_logs = _run_epoch(model, va_loader, cfg, device, optimizer=None)
        print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}")
        if val_loss + float(cfg.schedule.min_delta) < best_val:
            best_val = val_loss
            no_improve = 0
            save_world_checkpoint(model, run_dir, metadata={**metadata, "epoch": epoch, "best_val": best_val})
        else:
            no_improve += 1
            if no_improve >= int(cfg.schedule.patience):
                print("converged (patience reached)")
                break
        del train_logs, val_logs
    print(f"Training complete. Checkpoints: {run_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) in {4, 6} and sys.argv[2] == "--export-from-lightning":
        output = None
        if len(sys.argv) == 6:
            if sys.argv[4] != "--output-dir":
                print(
                    "Usage: python train_mwm.py CONFIG --export-from-lightning CHECKPOINT "
                    "[--output-dir OUTPUT_DIR]"
                )
                raise SystemExit(1)
            output = sys.argv[5]
        export_exact_lewm_lightning_checkpoint(sys.argv[1], sys.argv[3], output_dir=output)
    else:
        print(
            "Usage: python train_mwm.py CONFIG\n"
            "   or: python train_mwm.py CONFIG --export-from-lightning CHECKPOINT [--output-dir OUTPUT_DIR]"
        )
        raise SystemExit(1)
