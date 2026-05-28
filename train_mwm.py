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
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mwm.dependency_refs import dependency_refs
from mwm.data.stable_wm import load_dataset_metadata
from mwm.swm.restore import validate_restore_columns
from mwm.checkpoints import LEWM_BASE_ADAPTER_ARCH, save_world_checkpoint
from mwm.models.world_model import MatryoshkaWorldModel


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


def _as_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {k: _as_container(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {k: _as_container(v) for k, v in vars(value).items()}
    return value


def _resolve_lewm_base_adapter_model_cfg(cfg: Any, dataset: Any) -> dict[str, Any]:
    frameskip = int(cfg.data.get("frameskip", 1))
    action_dim = int(dataset.get_dim(str(cfg.data.action_key))) * frameskip
    image_shape_cfg = cfg.model.get("image_shape", "auto")
    image_shape = (
        (int(cfg.model.get("image_size", 224)), int(cfg.model.get("image_size", 224)))
        if str(image_shape_cfg).lower() == "auto"
        else tuple(int(x) for x in image_shape_cfg)
    )
    model_cfg = {
        "D": int(cfg.model.D),
        "K": tuple(int(k) for k in cfg.model.K),
        "action_dim": action_dim,
        "image_shape": tuple(int(x) for x in image_shape),
        "action_block": int(cfg.model.get("action_block", frameskip)),
    }
    return model_cfg


def _stable_checkpoint_config_path(checkpoint: str) -> Path:
    from stable_worldmodel.data import get_cache_dir

    root = Path(str(checkpoint)).expanduser()
    if root.exists():
        if root.is_dir():
            return root / "config.json"
        if root.name == "config.json":
            return root
        return root.parent / "config.json"
    return Path(get_cache_dir(None, sub_folder="checkpoints")) / str(checkpoint) / "config.json"


def _build_trainable_model_from_base(cfg: Any, model_cfg: dict[str, Any]) -> torch.nn.Module:
    from mwm.adapters.base import ComponentPolicy
    from mwm.adapters.lewm import build_mwm_lewm_from_stable_config
    from mwm.adapters.registry import family_for_target
    from mwm.adapters.stable_config import load_stable_wm_config, root_target, stable_config_sha256

    base = cfg.get("base", {}) if hasattr(cfg, "get") else {}
    base = _as_container(base)
    if not base:
        raise ValueError("Trainable Le-WM MWM requires a Stable-WM base checkpoint config.")

    config_path = _stable_checkpoint_config_path(str(base["checkpoint"]))
    source_config, loaded_path = load_stable_wm_config(config_path)
    detected_family = family_for_target(root_target(source_config))
    configured_family = str(base.get("family", detected_family))
    if configured_family != detected_family:
        raise ValueError(
            f"Configured Stable-WM base family {configured_family!r} does not match config target family {detected_family!r}."
        )
    if configured_family != "lewm":
        raise ValueError(f"Unsupported trainable Stable-WM base family {configured_family!r}.")

    mwm_cfg = _as_container(cfg.get("mwm", {}) if hasattr(cfg, "get") else {})
    loss_cfg = _as_container(cfg.get("loss", {}) if hasattr(cfg, "get") else {})
    model_section = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    recipe = {
        **dict(loss_cfg),
        "history_size": int(model_section.get("history_size", loss_cfg.get("history_size", 3))),
        "num_preds": int(model_section.get("num_preds", loss_cfg.get("num_preds", 1))),
        "action_preprocessing": "standard_scaler",
        "loss_scope": dict(mwm_cfg.get("loss_terms", {"regularizers": "shared_latent"})),
    }
    return build_mwm_lewm_from_stable_config(
        source_config=source_config,
        source_config_sha256=stable_config_sha256(loaded_path),
        training_recipe=recipe,
        K=tuple(int(k) for k in model_cfg["K"]),
        action_dim=int(model_cfg["action_dim"]),
        expected_D=int(model_cfg["D"]) if "D" in model_cfg else None,
        action_block=int(model_cfg.get("action_block", 1)),
        image_shape=tuple(int(x) for x in model_cfg["image_shape"]),
        normalize_imagenet=bool(model_cfg.get("normalize_imagenet", True)),
        component_policy=ComponentPolicy.from_mapping(mwm_cfg.get("component_policy", None)),
    )


def _metadata_for_model(metadata: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    merged = {**metadata, **dict(getattr(model, "metadata", {}) or {})}
    mwm_config = getattr(model, "mwm_config", None)
    if isinstance(mwm_config, dict):
        merged["model"] = _as_container(mwm_config)
    return merged


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


def _load_lewm_base_adapter_train_valid_datasets(cfg: Any) -> tuple[Any, Any, Any]:
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
    img_size = int(cfg.model.get("image_size", 224))
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


def _lewm_base_adapter_forward(module: Any, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    cfg = module.lewm_base_adapter_cfg
    if not isinstance(module.model, MatryoshkaWorldModel):
        raise RuntimeError("Le-WM training requires the MWM base-adapter model, not a raw Stable-WM object.")
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


def _run_lewm_base_adapter_training(
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
    checkpoint_cb = _lewm_base_adapter_checkpoint_callback(cfg)
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
        forward=_lewm_base_adapter_forward,
        optim=optimizers,
        hparams=_as_container(cfg),
    )
    module.lewm_base_adapter_cfg = cfg
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


def _prepare_trainer_root(run_dir: str | Path, cfg: Any, *, logs_root: str | Path = "logs") -> Path:
    trainer_root = Path(logs_root) / "mwm_training" / Path(run_dir).name
    if bool(cfg.train.get("clean_trainer_root", True)) and trainer_root.exists():
        shutil.rmtree(trainer_root)
    trainer_root.mkdir(parents=True, exist_ok=True)
    return trainer_root


def _lewm_base_adapter_checkpoint_callback(cfg: Any) -> ModelCheckpoint:
    checkpoint_steps = int(cfg.train.get("checkpoint_every_n_train_steps", 0) or 0)
    checkpoint_kwargs: dict[str, Any] = {"save_last": True, "save_top_k": 0}
    if checkpoint_steps > 0:
        checkpoint_kwargs.update({"every_n_train_steps": checkpoint_steps, "every_n_epochs": 0})
    return ModelCheckpoint(**checkpoint_kwargs)


def _prepare_lewm_base_adapter_context(cfg: Any) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    tr_ds, va_ds, base_ds = _load_lewm_base_adapter_train_valid_datasets(cfg)
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec = validate_restore_columns(str(cfg.env_id), _base_dataset(base_ds).column_names, import_path=restore_import_path)
    model_cfg = _resolve_lewm_base_adapter_model_cfg(cfg, _base_dataset(base_ds))
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
        "architecture_version": LEWM_BASE_ADAPTER_ARCH,
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
        "model": {"target": "mwm.adapters.lewm.build_mwm_lewm_from_stable_config"},
    }
    for key in ("action_low", "action_high"):
        if key in dataset_meta:
            metadata[key] = dataset_meta[key]
    return tr_ds, va_ds, base_ds, model_cfg, metadata


def _load_lewm_base_adapter_lightning_state(lewm: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
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
    tr_ds, va_ds, base_ds, model_cfg, metadata = _prepare_lewm_base_adapter_context(cfg)
    del tr_ds, va_ds
    try:
        model = _build_trainable_model_from_base(cfg, model_cfg)
        checkpoint = _load_lewm_base_adapter_lightning_state(model, checkpoint_path)
        train_info = {
            "epoch": int(checkpoint.get("epoch", 0)),
            "last_checkpoint": str(checkpoint_path),
            "exported_from_lightning_checkpoint": True,
        }
        save_world_checkpoint(model, run_dir, metadata={**_metadata_for_model(metadata, model), **train_info})
    finally:
        _close_dataset_handles(base_ds)
    print(f"Exported Le-WM base-adapter Lightning checkpoint to canonical MWM checkpoint: {run_dir}")


def main(cfg_path: str) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))
    torch.manual_seed(int(cfg.seed))
    backend = str(cfg.train.backend).lower()
    if backend != "stable_worldmodel_lewm":
        raise ValueError(
            "MWM training requires train.backend=stable_worldmodel_lewm so the adapter-owned "
            "Stable-WM base architecture and recipe are explicit."
        )
    run_dir = make_run_dir(
        str(cfg.train.checkpoint_dir),
        str(cfg.train.run_name),
        timestamp=bool(cfg.train.get("timestamp_run_dir", False)),
    )
    tr_ds, va_ds, base_ds, model_cfg, metadata = _prepare_lewm_base_adapter_context(cfg)
    try:
        model = _build_trainable_model_from_base(cfg, model_cfg)
        train_info = _run_lewm_base_adapter_training(model, tr_ds, va_ds, cfg, run_dir)
        save_world_checkpoint(model, run_dir, metadata={**_metadata_for_model(metadata, model), **train_info})
    finally:
        _close_dataset_handles(base_ds)
    print(f"Exact Le-WM training complete. Checkpoints: {run_dir}")


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
        export_lewm_base_adapter_lightning_checkpoint(sys.argv[1], sys.argv[3], output_dir=output)
    else:
        print(
            "Usage: python train_mwm.py CONFIG\n"
            "   or: python train_mwm.py CONFIG --export-from-lightning CHECKPOINT [--output-dir OUTPUT_DIR]"
        )
        raise SystemExit(1)
