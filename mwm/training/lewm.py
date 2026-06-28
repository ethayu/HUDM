from __future__ import annotations

import argparse
from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "DEFAULTS": ("mwm.training.lewm_config", "DEFAULTS"),
    "make_run_dir": ("mwm.training.lewm_config", "make_run_dir"),
    "export_lewm_base_adapter_lightning_checkpoint": (
        "mwm.training.lewm_export",
        "export_lewm_base_adapter_lightning_checkpoint",
    ),
}


def main(cfg_path: str, *, overrides: list[str] | None = None) -> None:
    import torch

    from mwm.checkpoint_io import save_world_checkpoint
    from mwm.config_cli import load_config
    from mwm.training import lewm_data, lewm_lightning, lewm_model
    from mwm.training.lewm_config import DEFAULTS, make_run_dir, validate_lewm_loss_config

    cfg = load_config(DEFAULTS, cfg_path, overrides or [])
    validate_lewm_loss_config(cfg.loss)
    torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))
    torch.manual_seed(int(cfg.seed))
    backend = str(cfg.train.backend).lower()
    if backend not in {"stable_worldmodel_lewm", "stable_worldmodel_prejepa", "stable_worldmodel_dino", "stable_worldmodel_dinowm"}:
        raise ValueError(
            "MWM training requires train.backend=stable_worldmodel_lewm or stable_worldmodel_prejepa so the adapter-owned "
            "Stable-WM base architecture and recipe are explicit."
        )
    run_dir = make_run_dir(
        str(cfg.train.checkpoint_dir),
        str(cfg.train.run_name),
        timestamp=bool(cfg.train.get("timestamp_run_dir", False)),
    )
    tr_ds, va_ds, base_ds, model_cfg, metadata = lewm_data.prepare_lewm_base_adapter_context(cfg)
    try:
        model = lewm_model.build_trainable_model_from_base(cfg, model_cfg)
        train_info = lewm_lightning.run_lewm_base_adapter_training(model, tr_ds, va_ds, cfg, run_dir)
        save_world_checkpoint(model, run_dir, metadata={**lewm_model.metadata_for_model(metadata, model), **train_info})
    finally:
        lewm_data.close_dataset_handles(base_ds)
    print(f"Exact Le-WM training complete. Checkpoints: {run_dir}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train an MWM checkpoint.")
    parser.add_argument("config", help="Training YAML config")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. train.batch_size=16")
    parser.add_argument("--export-from-lightning", metavar="CHECKPOINT", help="Export a Lightning checkpoint")
    parser.add_argument("--output-dir", help="Output directory for --export-from-lightning")
    args = parser.parse_args()
    if args.export_from_lightning:
        if args.set:
            parser.error("--set is only supported for training, not --export-from-lightning")
        from mwm.training.lewm_export import export_lewm_base_adapter_lightning_checkpoint

        export_lewm_base_adapter_lightning_checkpoint(
            args.config,
            args.export_from_lightning,
            output_dir=args.output_dir,
        )
    else:
        if args.output_dir:
            parser.error("--output-dir requires --export-from-lightning")
        main(args.config, overrides=args.set)


if __name__ == "__main__":
    _main()


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULTS",
    "export_lewm_base_adapter_lightning_checkpoint",
    "main",
    "make_run_dir",
]
