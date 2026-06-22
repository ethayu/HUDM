from __future__ import annotations

from mwm.config_cli import load_config as load_config
from mwm.training.lewm import export_lewm_base_adapter_lightning_checkpoint, main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an MWM checkpoint.")
    parser.add_argument("config", help="Training YAML config")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. train.batch_size=16")
    parser.add_argument("--export-from-lightning", metavar="CHECKPOINT", help="Export a Lightning checkpoint")
    parser.add_argument("--output-dir", help="Output directory for --export-from-lightning")
    args = parser.parse_args()
    if args.export_from_lightning:
        if args.set:
            parser.error("--set is only supported for training, not --export-from-lightning")
        export_lewm_base_adapter_lightning_checkpoint(
            args.config,
            args.export_from_lightning,
            output_dir=args.output_dir,
        )
    else:
        if args.output_dir:
            parser.error("--output-dir requires --export-from-lightning")
        main(args.config, overrides=args.set)
