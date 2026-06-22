from __future__ import annotations

from mwm.config_cli import load_config as load_config
from mwm.eval.runner import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an MWM checkpoint.")
    parser.add_argument("config", help="Evaluation YAML config")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. eval.seed=1")
    args = parser.parse_args()
    main(args.config, overrides=args.set)
