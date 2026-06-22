from __future__ import annotations

from mwm.config_cli import load_config as load_config
from mwm.benchmark.matrix import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an MWM benchmark matrix.")
    parser.add_argument("config", help="Benchmark YAML config")
    parser.add_argument("--roles", nargs="+", help="Optional role filter, e.g. upstream_lewm_converted")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. seed=1")
    args = parser.parse_args()
    main(args.config, roles=args.roles, overrides=args.set)
