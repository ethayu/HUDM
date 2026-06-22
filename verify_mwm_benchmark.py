from __future__ import annotations

from mwm.benchmark.verify import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify MWM benchmark artifacts.")
    parser.add_argument("config", nargs="?", default="configs/benchmark/scheduled_pusht.yaml", help="Benchmark YAML config")
    parser.add_argument("--static-only", action="store_true", help="Validate the config matrix and input checkpoint contracts")
    parser.add_argument("--no-checkpoints", action="store_true", help="Skip checkpoint contract checks in --static-only mode")
    parser.add_argument("--roles", nargs="+", help="Optional role filter, e.g. upstream_lewm_converted")
    args = parser.parse_args()
    main(args.config, static_only=args.static_only, roles=args.roles, check_checkpoints=not args.no_checkpoints)
