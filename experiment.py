from __future__ import annotations

from hudm.experiment import run_experiment


def main(cfg_path: str) -> None:
    run_experiment(cfg_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python experiment.py <path/to/config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
