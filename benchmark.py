from __future__ import annotations

from hudm.benchmark import run_benchmark


def main(cfg_path: str) -> None:
    run_benchmark(cfg_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python benchmark.py <path/to/config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
