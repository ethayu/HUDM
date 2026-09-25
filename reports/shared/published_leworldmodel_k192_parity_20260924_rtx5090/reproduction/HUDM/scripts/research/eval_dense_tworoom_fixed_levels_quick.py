from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.matrix import main as run_benchmark_matrix


def checkpoint_levels(checkpoint: Path) -> tuple[int, ...]:
    metadata_path = checkpoint / "world_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    levels = tuple(int(item["K"]) for item in metadata["head_architectures"])
    if not levels or tuple(sorted(set(levels))) != levels:
        raise ValueError(f"Checkpoint levels must be non-empty, unique, and sorted: {levels}")
    return levels


def build_config(
    checkpoint: Path,
    output_dir: Path,
    seed: int,
    episodes: int,
    num_envs: int,
    eval_config: Path,
    manifest: Path,
    expected_levels: tuple[int, ...],
) -> dict[str, Any]:
    levels = checkpoint_levels(checkpoint)
    if levels != expected_levels:
        raise ValueError(f"Expected K={expected_levels}, found K={levels} in {checkpoint}")
    runs = []
    for level_index, k in enumerate(levels):
        runs.append(
            {
                "name": f"fixed_k{k}",
                "role": f"fixed_k{k}",
                "checkpoint": str(checkpoint),
                "planner": {
                    "scheduler": {
                        "enabled": True,
                        "mpc": {"mode": "fixed", "level": level_index},
                        "cem": {"mode": "fixed", "level": "base"},
                        "rollout": {"mode": "fixed", "level": "base"},
                    }
                },
                "eval": {"episodes": episodes, "num_envs": num_envs, "seed": seed},
            }
        )
    level_label = "_".join(str(k) for k in levels)
    return {
        "output_dir": str(output_dir),
        "title": f"Dense TwoRoom K={list(levels)} quick paired benchmark",
        "env_id": "swm/TwoRoom-v1",
        "seed": seed,
        "eval_config": str(eval_config),
        "manifest": {
            "group": f"tworoom_dense_k{level_label}_quick_seed{seed}_n{episodes}",
            "path": str(manifest),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick paired benchmark over fixed widths of a dense TwoRoom model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-levels", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-config-only", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1 or args.num_envs > args.episodes:
        raise ValueError("Require 1 <= num_envs <= episodes.")
    for path in (args.checkpoint, args.eval_config, args.manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    config_path = args.output_dir.parent / f"{args.output_dir.name}_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        OmegaConf.to_yaml(
            build_config(
                checkpoint=args.checkpoint.resolve(),
                output_dir=args.output_dir.resolve(),
                seed=args.seed,
                episodes=args.episodes,
                num_envs=args.num_envs,
                eval_config=args.eval_config.resolve(),
                manifest=args.manifest.resolve(),
                expected_levels=tuple(args.expected_levels),
            )
        ),
        encoding="utf-8",
    )
    if not args.write_config_only:
        run_benchmark_matrix(str(config_path))


if __name__ == "__main__":
    main()
