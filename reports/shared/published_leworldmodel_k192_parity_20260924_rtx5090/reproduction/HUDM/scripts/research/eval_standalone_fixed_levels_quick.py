from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from eval_dense_fixed_levels_quick import checkpoint_identity, validate_manifest


def parse_checkpoint(value: str) -> tuple[int, Path]:
    try:
        level_text, path_text = value.split("=", 1)
        level = int(level_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("checkpoint must be formatted as K=PATH") from exc
    return level, Path(path_text)


def build_config(
    checkpoints: list[tuple[int, Path]],
    output_dir: Path,
    env_id: str,
    seed: int,
    episodes: int,
    num_envs: int,
    eval_config: Path,
    manifest: Path,
) -> dict[str, Any]:
    levels = tuple(level for level, _ in checkpoints)
    if not levels or tuple(sorted(set(levels))) != levels:
        raise ValueError(f"Checkpoint levels must be non-empty, unique, and sorted: {levels}")

    manifest_data = validate_manifest(manifest, env_id, seed, episodes)
    runs = []
    for level, checkpoint in checkpoints:
        checkpoint_env_id, checkpoint_levels = checkpoint_identity(checkpoint)
        if checkpoint_env_id != env_id:
            raise ValueError(
                f"Expected env_id={env_id}, found {checkpoint_env_id} in {checkpoint}"
            )
        if checkpoint_levels != (level,):
            raise ValueError(f"Expected singleton K={level}, found K={checkpoint_levels} in {checkpoint}")
        runs.append(
            {
                "name": f"standalone_k{level}",
                "role": f"standalone_k{level}",
                "checkpoint": str(checkpoint),
                "planner": {
                    "scheduler": {
                        "enabled": True,
                        "mpc": {"mode": "fixed", "level": 0},
                        "cem": {"mode": "fixed", "level": "base"},
                        "rollout": {"mode": "fixed", "level": "base"},
                    }
                },
                "eval": {
                    "episodes": episodes,
                    "budget": episodes,
                    "num_envs": num_envs,
                    "seed": seed,
                    "goal_offset": int(manifest_data["goal_offset"]),
                    "goal_indexing": str(manifest_data.get("goal_indexing", "exact")),
                },
            }
        )

    level_label = "_".join(str(level) for level in levels)
    env_label = env_id.rsplit("/", 1)[-1]
    return {
        "output_dir": str(output_dir),
        "title": f"Standalone {env_label} K={list(levels)} quick paired benchmark",
        "env_id": env_id,
        "seed": seed,
        "eval_config": str(eval_config),
        "manifest": {
            "group": f"{env_label}_standalone_k{level_label}_quick_seed{seed}_n{episodes}",
            "path": str(manifest),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick paired benchmark for independently trained singleton checkpoints."
    )
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-config-only", action="store_true")
    args = parser.parse_args()

    if args.episodes < 1 or args.num_envs < 1 or args.num_envs > args.episodes:
        raise ValueError("Require 1 <= num_envs <= episodes.")
    checkpoints = [(level, path.resolve()) for level, path in args.checkpoint]
    for _, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.eval_config, args.manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    config_path = args.output_dir.parent / f"{args.output_dir.name}_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        OmegaConf.to_yaml(
            build_config(
                checkpoints=checkpoints,
                output_dir=args.output_dir.resolve(),
                env_id=args.env_id,
                seed=args.seed,
                episodes=args.episodes,
                num_envs=args.num_envs,
                eval_config=args.eval_config.resolve(),
                manifest=args.manifest.resolve(),
            )
        ),
        encoding="utf-8",
    )
    if not args.write_config_only:
        from mwm.benchmark.matrix import main as run_benchmark_matrix

        run_benchmark_matrix(str(config_path))


if __name__ == "__main__":
    main()
