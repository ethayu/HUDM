from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.matrix import main as run_benchmark_matrix


def checkpoint_identity(checkpoint: Path) -> tuple[str, tuple[int, ...]]:
    metadata_path = checkpoint / "world_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    levels = tuple(int(item["K"]) for item in metadata["head_architectures"])
    if not levels or tuple(sorted(set(levels))) != levels:
        raise ValueError(f"Checkpoint levels must be non-empty, unique, and sorted: {levels}")
    return str(metadata["env_id"]), levels


def validate_manifest(manifest: Path, env_id: str, seed: int, episodes: int) -> dict[str, Any]:
    with manifest.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    actual = {
        "env_id": str(data.get("env_id")),
        "seed": int(data.get("seed", -1)),
        "pairs": len(data.get("pairs", [])),
        "eval_budget": int(data.get("eval_budget", -1)),
    }
    expected = {"env_id": env_id, "seed": seed, "pairs": episodes, "eval_budget": episodes}
    if actual != expected:
        raise ValueError(f"Manifest contract mismatch: {actual} != {expected}")
    goal_offset = int(data.get("goal_offset", -1))
    goal_indexing = str(data.get("goal_indexing", "exact"))
    if goal_offset < 1 or goal_indexing not in {"exact", "upstream_lewm_end_exclusive"}:
        raise ValueError(
            f"Manifest has invalid goal contract: goal_offset={goal_offset}, "
            f"goal_indexing={goal_indexing!r}"
        )
    return data


def build_config(
    checkpoint: Path,
    output_dir: Path,
    env_id: str,
    seed: int,
    episodes: int,
    num_envs: int,
    eval_config: Path,
    manifest: Path,
    expected_levels: tuple[int, ...],
    flop_accounting: str = "none",
) -> dict[str, Any]:
    checkpoint_env_id, levels = checkpoint_identity(checkpoint)
    if checkpoint_env_id != env_id:
        raise ValueError(f"Expected env_id={env_id}, found {checkpoint_env_id} in {checkpoint}")
    if levels != expected_levels:
        raise ValueError(f"Expected K={expected_levels}, found K={levels} in {checkpoint}")
    manifest_data = validate_manifest(manifest, env_id, seed, episodes)
    runs = []
    for level_index, k in enumerate(levels):
        runs.append(
            {
                "name": f"fixed_k{k}",
                "role": f"fixed_k{k}",
                "checkpoint": str(checkpoint),
                "planner": {
                    "flop_accounting": flop_accounting,
                    "scheduler": {
                        "enabled": True,
                        "mpc": {"mode": "fixed", "level": level_index},
                        "cem": {"mode": "fixed", "level": "base"},
                        "rollout": {"mode": "fixed", "level": "base"},
                    }
                },
                # The immutable paired manifest is authoritative for goal
                # indexing. Carry this contract into every cell explicitly so
                # an older eval template cannot reinterpret the same pairs.
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
    level_label = "_".join(str(k) for k in levels)
    env_label = env_id.rsplit("/", 1)[-1]
    return {
        "output_dir": str(output_dir),
        "title": f"Dense {env_label} K={list(levels)} quick paired benchmark",
        "env_id": env_id,
        "seed": seed,
        "eval_config": str(eval_config),
        "manifest": {
            "group": f"{env_label}_dense_k{level_label}_quick_seed{seed}_n{episodes}",
            "path": str(manifest),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick paired fixed-width benchmark for a dense checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--expected-levels", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-config-only", action="store_true")
    parser.add_argument(
        "--flop-accounting",
        choices=("none", "dynamics_audit"),
        default="none",
    )
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
                env_id=args.env_id,
                seed=args.seed,
                episodes=args.episodes,
                num_envs=args.num_envs,
                eval_config=args.eval_config.resolve(),
                manifest=args.manifest.resolve(),
                expected_levels=tuple(args.expected_levels),
                flop_accounting=args.flop_accounting,
            )
        ),
        encoding="utf-8",
    )
    if not args.write_config_only:
        run_benchmark_matrix(str(config_path))


if __name__ == "__main__":
    main()
