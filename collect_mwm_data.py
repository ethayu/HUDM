from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.config_cli import load_config
from mwm.data.metadata import write_dataset_metadata
from mwm.data.paths import local_path
from mwm.imports import import_object
from mwm.swm.envs import (
    make_swm_world,
    parse_env_kwargs,
    parse_image_shape,
    validate_continuous_box_action_space,
)
from mwm.swm.restore import restore_spec_for_env, validate_restore_columns


DEFAULTS = {
    "env_id": "swm/PushT-v1",
    "image_shape": 96,
    "max_episode_steps": 100,
    "num_envs": 1,
    "episodes": 10,
    "seed": 0,
    "output_path": "data/swm_dataset.lance",
    "format": "lance",
    "goal_conditioned": True,
    "env_kwargs": {},
    "policy": {"import_path": None},
    "restore": {"import_path": None},
}


def _record_dataset_to_path(world: Any, output_path: Path, episodes: int, seed: int, format: str | None = None) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to append to existing SWM dataset {output_path}. "
            "Choose a new output_path; append/resume compatibility is not supported."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = str(format or "lance")
    if fmt != "lance":
        raise ValueError(f"MWM v1 only supports Lance collection, got format={fmt!r}.")
    if not hasattr(world, "collect"):
        raise AttributeError("MWM data collection requires Stable-WM World.collect(...).")
    world.collect(output_path, episodes=int(episodes), seed=int(seed), format="lance")


def _build_policy(import_path: str | None, seed: int):
    if import_path:
        obj = import_object(import_path)
        return obj() if isinstance(obj, type) else obj
    from stable_worldmodel.policy import RandomPolicy

    return RandomPolicy(seed=int(seed))


def main(cfg_path: str, *, overrides: list[str] | None = None) -> None:
    cfg = load_config(DEFAULTS, cfg_path, overrides or [])
    env_id = str(cfg.env_id)
    image_shape = parse_image_shape(cfg.image_shape)
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    if restore_import_path is None:
        restore_spec_for_env(env_id)
    env_kwargs = parse_env_kwargs(OmegaConf.to_container(cfg.get("env_kwargs", {}), resolve=True))

    world = make_swm_world(
        env_id,
        num_envs=int(cfg.num_envs),
        image_shape=image_shape,
        max_episode_steps=int(cfg.max_episode_steps),
        goal_conditioned=bool(cfg.goal_conditioned),
        env_kwargs=env_kwargs,
    )
    try:
        action_low, action_high = validate_continuous_box_action_space(world.envs.single_action_space, env_id)
        policy = _build_policy(cfg.policy.get("import_path", None), int(cfg.seed))
        world.set_policy(policy)
        output_path = Path(str(cfg.output_path))
        data_format = str(cfg.get("format", "lance"))
        if data_format != "lance":
            raise ValueError(f"MWM v1 only supports Lance datasets, got format={data_format!r}.")
        _record_dataset_to_path(world, output_path, int(cfg.episodes), int(cfg.seed), data_format)
        from stable_worldmodel.data import load_dataset

        ds = load_dataset(local_path(output_path), format="lance")
        dataset_columns = list(ds.column_names)
        restore_spec = validate_restore_columns(env_id, dataset_columns, import_path=restore_import_path)
        metadata: dict[str, Any] = {
            "format": f"swm_{data_format}",
            "env_id": env_id,
            "image_shape": list(image_shape),
            "max_episode_steps": int(cfg.max_episode_steps),
            "num_envs": int(cfg.num_envs),
            "episodes": int(cfg.episodes),
            "seed": int(cfg.seed),
            "restore_spec": restore_spec.spec_id,
            "action_dim": int(action_low.size),
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
            "dataset": {
                "pixels_key": "pixels",
                "action_key": "action",
            },
        }
        write_dataset_metadata(output_path, metadata)
        print(f"Collected {cfg.episodes} episodes from {env_id} into {output_path}")
    finally:
        world.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect an MWM dataset.")
    parser.add_argument("config", help="Collection YAML config")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. seed=1")
    args = parser.parse_args()
    main(args.config, overrides=args.set)
