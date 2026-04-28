from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
from omegaconf import OmegaConf

from datasets.swm_hdf5 import swm_dataset_metadata_path
from hudm.swm_envs import (
    import_object,
    make_swm_world,
    parse_env_kwargs,
    parse_image_shape,
    validate_continuous_box_action_space,
)
from hudm.swm_restore import restore_spec_for_env, validate_restore_columns


DEFAULTS = {
    "env_id": "swm/PushT-v1",
    "image_shape": 96,
    "max_episode_steps": 100,
    "num_envs": 1,
    "episodes": 10,
    "seed": 0,
    "output_path": "data/swm_dataset.h5",
    "goal_conditioned": True,
    "env_kwargs": {},
    "policy": {"import_path": None},
    "restore": {"import_path": None},
}


def _record_dataset_to_path(world: Any, output_path: Path, episodes: int, seed: int) -> None:
    if output_path.suffix != ".h5":
        raise ValueError(f"SWM HDF5 output_path must end with .h5, got {output_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to append to existing SWM HDF5 dataset {output_path}. "
            "Choose a new output_path; append/resume compatibility is not supported in HUDM v1."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    world.record_dataset(
        dataset_name=output_path.stem,
        episodes=int(episodes),
        seed=int(seed),
        cache_dir=output_path.parent,
    )


def _build_policy(import_path: str | None, seed: int):
    if import_path:
        obj = import_object(import_path)
        return obj() if isinstance(obj, type) else obj
    from stable_worldmodel.policy import RandomPolicy

    return RandomPolicy(seed=int(seed))


def main(cfg_path: str) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
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
        _record_dataset_to_path(world, output_path, int(cfg.episodes), int(cfg.seed))
        with h5py.File(output_path, "r") as f:
            dataset_columns = [str(k) for k in f.keys() if str(k) not in {"ep_len", "ep_offset"}]
        restore_spec = validate_restore_columns(env_id, dataset_columns, import_path=restore_import_path)
        metadata: dict[str, Any] = {
            "format": "swm_hdf5",
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
        with open(swm_dataset_metadata_path(output_path), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(f"Collected {cfg.episodes} episodes from {env_id} into {output_path}")
    finally:
        world.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python collect_swm.py configs/collect_swm.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
