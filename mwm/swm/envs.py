from __future__ import annotations

import importlib
import json
from typing import Any, Callable

import numpy as np

from mwm.swm.restore import restore_spec_for_env


def parse_image_shape(value: int | str | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        out = (int(value), int(value))
    elif isinstance(value, str):
        text = value.lower().replace("x", ",")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) == 1:
            out = (int(parts[0]), int(parts[0]))
        elif len(parts) == 2:
            out = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError(f"Cannot parse image_shape from {value!r}")
    else:
        if len(value) != 2:
            raise ValueError(f"image_shape must have 2 entries, got {value!r}")
        out = (int(value[0]), int(value[1]))
    if out[0] != out[1]:
        raise ValueError(f"MWM v1 requires square image_shape, got {out}")
    if out[0] <= 0:
        raise ValueError(f"image_shape must be positive, got {out}")
    return out


def import_object(path: str) -> Any:
    module_name, sep, attr = str(path).partition(":")
    if not sep:
        module_name, sep, attr = str(path).rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import path must be 'module:attr' or 'module.attr', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def swm_extra_wrappers_for_env(env_id: str) -> list[Callable]:
    wrappers: list[Callable] = []
    try:
        spec = restore_spec_for_env(env_id)
    except ValueError:
        return wrappers
    del spec
    return wrappers


def parse_env_kwargs(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    return dict(json.loads(raw))


def make_swm_world(
    env_id: str,
    num_envs: int,
    image_shape: int | str | tuple[int, int] | list[int],
    max_episode_steps: int = 100,
    goal_conditioned: bool = True,
    env_kwargs: dict[str, Any] | None = None,
):
    import stable_worldmodel as swm

    shape = parse_image_shape(image_shape)
    return swm.World(
        env_id,
        num_envs=int(num_envs),
        image_shape=shape,
        max_episode_steps=int(max_episode_steps),
        goal_conditioned=bool(goal_conditioned),
        extra_wrappers=swm_extra_wrappers_for_env(env_id),
        **(env_kwargs or {}),
    )


def validate_continuous_box_action_space(action_space: Any, env_id: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        from gymnasium import spaces
    except Exception:  # pragma: no cover - gymnasium is provided by SWM
        spaces = None
    if spaces is not None and not isinstance(action_space, spaces.Box):
        raise ValueError(f"MWM v1 only supports continuous Box actions; {env_id} has {action_space}.")
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise ValueError(f"Action space for {env_id} does not expose low/high bounds: {action_space}.")
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    if low.shape != high.shape or low.size == 0:
        raise ValueError(f"Invalid action bounds for {env_id}: low={low.shape} high={high.shape}")
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError(f"MWM v1 requires finite action bounds for {env_id}.")
    return low, high


def infer_swm_action_space(
    env_id: str,
    image_shape: int | str | tuple[int, int] | list[int],
    max_episode_steps: int = 100,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    world = make_swm_world(
        env_id,
        num_envs=1,
        image_shape=image_shape,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs,
    )
    try:
        action_space = world.envs.single_action_space
        low, high = validate_continuous_box_action_space(action_space, env_id)
        return int(low.size), low, high
    finally:
        world.close()
