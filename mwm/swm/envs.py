from __future__ import annotations

import json
from types import MethodType
from typing import Any

import numpy as np


_REACHER_THRESHOLD_METHOD = "_mwm_set_reacher_qpos_threshold"


def _reacher_task(wrapped: Any, index: int) -> Any:
    unwrapped = getattr(wrapped, "unwrapped", wrapped)
    dm_env = getattr(unwrapped, "env", None)
    task = getattr(dm_env, "task", None)
    if task is None or not hasattr(task, "qpos_threshold"):
        raise RuntimeError(
            f"Reacher environment {index} does not expose env.task.qpos_threshold"
        )
    return unwrapped, task


def _set_reacher_qpos_threshold(self: Any, qpos_threshold: float) -> None:
    task = getattr(getattr(self, "env", None), "task", None)
    if task is None or not hasattr(task, "qpos_threshold"):
        raise RuntimeError("Reacher task was not available after environment reset/recompilation")
    task.qpos_threshold = float(qpos_threshold)


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
        extra_wrappers=[],
        **(env_kwargs or {}),
    )


def apply_swm_world_runtime_config(
    world: Any,
    env_id: str,
    raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply explicit post-construction environment settings with provenance.

    Stable WorldModel's Reacher qpos-match tolerance is not exposed as a Gym
    constructor argument.  It also changed from 0.1 to 0.05 between the
    historical LeWM evaluation code and the later packaged environment.  Keep
    that evaluation-semantic choice explicit instead of inheriting whichever
    dependency version happens to be installed.
    """

    config = dict(raw or {})
    supported = {"reacher_qpos_threshold"}
    unknown = sorted(set(config) - supported)
    if unknown:
        raise ValueError(f"Unsupported env.runtime settings: {unknown}")
    if "reacher_qpos_threshold" not in config:
        return {}

    threshold = float(config["reacher_qpos_threshold"])
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"reacher_qpos_threshold must be finite and positive, got {threshold}")
    if str(env_id) != "swm/ReacherDMControl-v0":
        raise ValueError(
            "env.runtime.reacher_qpos_threshold is only valid for "
            f"swm/ReacherDMControl-v0, got {env_id!r}"
        )

    pool = getattr(world, "envs", None)
    environments = getattr(pool, "envs", None)
    if not isinstance(environments, list) or not environments:
        raise RuntimeError("Stable WorldModel world does not expose a non-empty envs.envs list")
    previous: list[float] = []
    for index, wrapped in enumerate(environments):
        unwrapped, task = _reacher_task(wrapped, index)
        previous.append(float(task.qpos_threshold))
        task.qpos_threshold = threshold
        setattr(unwrapped, _REACHER_THRESHOLD_METHOD, MethodType(_set_reacher_qpos_threshold, unwrapped))
    return {
        "reacher_qpos_threshold": threshold,
        "dependency_default_qpos_thresholds": previous,
        "applied_env_count": len(environments),
    }


def swm_runtime_eval_callables(env_id: str, raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    config = dict(raw or {})
    if "reacher_qpos_threshold" not in config:
        return []
    if str(env_id) != "swm/ReacherDMControl-v0":
        raise ValueError(f"Reacher threshold runtime callable is invalid for {env_id!r}")
    threshold = float(config["reacher_qpos_threshold"])
    return [
        {
            "method": _REACHER_THRESHOLD_METHOD,
            "args": {
                "qpos_threshold": {
                    "value": threshold,
                    "in_dataset": False,
                }
            },
        }
    ]


def validate_swm_world_runtime_config(world: Any, provenance: dict[str, Any]) -> None:
    if "reacher_qpos_threshold" not in provenance:
        return
    expected = float(provenance["reacher_qpos_threshold"])
    environments = getattr(getattr(world, "envs", None), "envs", None)
    if not isinstance(environments, list) or not environments:
        raise RuntimeError("Stable WorldModel world does not expose a non-empty envs.envs list")
    observed = [float(_reacher_task(wrapped, index)[1].qpos_threshold) for index, wrapped in enumerate(environments)]
    if any(value != expected for value in observed):
        raise RuntimeError(
            f"Reacher qpos threshold did not survive reset/recompilation: expected {expected}, observed {observed}"
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
