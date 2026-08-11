from __future__ import annotations

from types import SimpleNamespace

import pytest

from mwm.swm.envs import (
    apply_swm_world_runtime_config,
    swm_runtime_eval_callables,
    validate_swm_world_runtime_config,
)


class _Wrapped:
    def __init__(self, threshold: float = 0.05) -> None:
        task = SimpleNamespace(qpos_threshold=threshold)
        self.unwrapped = SimpleNamespace(env=SimpleNamespace(task=task))


def _world(count: int = 2) -> SimpleNamespace:
    return SimpleNamespace(envs=SimpleNamespace(envs=[_Wrapped() for _ in range(count)]))


def test_reacher_qpos_threshold_override_is_explicit_and_reported() -> None:
    world = _world()

    provenance = apply_swm_world_runtime_config(
        world,
        "swm/ReacherDMControl-v0",
        {"reacher_qpos_threshold": 0.1},
    )

    assert provenance == {
        "reacher_qpos_threshold": 0.1,
        "dependency_default_qpos_thresholds": [0.05, 0.05],
        "applied_env_count": 2,
    }
    assert [env.unwrapped.env.task.qpos_threshold for env in world.envs.envs] == [0.1, 0.1]

    # Reacher may recompile its DM-Control task during reset, recreating the
    # dependency default. The post-reset callable must reapply the request.
    world.envs.envs[0].unwrapped.env.task = SimpleNamespace(qpos_threshold=0.05)
    spec = swm_runtime_eval_callables(
        "swm/ReacherDMControl-v0", {"reacher_qpos_threshold": 0.1}
    )[0]
    method = getattr(world.envs.envs[0].unwrapped, spec["method"])
    method(qpos_threshold=spec["args"]["qpos_threshold"]["value"])
    validate_swm_world_runtime_config(world, provenance)
    assert world.envs.envs[0].unwrapped.env.task.qpos_threshold == 0.1


def test_reacher_qpos_threshold_rejects_wrong_env_and_unknown_settings() -> None:
    with pytest.raises(ValueError, match="only valid"):
        apply_swm_world_runtime_config(
            _world(), "swm/PushT-v1", {"reacher_qpos_threshold": 0.1}
        )
    with pytest.raises(ValueError, match="Unsupported env.runtime"):
        apply_swm_world_runtime_config(_world(), "swm/ReacherDMControl-v0", {"typo": 1})
