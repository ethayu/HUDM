from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class OGBenchRestoreStateWrapper(gym.Wrapper):
    """Add ``restore_state = concat(qpos, qvel)`` to SWM OGBench info dicts."""

    @property
    def unwrapped(self):
        return self

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # SWM evaluate_from_dataset applies callables on env.unwrapped, so expose
        # the restore adapter on the underlying env as well as on this wrapper.
        setattr(self.env.unwrapped, "set_restore_state", self.set_restore_state)

    def _restore_state(self) -> np.ndarray:
        env = self.env.unwrapped
        data = getattr(env, "_data", None)
        model = getattr(env, "_model", None)
        if data is None or model is None:
            nested = getattr(env, "env", None)
            data = getattr(nested, "data", data)
            model = getattr(nested, "model", model)
        if data is None or model is None or not hasattr(data, "qpos") or not hasattr(data, "qvel"):
            raise RuntimeError(
                "OGBenchRestoreStateWrapper could not find MuJoCo qpos/qvel on the wrapped env."
            )
        return np.concatenate([np.asarray(data.qpos).reshape(-1), np.asarray(data.qvel).reshape(-1)]).astype(np.float32)

    @property
    def variation_space(self):
        return getattr(self.env.unwrapped, "variation_space", None)

    def set_restore_state(self, restore_state: Any) -> None:
        env = self.env.unwrapped
        data = getattr(env, "_data", None)
        model = getattr(env, "_model", None)
        if data is None or model is None:
            nested = getattr(env, "env", None)
            data = getattr(nested, "data", data)
            model = getattr(nested, "model", model)
        if data is None or model is None:
            raise RuntimeError("OGBenchRestoreStateWrapper could not restore MuJoCo state.")
        state = np.asarray(restore_state, dtype=np.float32).reshape(-1)
        nq = int(getattr(model, "nq"))
        nv = int(getattr(model, "nv"))
        if state.shape[0] != nq + nv:
            raise ValueError(f"restore_state has shape {state.shape}; expected ({nq + nv},)")
        if hasattr(env, "set_state"):
            env.set_state(state[:nq], state[nq:])
        else:
            data.qpos[:] = state[:nq]
            data.qvel[:] = state[nq:]
            if hasattr(env, "pre_step"):
                env.pre_step()
            if hasattr(env, "post_step"):
                env.post_step()

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        obs, info = self.env.reset(*args, **kwargs)
        info["restore_state"] = self._restore_state()
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["restore_state"] = self._restore_state()
        return obs, reward, terminated, truncated, info


def ogbench_restore_wrapper(env: gym.Env) -> OGBenchRestoreStateWrapper:
    return OGBenchRestoreStateWrapper(env)
