from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from hudm.runtime import resolve_dataset_seed
from pusht.pusht_wrapper import PushTWrapper


def sample_init_goal_states(
    env: PushTWrapper,
    cfg: DictConfig,
    wm_cfg: DictConfig | None,
    selection: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ds_cfg = cfg.init_goal.dataset
    src = str(cfg.init_goal.source).lower()
    if src == "random":
        init_state, goal_state = env.sample_random_init_goal_states(seed=ds_cfg.seed)
        return init_state, goal_state, {"source": "random"}

    zarr_path = ds_cfg.zarr_path
    if zarr_path is None:
        if wm_cfg is not None and getattr(wm_cfg, "data", None) is not None:
            zarr_path = wm_cfg.data.zarr_path
        else:
            raise ValueError(
                "plan.init_goal.dataset.zarr_path must be set when backend is gt_env or particle_sim "
                "and no world config is available."
            )
    split_ratio = ds_cfg.split_ratio
    if split_ratio is None:
        split_ratio = 0.8 if wm_cfg is None else float(getattr(wm_cfg.data, "split_ratio", 0.8))
    sample_seed = resolve_dataset_seed(getattr(ds_cfg, "seed", 0))

    init_state, goal_state, meta = env.sample_dataset_init_goal_states(
        dataset=str(zarr_path),
        trajectory_len=int(ds_cfg.trajectory_len),
        split=str(ds_cfg.split),
        split_ratio=float(split_ratio),
        seed=sample_seed,
        reconstruct_goal_state=int(getattr(ds_cfg, "reconstruct_goal_state", 0)),
        selection=selection,
    )
    print(
        "[init_goal] source=dataset "
        f"episode={meta['episode_index']} start={meta['start_index']} "
        f"goal={meta['goal_index']} len={meta['trajectory_len']} split={meta['split']} "
        f"seed={sample_seed}"
    )
    meta = dict(meta)
    meta["source"] = "dataset"
    meta["zarr_path"] = str(zarr_path)
    meta["split_ratio"] = float(split_ratio)
    meta["seed"] = int(sample_seed)
    return init_state, goal_state, meta


def set_goal_pose(env: PushTWrapper, goal_state: np.ndarray) -> None:
    goal_state = np.asarray(goal_state, dtype=np.float32)
    if goal_state.shape[0] < 5:
        raise ValueError(f"goal_state must have at least 5 dims, got {goal_state.shape}")
    env.set_task_goal(goal_state[2:5])


def set_start_pose(env: PushTWrapper, init_state: np.ndarray) -> None:
    init_state = np.asarray(init_state, dtype=np.float32)
    if init_state.shape[0] < 5:
        raise ValueError(f"init_state must have at least 5 dims, got {init_state.shape}")
    if hasattr(env, "set_task_start"):
        env.set_task_start(init_state[2:5])


def set_execution_fidelity_finest(env: PushTWrapper) -> None:
    if hasattr(env, "_planning_fidelity_num_levels") and hasattr(env, "set_planning_fidelity_level"):
        n_levels = int(getattr(env, "_planning_fidelity_num_levels", 1))
        env.set_planning_fidelity_level(max(0, n_levels - 1))


def load_selected_rollout(
    env: PushTWrapper,
    cfg: DictConfig,
    wm_cfg: DictConfig | None,
    selection: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if str(cfg.init_goal.source).lower() != "dataset":
        raise ValueError("Explicit rollout selection requires init_goal.source=dataset.")
    return sample_init_goal_states(env, cfg, wm_cfg=wm_cfg, selection=selection)
