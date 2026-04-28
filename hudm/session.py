from __future__ import annotations

from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig

from hudm.artifacts import save_plan_result
from hudm.config import resolve_plan_spec
from hudm.runtime import build_plan_runtime, print_plan_runtime_summary, resolve_dataset_seed
from hudm.session_exec import run_closed_loop
from hudm.session_helpers import (
    load_selected_rollout,
    sample_init_goal_states,
    set_execution_fidelity_finest,
    set_start_pose,
)
from hudm.specs import PlanSpec


def load_plan_spec(cfg_path: str) -> PlanSpec:
    return resolve_plan_spec(cfg_path)


def load_plan_cfg(cfg_path: str) -> DictConfig:
    cfg = resolve_plan_spec(cfg_path).runtime_cfg
    cfg.init_goal.dataset.seed = resolve_dataset_seed(getattr(cfg.init_goal.dataset, "seed", 0))
    return cfg


def _execution_env_from_runtime(runtime: dict[str, Any]) -> object:
    if str(runtime["backend"]) == "particle_sim":
        execution_env = getattr(runtime["planner"], "backend", None)
        if execution_env is None:
            raise ValueError("particle_sim execution runtime requires planner.backend.")
        return execution_env
    return runtime["env"]


def _prepare_execution_env(env: object, init_state: np.ndarray, goal_state: np.ndarray):
    try:
        return env.prepare(seed=0, init_state=init_state, goal_state=goal_state, with_visual=False)
    except TypeError as exc:
        if "with_visual" not in str(exc):
            raise
        return env.prepare(seed=0, init_state=init_state, goal_state=goal_state)


def _step_execution_env(env: object, action: np.ndarray):
    try:
        return env.step(action, with_visual=False)
    except TypeError as exc:
        if "with_visual" not in str(exc):
            raise
        return env.step(action)


def _angle_diff(a: float, b: float) -> float:
    d = float(a) - float(b)
    return float(abs(np.arctan2(np.sin(d), np.cos(d))))


def _agent_in_frame(env: object, state: np.ndarray) -> bool:
    s = np.asarray(state, dtype=np.float32)
    if s.shape[0] < 2:
        return False
    frame_max = float(getattr(env, "window_size", 512.0))
    return bool(np.all(s[:2] >= 0.0) and np.all(s[:2] <= frame_max))


def _meta_float(meta: dict[str, Any], key: str) -> float:
    raw = meta.get(key, np.nan)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _recalculate_goal_for_execution_env(
    execution_env: object,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    sample_meta: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if str(sample_meta.get("source", "")).lower() != "dataset":
        return goal_state, sample_meta
    raw_actions = sample_meta.get("actions", None)
    if raw_actions is None:
        raw_actions = sample_meta.get("gt_action_trajectory", None)
    if raw_actions is None:
        return goal_state, sample_meta

    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] <= 0:
        return goal_state, sample_meta

    init_arr = np.asarray(init_state, dtype=np.float32)
    previous_goal = np.asarray(goal_state, dtype=np.float32)
    set_start_pose(execution_env, init_arr)
    _, cur_state = _prepare_execution_env(execution_env, init_arr, previous_goal)
    set_execution_fidelity_finest(execution_env)
    trajectory = [np.asarray(cur_state, dtype=np.float32).copy()]
    for action in actions:
        _, _, _, info = _step_execution_env(execution_env, np.asarray(action, dtype=np.float32))
        if not isinstance(info, dict) or "state" not in info:
            raise ValueError("Execution env step must return info['state'] for goal recalculation.")
        cur_state = np.asarray(info["state"], dtype=np.float32)
        trajectory.append(cur_state.copy())

    trajectory_arr = np.asarray(trajectory, dtype=np.float32)
    recalculated_goal = trajectory_arr[-1].copy()
    meta = dict(sample_meta)
    meta["goal_recalculated_for_execution_env"] = True
    meta["goal_recalculation_source"] = "baseline_execution_env_action_rollout"
    meta["goal_recalculation_action_count"] = int(actions.shape[0])
    meta["pre_recalculation_goal_state"] = previous_goal.tolist()
    meta["pre_recalculation_pos_diff"] = _meta_float(meta, "pos_diff")
    meta["pre_recalculation_angle_diff"] = _meta_float(meta, "angle_diff")
    meta["gt_state_trajectory"] = trajectory_arr.tolist()
    meta["gt_state_trajectory_source"] = "baseline_execution_env_action_rollout"
    meta["used_action_rollout"] = True
    meta["pos_diff"] = float(np.linalg.norm(recalculated_goal[2:4] - init_arr[2:4]))
    meta["angle_diff"] = _angle_diff(float(recalculated_goal[4]), float(init_arr[4]))
    meta["goal_agent_in_frame"] = _agent_in_frame(execution_env, recalculated_goal)
    print(
        "[init_goal] recalculated goal in execution env "
        f"steps={int(actions.shape[0])} pos_diff={meta['pos_diff']:.3f} "
        f"angle_diff={meta['angle_diff']:.3f}"
    )
    return recalculated_goal, meta


def run_plan_session(
    cfg: DictConfig,
    rollout_selection: Optional[dict] = None,
    schedule_name: Optional[str] = None,
    print_summary: bool = True,
    execution_cfg: Optional[DictConfig] = None,
) -> dict:
    runtime = build_plan_runtime(cfg)
    runtime_cfg = runtime.get("cfg", cfg)
    env = runtime["env"]
    wm_cfg = runtime["wm_cfg"]
    execution_runtime = None
    if execution_cfg is not None:
        execution_runtime = build_plan_runtime(execution_cfg)
    execution_env = _execution_env_from_runtime(execution_runtime or runtime)
    if rollout_selection is None:
        init_state, goal_state, sample_meta = sample_init_goal_states(env, runtime_cfg, wm_cfg=wm_cfg)
    else:
        init_state, goal_state, sample_meta = load_selected_rollout(
            env,
            runtime_cfg,
            wm_cfg=wm_cfg,
            selection=rollout_selection,
        )
    goal_state, sample_meta = _recalculate_goal_for_execution_env(
        execution_env,
        init_state,
        goal_state,
        sample_meta,
    )
    if str(sample_meta.get("source", "")).lower() == "dataset":
        gt_len = int(sample_meta.get("trajectory_len", -1))
        plan_steps = int(cfg.mpc.steps)
        if gt_len > 0 and gt_len != plan_steps:
            print(
                "[warn] Dataset GT trajectory length differs from plan.mpc.steps: "
                f"gt_len={gt_len}, planned_steps={plan_steps}. "
                "Set them equal for length-matched visual comparison."
            )
    if bool(print_summary):
        print_plan_runtime_summary(runtime, runtime_cfg)
    planner = runtime["planner"]
    if hasattr(planner, "reset"):
        planner.reset()
    success, traj, frames, planner_frames, run_stats, trace = run_closed_loop(
        env=runtime["env"],
        wm=runtime["wm"],
        planner=runtime["planner"],
        backend=runtime["backend"],
        cfg=runtime_cfg,
        init_state=init_state,
        goal_state=goal_state,
        device=runtime["device"],
        init_goal_meta=sample_meta,
        execution_env=execution_env,
    )
    runtime["execution_env"] = execution_env
    if execution_runtime is not None:
        runtime["execution_backend"] = execution_runtime["backend"]
        runtime["execution_cfg"] = execution_runtime["cfg"]
    return {
        "cfg": runtime_cfg,
        "runtime": runtime,
        "success": bool(success),
        "trajectory": traj,
        "frames": frames,
        "planner_frames": planner_frames,
        "run_stats": run_stats,
        "trace": trace,
        "init_state": np.asarray(init_state, dtype=np.float32),
        "goal_state": np.asarray(goal_state, dtype=np.float32),
        "sample_meta": sample_meta,
        "schedule_name": schedule_name,
    }


def run_plan_spec(
    spec_or_path: str | PlanSpec,
    *,
    rollout_selection: Optional[dict] = None,
    schedule_name: Optional[str] = None,
    print_summary: bool = True,
) -> dict:
    spec = load_plan_spec(spec_or_path) if isinstance(spec_or_path, str) else spec_or_path
    return run_plan_session(
        spec.runtime_cfg,
        rollout_selection=rollout_selection,
        schedule_name=schedule_name,
        print_summary=print_summary,
    )


__all__ = [
    "load_plan_cfg",
    "load_plan_spec",
    "run_plan_session",
    "run_plan_spec",
    "save_plan_result",
]
