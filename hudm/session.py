from __future__ import annotations

from typing import Optional

import numpy as np
from omegaconf import DictConfig

from hudm.artifacts import save_plan_result
from hudm.config import resolve_plan_spec
from hudm.runtime import build_plan_runtime, print_plan_runtime_summary, resolve_dataset_seed
from hudm.session_exec import run_closed_loop
from hudm.session_helpers import load_selected_rollout, sample_init_goal_states
from hudm.specs import PlanSpec


def load_plan_spec(cfg_path: str) -> PlanSpec:
    return resolve_plan_spec(cfg_path)


def load_plan_cfg(cfg_path: str) -> DictConfig:
    cfg = resolve_plan_spec(cfg_path).runtime_cfg
    cfg.init_goal.dataset.seed = resolve_dataset_seed(getattr(cfg.init_goal.dataset, "seed", 0))
    return cfg


def run_plan_session(
    cfg: DictConfig,
    rollout_selection: Optional[dict] = None,
    schedule_name: Optional[str] = None,
    print_summary: bool = True,
) -> dict:
    runtime = build_plan_runtime(cfg)
    env = runtime["env"]
    wm_cfg = runtime["wm_cfg"]
    if rollout_selection is None:
        init_state, goal_state, sample_meta = sample_init_goal_states(env, cfg, wm_cfg=wm_cfg)
    else:
        init_state, goal_state, sample_meta = load_selected_rollout(
            env,
            cfg,
            wm_cfg=wm_cfg,
            selection=rollout_selection,
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
        print_plan_runtime_summary(runtime, cfg)
    planner = runtime["planner"]
    if hasattr(planner, "reset"):
        planner.reset()
    success, traj, frames, planner_frames, run_stats, trace = run_closed_loop(
        env=runtime["env"],
        wm=runtime["wm"],
        planner=runtime["planner"],
        backend=runtime["backend"],
        cfg=cfg,
        init_state=init_state,
        goal_state=goal_state,
        device=runtime["device"],
    )
    return {
        "cfg": cfg,
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
