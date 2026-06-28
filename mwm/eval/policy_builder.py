from __future__ import annotations

import math
from typing import Any

import torch
from omegaconf import OmegaConf
from stable_worldmodel.policy import PlanConfig

from mwm.planning.scheduled_cem import MWMScheduledCEMSolver
from mwm.preprocessing.images import mwm_image_input_transform
from mwm.eval.policy import MWMWorldModelPolicy


def build_mwm_policy(
    model: Any,
    metadata: dict[str, Any],
    cfg: Any,
    device: torch.device,
    action_space: Any,
    process: dict[str, Any],
) -> Any:
    if not hasattr(model, "get_cost_with_fidelity"):
        raise TypeError("MWM evaluator requires models with get_cost_with_fidelity(...).")
    configured_action_block = int(metadata.get("action_block", metadata.get("model", {}).get("action_block", 1)))
    action_block = int(cfg.planner.get("action_block", configured_action_block))
    raw_batch_size = cfg.planner.get("batch_size", "auto")
    planner_batch_size = (
        int(cfg.eval.num_envs)
        if raw_batch_size is None or str(raw_batch_size).lower() == "auto"
        else int(raw_batch_size)
    )
    raw_topk = cfg.planner.get("topk", None)
    topk = (
        max(1, int(raw_topk))
        if raw_topk is not None
        else max(1, int(round(int(cfg.planner.pop_size) * float(cfg.planner.elite_frac))))
    )
    raw_pop_schedule = cfg.planner.get("pop_schedule", None)
    pop_schedule = OmegaConf.to_container(raw_pop_schedule, resolve=True) if raw_pop_schedule else None
    replan_interval = max(1, int(cfg.planner.receding_horizon) * int(action_block))
    eval_budget = int(cfg.eval.get("budget", cfg.planner.horizon))
    max_replans = max(1, int(math.ceil(eval_budget / replan_interval)))
    solver = MWMScheduledCEMSolver(
        model,
        batch_size=max(1, planner_batch_size),
        num_samples=int(cfg.planner.pop_size),
        var_scale=float(cfg.planner.init_std),
        n_steps=int(cfg.planner.n_iter),
        topk=topk,
        scheduler=OmegaConf.to_container(cfg.planner.scheduler, resolve=True),
        device=device,
        seed=int(cfg.planner.seed if cfg.planner.get("seed", None) is not None else cfg.eval.seed),
        clamp_actions=bool(cfg.planner.get("clamp_actions", False)),
        std_unbiased=bool(cfg.planner.get("std_unbiased", True)),
        pop_schedule=pop_schedule,
        elite_frac=float(cfg.planner.elite_frac) if pop_schedule is not None else None,
        max_replans=max_replans,
    )
    plan_cfg = PlanConfig(
        horizon=int(cfg.planner.horizon),
        receding_horizon=int(cfg.planner.receding_horizon),
        action_block=action_block,
        warm_start=bool(cfg.planner.warm_start),
    )
    image_transform = {"pixels": mwm_image_input_transform, "goal": mwm_image_input_transform}
    return MWMWorldModelPolicy(model=model, solver=solver, config=plan_cfg, process=process, transform=image_transform)


__all__ = ["build_mwm_policy"]
