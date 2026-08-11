from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from mwm.data.sampling import StartGoalPair
from mwm.eval.policy_builder import build_mwm_policy
from mwm.eval.runtime import effective_goal_offset, normalize_goal_indexing
from mwm.eval.videos import collect_video_paths
from mwm.io import jsonable
from mwm.swm.envs import (
    apply_swm_world_runtime_config,
    make_swm_world,
    parse_env_kwargs,
    swm_runtime_eval_callables,
    validate_continuous_box_action_space,
    validate_swm_world_runtime_config,
)


def run_batch(
    *,
    env_id: str,
    image_shape: tuple[int, int],
    model: Any,
    metadata: dict[str, Any],
    dataset: Any,
    pairs: list[StartGoalPair],
    cfg: Any,
    device: torch.device,
    eval_callables: list[dict[str, Any]],
    batch_index: int,
    process: dict[str, Any],
) -> dict[str, Any]:
    env_kwargs = parse_env_kwargs(OmegaConf.to_container(cfg.env.get("kwargs", {}), resolve=True))
    world = make_swm_world(
        env_id,
        num_envs=len(pairs),
        image_shape=image_shape,
        max_episode_steps=int(cfg.env.max_episode_steps),
        goal_conditioned=bool(cfg.env.goal_conditioned),
        env_kwargs=env_kwargs,
    )
    try:
        env_runtime = apply_swm_world_runtime_config(
            world,
            env_id,
            OmegaConf.to_container(cfg.env.get("runtime", {}), resolve=True),
        )
        low, _ = validate_continuous_box_action_space(world.envs.single_action_space, env_id)
        if "action_dim" not in metadata:
            metadata["action_dim"] = int(low.shape[0])
        if low.shape[0] != int(metadata["action_dim"]):
            raise ValueError(f"Env action_dim={low.shape[0]} does not match checkpoint action_dim={metadata['action_dim']}.")
        policy = build_mwm_policy(model, metadata, cfg, device, world.envs.single_action_space, process)
        world.set_policy(policy)
        if hasattr(policy, "reset_trace"):
            policy.reset_trace()
        batch_video_path = Path(str(cfg.eval.video_path)) / f"batch_{int(batch_index):04d}"
        goal_indexing = normalize_goal_indexing(str(cfg.eval.get("goal_indexing", "exact")))
        runtime_goal_offset = effective_goal_offset(int(cfg.eval.goal_offset), goal_indexing)
        runtime_raw = OmegaConf.to_container(cfg.env.get("runtime", {}), resolve=True)
        runtime_callables = [*swm_runtime_eval_callables(env_id, runtime_raw), *eval_callables]
        eval_kwargs = dict(
            dataset=dataset,
            episodes_idx=[p.episode for p in pairs],
            start_steps=[p.start_step for p in pairs],
            eval_budget=int(cfg.eval.budget),
            callables=runtime_callables,
        )
        swm_results = world.evaluate(
            **eval_kwargs,
            goal_offset=runtime_goal_offset,
            video=str(batch_video_path) if bool(cfg.eval.save_video) else None,
        )
        validate_swm_world_runtime_config(world, env_runtime)
        if "reacher_qpos_threshold" in env_runtime:
            env_runtime["post_reset_validation"] = "passed"
        videos = [str(p) for p in collect_video_paths(batch_video_path)] if bool(cfg.eval.save_video) else []
        diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
        review_trace = policy.review_trace() if hasattr(policy, "review_trace") else {}
        return {
            "pairs": [
                {
                    "episode": p.episode,
                    "start_step": p.start_step,
                    "goal_step": p.goal_step,
                    "start_row": p.start_row,
                    "goal_row": p.goal_row,
                }
                for p in pairs
            ],
            "swm_results": jsonable(swm_results),
            "planning_diagnostics": diagnostics,
            "review_trace": jsonable(review_trace),
            "env_runtime": env_runtime,
            "goal_indexing": {
                "mode": goal_indexing,
                "requested_offset": int(cfg.eval.goal_offset),
                "effective_offset": runtime_goal_offset,
            },
            "videos": videos,
        }
    finally:
        world.close()


def combine_swm_results(batches: list[dict[str, Any]]) -> dict[str, Any]:
    successes: list[bool] = []
    seeds: list[Any] = []
    for batch in batches:
        results = batch.get("swm_results", {})
        successes.extend(bool(x) for x in results.get("episode_successes", []))
        batch_seeds = results.get("seeds")
        if batch_seeds is not None:
            seeds.extend(batch_seeds if isinstance(batch_seeds, list) else [batch_seeds])
    return {
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "episode_successes": successes,
        "seeds": seeds or None,
    }


def combine_mwm_diagnostics(batches: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [dict(batch.get("planning_diagnostics", {}).get("summary", {})) for batch in batches]
    traces = [diag for batch in batches for diag in batch.get("planning_diagnostics", {}).get("trace", [])]
    total_replans = int(sum(int(s.get("replans", 0)) for s in summaries))
    total_actions = int(sum(int(s.get("actions_recorded", s.get("action_calls", 0))) for s in summaries))
    total_time = float(sum(float(s.get("total_plan_time_sec", 0.0)) for s in summaries))
    total_policy_time = float(sum(float(s.get("total_policy_time_sec", 0.0)) for s in summaries))
    total_latent_work = int(sum(int(s.get("latent_work_total", 0)) for s in summaries))
    total_bits = int(sum(int(s.get("total_bits_used_estimate", 0)) for s in summaries))
    total_dynamics_flops = int(sum(int(s.get("dynamics_flops_total", 0)) for s in summaries))
    total_flop_audit_errors = int(sum(int(s.get("flop_audit_error_count", 0)) for s in summaries))
    total_cem_cost_calls = int(sum(int(s.get("cem_cost_calls", 0)) for s in summaries))
    total_candidate_action_values = int(sum(int(s.get("candidate_action_values", 0)) for s in summaries))
    level_counts: dict[str, int] = {}
    k_counts: dict[str, int] = {}
    rollout_semantics_counts: dict[str, int] = {}
    policy_semantics_counts: dict[str, int] = {}
    for diag in traces:
        key = str(diag.get("base_level_idx", "unknown"))
        level_counts[key] = level_counts.get(key, 0) + 1
        k_key = str(diag.get("base_k", "unknown"))
        k_counts[k_key] = k_counts.get(k_key, 0) + 1
        semantics_key = str(diag.get("model_rollout_semantics", "unknown"))
        rollout_semantics_counts[semantics_key] = rollout_semantics_counts.get(semantics_key, 0) + 1
        policy_key = str(diag.get("policy_semantics", "unknown"))
        policy_semantics_counts[policy_key] = policy_semantics_counts.get(policy_key, 0) + 1
    return {
        "summary": {
            "actions_recorded": total_actions,
            "replans": total_replans,
            "total_plan_time_sec": total_time,
            "mean_plan_time_sec": total_time / total_replans if total_replans else 0.0,
            "total_policy_time_sec": total_policy_time,
            "mean_policy_time_sec": total_policy_time / total_actions if total_actions else 0.0,
            "latent_work_total": total_latent_work,
            "total_bits_used_estimate": total_bits,
            "dynamics_flops_total": total_dynamics_flops,
            "flop_audit_error_count": total_flop_audit_errors,
            "cem_cost_calls": total_cem_cost_calls,
            "candidate_action_values": total_candidate_action_values,
        },
        "trace": traces,
        "schedule_level_counts": level_counts,
        "schedule_k_counts": k_counts,
        "rollout_semantics_counts": rollout_semantics_counts,
        "policy_semantics_counts": policy_semantics_counts,
        "plans": total_replans,
        "steps": total_actions,
        "latent_work_total": total_latent_work,
        "bits_used_total": total_bits,
        "dynamics_flops_total": total_dynamics_flops,
        "flop_audit_error_count": total_flop_audit_errors,
        "cem_cost_calls": total_cem_cost_calls,
        "candidate_action_values": total_candidate_action_values,
        "plan_time_total_sec": total_time,
        "policy_time_total_sec": total_policy_time,
    }


def combine_policy_diagnostics(batches: list[dict[str, Any]]) -> dict[str, Any]:
    return combine_mwm_diagnostics(batches)


__all__ = ["combine_mwm_diagnostics", "combine_policy_diagnostics", "combine_swm_results", "run_batch"]
