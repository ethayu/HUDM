from __future__ import annotations

import math
from typing import Any


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _contains_batch_env(diag: dict[str, Any], batch_env: int) -> bool:
    start = _int(diag.get("batch_start", 0))
    end = _int(diag.get("batch_end", start + 1))
    return start <= int(batch_env) < end


def _finite_action(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_finite_action(item) for item in value)
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def executed_action_prefix(action_trace: list[Any]) -> list[Any]:
    actions: list[Any] = []
    for action in action_trace:
        if not _finite_action(action):
            break
        actions.append(action)
    return actions


def _final_cem_by_replan(planning_trace: list[dict[str, Any]], *, batch_env: int) -> dict[int, dict[str, Any]]:
    final: dict[int, dict[str, Any]] = {}
    fallback_idx = 0
    fallback_last_cem: int | None = None
    for diag in planning_trace:
        if not isinstance(diag, dict) or not _contains_batch_env(diag, batch_env):
            continue
        if "mpc_iter" in diag:
            replan_idx = _int(diag.get("mpc_iter"))
        else:
            cem_iter = _int(diag.get("cem_iter", 0))
            if fallback_last_cem is not None and cem_iter <= fallback_last_cem:
                fallback_idx += 1
            fallback_last_cem = cem_iter
            replan_idx = fallback_idx
        current = final.get(replan_idx)
        if current is None or _int(diag.get("cem_iter", 0)) >= _int(current.get("cem_iter", 0)):
            final[replan_idx] = diag
    return final


def fidelity_trace_from_planning_trace(
    *,
    planning_trace: list[dict[str, Any]],
    batch_env: int,
    eval_budget: int,
    action_block: int,
    replan_interval: int,
    k_values: list[int],
) -> list[dict[str, Any]]:
    """Expand final-CEM rollout levels onto primitive environment timesteps."""

    budget = max(0, int(eval_budget))
    block = max(1, int(action_block))
    interval = max(1, int(replan_interval))
    final_by_replan = _final_cem_by_replan(list(planning_trace), batch_env=int(batch_env))
    if not final_by_replan:
        return []

    rows: list[dict[str, Any]] = []
    for t in range(budget):
        replan_idx = int(t // interval)
        decision = final_by_replan.get(replan_idx)
        if decision is None:
            previous = [idx for idx in final_by_replan if idx <= replan_idx]
            if not previous:
                continue
            decision = final_by_replan[max(previous)]
        rollout_levels = decision.get("rollout_level_indices") or decision.get("model_rollout_level_indices")
        if not isinstance(rollout_levels, list) or not rollout_levels:
            rollout_levels = [decision.get("base_level_idx", 0)]
        block_idx = int((t - replan_idx * interval) // block)
        level_idx = _int(rollout_levels[min(block_idx, len(rollout_levels) - 1)])
        k_value = int(k_values[level_idx]) if 0 <= level_idx < len(k_values) else level_idx
        rows.append(
            {
                "t": int(t),
                "replan_idx": int(replan_idx),
                "block_idx": int(block_idx),
                "level_idx": int(level_idx),
                "K": int(k_value),
            }
        )
    return rows


def review_rollouts_for_batches(
    *,
    batches: list[dict[str, Any]],
    successes: list[Any],
    eval_budget: int,
    action_block: int,
    receding_horizon: int,
    k_values: list[int],
) -> list[dict[str, Any]]:
    replan_interval = max(1, int(action_block) * int(receding_horizon))
    out: list[dict[str, Any]] = []
    episode_index = 0
    for batch_index, batch in enumerate(batches):
        planning_trace = list(batch.get("planning_diagnostics", {}).get("trace", []))
        action_by_env = list(batch.get("review_trace", {}).get("action_trace", []))
        for batch_env, pair in enumerate(batch.get("pairs", [])):
            raw_action_trace = action_by_env[batch_env] if batch_env < len(action_by_env) else []
            action_trace = executed_action_prefix(raw_action_trace if isinstance(raw_action_trace, list) else [])
            executed_steps = min(int(eval_budget), len(action_trace))
            success = bool(successes[episode_index]) if episode_index < len(successes) else None
            row = {
                "episode_index": int(episode_index),
                "batch": int(batch_index),
                "batch_env": int(batch_env),
                "dataset_episode": pair.get("episode"),
                "start_step": pair.get("start_step"),
                "goal_step": pair.get("goal_step"),
                "start_row": pair.get("start_row"),
                "goal_row": pair.get("goal_row"),
                "success": success,
                "evaluation_budget": int(eval_budget),
                "actions_recorded": int(executed_steps),
                "terminated_early": bool(success is True and executed_steps < int(eval_budget)),
                "action_trace": action_trace,
                "fidelity_trace": fidelity_trace_from_planning_trace(
                    planning_trace=planning_trace,
                    batch_env=batch_env,
                    eval_budget=int(executed_steps),
                    action_block=int(action_block),
                    replan_interval=replan_interval,
                    k_values=[int(k) for k in k_values],
                ),
            }
            out.append(row)
            episode_index += 1
    return out


__all__ = [
    "executed_action_prefix",
    "fidelity_trace_from_planning_trace",
    "review_rollouts_for_batches",
]
