from __future__ import annotations

from typing import Any


def fixed_level_rollout_indices(decision: Any, horizon: int) -> tuple[int, list[int]]:
    base_level_idx = int(getattr(decision, "base_level_idx", 0))
    rollout_levels = [int(x) for x in getattr(decision, "rollout_level_indices", [base_level_idx] * int(horizon))]
    return base_level_idx, rollout_levels


def validate_fixed_level_rollout(decision: Any, horizon: int) -> tuple[int, list[int]]:
    base_level_idx, rollout_levels = fixed_level_rollout_indices(decision, horizon)
    if any(level_idx != base_level_idx for level_idx in rollout_levels):
        raise ValueError(
            "Base-adaptive MWM scheduled evaluation operates entirely at the selected K level; "
            f"got base={base_level_idx}, rollout={rollout_levels}."
        )
    return base_level_idx, rollout_levels


__all__ = ["fixed_level_rollout_indices", "validate_fixed_level_rollout"]
