from __future__ import annotations

from typing import Any, Callable


def rollout_schedule_indices(decision: Any, horizon: int, *, num_levels: int | None = None) -> tuple[int, list[int]]:
    base_level_idx = int(getattr(decision, "base_level_idx", 0))
    rollout_levels = [int(x) for x in getattr(decision, "rollout_level_indices", [base_level_idx] * int(horizon))]
    if len(rollout_levels) != int(horizon):
        raise ValueError(f"Expected rollout_level_indices to have horizon={int(horizon)} entries, got {len(rollout_levels)}.")
    if num_levels is not None:
        for field_name, idx in [("base_level_idx", base_level_idx)] + [
            (f"rollout_level_indices[{i}]", level) for i, level in enumerate(rollout_levels)
        ]:
            if idx < 0 or idx >= int(num_levels):
                raise ValueError(f"{field_name}={idx} is outside [0, {int(num_levels) - 1}].")
    for prev, cur in zip(rollout_levels, rollout_levels[1:]):
        if cur > prev:
            raise ValueError(f"rollout cannot move from lower to higher fidelity within one rollout: {rollout_levels}")
    return base_level_idx, rollout_levels


def active_rollout_levels(rollout_levels: list[int], *, horizon: int, history: int) -> list[int]:
    if int(horizon) < int(history):
        raise ValueError(f"Action horizon {int(horizon)} is shorter than history {int(history)}.")
    start = max(0, int(history) - 1)
    active = [int(x) for x in rollout_levels[start:]]
    expected = int(horizon) - int(history) + 1
    if len(active) != expected:
        raise ValueError(f"Expected {expected} active rollout levels, got {len(active)} from {rollout_levels}.")
    return active


def terminal_rollout_level(rollout_levels: list[int], *, horizon: int, history: int) -> int:
    return int(active_rollout_levels(rollout_levels, horizon=horizon, history=history)[-1])


def latent_work_for_levels(
    *,
    batch: int,
    samples: int,
    levels: list[int],
    level_width: Callable[[int], int],
    multiplier: int = 1,
) -> int:
    width_total = sum(int(level_width(int(level))) for level in levels)
    return int(batch) * int(samples) * int(width_total) * int(multiplier)


def fixed_level_rollout_indices(decision: Any, horizon: int) -> tuple[int, list[int]]:
    return rollout_schedule_indices(decision, horizon)


def validate_fixed_level_rollout(decision: Any, horizon: int) -> tuple[int, list[int]]:
    base_level_idx, rollout_levels = rollout_schedule_indices(decision, horizon)
    if any(level_idx != base_level_idx for level_idx in rollout_levels):
        raise ValueError(
            "Expected a fixed rollout schedule for this call; "
            f"got base={base_level_idx}, rollout={rollout_levels}."
        )
    return base_level_idx, rollout_levels


__all__ = [
    "active_rollout_levels",
    "fixed_level_rollout_indices",
    "latent_work_for_levels",
    "rollout_schedule_indices",
    "terminal_rollout_level",
    "validate_fixed_level_rollout",
]
