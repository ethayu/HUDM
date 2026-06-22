from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StartGoalPair:
    episode: int
    start_step: int
    goal_step: int
    start_row: int
    goal_row: int


def _pair_from_row(row: int, *, offsets: np.ndarray, goal_offset_steps: int) -> StartGoalPair:
    ep = int(np.searchsorted(offsets, int(row), side="right") - 1)
    start = int(row - offsets[ep])
    goal = start + int(goal_offset_steps)
    return StartGoalPair(
        episode=ep,
        start_step=start,
        goal_step=goal,
        start_row=int(offsets[ep] + start),
        goal_row=int(offsets[ep] + goal),
    )


def sample_start_goal_pairs(
    dataset: Any,
    *,
    count: int,
    goal_offset_steps: int,
    seed: int,
    mode: str = "mwm",
) -> list[StartGoalPair]:
    lengths = np.asarray(getattr(dataset, "lengths"), dtype=np.int64)
    offsets = np.asarray(getattr(dataset, "offsets"), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    sample_mode = str(mode).lower()
    if sample_mode in {"stable_worldmodel", "stable-wm", "swm", "upstream"}:
        valid_rows: list[np.ndarray] = []
        for ep, length in enumerate(lengths.tolist()):
            max_start = int(length) - int(goal_offset_steps) - 1
            if max_start >= 0:
                valid_rows.append(offsets[int(ep)] + np.arange(max_start + 1, dtype=np.int64))
        if not valid_rows:
            raise ValueError(f"No valid start-goal pairs with goal_offset_steps={goal_offset_steps}.")
        rows = np.concatenate(valid_rows)
        population = len(rows)
        replace = int(count) > population
        choices = rng.choice(population, size=int(count), replace=replace)
        return [
            _pair_from_row(int(row), offsets=offsets, goal_offset_steps=int(goal_offset_steps))
            for row in np.sort(rows[choices])
        ]
    if sample_mode != "mwm":
        raise ValueError(f"Unknown start-goal sampling mode {mode!r}.")
    valid: list[tuple[int, int]] = []
    for ep, length in enumerate(lengths.tolist()):
        max_start = int(length) - int(goal_offset_steps) - 1
        if max_start >= 0:
            valid.extend((int(ep), start) for start in range(max_start + 1))
    if not valid:
        raise ValueError(f"No valid start-goal pairs with goal_offset_steps={goal_offset_steps}.")
    replace = int(count) > len(valid)
    choices = rng.choice(len(valid), size=int(count), replace=replace)
    pairs = []
    for idx in choices:
        ep, start = valid[int(idx)]
        goal = start + int(goal_offset_steps)
        pairs.append(
            StartGoalPair(
                episode=ep,
                start_step=int(start),
                goal_step=int(goal),
                start_row=int(offsets[ep] + start),
                goal_row=int(offsets[ep] + goal),
            )
        )
    return pairs


__all__ = ["StartGoalPair", "sample_start_goal_pairs"]
