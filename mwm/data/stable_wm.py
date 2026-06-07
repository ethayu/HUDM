from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


class MWMTrainSampleTransform:
    """Transform Stable-WM samples into MWM ``x`` / ``a`` training batches."""

    def __init__(self, pixels_key: str = "pixels", action_key: str = "action", normalize_pixels: bool = True) -> None:
        self.pixels_key = str(pixels_key)
        self.action_key = str(action_key)
        self.normalize_pixels = bool(normalize_pixels)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.pixels_key not in sample:
            raise KeyError(f"Stable-WM sample is missing pixels key {self.pixels_key!r}")
        if self.action_key not in sample:
            raise KeyError(f"Stable-WM sample is missing action key {self.action_key!r}")
        x = torch.as_tensor(sample[self.pixels_key])
        if x.ndim != 4:
            raise ValueError(f"pixels must be a 4D trajectory, got {tuple(x.shape)}")
        if x.shape[-1] in {1, 3, 4}:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if self.normalize_pixels and x.numel() and float(x.max().item()) > 1.5:
            x = x / 255.0
        a = self._action_sequence(torch.as_tensor(sample[self.action_key]).float(), num_frames=int(x.shape[0]))
        out = dict(sample)
        out["x"] = x
        out["a"] = a
        out["mask"] = torch.ones(x.shape[0], dtype=torch.bool)
        return out

    @staticmethod
    def _action_sequence(action: torch.Tensor, *, num_frames: int) -> torch.Tensor:
        if action.ndim == 0:
            raise ValueError("action must have at least one dimension.")
        action = action.reshape(int(action.shape[0]), -1)
        target_actions = max(0, int(num_frames) - 1)
        if target_actions == 0:
            return action[:0]
        if int(action.shape[0]) == target_actions:
            return action
        if int(action.shape[0]) == int(num_frames):
            return action[:target_actions]
        if int(action.shape[0]) >= int(num_frames) and int(action.shape[0]) % int(num_frames) == 0:
            return action.reshape(int(num_frames), -1)[:target_actions]
        if int(action.shape[0]) >= target_actions:
            return action[:target_actions]
        raise ValueError(
            f"Not enough action rows for {num_frames} frames: got action shape {tuple(action.shape)}."
        )


def load_stable_wm_dataset_for_mwm(
    name: str,
    *,
    format: str | None = None,
    frameskip: int = 1,
    num_steps: int = 4,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    from stable_worldmodel.data import load_dataset

    return load_dataset(
        str(name),
        format=format,
        frameskip=int(frameskip),
        num_steps=int(num_steps),
        transform=transform or MWMTrainSampleTransform(),
        **kwargs,
    )


def dataset_metadata_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_suffix(p.suffix + ".metadata.json") if p.suffix else p / "metadata.json"


def load_dataset_metadata(path: str | Path, *, required: bool = False) -> dict[str, Any]:
    meta_path = dataset_metadata_path(path)
    if not meta_path.is_file():
        if required:
            raise FileNotFoundError(f"Missing dataset metadata: {meta_path}")
        return {}
    return dict(json.loads(meta_path.read_text(encoding="utf-8")))


def write_dataset_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    meta_path = dataset_metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


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
        return [_pair_from_row(int(row), offsets=offsets, goal_offset_steps=int(goal_offset_steps)) for row in np.sort(rows[choices])]
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


__all__ = [
    "MWMTrainSampleTransform",
    "StartGoalPair",
    "dataset_metadata_path",
    "load_dataset_metadata",
    "load_stable_wm_dataset_for_mwm",
    "sample_start_goal_pairs",
    "write_dataset_metadata",
]
