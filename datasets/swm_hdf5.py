from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import h5py

try:  # SWM HDF5 files may be compressed with hdf5plugin filters.
    import hdf5plugin  # noqa: F401
except Exception:  # pragma: no cover - optional runtime dependency
    hdf5plugin = None

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_hdf5_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.h5")) + sorted(p.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No .h5/.hdf5 file found in {p}")
        if len(files) > 1:
            raise ValueError(f"Multiple HDF5 files found in {p}; pass the file path explicitly.")
        return files[0]
    if not p.exists():
        raise FileNotFoundError(f"SWM HDF5 dataset not found: {p}")
    return p


def swm_dataset_metadata_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_suffix(p.suffix + ".metadata.json")


def load_swm_dataset_metadata(path: str | Path, *, required: bool = False) -> dict[str, Any]:
    meta_path = swm_dataset_metadata_path(path)
    if not meta_path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing SWM dataset metadata: {meta_path}. "
                "Use collect_swm.py or create a matching .metadata.json sidecar before planning."
            )
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


@dataclass(frozen=True)
class SWMStartGoalPair:
    episode: int
    start_step: int
    goal_step: int
    start_row: int
    goal_row: int


class SWMHDF5Episodes(Dataset):
    """Training/eval adapter for SWM HDF5 datasets.

    SWM records one row per reset/step info dict and left-shifts actions before
    writing so each row holds the action to apply at that observation. For a
    window of ``L`` observations, the dynamics actions are rows ``start`` through
    ``start + L - 2``.
    """

    def __init__(
        self,
        path: str | Path,
        horizon: int,
        split: str = "train",
        split_ratio: float = 0.8,
        seed: int = 0,
        pixels_key: str = "pixels",
        action_key: str = "action",
        frameskip: int = 1,
    ) -> None:
        self.path = resolve_hdf5_path(path)
        self.horizon = int(horizon)
        self.num_steps = self.horizon
        self.frameskip = int(frameskip)
        self.span = self.num_steps * self.frameskip
        self.split = str(split)
        self.split_ratio = float(split_ratio)
        self.seed = int(seed)
        self.pixels_key = str(pixels_key)
        self.action_key = str(action_key)
        self._file: h5py.File | None = None

        if self.horizon < 2:
            raise ValueError(f"horizon must be at least 2, got {self.horizon}")
        if self.frameskip != 1:
            raise ValueError(f"HUDM SWM HDF5 adapter currently supports RGB frames with frameskip=1, got {self.frameskip}")
        if self.split not in {"train", "valid", "all"}:
            raise ValueError(f"split must be train, valid, or all; got {self.split!r}")
        if not (0.0 < self.split_ratio < 1.0):
            raise ValueError(f"split_ratio must be in (0,1), got {self.split_ratio}")

        with h5py.File(self.path, "r") as f:
            required = {"ep_len", "ep_offset", self.pixels_key, self.action_key}
            missing = sorted(k for k in required if k not in f)
            if missing:
                raise ValueError(f"{self.path} is missing required SWM columns: {missing}")
            self.lengths = np.asarray(f["ep_len"][:], dtype=np.int64)
            self.offsets = np.asarray(f["ep_offset"][:], dtype=np.int64)
            self.all_column_names = [str(k) for k in f.keys()]
            self.column_names = [k for k in self.all_column_names if k not in {"ep_len", "ep_offset"}]
            self.total_rows = int(f[self.pixels_key].shape[0])
            self.image_shape = self._infer_image_shape(np.asarray(f[self.pixels_key][0]))
            self.action_dim = int(np.asarray(f[self.action_key][0]).reshape(-1).shape[0])

        self.episode_indices = self._split_episode_indices()
        self.clip_indices = [
            (int(ep), int(start))
            for ep in self.episode_indices
            for start in range(0, max(0, int(self.lengths[ep]) - self.horizon + 1))
        ]
        if not self.clip_indices:
            raise ValueError(
                f"No {self.split} windows of horizon {self.horizon} found in {self.path}."
            )

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()

    def _split_episode_indices(self) -> np.ndarray:
        indices = np.arange(len(self.lengths), dtype=np.int64)
        if self.split == "all":
            return indices
        rng = np.random.default_rng(self.seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * self.split_ratio))
        selected = shuffled[:n_train] if self.split == "train" else shuffled[n_train:]
        return np.sort(selected)

    @staticmethod
    def _infer_image_shape(sample: np.ndarray) -> tuple[int, int]:
        if sample.ndim != 3:
            raise ValueError(f"Expected image sample with 3 dims, got shape {sample.shape}")
        if sample.shape[-1] == 3:
            return int(sample.shape[0]), int(sample.shape[1])
        if sample.shape[0] == 3:
            return int(sample.shape[1]), int(sample.shape[2])
        raise ValueError(f"HUDM SWM eval supports RGB-only images; cannot infer RGB channel axis for {sample.shape}")

    @staticmethod
    def _pixels_to_tensor(arr: np.ndarray) -> torch.Tensor:
        arr = np.asarray(arr)
        if arr.ndim != 4:
            raise ValueError(f"Expected pixel window with shape (T,H,W,C) or (T,C,H,W), got {arr.shape}")
        if arr.shape[-1] == 3:
            arr = np.moveaxis(arr, -1, 1)
        elif arr.shape[1] != 3:
            raise ValueError(f"HUDM SWM eval supports RGB-only pixel windows; got {arr.shape}")
        arr = arr.astype(np.float32, copy=False)
        if arr.size and float(np.nanmax(arr)) > 2.0:
            arr = arr / 255.0
        return torch.from_numpy(np.ascontiguousarray(arr))

    @staticmethod
    def _array_to_swm_tensor(key: str, data: np.ndarray) -> torch.Tensor:
        arr = np.asarray(data)
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        if key.startswith("pixels") and arr.ndim == 4:
            if arr.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            elif arr.shape[1] != 3:
                raise ValueError(f"HUDM SWM eval supports RGB-only pixel chunks; got {arr.shape}")
        return tensor

    def _read_rows(self, key: str, start: int, end: int) -> np.ndarray:
        f = self._open()
        if key not in f:
            raise KeyError(f"Column {key!r} is not present in {self.path}")
        return np.asarray(f[key][int(start) : int(end)])

    def load_episode(self, episode_idx: int) -> dict[str, Any]:
        return self._load_slice(int(episode_idx), 0, int(self.lengths[int(episode_idx)]))

    def get_col_data(self, col: str) -> np.ndarray:
        f = self._open()
        if col not in f:
            raise KeyError(f"Column {col!r} is not present in {self.path}")
        return np.asarray(f[col][:])

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return int(np.prod(data.shape[1:]).item()) if data.ndim > 1 else 1

    def get_row_data(self, row_idx: int | list[int]) -> dict[str, Any]:
        f = self._open()
        return {col: np.asarray(f[col][row_idx]) for col in self.column_names}

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict[str, Any]:
        """SWM Dataset-compatible episode-local slice loader.

        ``stable_worldmodel.World.evaluate_from_dataset`` expects dataset
        chunks with numeric columns as tensors and image columns in CHW order.
        """
        ep = int(ep_idx)
        start_i = int(start)
        end_i = int(end)
        if ep < 0 or ep >= len(self.lengths):
            raise IndexError(f"Episode index {ep} is out of bounds for {len(self.lengths)} episodes")
        if start_i < 0 or end_i <= start_i or end_i > int(self.lengths[ep]):
            raise ValueError(
                f"Invalid slice for episode {ep}: start={start_i}, end={end_i}, length={int(self.lengths[ep])}"
            )
        row0 = int(self.offsets[ep] + start_i)
        row1 = int(self.offsets[ep] + end_i)
        f = self._open()
        steps: dict[str, Any] = {}
        for col in self.column_names:
            data = np.asarray(f[col][row0:row1])
            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                val = data[0] if len(data) > 0 else b""
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = self._array_to_swm_tensor(col, data)
        return steps

    def load_chunk(self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray) -> list[dict[str, Any]]:
        chunk: list[dict[str, Any]] = []
        for ep, s, e in zip(episodes_idx, start, end):
            steps = self._load_slice(int(ep), int(s), int(e))
            if self.action_key in steps:
                steps[self.action_key] = steps[self.action_key].reshape(int(e) - int(s), -1)
            chunk.append(steps)
        return chunk

    def get_row(self, row: int, keys: Iterable[str] | None = None) -> dict[str, Any]:
        f = self._open()
        selected = list(keys) if keys is not None else self.column_names
        out: dict[str, Any] = {}
        for key in selected:
            if key not in f:
                raise KeyError(f"Column {key!r} is not present in {self.path}")
            out[key] = np.asarray(f[key][int(row)])
        return out

    def get_rows(self, rows: Iterable[int], keys: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return [self.get_row(int(row), keys=keys) for row in rows]

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep, start = self.clip_indices[int(idx)]
        row0 = int(self.offsets[ep] + start)
        row1 = row0 + self.horizon
        pixels = self._pixels_to_tensor(self._read_rows(self.pixels_key, row0, row1))
        actions = self._read_rows(self.action_key, row0, row1 - 1)
        actions = np.nan_to_num(actions.reshape(self.horizon - 1, -1), nan=0.0).astype(np.float32)
        mask = torch.ones(self.horizon, dtype=torch.bool)
        return {
            "x": pixels,
            "a": torch.from_numpy(actions),
            "mask": mask,
        }

    def sample_eval_start_goal_pairs(
        self,
        count: int,
        goal_offset_steps: int,
        seed: int = 0,
        episodes: Iterable[int] | None = None,
    ) -> list[SWMStartGoalPair]:
        """Sample starts using SWM ``evaluate_from_dataset`` goal semantics.

        SWM loads ``dataset.load_chunk(ep, start, start + goal_offset_steps)``
        and uses the last loaded row as the goal, so the goal row is
        ``start + goal_offset_steps - 1``.
        """
        goal_offset_steps = int(goal_offset_steps)
        if goal_offset_steps <= 0:
            raise ValueError(f"goal_offset_steps must be > 0, got {goal_offset_steps}")
        candidates: list[tuple[int, int]] = []
        eps = list(int(e) for e in (episodes if episodes is not None else self.episode_indices))
        for ep in eps:
            max_start = int(self.lengths[ep]) - goal_offset_steps
            for start in range(max_start + 1):
                candidates.append((ep, start))
        if not candidates:
            raise ValueError(
                f"No dataset start-goal pairs with goal_offset_steps={goal_offset_steps} in split {self.split}."
            )
        rng = np.random.default_rng(int(seed))
        replace = int(count) > len(candidates)
        choices = rng.choice(len(candidates), size=int(count), replace=replace)
        pairs: list[SWMStartGoalPair] = []
        for choice in np.atleast_1d(choices):
            ep, start = candidates[int(choice)]
            goal = start + goal_offset_steps - 1
            pairs.append(
                SWMStartGoalPair(
                    episode=int(ep),
                    start_step=int(start),
                    goal_step=int(goal),
                    start_row=int(self.offsets[ep] + start),
                    goal_row=int(self.offsets[ep] + goal),
                )
            )
        return pairs

    def read_pixels_by_rows(self, rows: Iterable[int]) -> np.ndarray:
        row_list = [int(r) for r in rows]
        f = self._open()
        data = np.stack([np.asarray(f[self.pixels_key][row]) for row in row_list], axis=0)
        if data.ndim != 4:
            raise ValueError(f"Expected pixel rows to have 4 dims, got {data.shape}")
        return data
