from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import zarr
except Exception:
    zarr = None  # lazily checked at __init__


class ZarrPushTEpisodes(Dataset):
    """
    Episode dataset over a PushT Zarr store.

    Each item returns a dict with:
      - x: (T, 3, H, W)   images normalized to [-1, 1]
      - a: (T-1, A)       actions (relative or absolute, see action_mode)
      - length: T
    """

    _ACTION_SCALE = 100.0

    def __init__(
        self,
        zarr_root: str,
        split: str = "train",
        split_ratio: float = 0.8,
        action_mode: str = "relative",  # "relative" | "absolute"
        image_key: str = "img",
        action_key: str = "action",
        state_key: str = "state",
        meta_episode_key: str = "episode_ends",
    ):
        if zarr is None:
            raise ImportError("zarr not installed. pip install zarr")
        self.store_path = zarr_root
        if not (0.0 < float(split_ratio) < 1.0):
            raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")
        if action_mode not in {"relative", "absolute"}:
            raise ValueError(f"action_mode must be 'relative' or 'absolute', got {action_mode}")
        self.action_mode = action_mode
        self.image_key = image_key
        self.action_key = action_key
        self.state_key = state_key
        self.meta_episode_key = meta_episode_key

        # Open Zarr arrays
        root = zarr.open_group(zarr_root, mode="r")
        data = root["data"]
        meta = root["meta"]
        self.img = data[self.image_key]       # (N, H, W, C), float32
        self.act = data[self.action_key]      # (N, A), float32
        self.state = data[self.state_key]     # (N, 5), float32
        self.ends = meta[self.meta_episode_key][:]  # (E,), int64 ends
        if len(self.ends) == 0:
            raise ValueError(f"No episodes found in {zarr_root}")
        if not np.all(self.ends[1:] > self.ends[:-1]):
            raise ValueError(f"episode_ends in {zarr_root} must be strictly increasing")

        # Compute episode start indices
        self.starts = np.zeros_like(self.ends)
        self.starts[0] = 0
        for i in range(1, len(self.ends)):
            self.starts[i] = self.ends[i - 1] + 1

        # Split episodes
        n_ep = len(self.ends)
        n_train = int(split_ratio * n_ep)
        if split.lower() in ("train",):
            self.ep_idx = np.arange(0, n_train)
        else:
            self.ep_idx = np.arange(n_train, n_ep)

    def __len__(self):
        return len(self.ep_idx)

    def __getitem__(self, idx: int):
        ei = self.ep_idx[idx]
        s = int(self.starts[ei])
        e = int(self.ends[ei])
        # Frames and states
        x = self.img[s : e + 1]
        state = self.state[s : e + 1]
        # Actions align with transitions s_t -> s_{t+1}
        a_abs = self.act[s:e]
        agent_pos = state[:-1, :2]
        if self.action_mode == "relative":
            a = (a_abs - agent_pos) / self._ACTION_SCALE
        else:
            a = a_abs

        x = np.clip(x, 0.0, 255.0).astype(np.float32)
        x = x / 255.0
        x = x * 2.0 - 1.0
        x_t = torch.from_numpy(x).permute(0, 3, 1, 2)
        a_t = torch.from_numpy(a.astype(np.float32))

        return {
            "x": x_t,
            "a": a_t,
            "length": x_t.shape[0],
        }


def collate_episodes(batch: List[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    bsz = len(batch)
    c, h, w = batch[0]["x"].shape[1:]
    a_dim = batch[0]["a"].shape[1] if batch[0]["a"].numel() > 0 else 2

    x = torch.zeros((bsz, max_len, c, h, w), dtype=batch[0]["x"].dtype)
    a = torch.zeros((bsz, max_len - 1, a_dim), dtype=batch[0]["a"].dtype)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["length"]
        x[i, :L] = item["x"]
        mask[i, :L] = True
        if L > 1:
            a[i, : L - 1] = item["a"]

    return {
        "x": x,
        "a": a,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
