import os
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import zarr
except Exception:
    zarr = None  # lazily checked at __init__


class ZarrPushTWindows(Dataset):
    """
    Windowed dataset over a PushT Zarr store producing image sequences and actions.

    Each item returns a dict with:
      - x_t:      (3, H, W)
      - x_t1:     (3, H, W)
      - x_tfut:   (3, H, W)  # at t+T
      - a_t:      (A,)       # relative action, scaled by action_scale (not clipped)
      - a_seq:    (T, A)     # relative actions from t..t+T-1, scaled by action_scale (not clipped)

    Windows are uniformly sampled over time indices within split episodes.
    """

    _ACTION_SCALE = 100.0

    def __init__(
        self,
        zarr_root: str,
        split: str = "train",
        split_ratio: float = 0.8,
        horizon_T: int = 8,
        image_key: str = "img",
        action_key: str = "action",
        state_key: str = "state",
        meta_episode_key: str = "episode_ends",
    ):
        if zarr is None:
            raise ImportError("zarr not installed. pip install zarr")
        self.store_path = zarr_root
        self.T = int(horizon_T)
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

        # Precompute all valid (episode, t) windows within chosen split
        self.indices: List[Tuple[int, int]] = []
        for ei in self.ep_idx:
            s = int(self.starts[ei])
            e = int(self.ends[ei])
            # valid t satisfy t+T <= e (since x_tfut is at t+T)
            max_t = e - self.T
            if max_t < s:
                continue
            for t in range(s, max_t + 1):
                self.indices.append((ei, t))

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
        img = np.clip(img, 0.0, 255.0).astype(np.float32)
        img = img / 255.0
        img = img * 2.0 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx: int):
        ei, t = self.indices[idx]
        # Direct indexing over flattened arrays
        x_t   = self.img[t]
        x_t1  = self.img[t + 1]
        x_tf  = self.img[t + self.T]
        a_seq_abs = self.act[t : t + self.T]
        agent_pos = self.state[t : t + self.T, :2]

        # Absolute pixel targets -> relative delta -> normalize by action_scale.
        a_seq = (a_seq_abs - agent_pos) / self._ACTION_SCALE
        a_t = a_seq[0]

        # Convert images to tensors in [-1,1]
        x_t = self._img_to_tensor(x_t)
        x_t1 = self._img_to_tensor(x_t1)
        x_tf = self._img_to_tensor(x_tf)

        sample = {
            "x_t": x_t,
            "x_t1": x_t1,
            "x_tfut": x_tf,
            "a_t": torch.from_numpy(a_t.astype(np.float32)),
            "a_seq": torch.from_numpy(a_seq.astype(np.float32)),
        }
        return sample
