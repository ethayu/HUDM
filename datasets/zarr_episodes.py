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
      - a: (T-1, A)       relative actions normalized by action_scale
      - length: T
    """

    _ACTION_SCALE = 100.0

    def __init__(
        self,
        zarr_root: str,
        split: str = "train",
        split_ratio: float = 0.8,
        image_key: str = "img",
        action_key: str = "action",
        state_key: str = "state",
        meta_episode_key: str = "episode_ends",
    ):
        if zarr is None:
            raise ImportError("zarr not installed. pip install zarr")
        self.store_path = zarr_root
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

    def __len__(self):
        return len(self.ep_idx)

    @staticmethod
    def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
        img = np.clip(img, 0.0, 255.0).astype(np.float32)
        img = img / 255.0
        img = img * 2.0 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

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
        a = (a_abs - agent_pos) / self._ACTION_SCALE

        x_t = torch.stack([self._img_to_tensor(frame) for frame in x], dim=0)
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
