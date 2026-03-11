from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
import random
try:
    import zarr
except Exception:
    zarr = None  # lazily checked at __init__

ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])

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
        action_mode: str = "relative",
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
        if str(action_mode).lower() != "relative":
            raise ValueError(
                "ZarrPushTEpisodes currently supports only action_mode='relative'. "
                f"Got {action_mode!r}."
            )
        self.action_mode = "relative"

        # Open Zarr arrays
        root = zarr.open_group(zarr_root, mode="r")
        data = root["data"]
        meta = root["meta"]
        attrs = root.attrs
        self.img = data[self.image_key]       # (N, H, W, C), float32
        self.act = data[self.action_key]      # (N, A), float32
        self.action = self.act
        self.actions = self.act
        self.state = data[self.state_key]     # (N, 5), float32
        self.action_abs = data.get("action_abs", None)
        self.action_format = str(attrs.get("action_format", "unknown")).strip().lower()
        self.action_abs_format = str(attrs.get("action_abs_format", "unknown")).strip().lower()
        self.env_action_scale = float(attrs.get("env_action_scale", self._ACTION_SCALE))
        self.env_relative = bool(attrs.get("env_relative", True))
        self.ends = meta[self.meta_episode_key][:]  # (E,), int64 ends
        self.ends[-1] = self.ends[-1] - 1

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
        #img = np.clip(img, 0.0, 1.0).astype(np.float32)
        img = img * 2.0 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx: int):
        ei = self.ep_idx[idx]
        s = int(self.starts[ei])
        e = int(self.ends[ei])
        # Frames and states
        x = self.img[s : e + 1]
        state = self.state[s : e + 1, :] 
        # Actions align with transitions s_t -> s_{t+1}
        a_abs = self.act[s:e]
        agent_pos = state[:-1, :2]
        #print(agent_pos.shape, a_abs.shape,s,e,self.act.shape,state.shape)
        a = a_abs# (a_abs - agent_pos) / self._ACTION_SCALE #

        x_t = torch.stack([self._img_to_tensor(frame) for frame in x], dim=0)
        a_t = torch.from_numpy(a.astype(np.float32))
        state_t = torch.from_numpy(state.astype(np.float32))
        #a_t = (a_t - ACTION_MEAN) / ACTION_STD
        #state_t = (state_t - STATE_MEAN) / STATE_STD

        return {
            "x": x_t,
            "a": a_t,
            "state": state_t,
            "length": x_t.shape[0],
        }

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []
        # sample init_states from dset
        traj_len = traj_len+1
        max_offset = -1
        while max_offset < 0:  # filter out traj that are not long enough
            traj_id = random.randint(0, len(self.ep_idx) - 1)
            traj_dict = self.__getitem__(traj_id)
            max_offset = traj_dict["state"].shape[0] - traj_len
        states = traj_dict["state"].numpy()
        offset = random.randint(0, max_offset)
        states = states[offset : offset + traj_len]
        actions = traj_dict["a"][offset : offset + traj_len - 1]
        observations = traj_dict["x"][offset : offset + traj_len]
        return observations, states, actions


def collate_episodes(batch: List[dict], length: int | None = None) -> dict:
    lengths = [item["length"] for item in batch]
    if length is None or int(length) <= 0:
        max_len = max(lengths)
    else:
        max_len = min(int(length), max(lengths))

    bsz = len(batch)
    c, h, w = batch[0]["x"].shape[1:]
    a_dim = batch[0]["a"].shape[1] if batch[0]["a"].numel() > 0 else 2
    d = batch[0]["state"].shape[1] if batch[0]["state"].numel() > 0 else 5

    x = torch.zeros((bsz, max_len, c, h, w), dtype=batch[0]["x"].dtype)
    state = torch.zeros((bsz, max_len, d), dtype=batch[0]["state"].dtype)
    a = torch.zeros((bsz, max_len - 1, a_dim), dtype=batch[0]["a"].dtype)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        item_len = int(item["x"].shape[0])
        L = min(item_len, max_len)
        start = 0 if item_len == L else random.randint(0, item_len - L)
        x[i, :L] = item["x"][start:start+L]
        state[i, :L] = item["state"][start:start+L]
        mask[i, :L] = True
        if L > 1:
            a[i, : L - 1] = item["a"][start:start+L-1]

    return {
        "x": x,
        "a": a,
        "s": state,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
