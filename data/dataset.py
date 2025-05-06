# dynamics_model/new_model/data/dataset.py
"""
Dataset loader wrapping pusht_dset.load_pusht_slice_train_val into simple (state, action, next_state) tuples.
"""
import torch
from torch.utils.data import Dataset
from datasets.pusht_dset import load_pusht_slice_train_val

def load_pusht_dataset(data_cfg):
    """
    Always slices windows of H+1 frames (num_hist + num_pred=1),
    then wraps them to pad early steps.
    """
    train_slices, val_slices = load_pusht_slice_train_val(
        n_rollout=data_cfg.n_rollout,
        data_path=data_cfg.path,
        normalize_action=data_cfg.normalize_action,
        split_ratio=data_cfg.split_ratio,
        num_hist=data_cfg.num_hist,
        num_pred=1,              
        frameskip=data_cfg.frameskip,
        with_velocity=data_cfg.with_velocity,
    )[0].values()              
    return (
        _PadRolloutDataset(train_slices, data_cfg.num_hist),
        _PadRolloutDataset(val_slices,   data_cfg.num_hist),
    )

class _PadRolloutDataset(Dataset):
    """
    Wraps each sliced window of length H+1 into H+1 single‐step
    examples, padding t< H by repeating the first state.
    """
    def __init__(self, slice_dataset, num_hist):
        self.base     = slice_dataset
        self.num_hist = num_hist
        seq_len = self.base.get_seq_length(0)  # = num_hist + 1

        # index every t in [0..seq_len)
        self.index_map = [
            (slice_idx, t)
            for slice_idx in range(len(self.base))
            for t in range(seq_len)
        ]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        slice_idx, t = self.index_map[idx]
        obs, act_seq, state_seq = self.base[slice_idx]
        H = self.num_hist

        # --- build history states ---
        if t >= H:
            past_states = state_seq[t-H : t]         # (H, state_dim)
        else:
            missing     = H - t
            first_state = state_seq[0].unsqueeze(0)  # (1, state_dim)
            pad_states  = first_state.expand(missing, -1)
            past_states = torch.cat([pad_states, state_seq[:t]], dim=0)

        # --- build history actions (zeros for missing)---
        if t >= H:
            past_actions = act_seq[t-H : t]         # (H, action_dim)
        else:
            missing      = H - t
            zero_action  = torch.zeros_like(act_seq[0]).unsqueeze(0)
            pad_actions  = zero_action.expand(missing, -1)
            past_actions = torch.cat([pad_actions, act_seq[:t]], dim=0)

        return {
            'state':      past_states.float(),
            'action':     past_actions.float(),
            'next_state': state_seq[t].float(),
        }
        