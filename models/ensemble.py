# dynamics_model/new_model/ensemble.py
"""
Ensemble wrapper around SequenceAwareMaskedDynamicsModel
with variance-based per-dimension dropout.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional
from .masked_dynamics import build_masked_dynamics_model

class MaskedDynamicsEnsemble(nn.Module):
    """N independent models for epistemic uncertainty."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.num_models = cfg.model.num_models
        self.window_H   = cfg.data.num_hist
        self.state_dim  = cfg.model.state_dim

        self.models = nn.ModuleList(
            [build_masked_dynamics_model(cfg) for _ in range(self.num_models)]
        )

    def forward(
        self,
        state_hist:  torch.Tensor,   # (B, H, D)
        action_hist: torch.Tensor,   # (B, H, A)
        mask_hist:   torch.BoolTensor  # (B, H, D)
    ):
        preds = [m(state_hist, action_hist, mask_hist) for m in self.models]
        stacked = torch.stack(preds, dim=0)          # (M,B,D)
        return stacked.mean(0), stacked.var(0, unbiased=False)

    # --------------------------------------------------------------------- #
    # Rollout K steps with shared variance-based dropout                    #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def rollout_with_dropout(
        self,
        init_state_hist: torch.Tensor,   # (B, H, D)
        init_action_hist: torch.Tensor,  # (B, H, A)     (last entry = “dummy”)
        init_mask_hist: torch.BoolTensor,# (B, H, D)     (1 = observed)
        actions: torch.Tensor,           # (B, K, A) planned controls
        var_threshold: float,
        return_vars: bool = False,
        gt_states: Optional[torch.Tensor] = None,  # (B, K, D) ground truth states
    ):
        B, K, A = actions.shape
        D, H    = self.state_dim, self.window_H
        M       = self.num_models
        
        if init_mask_hist is None:
            init_mask_hist = torch.ones_like(init_state_hist, dtype=torch.bool)

        # Circular buffers (index 0 = oldest, H–1 = most recent)
        s_buf = init_state_hist.clone()      # (B,H,D)
        a_buf = init_action_hist.clone()
        m_buf = init_mask_hist.clone()       # bool
        
        a_buf[:, -1] = actions[:, 0]   # replace dummy action with first planned action
        
        states_traj = []
        masks_traj  = []

        for k in range(K):
            # Model step
            if gt_states is not None:
                if k >= H - 1:
                    s_buf = gt_states[k - H + 1:k + 1].unsqueeze(0) # (B,H,D)
                else:
                    missing = H - k - 1
                    first_state = gt_states[0]
                    pad_states = first_state.expand(1, missing, -1)
                    s_buf = torch.cat([pad_states, gt_states[:k + 1].unsqueeze(0)], dim=1)
                m_buf = torch.ones_like(s_buf, dtype=torch.bool)
            pred_mu, pred_var = self.forward(s_buf, a_buf, m_buf)

            # Update mask: keep dims with low ensemble variance
            new_mask = m_buf[:, -1] & (pred_var <= var_threshold)   # (B,D)

            # Push next state & mask into buffers
            s_next = pred_mu
            if k < K - 1:
                a_next = actions[:, k + 1]
                s_buf = torch.cat([s_buf[:, 1:],  s_next.unsqueeze(1)],  dim=1)
                a_buf = torch.cat([a_buf[:, 1:],  a_next.unsqueeze(1)],  dim=1)
                m_buf = torch.cat([m_buf[:, 1:], new_mask.unsqueeze(1)], dim=1)

            states_traj.append(s_next)
            masks_traj.append(new_mask)

        # shapes → (B,K,D)
        if return_vars:
            return torch.stack(states_traj, dim=1), torch.stack(masks_traj, dim=1), pred_var
        else:
            return torch.stack(states_traj, dim=1), torch.stack(masks_traj, dim=1)
