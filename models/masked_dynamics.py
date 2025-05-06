# dynamics_model/new_model/masked_dynamics.py
"""
Sequence-aware transformer that predicts s_{t+1} from a length-H
history of (state, action) and a per-dim reliability mask.
"""

import torch
import torch.nn as nn
from typing import Optional
from omegaconf import DictConfig

class SequenceAwareMaskedDynamics(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int,
        feedforward_dim: int,
        n_heads: int,
        n_layers: int,
        window_H: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.D, self.A, self.H = state_dim, action_dim, window_H
        E = embedding_dim

        # --- token embedders ------------------------------------------------
        self.state_fc  = nn.Linear(1, E)
        self.action_fc = nn.Linear(action_dim, E)
        self.next_token = nn.Parameter(torch.empty(1, 1, E))
        nn.init.normal_(self.next_token, mean=0.0, std=0.02)

        # positional + role encodings
        self.time_pos = nn.Embedding(window_H, E)
        self.role_enc = nn.Embedding(self.D + 1, E)  # D state roles + 1 action

        # reliability (observed / masked)
        self.rel_enc = nn.Embedding(2, E)

        # --- transformer ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=E, nhead=n_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # --- output head -----------------------------------------------------
        self.out = nn.Sequential(
            nn.Linear(E, 4*E), nn.ReLU(), nn.Linear(4*E, state_dim)
        )

    # --------------------------------------------------------------------- #
    def forward(
        self,
        state_hist:  torch.Tensor,         # (B,H,D)
        action_hist: torch.Tensor,         # (B,H,A)
        mask_hist:   Optional[torch.BoolTensor] = None  # (B,H,D) 1 = observed, 0 = masked
    ) -> torch.Tensor:
        B, H, D = state_hist.shape
        device  = state_hist.device
        E       = self.next_token.size(-1)

        # ----- build tokens per time-step -----------------------------------
        # State tokens: (B,H,D,E)
        st_tok = self.state_fc(state_hist.unsqueeze(-1))

        # Action tokens: (B,H,1,E)
        act_tok = self.action_fc(action_hist).unsqueeze(2)

        # Concatenate along dim-within-step → shape (B,H,D+1,E)
        step_tok = torch.cat([st_tok, act_tok], dim=2)

        # Add role & time embeddings
        role_ids = torch.arange(D+1, device=device)          # (D+1,)
        role_emb = self.role_enc(role_ids)                   # (D+1,E)
        step_tok = step_tok + role_emb.view(1,1,D+1,E)

        time_ids = torch.arange(H-1, -1, -1, device=device)  # 0=oldest
        step_tok = step_tok + self.time_pos(time_ids).view(1,H,1,E)

        # Reliability embedding
        if mask_hist is not None:
            rel = self.rel_enc(mask_hist.long())             # (B,H,D,E)
            rel_act = self.rel_enc(torch.ones(B,H,1, dtype=torch.long,
                                              device=device))
            step_tok = step_tok + torch.cat([rel, rel_act], dim=2)

        # Flatten time dimension: (B, H*(D+1), E)
        tokens = step_tok.view(B, H*(D+1), E)

        # Prepend [NEXT] query
        next_tok = self.next_token.repeat(B,1,1)             # (B,1,E)
        tokens   = torch.cat([next_tok, tokens], dim=1)      # (B,1+H*(D+1),E)

        # ----- padding mask (True = ignore) -------------------------------
        if mask_hist is None:
            pad = None
        else:
            pad_state = ~mask_hist               # (B,H,D)
            pad_action = torch.zeros(B, H, 1, dtype=torch.bool, device=device)
            pad = torch.cat([pad_state, pad_action], dim=2).view(B, H*(D+1))   # (B,H*(D+1))
            pad = torch.cat([torch.zeros(B,1,dtype=torch.bool, # add mask for [NEXT]
                                         device=device), pad], dim=1)

        enc = self.transformer(tokens, src_key_padding_mask=pad)  # (B,N,E)
        pred = self.out(enc[:,0])                                 # (B,D)
        return state_hist[:, -1] + pred

# ------------------------------------------------------------------------- #
def build_masked_dynamics_model(cfg: DictConfig) -> SequenceAwareMaskedDynamics:
    m = cfg.model
    return SequenceAwareMaskedDynamics(
        state_dim      = m.state_dim,
        action_dim     = m.action_dim,
        embedding_dim  = m.embedding_dim,
        feedforward_dim= m.feedforward_dim,
        n_heads        = m.n_heads,
        n_layers       = m.n_layers,
        window_H       = cfg.data.num_hist,
        dropout        = m.dropout,
    )
