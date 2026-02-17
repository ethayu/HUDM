from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class WorldModelEnsemble(nn.Module):
    """
    Ensemble wrapper for hierarchical world models.

    Notes:
    - `encode` uses the primary member (index 0) to keep one latent frame.
    - `predict_next_stats` returns ensemble mean/variance over next-latent predictions.
    """

    def __init__(self, members: Iterable[nn.Module]):
        super().__init__()
        members = list(members)
        if len(members) < 1:
            raise ValueError("WorldModelEnsemble requires at least one member.")
        self.members = nn.ModuleList(members)
        self.primary = self.members[0]

        self.K = list(self.primary.K)
        self.D = int(self.primary.D)
        self.num_members = len(self.members)

        for idx, m in enumerate(self.members):
            if list(m.K) != self.K:
                raise ValueError(f"Member {idx} has K={list(m.K)} but expected K={self.K}")
            if int(m.D) != self.D:
                raise ValueError(f"Member {idx} has D={int(m.D)} but expected D={self.D}")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.primary.encode(x)

    @torch.no_grad()
    def predict_next_stats(
        self, level: int, z: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds: List[torch.Tensor] = []
        for m in self.members:
            preds.append(m.predict_next(level, z, a))
        stacked = torch.stack(preds, dim=0)  # (M,B,k)
        mu = stacked.mean(dim=0)
        var = stacked.var(dim=0, unbiased=False)
        return mu, var

    @torch.no_grad()
    def predict_next(self, level: int, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mu, _ = self.predict_next_stats(level, z, a)
        return mu
