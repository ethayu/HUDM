from __future__ import annotations

import torch
import torch.nn as nn


class TransitionPackage(nn.Module):
    """Per-level base transition package used by base-adaptive MWM models."""

    def __init__(self, action_encoder: nn.Module, predictor: nn.Module, pred_proj: nn.Module) -> None:
        super().__init__()
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.pred_proj = pred_proj

    def predict(self, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, self.action_encoder(action))
        flat = preds.reshape(-1, preds.shape[-1])
        preds = self.pred_proj(flat).reshape(*preds.shape[:-1], -1)
        return preds


__all__ = ["TransitionPackage"]
