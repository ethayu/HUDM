from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from mwm.models.losses import matryoshka_base_loss


def matryoshka_training_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    level_weights: Sequence[float] | None = None,
    rollout_weight: float = 1.0,
    sigreg: nn.Module | None = None,
    sigreg_weight: float = 0.0,
    sigreg_scope: str = "shared_latent",
) -> dict[str, torch.Tensor]:
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    emb = model._encode_pixels(batch["pixels"], already_preprocessed=True)
    actions = batch["action"]
    pred_losses: list[torch.Tensor] = []
    for level_idx in range(model.num_levels):
        k = model.K[level_idx]
        pred_emb = model._predict_prefix(
            level_idx,
            emb[:, : model.history_size, :k],
            actions[:, : model.history_size],
        )
        tgt_emb = emb[:, model.num_preds :, :k]
        pred_losses.append((pred_emb - tgt_emb).pow(2).mean())

    return matryoshka_base_loss(
        pred_losses,
        latents=emb,
        K=model.K,
        level_weights=level_weights,
        primary_log_prefix="pred_loss",
        primary_aliases=("pred_loss", "rollout_loss"),
        rollout_weight=rollout_weight,
        regularizer=sigreg,
        regularizer_weight=sigreg_weight,
        regularizer_scope=sigreg_scope,
    )


__all__ = ["matryoshka_training_loss"]
