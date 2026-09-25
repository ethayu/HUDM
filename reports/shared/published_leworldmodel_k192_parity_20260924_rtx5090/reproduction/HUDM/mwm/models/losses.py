from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def weighted_level_mean(
    level_losses: Sequence[torch.Tensor],
    *,
    level_weights: Sequence[float] | None = None,
    log_prefix: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Aggregate per-level objective terms with the shared MWM weighting rule."""

    losses = list(level_losses)
    if not losses:
        raise ValueError("weighted_level_mean requires at least one level loss.")
    weights = list(level_weights or [1.0] * len(losses))
    if len(weights) != len(losses):
        raise ValueError(f"level_weights has {len(weights)} entries for {len(losses)} levels")
    denom = float(sum(weights)) if sum(weights) else 1.0
    total = losses[0].new_tensor(0.0)
    logs: dict[str, torch.Tensor] = {}
    for level_idx, (loss, weight) in enumerate(zip(losses, weights)):
        logs[f"{log_prefix}_l{level_idx}"] = loss.detach()
        total = total + float(weight) * loss / denom
    return total, logs


def latent_regularizer_loss(
    latents: torch.Tensor,
    *,
    K: Sequence[int],
    regularizer: nn.Module,
    scope: str = "shared_latent",
    level_weights: Sequence[float] | None = None,
    log_prefix: str = "sigreg_loss",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a latent regularizer using MWM's shared/per-level scope policy."""

    if latents.ndim != 3:
        raise ValueError(f"latent_regularizer_loss expects (B,T,D) latents, got {tuple(latents.shape)}")
    if scope == "shared_latent":
        total = regularizer(latents.transpose(0, 1))
        return total, {log_prefix: total.detach()}
    if scope != "per_level_prefix":
        raise ValueError(f"Unknown regularizer scope {scope!r}")
    losses = [regularizer(latents[..., : int(k)].transpose(0, 1)) for k in K]
    total, logs = weighted_level_mean(losses, level_weights=level_weights, log_prefix=log_prefix)
    logs[log_prefix] = total.detach()
    return total, logs


def matryoshka_base_loss(
    level_losses: Sequence[torch.Tensor],
    *,
    latents: torch.Tensor | None = None,
    K: Sequence[int] | None = None,
    level_weights: Sequence[float] | None = None,
    primary_log_prefix: str = "pred_loss",
    primary_aliases: Sequence[str] = (),
    rollout_weight: float = 1.0,
    regularizer: nn.Module | None = None,
    regularizer_weight: float = 0.0,
    regularizer_scope: str = "shared_latent",
    regularizer_log_prefix: str = "sigreg_loss",
) -> dict[str, torch.Tensor]:
    """Aggregate base-provided level losses and optional shared regularization."""

    primary_loss, logs = weighted_level_mean(
        level_losses,
        level_weights=level_weights,
        log_prefix=primary_log_prefix,
    )
    for alias in primary_aliases:
        logs[str(alias)] = primary_loss.detach()
    loss = float(rollout_weight) * primary_loss
    if regularizer is not None and float(regularizer_weight):
        if latents is None:
            raise ValueError("matryoshka_base_loss requires latents when regularizer_weight is non-zero.")
        if K is None:
            raise ValueError("matryoshka_base_loss requires K when regularizer_weight is non-zero.")
        reg_loss, reg_logs = latent_regularizer_loss(
            latents,
            K=K,
            regularizer=regularizer,
            scope=regularizer_scope,
            level_weights=level_weights,
            log_prefix=regularizer_log_prefix,
        )
        loss = loss + float(regularizer_weight) * reg_loss
        logs.update(reg_logs)
    logs["loss"] = loss
    return logs


__all__ = [
    "latent_regularizer_loss",
    "matryoshka_base_loss",
    "weighted_level_mean",
]
