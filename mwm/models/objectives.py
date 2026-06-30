from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from mwm.models.losses import matryoshka_base_loss, weighted_level_mean
from mwm.preprocessing.images import image_tensor_to_bchw


@contextmanager
def _temporarily_freeze_parameters(modules: Iterable[nn.Module]):
    states: list[tuple[nn.Parameter, bool]] = []
    for module in modules:
        for param in module.parameters():
            states.append((param, bool(param.requires_grad)))
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, requires_grad in states:
            param.requires_grad_(requires_grad)


def _reconstruction_target(pixels: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if pixels.ndim < 5:
        raise ValueError(f"Reconstruction pixels must include batch, time, and image dimensions, got {tuple(pixels.shape)}")
    prefix_shape = tuple(pixels.shape[:-3])
    flat = pixels.reshape(-1, *pixels.shape[-3:])
    flat = image_tensor_to_bchw(flat).to(device=device, dtype=dtype)
    return flat.reshape(*prefix_shape, *flat.shape[-3:])


def _decoder_reconstruction_losses(
    model: Any,
    emb: torch.Tensor,
    target: torch.Tensor,
    *,
    detach_latents: bool,
) -> list[torch.Tensor]:
    losses: list[torch.Tensor] = []
    batch_shape = tuple(emb.shape[:-1])
    target_flat = target.reshape(-1, *target.shape[-3:])
    for level_idx, k in enumerate(model.K):
        latent = emb[..., : int(k)]
        if detach_latents:
            latent = latent.detach()
        recon = model.decode(level_idx, latent.reshape(-1, int(k)))
        recon = recon.reshape(*batch_shape, *recon.shape[-3:])
        recon_flat = recon.reshape(-1, *recon.shape[-3:])
        losses.append((recon_flat - target_flat).pow(2).mean())
    return losses


def matryoshka_training_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    level_weights: Sequence[float] | None = None,
    rollout_weight: float = 1.0,
    recon_latent_weight: float = 0.0,
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

    logs = matryoshka_base_loss(
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
    target = _reconstruction_target(batch["pixels"], dtype=emb.dtype, device=emb.device)
    recon_losses = _decoder_reconstruction_losses(model, emb, target, detach_latents=True)
    recon_loss, recon_logs = weighted_level_mean(
        recon_losses,
        level_weights=level_weights,
        log_prefix="recon_loss",
    )
    logs["loss"] = logs["loss"] + recon_loss
    logs["recon_loss"] = recon_loss.detach()
    logs.update(recon_logs)

    if float(recon_latent_weight):
        with _temporarily_freeze_parameters(model.decoders):
            latent_recon_losses = _decoder_reconstruction_losses(model, emb, target, detach_latents=False)
        latent_recon_loss, latent_recon_logs = weighted_level_mean(
            latent_recon_losses,
            level_weights=level_weights,
            log_prefix="recon_latent_loss",
        )
        logs["loss"] = logs["loss"] + float(recon_latent_weight) * latent_recon_loss
        logs["recon_latent_loss"] = latent_recon_loss.detach()
        logs.update(latent_recon_logs)
    return logs


__all__ = ["matryoshka_training_loss"]
