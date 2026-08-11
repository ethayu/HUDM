from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from mwm.models.losses import matryoshka_base_loss, weighted_level_mean
from mwm.preprocessing.images import image_tensor_to_bchw


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
    decoder_training_enabled: bool = True,
    sigreg: nn.Module | None = None,
    sigreg_weight: float = 0.0,
    sigreg_scope: str = "shared_latent",
    random_prefix_weight: float = 0.0,
    sample_random_prefixes: bool = False,
) -> dict[str, torch.Tensor]:
    if float(random_prefix_weight) < 0:
        raise ValueError(f"random_prefix_weight must be non-negative, got {random_prefix_weight}.")
    if float(recon_latent_weight):
        raise ValueError(
            "loss.recon_latent_weight must be 0 with optimizer-isolated decoder training; "
            "decoder reconstruction cannot contribute gradients to the encoder."
        )
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

    random_loss: torch.Tensor | None = None
    sampled_k: int | None = None
    if sample_random_prefixes and bool(model.training) and float(random_prefix_weight) > 0:
        if not bool(getattr(model, "supports_arbitrary_k", False)):
            raise ValueError("Random prefix training requires a model with shared arbitrary-K dynamics.")
        sampling = dict(getattr(model, "shared_dynamics", {}).get("prefix_sampling", {}))
        mode = str(sampling.get("mode", "discrete_log_uniform_non_anchor"))
        samples_per_batch = int(sampling.get("samples_per_batch", 1))
        if mode != "discrete_log_uniform_non_anchor":
            raise ValueError(f"Unsupported shared dynamics prefix sampling mode {mode!r}.")
        if samples_per_batch != 1:
            raise ValueError("slimmable_transformer_v1 requires prefix_sampling.samples_per_batch=1.")
        anchors = {int(k) for k in model.K}
        candidates = [k for k in range(int(model.min_k), int(model.D) + 1) if k not in anchors]
        if not candidates:
            raise ValueError("Random prefix training requires at least one non-anchor K in the supported range.")
        candidate_tensor = torch.tensor(candidates, device=emb.device, dtype=torch.long)
        probabilities = candidate_tensor.to(dtype=torch.float32).reciprocal()
        sampled_index = torch.multinomial(probabilities, num_samples=1).item()
        sampled_k = int(candidates[int(sampled_index)])
        random_pred = model.predict_at_k(
            emb[:, : model.history_size, :sampled_k],
            actions[:, : model.history_size],
            k=sampled_k,
        )
        random_target = emb[:, model.num_preds :, :sampled_k]
        random_loss = (random_pred - random_target).pow(2).mean()

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
    if random_loss is not None and sampled_k is not None:
        anchor_weights = list(level_weights or [1.0] * len(pred_losses))
        if len(anchor_weights) != len(pred_losses):
            raise ValueError(f"level_weights has {len(anchor_weights)} entries for {len(pred_losses)} levels")
        anchor_denom = float(sum(anchor_weights))
        combined_denom = anchor_denom + float(random_prefix_weight)
        if anchor_denom <= 0 or combined_denom <= 0:
            raise ValueError("Anchor and random prefix weights must have positive sums.")
        anchor_numerator = pred_losses[0].new_tensor(0.0)
        for loss, weight in zip(pred_losses, anchor_weights):
            anchor_numerator = anchor_numerator + float(weight) * loss
        combined_prediction_loss = (
            anchor_numerator + float(random_prefix_weight) * random_loss
        ) / combined_denom
        anchor_prediction_loss = anchor_numerator / anchor_denom
        logs["loss"] = logs["loss"] + float(rollout_weight) * (
            combined_prediction_loss - anchor_prediction_loss
        )
        logs["pred_loss"] = combined_prediction_loss.detach()
        logs["rollout_loss"] = combined_prediction_loss.detach()
        logs["pred_loss_random"] = random_loss.detach()
        logs["sampled_k"] = emb.new_tensor(sampled_k, dtype=torch.long)
    logs["world_loss"] = logs["loss"].detach()
    if bool(decoder_training_enabled):
        target = _reconstruction_target(batch["pixels"], dtype=emb.dtype, device=emb.device)
        recon_losses = _decoder_reconstruction_losses(model, emb, target, detach_latents=True)
        decoder_loss, recon_logs = weighted_level_mean(
            recon_losses,
            level_weights=level_weights,
            log_prefix="recon_loss",
        )
        logs["decoder_loss"] = decoder_loss
        logs["recon_loss"] = decoder_loss.detach()
        logs.update(recon_logs)
    return logs


__all__ = ["matryoshka_training_loss"]
