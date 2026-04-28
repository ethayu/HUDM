from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.world.model import HierWorldModel


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    loss_per = diff.pow(2).mean(dim=-1)
    mask_f = mask.float()
    return (loss_per * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def run_epoch(
    model: HierWorldModel,
    loader: DataLoader,
    cfg,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    train: bool = True,
) -> Tuple[float, Dict[str, float]]:
    if train:
        model.train()
    else:
        model.eval()

    recon_w = float(getattr(cfg.loss, "recon_weight", 1.0))
    teacher_w = float(getattr(cfg.loss, "teacher_weight", 1.0))
    rollout_w = float(getattr(cfg.loss, "rollout_weight", 1.0))

    total_loss = 0.0
    total_count = 0
    totals: Dict[str, float] = {}

    for batch in loader:
        x = batch["x"].to(device)
        a = batch["a"].to(device)
        mask = batch["mask"].to(device)
        batch_size, seq_len = mask.shape
        channels, height, width = x.shape[2:]

        x_flat = x.view(batch_size * seq_len, channels, height, width)
        z_flat = model.encoder(x_flat)
        z = z_flat.view(batch_size, seq_len, -1)
        mask_flat = mask.view(batch_size * seq_len)

        a_null = torch.zeros((batch_size, 1, a.shape[-1]), device=device, dtype=a.dtype)
        a_full = torch.cat([a_null, a], dim=1)

        z_in = torch.cat([z[:, :1, :], z[:, :-1, :]], dim=1)
        z_in_flat = z_in.view(batch_size * seq_len, -1)
        a_full_flat = a_full.view(batch_size * seq_len, -1)

        loss = torch.zeros((), device=device)
        logs: Dict[str, float] = {}

        for li, k in enumerate(model.K):
            if model.decoder_mode == "per_level":
                z_k_flat = z_flat[:, :k]
                x_hat_flat = model.decoders[li](z_k_flat)
            else:
                z_pad_flat = model._pad_prefix(z_flat, k)
                x_hat_flat = model.decoder(z_pad_flat)
            recon = F.mse_loss(x_hat_flat[mask_flat], x_flat[mask_flat])

            if model.dynamics_mode == "per_level":
                z_in_k_flat = z_in_flat[:, :k]
                pred_flat = model.dynamics[li].step(z_in_k_flat, a_full_flat)
                pred = pred_flat.view(batch_size, seq_len, k)
            else:
                z_in_pad_flat = model._pad_prefix(z_in_flat, k)
                pred_full_flat = model.dynamics.step(z_in_pad_flat, a_full_flat)
                pred = pred_full_flat[:, :k].view(batch_size, seq_len, k)
            teacher = masked_mse(pred, z[:, :, :k], mask)

            if model.dynamics_mode == "per_level":
                z_prev = z[:, 0, :k]
                preds = []
                for t in range(seq_len):
                    z_prev = model.dynamics[li].step(z_prev, a_full[:, t, :])
                    preds.append(z_prev)
                pred_roll = torch.stack(preds, dim=1)
            else:
                z_prev = model._pad_prefix(z[:, 0, :], k)
                preds = []
                for t in range(seq_len):
                    z_prev = model.dynamics.step(z_prev, a_full[:, t, :])
                    if k < model.D:
                        z_prev = torch.cat([z_prev[..., :k], z_prev.new_zeros(z_prev[..., k:].shape)], dim=-1)
                    preds.append(z_prev[..., :k])
                pred_roll = torch.stack(preds, dim=1)
            rollout = masked_mse(pred_roll, z[:, :, :k], mask)

            loss = loss + recon_w * recon + teacher_w * teacher + rollout_w * rollout
            logs[f"recon_l{li}"] = float(recon.item())
            logs[f"teacher_l{li}"] = float(teacher.item())
            logs[f"rollout_l{li}"] = float(rollout.item())

        if train:
            if optimizer is None:
                raise ValueError("optimizer is required when train=True")
            for p in model.parameters():
                p.grad = None
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_count += 1
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value

    avg_loss = total_loss / max(1, total_count)
    avg_logs = {k: v / max(1, total_count) for k, v in totals.items()}
    return avg_loss, avg_logs
