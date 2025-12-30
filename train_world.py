import os
import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb

from models.world.model import HierWorldModel
from datasets.zarr_episodes import ZarrPushTEpisodes, collate_episodes
from datasets.mixed_zarr import build_mixed_zarr_episodes


def make_run_dir(root: str, tag: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{tag}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


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
        x = batch["x"].to(device)          # (B,L,C,H,W)
        a = batch["a"].to(device)          # (B,L-1,A)
        mask = batch["mask"].to(device)    # (B,L)
        B, L = mask.shape
        C, H, W = x.shape[2:]

        x_flat = x.view(B * L, C, H, W)
        z_flat = model.encoder(x_flat)
        z = z_flat.view(B, L, -1)
        mask_flat = mask.view(B * L)

        # Null action at t=0
        a_null = torch.zeros((B, 1, a.shape[-1]), device=device, dtype=a.dtype)
        a_full = torch.cat([a_null, a], dim=1)  # (B,L,A)

        # Teacher forcing inputs/targets
        z_in = torch.cat([z[:, :1, :], z[:, :-1, :]], dim=1)  # (B,L,D)
        z_in_flat = z_in.view(B * L, -1)
        a_full_flat = a_full.view(B * L, -1)

        loss = torch.zeros((), device=device)
        logs: Dict[str, float] = {}

        for li, k in enumerate(model.K):
            # Reconstruction loss (per step)
            if model.decoder_mode == "per_level":
                z_k_flat = z_flat[:, :k]
                x_hat_flat = model.decoders[li](z_k_flat)
            else:
                z_pad_flat = model._pad_prefix(z_flat, k)
                x_hat_flat = model.decoder(z_pad_flat)
            recon = F.mse_loss(x_hat_flat[mask_flat], x_flat[mask_flat])

            # Teacher forcing loss over all steps
            if model.dynamics_mode == "per_level":
                z_in_k_flat = z_in_flat[:, :k]
                pred_flat = model.dynamics[li].step(z_in_k_flat, a_full_flat)
                pred = pred_flat.view(B, L, k)
            else:
                z_in_pad_flat = model._pad_prefix(z_in_flat, k)
                pred_full_flat = model.dynamics.step(z_in_pad_flat, a_full_flat)
                pred = pred_full_flat[:, :k].view(B, L, k)
            teacher = masked_mse(pred, z[:, :, :k], mask)

            # Autoregressive rollout loss (null at t=0)
            if model.dynamics_mode == "per_level":
                z_prev = z[:, 0, :k]
                preds = []
                for t in range(L):
                    z_prev = model.dynamics[li].step(z_prev, a_full[:, t, :])
                    preds.append(z_prev)
                pred_roll = torch.stack(preds, dim=1)  # (B,L,k)
            else:
                z_prev = model._pad_prefix(z[:, 0, :], k)
                preds = []
                for t in range(L):
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


def save_checkpoint(model: HierWorldModel, run_dir: str) -> None:
    torch.save(model.encoder.state_dict(), os.path.join(run_dir, "encoder.pt"))
    if model.decoder_mode == "per_level":
        for li in range(len(model.K)):
            torch.save(model.decoders[li].state_dict(), os.path.join(run_dir, f"decoder_l{li}.pt"))
    else:
        torch.save(model.decoder.state_dict(), os.path.join(run_dir, "decoder.pt"))
    if model.dynamics_mode == "per_level":
        for li in range(len(model.K)):
            torch.save(model.dynamics[li].state_dict(), os.path.join(run_dir, f"dyn_l{li}.pt"))
    else:
        torch.save(model.dynamics.state_dict(), os.path.join(run_dir, "dyn.pt"))


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.train.no_cuda else 'cpu')

    # Build model
    K: List[int] = list(cfg.model.K)
    D: int = int(cfg.model.D)
    assert max(K) == D, "Largest K must equal D"
    wm = HierWorldModel(
        K=K,
        D=D,
        action_dim=cfg.data.action_dim,
        decoder_mode=str(getattr(cfg.model, "decoder_mode", "per_level")),
        dynamics_mode=str(getattr(cfg.model, "dynamics_mode", "per_level")),
    ).to(device)

    # Data
    if getattr(cfg.data, 'synthetic', None) and getattr(cfg.data.synthetic, 'enable', False):
        tr_ds, va_ds = build_mixed_zarr_episodes(cfg)
    else:
        tr_ds = ZarrPushTEpisodes(cfg.data.zarr_path, split='train', split_ratio=cfg.data.split_ratio)
        va_ds = ZarrPushTEpisodes(cfg.data.zarr_path, split='valid', split_ratio=cfg.data.split_ratio)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_episodes,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_episodes,
    )

    # Run dir and metadata
    run_dir = make_run_dir(cfg.train.checkpoint_dir, cfg.train.run_name)
    with open(os.path.join(run_dir, 'world.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # WandB
    if cfg.get('wandb', {}).get('enable', False):
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=run_dir,
        )

    optimizer = torch.optim.Adam(wm.parameters(), lr=cfg.optim.lr)

    best_val = float("inf")
    no_improve = 0
    max_epochs = int(getattr(cfg.schedule, "max_epochs", 30))
    patience = int(cfg.schedule.patience)
    min_delta = float(cfg.schedule.min_delta)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_logs = run_epoch(wm, tr_loader, cfg, device, optimizer=optimizer, train=True)
        with torch.no_grad():
            val_loss, val_logs = run_epoch(wm, va_loader, cfg, device, train=False)

        print(f"epoch {epoch}  train {train_loss:.4f}  val {val_loss:.4f}")
        if cfg.get('wandb', {}).get('enable', False):
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                **{f"train/{k}": v for k, v in train_logs.items()},
                **{f"val/{k}": v for k, v in val_logs.items()},
                "epoch": epoch,
            })

        if val_loss + min_delta < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(wm, run_dir)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("converged (patience reached)")
                break

    print("Training complete.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_world.py configs/world.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
