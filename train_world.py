import os
import datetime
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb

from models.world.model import HierWorldModel
from datasets.zarr_rollouts import ZarrPushTWindows
from datasets.mixed_zarr import build_mixed_zarr_windows


def make_run_dir(root: str, tag: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{tag}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def level_train_loop(level: int, model: HierWorldModel, cfg, train_loader, val_loader, device, run_dir: str) -> None:
    k = model.K[level]
    E = model.encoder
    D = model.decoders[level]
    F = model.dynamics[level]

    E.to(device); D.to(device); F.to(device)
    E.train(); D.train(); F.train()

    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()

    opt_E = torch.optim.Adam(E.parameters(), lr=cfg.optim.lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.optim.lr)
    opt_F = torch.optim.Adam(F.parameters(), lr=cfg.optim.lr)

    best_val = float('inf')
    no_improve = 0
    max_epochs = int(cfg.schedule.max_epochs_per_level)
    patience = int(cfg.schedule.patience)
    min_delta = float(cfg.schedule.min_delta)
    beta = float(cfg.loss.beta_null)
    T = int(cfg.data.horizon_T)

    for epoch in range(1, max_epochs + 1):
        total = 0.0
        count = 0
        comp_totals = {"recon": 0.0, "teacher": 0.0, "roll": 0.0, "null": 0.0}
        for batch in train_loader:
            x_t = batch['x_t'].to(device)
            x_t1 = batch['x_t1'].to(device)
            x_tf = batch['x_tfut'].to(device)
            a_t = batch['a_t'].to(device)
            a_seq = batch['a_seq'].to(device)  # (B,T,A)

            # encode
            z_t = E(x_t)
            z_t1 = E(x_t1)
            z_tf = E(x_tf)
            z_t_k = z_t[:, :k]
            z_t1_k = z_t1[:, :k]
            z_tf_k = z_tf[:, :k]

            # recon
            x_hat = D(z_t_k)
            L_recon = loss_mse(x_hat, x_t)

            # teacher forcing
            z_pred_next = F.step(z_t_k, a_t)
            L_teacher = loss_l1(z_pred_next, z_t1_k)

            # rollout to t+T
            z_roll = F.rollout(z_t_k, a_seq, detach_each_step=True)
            L_roll = loss_l1(z_roll, z_tf_k)

            # null action
            a_null = torch.zeros_like(a_t)
            z_null = F.step(z_t_k, a_null)
            L_null = loss_l1(z_null, z_t_k)

            loss = L_recon + L_teacher + L_roll + beta * L_null

            opt_E.zero_grad(); opt_D.zero_grad(); opt_F.zero_grad()
            loss.backward()
            opt_E.step(); opt_D.step(); opt_F.step()

            total += loss.item(); count += 1
            comp_totals["recon"] += float(L_recon.item())
            comp_totals["teacher"] += float(L_teacher.item())
            comp_totals["roll"] += float(L_roll.item())
            comp_totals["null"] += float(L_null.item())

        # validation
        E.eval(); D.eval(); F.eval()
        with torch.no_grad():
            vtot = 0.0; vcount = 0
            vcomps = {"recon": 0.0, "teacher": 0.0, "roll": 0.0, "null": 0.0}
            for batch in val_loader:
                x_t = batch['x_t'].to(device)
                x_t1 = batch['x_t1'].to(device)
                x_tf = batch['x_tfut'].to(device)
                a_t = batch['a_t'].to(device)
                a_seq = batch['a_seq'].to(device)
                z_t = E(x_t); z_t1 = E(x_t1); z_tf = E(x_tf)
                z_t_k = z_t[:, :k]; z_t1_k = z_t1[:, :k]; z_tf_k = z_tf[:, :k]
                x_hat = D(z_t_k); L_recon = loss_mse(x_hat, x_t)
                z_pred_next = F.step(z_t_k, a_t); L_teacher = nn.functional.l1_loss(z_pred_next, z_t1_k)
                z_roll = F.rollout(z_t_k, a_seq, detach_each_step=True); L_roll = nn.functional.l1_loss(z_roll, z_tf_k)
                a_null = torch.zeros_like(a_t); z_null = F.step(z_t_k, a_null); L_null = nn.functional.l1_loss(z_null, z_t_k)
                loss = L_recon + L_teacher + L_roll + beta * L_null
                vtot += loss.item(); vcount += 1
                vcomps["recon"] += float(L_recon.item())
                vcomps["teacher"] += float(L_teacher.item())
                vcomps["roll"] += float(L_roll.item())
                vcomps["null"] += float(L_null.item())
        E.train(); D.train(); F.train()

        avg_train = total / max(1, count)
        avg_val = vtot / max(1, vcount)
        avg_train_comps = {k: v / max(1, count) for k, v in comp_totals.items()}
        avg_val_comps = {k: v / max(1, vcount) for k, v in vcomps.items()}
        print(f"[level {level}] epoch {epoch}  train {avg_train:.4f}  val {avg_val:.4f}")
        if cfg.get('wandb', {}).get('enable', False):
            wandb.log({
                f"train_l{level}/loss": avg_train,
                f"val_l{level}/loss": avg_val,
                **{f"train_l{level}/{k}": v for k, v in avg_train_comps.items()},
                **{f"val_l{level}/{k}": v for k, v in avg_val_comps.items()},
                "epoch": epoch,
            })

        # early-stopping style convergence
        if avg_val + min_delta < best_val:
            best_val = avg_val
            no_improve = 0
            # checkpoint current best
            torch.save(E.state_dict(), os.path.join(run_dir, 'encoder.pt'))
            torch.save(D.state_dict(), os.path.join(run_dir, f'decoder_l{level}.pt'))
            torch.save(F.state_dict(), os.path.join(run_dir, f'dyn_l{level}.pt'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[level {level}] converged (patience reached)")
                break


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.train.no_cuda else 'cpu')

    # Build model
    K: List[int] = list(cfg.model.K)
    D: int = int(cfg.model.D)
    assert max(K) == D, "Largest K must equal D"
    wm = HierWorldModel(K=K, D=D, action_dim=cfg.data.action_dim).to(device)

    # Data
    if getattr(cfg.data, 'synthetic', None) and getattr(cfg.data.synthetic, 'enable', False):
        tr_ds, va_ds = build_mixed_zarr_windows(cfg)
    else:
        tr_ds = ZarrPushTWindows(cfg.data.zarr_path, split='train', split_ratio=cfg.data.split_ratio, horizon_T=cfg.data.horizon_T)
        va_ds = ZarrPushTWindows(cfg.data.zarr_path, split='valid', split_ratio=cfg.data.split_ratio, horizon_T=cfg.data.horizon_T)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    va_loader = DataLoader(va_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

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

    # Train per level
    for level in range(len(K)):
        print(f"=== Training level {level} (k={K[level]}) ===")
        level_train_loop(level, wm, cfg, tr_loader, va_loader, device, run_dir)

    print("Training complete.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_world.py configs/world.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
