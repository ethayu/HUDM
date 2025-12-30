#!/usr/bin/env python3
"""
Visualize world-model decoder reconstructions per level against ground truth.

Produces a grid image where each row corresponds to a level ℓ and shows
columns of [GT | Recon_ℓ] for a few samples.

Usage:
  python scripts/visualize_world_decoder.py configs/world.yaml --count 5 --out rollouts/decoder_grid.png
"""

import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

from models.world.model import HierWorldModel
from datasets.zarr_episodes import ZarrPushTEpisodes


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--count', type=int, default=4)
    ap.add_argument('--out', type=str, default='rollouts/decoder_grid.png')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.train.no_cuda else 'cpu')

    # Build model and try to load decoders/encoder from latest run dir under checkpoint
    wm = HierWorldModel(
        K=list(cfg.model.K),
        D=int(cfg.model.D),
        action_dim=cfg.data.action_dim,
        decoder_mode=str(getattr(cfg.model, "decoder_mode", "per_level")),
        dynamics_mode=str(getattr(cfg.model, "dynamics_mode", "per_level")),
    ).to(device)

    # Attempt to locate latest run dir
    ckpt_root = cfg.train.checkpoint_dir
    run_dirs = [os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))]
    if not run_dirs:
        print('No checkpoint run directories found; using random weights')
    else:
        latest = max(run_dirs, key=os.path.getmtime)
        enc_p = os.path.join(latest, 'encoder.pt')
        if os.path.isfile(enc_p):
            wm.encoder.load_state_dict(torch.load(enc_p, map_location=device))
        decoder_mode = str(getattr(cfg.model, "decoder_mode", "per_level"))
        if decoder_mode == "shared":
            dp = os.path.join(latest, "decoder.pt")
            if os.path.isfile(dp):
                wm.decoder.load_state_dict(torch.load(dp, map_location=device))
        else:
            for li in range(len(cfg.model.K)):
                dp = os.path.join(latest, f'decoder_l{li}.pt')
                if os.path.isfile(dp):
                    wm.decoders[li].load_state_dict(torch.load(dp, map_location=device))

    wm.eval()

    ds = ZarrPushTEpisodes(cfg.data.zarr_path, split='valid', split_ratio=cfg.data.split_ratio)
    # collect N samples from mid-episode frames
    idxs = np.linspace(0, len(ds) - 1, num=min(args.count, len(ds)), dtype=int)
    xs = []
    recons_by_level = [[] for _ in cfg.model.K]
    for i in idxs:
        sample = ds[i]
        x_seq = sample["x"]
        t_mid = x_seq.shape[0] // 2
        x_t = x_seq[t_mid].unsqueeze(0).to(device)
        z = wm.encode(x_t)
        xs.append(x_seq[t_mid])
        for li, k in enumerate(cfg.model.K):
            recon = wm.decode(li, z).squeeze(0).cpu()
            recons_by_level[li].append(recon)

    # Create grid: for each level a row of [GT | Recon] pairs
    rows = []
    # normalised in [-1,1]; save_image expects [0,1], but make_grid handles; we will clamp after denorm
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0,1)

    gt_row = torch.stack([denorm(x) for x in xs])
    # For each level: interleave GT and Recon
    for li, k in enumerate(cfg.model.K):
        interleaved = []
        for j in range(len(xs)):
            interleaved.append(denorm(xs[j]))
            interleaved.append(denorm(recons_by_level[li][j]))
        grid = make_grid(torch.stack(interleaved), nrow=2*len(xs), padding=2)
        rows.append(grid)

    # Stack rows vertically
    full = torch.cat(rows, dim=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_image(full, args.out)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
