"""Read-only diagnostic (no training, no checkpoint/data mutation).

Question: why is dense-matched K=96 (jointly-trained checkpoint's K=96 slice)
so much better at planning than individually-trained K=96, given that
individually-trained K=96 actually has the LOWER (better) validation rollout
loss of the two (per checkpoints_paper_k_sweep_20260728/training_val_losses.json)?
That mismatch (lower loss, much worse task performance) is the classic
signature of representation collapse: a latent that barely varies across
different real states is trivially easy to predict (hence low loss) but
carries little information usable for planning.

This directly tests that hypothesis: encode a shared batch of real TwoRoom
frames with BOTH checkpoints' own encoders, take each one's first 96 latent
dims (the only dims individual_k96 ever trains -- its raw encoder output is
also 192-dim per the base architecture, but dims 96:192 are never touched by
any loss), and compare:
  - per-dimension variance across the batch (are individual_k96's dims
    collapsed toward near-constant values?)
  - effective rank of the 96x96 covariance matrix (how many of the 96 dims
    actually carry independent variance, via participation ratio and count
    of singular values needed for 95% of variance)
  - typical pairwise distance between DIFFERENT states' latents (are
    different real states mapped close together = collapse, or well
    separated?)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/aurora/HUDM-mwm-ethan")
LANCE_PATH = REPO_ROOT / "data/upstream/tworoom.lance"
NUM_SAMPLES = 400
SEED = 0

sys.path.insert(0, str(REPO_ROOT))

from mwm.checkpoint_io import load_world_model_from_checkpoint  # noqa: E402


def pick_device() -> torch.device:
    # Both GPUs are busy with other sweeps tonight; this analysis is tiny
    # (encoding 400 frames with 2 models), so run on CPU to avoid contention.
    return torch.device("cpu")


def latent_stats(z: torch.Tensor) -> dict:
    z = z.double()
    n = z.shape[0]
    mean = z.mean(dim=0, keepdim=True)
    centered = z - mean
    cov = (centered.T @ centered) / (n - 1)
    var_per_dim = torch.diag(cov)
    total_var = var_per_dim.sum().item()

    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    eigvals_sorted = torch.sort(eigvals, descending=True).values
    total = eigvals_sorted.sum()
    frac = eigvals_sorted / total.clamp(min=1e-12)
    cum = torch.cumsum(frac, dim=0)
    rank95 = int((cum < 0.95).sum().item()) + 1

    participation_ratio = (eigvals_sorted.sum() ** 2 / (eigvals_sorted ** 2).sum().clamp(min=1e-12)).item()

    idx = np.random.default_rng(SEED).choice(n, size=min(200, n), replace=False)
    sample = z[idx]
    d = torch.cdist(sample, sample)
    off_diag = d[~torch.eye(len(sample), dtype=torch.bool)]
    mean_pairwise_dist = off_diag.mean().item()

    return dict(
        total_var=total_var,
        mean_var_per_dim=(total_var / z.shape[1]),
        min_var_dim=var_per_dim.min().item(),
        max_var_dim=var_per_dim.max().item(),
        rank95_of_96=rank95,
        participation_ratio=participation_ratio,
        mean_pairwise_dist=mean_pairwise_dist,
    )


def main() -> None:
    device = pick_device()
    print(f"Using device: {device}")

    from stable_worldmodel.data import load_dataset

    raw_ds = load_dataset(
        str(LANCE_PATH), format="lance", frameskip=1, num_steps=1,
        keys_to_load=["pixels"], keys_to_cache=[],
    )
    print(f"Dataset rows available: {len(raw_ds)}")

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(raw_ds), size=NUM_SAMPLES, replace=False)
    samples = [raw_ds[int(i)] for i in chosen]
    pixels = torch.stack([s["pixels"] for s in samples]).to(device)  # (N, 1, C, H, W)
    pixels = pixels[:, 0]  # (N, C, H, W)
    print(f"pixels shape={tuple(pixels.shape)} dtype={pixels.dtype}")

    checkpoints = {
        "individual_k96": REPO_ROOT / "checkpoints_mwm/mwm_paper10_tworoom_k96_release20260728",
        "dense_k96_slice": REPO_ROOT / "checkpoints_mwm/mwm_paper10_tworoom_k96_120_144_168_192_release20260728",
    }

    results = {}
    with torch.no_grad():
        for name, ckpt_dir in checkpoints.items():
            model, metadata, _epoch = load_world_model_from_checkpoint(ckpt_dir, epoch=None, device=device)
            model.eval()
            emb = model.encode(pixels, already_preprocessed=False)  # (N, D)
            emb96 = emb[..., :96]
            stats = latent_stats(emb96)
            results[name] = stats
            print(f"\n=== {name} (D={model.D}, K={model.K}) ===")
            for k, v in stats.items():
                print(f"  {k}: {v:.6g}")
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

    out_path = REPO_ROOT / "analysis_output_tworoom_k96_latent_collapse.json"
    with open(out_path, "w") as f:
        json.dump({"n_samples": NUM_SAMPLES, "seed": SEED, "results": results}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
