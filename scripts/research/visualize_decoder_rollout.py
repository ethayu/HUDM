"""Visualize per-level dynamics rollouts decoded back to pixels.

For one manifest pair's episode, encode the start frame with the shared
encoder, then step each level's own transition model forward with the
recorded dataset actions and decode every predicted embedding with that
level's decoder. One model step = ``action_block`` env steps, so row i is
env step ``start_step + i * action_block``.

Modes:
  autoregressive  -- only t=0 is encoded; each prediction is fed back in
                     (open-loop imagination, errors compound).
  teacher_forced  -- each t+1 is predicted from the *encoded* real frames
                     up to t (within the model's history window), i.e.
                     one-step prediction error only.

Actions are preprocessed exactly as in training: per-base-dim z-score over
the full dataset action column (``mwm.data.transforms.ZScoreScaler``), then
``action_block`` consecutive raw actions are flattened into one model action.

Usage:
    PYTHONPATH=. python scripts/research/visualize_decoder_rollout.py \
        --checkpoint checkpoints_mwm/mwm_paper10_pusht_k48_72_96_120_144_168_192_sepopt_actckpt_20260917 \
        --manifest configs/manifest/data/release20260728/pusht_goal25_exact_seed42_n200.json \
        --pair 0 --num-steps 5 --mode autoregressive \
        --out reports/shared/dense_sepopt_k48to192_decoder_level_viz/pusht_k48to192_sepopt_dynrollout_ar_pair0.png
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import lance
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mwm.checkpoint_io import load_world_model_from_checkpoint
from mwm.data.transforms import ZScoreScaler
from mwm.preprocessing.images import IMAGENET_MEAN, IMAGENET_STD


def unnormalize(recon: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=recon.dtype, device=recon.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=recon.dtype, device=recon.device).view(1, 3, 1, 1)
    return (recon * std + mean).clamp(0.0, 1.0)


def column_to_numpy(ds: "lance.LanceDataset", col: str) -> np.ndarray:
    arr = ds.to_table(columns=[col]).column(col).combine_chunks()
    n = len(arr)
    while str(arr.type).startswith(("fixed_size_list", "list", "large_list")):
        arr = arr.flatten()
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(n, -1)


def load_episode_clip(manifest_path: Path, pair_index: int, num_steps: int, block: int):
    manifest = json.loads(manifest_path.read_text())
    dataset_path = manifest["dataset_path"]
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path {dataset_path!r} not found; run from the repo root.")
    ds = lance.dataset(dataset_path)
    pair = manifest["pairs"][pair_index]
    start = int(pair["start_row"])
    span = num_steps * block
    rows = list(range(start, start + span + 1))
    table = ds.take(rows, columns=["pixels", "action", "episode_idx", "step_idx"])
    episodes = table.column("episode_idx").to_pylist()
    steps = table.column("step_idx").to_pylist()
    if any(ep != pair["episode"] for ep in episodes) or steps != list(range(pair["start_step"], pair["start_step"] + span + 1)):
        raise ValueError(f"Rows {rows[0]}..{rows[-1]} are not contiguous steps of episode {pair['episode']}; reduce --num-steps.")
    obs_idx = [i * block for i in range(num_steps + 1)]
    pixels = table.column("pixels").to_pylist()
    frames = [np.array(Image.open(io.BytesIO(pixels[i])).convert("RGB")) for i in obs_idx]
    raw_actions = np.asarray(table.column("action").to_pylist()[:span], dtype=np.float32)
    raw_actions = raw_actions.reshape(span, -1)
    if np.isnan(raw_actions).any():
        raise ValueError("NaN actions inside the requested clip (episode end?); reduce --num-steps.")
    frame_steps = [steps[i] for i in obs_idx]
    return ds, pair, frames, raw_actions, frame_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair", type=int, default=0, help="Manifest pair index (episode + start step).")
    parser.add_argument("--num-steps", type=int, default=5, help="Model steps to roll out (each = action_block env steps).")
    parser.add_argument("--mode", choices=["autoregressive", "teacher_forced"], default="autoregressive")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, metadata, _ = load_world_model_from_checkpoint(args.checkpoint, epoch=None, device=device)
    model = model.to(device).eval()
    k_levels = list(model.K)
    block = int(model.action_block)
    history_size = int(model.history_size)

    ds, pair, frames, raw_actions, frame_steps = load_episode_clip(Path(args.manifest), args.pair, args.num_steps, block)

    scaler = ZScoreScaler().fit(column_to_numpy(ds, "action"))
    actions = torch.as_tensor(scaler(raw_actions), dtype=torch.float32).reshape(args.num_steps, -1)  # (T, block*adim)
    if actions.shape[-1] != int(model.action_dim):
        raise ValueError(f"Blocked action dim {actions.shape[-1]} != model.action_dim {model.action_dim}.")
    actions = actions.to(device)

    pixels = torch.stack([torch.from_numpy(f) for f in frames]).to(device)  # (T+1, H, W, 3)
    T = args.num_steps
    with torch.no_grad():
        enc = model.encode(pixels, already_preprocessed=False)  # (T+1, D) encoded real frames
        recon_rows: list[list[np.ndarray]] = [[] for _ in range(T + 1)]
        latent_err: list[list[float]] = [[] for _ in range(T + 1)]
        for level_idx, k in enumerate(k_levels):
            if args.mode == "autoregressive":
                infos = {"pixels": pixels[:1].view(1, 1, 1, *pixels.shape[1:]), "emb": enc[:1].view(1, 1, 1, -1)}
                out = model.rollout_at_level(infos, actions.view(1, 1, T, -1), level_idx)
                pred = out["predicted_emb"].view(-1, k)[: T + 1]  # (T+1, k); row 0 = encoded start
            else:
                preds = [enc[0, :k]]
                for t in range(T):
                    lo = max(0, t + 1 - history_size)
                    ctx = enc[lo : t + 1, :k].unsqueeze(0)
                    step = model._predict_prefix(level_idx, ctx, actions[lo : t + 1].unsqueeze(0))
                    preds.append(step[0, -1])
                pred = torch.stack(preds)
            recon = unnormalize(model.decode(level_idx, pred)).cpu()
            for t in range(T + 1):
                recon_rows[t].append(recon[t].permute(1, 2, 0).numpy())
                latent_err[t].append(float((pred[t] - enc[t, :k]).pow(2).mean()))

    n_rows, n_cols = T + 1, 1 + len(k_levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.3 * n_rows), squeeze=False)
    for t in range(n_rows):
        ax = axes[t][0]
        ax.imshow(frames[t])
        tag = " (start, encoded)" if t == 0 else (" (goal)" if frame_steps[t] == pair.get("goal_step") else "")
        ax.set_ylabel(f"t=+{t * block}{tag}\nstep {frame_steps[t]}", fontsize=8)
        if t == 0:
            ax.set_title("ground truth", fontsize=10)
        for col, k in enumerate(k_levels, start=1):
            ax = axes[t][col]
            ax.imshow(recon_rows[t][col - 1])
            if t == 0:
                ax.set_title(f"K={k}", fontsize=10)
            else:
                ax.text(0.02, 0.02, f"latent mse {latent_err[t][col - 1]:.3f}", transform=ax.transAxes,
                        fontsize=6, color="white", bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        for ax in axes[t]:
            ax.set_xticks([])
            ax.set_yticks([])

    mode_desc = (
        "autoregressive: encode t=0 only, feed predictions back"
        if args.mode == "autoregressive"
        else f"teacher-forced: predict t+1 from encoded real frames (history {history_size})"
    )
    fig.suptitle(
        f"Per-level dynamics rollout, decoded -- {Path(args.checkpoint).name}\n"
        f"episode {pair['episode']} | {mode_desc} | 1 model step = {block} env steps",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
