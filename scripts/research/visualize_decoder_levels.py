"""Visualize OGBench Cube frame reconstructions across matryoshka K levels.

Loads a canonical MWM checkpoint, encodes one or more real OGBench Cube
frames (from the Lance dataset behind an eval manifest) with the shared
encoder, then decodes the same latent with each per-level decoder
(K=48..192) to show how reconstruction quality scales with fidelity.

Usage:
    python scripts/research/visualize_decoder_levels.py \
        --checkpoint checkpoints_dense_k48to192_sepopt_20260917/checkpoints_mwm/mwm_paper10_ogb_cube_k48_72_96_120_144_168_192_sepopt_actckpt_20260917 \
        --manifest configs/manifest/data/release20260728/ogb_cube_goal25_exact_seed42_n200.json \
        --num-samples 4 \
        --out reports/shared/dense_sepopt_k48to192_decoder_level_viz/ogb_cube_k48to192_sepopt_start.png
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
from mwm.preprocessing.images import IMAGENET_MEAN, IMAGENET_STD


def _open_manifest(manifest_path: Path) -> tuple[dict, "lance.LanceDataset"]:
    manifest = json.loads(manifest_path.read_text())
    dataset_path = manifest["dataset_path"]
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Dataset path {dataset_path!r} from manifest not found relative to cwd; "
            "run this script from the repo root."
        )
    return manifest, lance.dataset(dataset_path)


def _decode_pixels(table) -> list[np.ndarray]:
    return [np.array(Image.open(io.BytesIO(v.as_py())).convert("RGB")) for v in table.column("pixels")]


def load_frames(manifest_path: Path, num_samples: int, row_field: str, start_index: int = 0) -> tuple[list[np.ndarray], list[str]]:
    manifest, ds = _open_manifest(manifest_path)
    pairs = manifest["pairs"][start_index : start_index + num_samples]
    rows = [pair[row_field] for pair in pairs]
    frames = _decode_pixels(ds.take(rows, columns=["pixels"]))
    step_field = row_field.replace("_row", "_step")
    labels = [f"ep {pair['episode']}\nstep {pair.get(step_field, '')}" for pair in pairs]
    return frames, labels


def load_rollout_frames(
    manifest_path: Path, pair_index: int, num_timesteps: int, stride: int
) -> tuple[list[np.ndarray], list[str], dict]:
    """Frames at start_row + i*stride from one episode (rows are contiguous per episode)."""
    manifest, ds = _open_manifest(manifest_path)
    pair = manifest["pairs"][pair_index]
    rows = [pair["start_row"] + i * stride for i in range(num_timesteps)]
    table = ds.take(rows, columns=["pixels", "episode_idx", "step_idx"])
    episodes = table.column("episode_idx").to_pylist()
    steps = table.column("step_idx").to_pylist()
    expected_steps = [pair["start_step"] + i * stride for i in range(num_timesteps)]
    if any(ep != pair["episode"] for ep in episodes) or steps != expected_steps:
        raise ValueError(
            f"Rows {rows} are not contiguous timesteps of episode {pair['episode']} "
            f"(got episodes={episodes}, steps={steps}); reduce --num-timesteps/--stride."
        )
    goal_step = pair.get("goal_step")
    labels = []
    for i, step in enumerate(steps):
        tag = " (start)" if i == 0 else (" (goal)" if step == goal_step else "")
        labels.append(f"t=+{i * stride}{tag}\nstep {step}")
    return _decode_pixels(table), labels, pair


def unnormalize(recon: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=recon.dtype, device=recon.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=recon.dtype, device=recon.device).view(1, 3, 1, 1)
    return (recon * std + mean).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Canonical MWM checkpoint directory.")
    parser.add_argument("--manifest", required=True, help="Eval manifest JSON with Lance row pointers.")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of manifest pairs to visualize.")
    parser.add_argument("--start-index", type=int, default=0, help="First manifest pair index to visualize.")
    parser.add_argument(
        "--row-field",
        default="start_row",
        choices=["start_row", "goal_row"],
        help="Which manifest row pointer to visualize per pair.",
    )
    parser.add_argument(
        "--rollout-pair",
        type=int,
        default=None,
        help="If set, rows are successive timesteps of this manifest pair's episode instead of one frame per pair.",
    )
    parser.add_argument("--num-timesteps", type=int, default=6, help="Rows in --rollout-pair mode.")
    parser.add_argument("--stride", type=int, default=5, help="Env steps between rows in --rollout-pair mode.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, metadata, _ = load_world_model_from_checkpoint(args.checkpoint, epoch=None, device=device)
    model = model.to(device)
    k_levels = list(model.K)

    if args.rollout_pair is not None:
        frames, labels, pair = load_rollout_frames(Path(args.manifest), args.rollout_pair, args.num_timesteps, args.stride)
        title_suffix = f" | episode {pair['episode']}, every {args.stride} steps"
    else:
        frames, labels = load_frames(Path(args.manifest), args.num_samples, args.row_field, args.start_index)
        title_suffix = ""

    pixels = torch.stack([torch.from_numpy(f) for f in frames]).to(device)  # (N, H, W, 3) uint8
    with torch.no_grad():
        emb = model.encode(pixels, already_preprocessed=False)  # (N, D)
        recon_by_level = []
        for level_idx in range(len(k_levels)):
            recon = model.decode(level_idx, emb)  # (N, 3, H, W), ImageNet-normalized space
            recon_by_level.append(unnormalize(recon).cpu())

    n_rows = len(frames)
    n_cols = 1 + len(k_levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False)

    for row in range(n_rows):
        axes[row][0].imshow(frames[row])
        axes[row][0].set_title("ground truth" if row == 0 else "", fontsize=10)
        axes[row][0].set_ylabel(labels[row], fontsize=8)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        for col, (k, recon) in enumerate(zip(k_levels, recon_by_level), start=1):
            img = recon[row].permute(1, 2, 0).numpy()
            axes[row][col].imshow(img)
            if row == 0:
                axes[row][col].set_title(f"K={k}", fontsize=10)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    fig.suptitle(f"Decoder reconstructions by K level -- {Path(args.checkpoint).name}{title_suffix}", fontsize=11)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
