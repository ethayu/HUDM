#!/usr/bin/env python3
"""
Visualize fidelity operators on a few frames from a zarr dataset.

Usage:
  python scripts/visualize_fidelity.py --config configs/world.yaml --num-frames 3
  python scripts/visualize_fidelity.py --zarr /path/to/data.zarr --indices 0,10,20
  python scripts/visualize_fidelity.py --config configs/world.yaml --save /tmp/fidelity.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

# Ensure repo root is on PYTHONPATH so "planning" can be imported when
# running this script from arbitrary working directories.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

try:
    import zarr
except Exception:
    zarr = None

from planning.fidelity import apply_fidelity


def to_float01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if np.nanmax(img) > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def parse_indices(indices: str) -> List[int]:
    return [int(x.strip()) for x in indices.split(",") if x.strip()]


def load_frames(zarr_path: str, indices: List[int]) -> List[np.ndarray]:
    if zarr is None:
        raise ImportError("zarr not installed. pip install zarr")
    root = zarr.open_group(zarr_path, mode="r")
    img = root["data"]["img"]
    frames = [img[i] for i in indices]
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to world.yaml for data.zarr_path")
    ap.add_argument("--zarr", help="Path to zarr dataset (overrides --config)")
    ap.add_argument("--indices", help="Comma-separated frame indices")
    ap.add_argument("--num-frames", type=int, default=3, help="Random frames to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="blur_avgpool,blur_quantize",
                    help="Comma-separated fidelity modes")
    ap.add_argument("--levels", default="0.0,0.5,1.0",
                    help="Comma-separated fidelity levels (0=low,1=high)")
    ap.add_argument("--blur-sigma-max", type=float, default=2.0)
    ap.add_argument("--pool-scale-max", type=int, default=4)
    ap.add_argument("--quantize-levels-min", type=int, default=8)
    ap.add_argument("--quantize-levels-max", type=int, default=256)
    ap.add_argument("--save", help="Optional output path to save the figure")
    args = ap.parse_args()

    if args.zarr:
        zarr_path = args.zarr
    else:
        if not args.config:
            raise SystemExit("Provide --config or --zarr")
        cfg = OmegaConf.load(args.config)
        zarr_path = cfg.data.zarr_path

    if zarr is None:
        raise SystemExit("zarr is required for this script.")

    root = zarr.open_group(zarr_path, mode="r")
    img = root["data"]["img"]
    n_frames = img.shape[0]

    if args.indices:
        indices = parse_indices(args.indices)
    else:
        rng = np.random.RandomState(args.seed)
        count = min(args.num_frames, n_frames)
        indices = rng.choice(n_frames, size=count, replace=False).tolist()

    frames = load_frames(zarr_path, indices)
    frames = [to_float01(f) for f in frames]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    levels = [float(x.strip()) for x in args.levels.split(",") if x.strip()]

    variants = [("original", None)]
    for mode in modes:
        for level in levels:
            variants.append((f"{mode}\nlevel={level:.2f}", (mode, level)))

    n_rows = len(frames)
    n_cols = len(variants)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for r, frame in enumerate(frames):
        for c, (title, spec) in enumerate(variants):
            ax = axes[r][c]
            if spec is None:
                img_show = frame
            else:
                mode, level = spec
                img_t = torch.as_tensor(frame)
                img_t = apply_fidelity(
                    img_t,
                    level,
                    mode=mode,
                    blur_sigma_max=args.blur_sigma_max,
                    pool_scale_max=args.pool_scale_max,
                    quantize_levels_min=args.quantize_levels_min,
                    quantize_levels_max=args.quantize_levels_max,
                )
                img_show = img_t.detach().cpu().numpy()
            ax.imshow(img_show)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.text(
                    0.02,
                    0.98,
                    f"idx {indices[r]}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
                    color="white",
                )

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"[save] Figure written to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
