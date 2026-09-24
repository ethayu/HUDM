"""Paper figure: reconstructions and autoregressive predictions across MWM levels.

Crops tiles out of the <env>_k48to192_sepopt_dynrollout_ar_pair0.png grids in
this folder (no model is re-run). Per environment, two rows:
  - Recon.: t=+0 row -- the real start frame encoded once and decoded at each level.
  - Pred. t=+25: last row -- each level's dynamics rolled out open-loop for 5 model
    steps (25 env steps) from the same start frame with the recorded actions.
The first column is the ground-truth frame. Reacher uses pair 1
instead of pair 0 (see OVERRIDES). The per-panel "latent mse" labels
are cropped away (they are not comparable across levels; see README).

Usage: python generate_paper_figure.py [OUTPUT]  (default: decoder_levels.pdf here)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

# Tile frame positions (px) in the 2640x2070 dynrollout grids: 8 columns
# (ground truth, K=48..192) x 6 rows (t=+0, +5, ..., +25).
# Inset crop (top, bottom, left, right) drops the frame and the bottom-left
# "latent mse" label -> 279x279 tiles.
ROLLOUT_GRID = dict(col_x=[62, 384, 706, 1029, 1351, 1673, 1995, 2317],
                    row_y=[120, 444, 769, 1093, 1417, 1741], tile=300, crop=(2, 19, 10, 11))
# Tile frame positions in the 2640x1980 reconstruction-only grids (<env>_..._goal_p1.png
# etc.): no labels, so a uniform inset.
RECON_GRID = dict(col_x=[67, 390, 712, 1034, 1356, 1678, 2000, 2322],
                  row_y=[98, 412, 725, 1039, 1353, 1666], tile=292, crop=(6, 6, 6, 6))

# Per-(env, row) source overrides: (file, grid, source row). Reacher uses the
# autoregressive pair-1 grid (ep 5050): t=+0 and t=+25 rows, instead of pair 0.
OVERRIDES = {
    ("reacher", 0): ("reacher_k48to192_sepopt_dynrollout_ar_pair1.png", ROLLOUT_GRID, 0),
    ("reacher", 1): ("reacher_k48to192_sepopt_dynrollout_ar_pair1.png", ROLLOUT_GRID, 5),
}

ENVS = [
    ("tworoom", "TwoRoom"),
    ("pusht", "PushT"),
    ("ogb_cube", "OGBench-Cube"),
    ("reacher", "Reacher"),
]
ROWS = [(0, "Recon."), (5, "Pred.\n$t{+}25$")]
COL_LABELS = ["Ground\ntruth"] + [f"$d={k}$" for k in (48, 72, 96, 120, 144, 168, 192)]


def load(name: str) -> np.ndarray:
    return np.asarray(Image.open(HERE / name).convert("RGB"))


def tile(img: np.ndarray, grid: dict, row: int, col: int) -> np.ndarray:
    x, y, t = grid["col_x"][col], grid["row_y"][row], grid["tile"]
    top, bottom, left, right = grid["crop"]
    return img[y + top:y + t - bottom, x + left:x + t - right]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "decoder_levels.pdf"
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})

    n_rows, n_cols = len(ENVS) * len(ROWS), len(COL_LABELS)
    fig = plt.figure(figsize=(5.5, 5.75))
    # Extra vertical gap between environments, small gap within one.
    heights = []
    for i in range(len(ENVS)):
        heights += [1.0, 1.0] + ([0.12] if i < len(ENVS) - 1 else [])
    gs = fig.add_gridspec(len(heights), n_cols, height_ratios=heights,
                          left=0.115, right=0.995, top=0.955, bottom=0.005, wspace=0.04, hspace=0.04)

    r = 0
    for e, (env, env_label) in enumerate(ENVS):
        rollout = load(f"{env}_k48to192_sepopt_dynrollout_ar_pair0.png")
        for j, (src_row, row_label) in enumerate(ROWS):
            img, grid = rollout, ROLLOUT_GRID
            if (env, j) in OVERRIDES:
                name, grid, src_row = OVERRIDES[(env, j)]
                img = load(name)
            for c in range(n_cols):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(tile(img, grid, src_row, c), interpolation="lanczos")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4); s.set_color("0.6")
                if r == 0:
                    ax.set_title(COL_LABELS[c], fontsize=7, pad=2)
                if c == 0:
                    ax.set_ylabel(row_label, fontsize=6.5, labelpad=2)
                    if j == 0:
                        ax.text(-0.62, -0.02, env_label, transform=ax.transAxes, rotation=90,
                                ha="center", va="center", fontsize=7.5, fontweight="bold")
            r += 1
        r += 1  # spacer row
    fig.savefig(out, dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
