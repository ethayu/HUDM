"""Same masking probe as debug_tworoom_k192_masked_to_k96.py, but against the
DENSE multi-K checkpoint's K=192 level (level_idx=4) instead of the
individually-trained K=192 checkpoint. See the paired config's docstring for
the motivation: does the matryoshka-trained K=192 tail behave differently
when starved of its [96:192] dims than the individually-trained one did?

Usage: python debug_tworoom_densek192_masked_to_k96.py
"""
from __future__ import annotations

import torch

from mwm.models.lewm import LeWMMatryoshkaWorldModel

MASK_FROM_DIM = 96

_original_encode_pixels = LeWMMatryoshkaWorldModel._encode_pixels


def _masked_encode_pixels(self, pixels: torch.Tensor, *, already_preprocessed: bool = False) -> torch.Tensor:
    emb = _original_encode_pixels(self, pixels, already_preprocessed=already_preprocessed)
    emb = emb.clone()
    emb[..., MASK_FROM_DIM:] = 0.0
    return emb


def main() -> None:
    LeWMMatryoshkaWorldModel._encode_pixels = _masked_encode_pixels
    print(f"[debug] patched LeWMMatryoshkaWorldModel._encode_pixels to zero dims [{MASK_FROM_DIM}:]")

    from mwm.benchmark import matrix

    cfg_path = "configs/research/release20260728_tworoom_goal25_debug_densek192_masked_to_k96.yaml"
    matrix.main(cfg_path, resume=False)


if __name__ == "__main__":
    main()
