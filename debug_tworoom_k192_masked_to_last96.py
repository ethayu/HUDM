"""Mirror of debug_tworoom_k192_masked_to_k96.py: zero out dims [0:96] of the
individually-trained K=192 TwoRoom checkpoint's latent (keeping only
[96:192]), to test whether task-relevant information is concentrated in the
leading 96 dims specifically or spread more evenly across all 192.

Usage: python debug_tworoom_k192_masked_to_last96.py
"""
from __future__ import annotations

import torch

from mwm.models.lewm import LeWMMatryoshkaWorldModel

MASK_UP_TO_DIM = 96

_original_encode_pixels = LeWMMatryoshkaWorldModel._encode_pixels


def _masked_encode_pixels(self, pixels: torch.Tensor, *, already_preprocessed: bool = False) -> torch.Tensor:
    emb = _original_encode_pixels(self, pixels, already_preprocessed=already_preprocessed)
    emb = emb.clone()
    emb[..., :MASK_UP_TO_DIM] = 0.0
    return emb


def main() -> None:
    LeWMMatryoshkaWorldModel._encode_pixels = _masked_encode_pixels
    print(f"[debug] patched LeWMMatryoshkaWorldModel._encode_pixels to zero dims [:{MASK_UP_TO_DIM}]")

    from mwm.benchmark import matrix

    cfg_path = "configs/research/release20260728_tworoom_goal25_debug_k192_masked_to_last96.yaml"
    matrix.main(cfg_path, resume=False)


if __name__ == "__main__":
    main()
