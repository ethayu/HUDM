"""Debugging probe: run TwoRoom's individually-trained K=192 checkpoint with
dims [96:192] of its latent zeroed out immediately after encoding, so every
downstream consumer (transition, decoder, CEM cost) only ever sees
information in the first 96 dims -- while still receiving a correctly-shaped
192-dim tensor (the K=192 checkpoint has only one tail, built for 192-dim
input; there is no built-in way to request K=96 from a single-level
checkpoint, see the config's docstring for why).

This tests whether the K=96-vs-K=192 gap seen throughout the k_matched
comparison is about REPRESENTATION CONTENT (does a well-trained K=192 model
happen to pack useful low-frequency information into its leading 96 dims,
the way the jointly-trained dense checkpoint's K=96 slice does) or purely
about DEDICATED TRAINING AT K=96 (a checkpoint literally never optimized to
be useful when truncated this way).

Usage: python debug_tworoom_k192_masked_to_k96.py
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

    cfg_path = "configs/research/release20260728_tworoom_goal25_debug_k192_masked_to_k96.yaml"
    matrix.main(cfg_path, resume=False)


if __name__ == "__main__":
    main()
