from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _channel_first_image_shape(image_shape: Sequence[int], *, default_channels: int = 3) -> tuple[int, int, int]:
    values = tuple(int(x) for x in image_shape)
    if len(values) == 2:
        height, width = values
        return int(default_channels), int(height), int(width)
    if len(values) == 3:
        if values[0] in {1, 3, 4}:
            channels, height, width = values
            return int(channels), int(height), int(width)
        if values[-1] in {1, 3, 4}:
            height, width, channels = values
            return int(channels), int(height), int(width)
    raise ValueError(f"image_shape must be (H, W), (C, H, W), or (H, W, C), got {values}.")


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvImageDecoder(nn.Module):
    """Convolutional latent-prefix decoder for visualizing MWM representation levels."""

    def __init__(
        self,
        *,
        latent_dim: int,
        image_shape: Sequence[int],
        base_channels: int = 256,
        start_size: int = 7,
    ) -> None:
        super().__init__()
        if int(latent_dim) <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if int(base_channels) <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")
        if int(start_size) <= 0:
            raise ValueError(f"start_size must be positive, got {start_size}.")
        out_channels, height, width = _channel_first_image_shape(image_shape)
        if height <= 0 or width <= 0:
            raise ValueError(f"image height/width must be positive, got {(height, width)}.")

        self.latent_dim = int(latent_dim)
        self.image_shape = (int(out_channels), int(height), int(width))
        self.base_channels = int(base_channels)
        self.start_size = int(start_size)

        self.input = nn.Sequential(
            nn.Linear(self.latent_dim, self.base_channels * self.start_size * self.start_size),
            nn.GELU(),
        )

        blocks: list[nn.Module] = []
        channels = self.base_channels
        spatial = self.start_size
        target_spatial = max(height, width)
        while spatial < target_spatial:
            next_channels = max(32, channels // 2)
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, next_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_group_count(next_channels), next_channels),
                    nn.GELU(),
                    nn.Conv2d(next_channels, next_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_group_count(next_channels), next_channels),
                    nn.GELU(),
                )
            )
            channels = next_channels
            spatial *= 2
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {int(latent.shape[-1])}.")
        flat = latent.reshape(-1, self.latent_dim)
        x = self.input(flat)
        x = x.reshape(flat.shape[0], self.base_channels, self.start_size, self.start_size)
        for block in self.blocks:
            x = block(x)
        _, height, width = self.image_shape
        if tuple(x.shape[-2:]) != (height, width):
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        x = self.output(x)
        return x.reshape(*latent.shape[:-1], *self.image_shape)


__all__ = ["ConvImageDecoder"]
