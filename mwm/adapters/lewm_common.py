from __future__ import annotations

import torch
import torch.nn as nn


class ImageNetPreprocess(nn.Module):
    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ImageNetPreprocess expects BCHW images, got {tuple(x.shape)}")
        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.numel() and float(x.detach().max().item()) > 2.0:
            x = x / 255.0
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)


__all__ = ["ImageNetPreprocess"]
