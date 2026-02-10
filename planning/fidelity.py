from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _to_nchw(img: torch.Tensor) -> torch.Tensor:
    # Expect HWC float tensor
    if img.ndim != 3:
        raise ValueError(f"Expected HWC image tensor, got shape {tuple(img.shape)}")
    h, w, c = img.shape
    if c not in (1, 3):
        raise ValueError(f"Expected HWC with 1 or 3 channels, got C={c}")
    return img.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W


def _to_hwc(img_nchw: torch.Tensor) -> torch.Tensor:
    if img_nchw.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape {tuple(img_nchw.shape)}")
    return img_nchw.squeeze(0).permute(1, 2, 0)  # H,W,C


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    sigma = float(sigma)
    k = max(3, int(2 * math.ceil(3 * sigma) + 1))
    x = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    img_nchw = _to_nchw(img)
    c = img_nchw.shape[1]
    kernel_x = kernel.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    kernel_y = kernel.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    img_nchw = F.conv2d(img_nchw, kernel_x, padding=(0, k // 2), groups=c)
    img_nchw = F.conv2d(img_nchw, kernel_y, padding=(k // 2, 0), groups=c)
    return _to_hwc(img_nchw)


def avg_pool_block(img: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return img
    img_nchw = _to_nchw(img)
    h, w = img_nchw.shape[-2:]
    h2 = max(1, h // scale)
    w2 = max(1, w // scale)
    down = F.interpolate(img_nchw, size=(h2, w2), mode="area")
    up = F.interpolate(down, size=(h, w), mode="nearest")
    return _to_hwc(up)


def quantize(img: torch.Tensor, levels: int) -> torch.Tensor:
    levels = int(levels)
    if levels <= 1:
        return torch.zeros_like(img)
    return torch.round(img * (levels - 1)) / (levels - 1)


def linear_schedule(start: float, end: float, progress: float) -> float:
    p = min(1.0, max(0.0, float(progress)))
    return float(start) + (float(end) - float(start)) * p


def fidelity_level(
    task_progress: float,
    cem_progress: float,
    task_start: float,
    task_end: float,
    cem_start: float,
    cem_end: float,
) -> float:
    task_level = linear_schedule(task_start, task_end, task_progress)
    cem_level = linear_schedule(cem_start, cem_end, cem_progress)
    return 0.5 * (task_level + cem_level)


def apply_fidelity(
    img: torch.Tensor,
    level: float,
    mode: str,
    blur_sigma_max: float,
    pool_scale_max: int,
    quantize_levels_min: int,
    quantize_levels_max: int,
) -> torch.Tensor:
    level = min(1.0, max(0.0, float(level)))
    sigma = (1.0 - level) * float(blur_sigma_max)
    if mode == "blur_avgpool":
        scale = int(round(1 + (1.0 - level) * (pool_scale_max - 1)))
        img = gaussian_blur(img, sigma)
        img = avg_pool_block(img, scale)
        return img
    if mode == "blur_quantize":
        levels = int(round(quantize_levels_min + level * (quantize_levels_max - quantize_levels_min)))
        img = gaussian_blur(img, sigma)
        img = quantize(img, levels)
        return img
    raise ValueError(f"Unknown fidelity mode: {mode}")
