from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


MWM_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetPreprocess(nn.Module):
    def __init__(
        self,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
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


def mwm_image_input_transform(image: torch.Tensor) -> torch.Tensor:
    """Stable-WM policy transform for MWM image inputs.

    The model owns Le-WM/ImageNet normalization in its preprocess module; this
    policy transform only standardizes layout, range, and image size.
    """

    tensor = torch.as_tensor(image)
    if tensor.ndim == 3 and tensor.shape[0] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
    else:
        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() and torch.isfinite(tensor).all().item() and tensor.max().item() > 2.0:
            tensor = tensor / 255.0
    if tuple(tensor.shape[-2:]) != MWM_IMAGE_SIZE:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=MWM_IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    return tensor


def imagenet_image_input_transform(image: torch.Tensor) -> torch.Tensor:
    """Policy transform for imported Stable-WM models trained with ImageNet normalization."""

    tensor = mwm_image_input_transform(image)
    if tensor.numel() and torch.isfinite(tensor).all().item() and tensor.min().item() < -0.5:
        return tensor
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def image_tensor_to_bchw(image: torch.Tensor) -> torch.Tensor:
    """Return an image tensor with channel dimension before height/width."""

    if image.ndim < 3:
        raise ValueError(f"Image tensors must have at least 3 dimensions, got {tuple(image.shape)}")
    if image.shape[-3] in {1, 3, 4}:
        return image
    if image.shape[-1] in {1, 3, 4}:
        return image.movedim(-1, -3)
    return image


def maybe_apply_image_preprocess(
    pixels: torch.Tensor,
    preprocess: nn.Module | None,
    *,
    already_preprocessed: bool = False,
) -> torch.Tensor:
    x = pixels.float()
    if already_preprocessed or preprocess is None:
        return x
    if x.numel() and torch.isfinite(x).all().item() and float(x.detach().min().cpu().item()) < -0.5:
        return x
    return preprocess(x)


def stable_pretraining_image_transforms(*, pixels_key: str, image_size: int) -> list[Any]:
    from stable_pretraining import data as dt

    imagenet_stats = dt.dataset_stats.ImageNet
    return [
        dt.transforms.ToImage(**imagenet_stats, source=str(pixels_key), target=str(pixels_key)),
        dt.transforms.Resize(int(image_size), source=str(pixels_key), target=str(pixels_key)),
    ]


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MWM_IMAGE_SIZE",
    "ImageNetPreprocess",
    "image_tensor_to_bchw",
    "imagenet_image_input_transform",
    "maybe_apply_image_preprocess",
    "mwm_image_input_transform",
    "stable_pretraining_image_transforms",
]
