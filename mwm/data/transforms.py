from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mwm.preprocessing.images import stable_pretraining_image_transforms


class MWMTrainSampleTransform:
    """Transform Stable-WM samples into MWM ``x`` / ``a`` training batches."""

    def __init__(self, pixels_key: str = "pixels", action_key: str = "action", normalize_pixels: bool = True) -> None:
        self.pixels_key = str(pixels_key)
        self.action_key = str(action_key)
        self.normalize_pixels = bool(normalize_pixels)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.pixels_key not in sample:
            raise KeyError(f"Stable-WM sample is missing pixels key {self.pixels_key!r}")
        if self.action_key not in sample:
            raise KeyError(f"Stable-WM sample is missing action key {self.action_key!r}")
        x = torch.as_tensor(sample[self.pixels_key])
        if x.ndim != 4:
            raise ValueError(f"pixels must be a 4D trajectory, got {tuple(x.shape)}")
        if x.shape[-1] in {1, 3, 4}:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if self.normalize_pixels and x.numel() and float(x.max().item()) > 1.5:
            x = x / 255.0
        a = self._action_sequence(torch.as_tensor(sample[self.action_key]).float(), num_frames=int(x.shape[0]))
        out = dict(sample)
        out["x"] = x
        out["a"] = a
        out["mask"] = torch.ones(x.shape[0], dtype=torch.bool)
        return out

    @staticmethod
    def _action_sequence(action: torch.Tensor, *, num_frames: int) -> torch.Tensor:
        if action.ndim == 0:
            raise ValueError("action must have at least one dimension.")
        action = action.reshape(int(action.shape[0]), -1)
        target_actions = max(0, int(num_frames) - 1)
        if target_actions == 0:
            return action[:0]
        if int(action.shape[0]) == target_actions:
            return action
        if int(action.shape[0]) == int(num_frames):
            return action[:target_actions]
        if int(action.shape[0]) >= int(num_frames) and int(action.shape[0]) % int(num_frames) == 0:
            return action.reshape(int(num_frames), -1)[:target_actions]
        if int(action.shape[0]) >= target_actions:
            return action[:target_actions]
        raise ValueError(f"Not enough action rows for {num_frames} frames: got action shape {tuple(action.shape)}.")


class ZScoreScaler:
    def __init__(self, eps: float = 1e-8) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.eps = float(eps)

    def fit(self, values: Any) -> "ZScoreScaler":
        arr = np.asarray(values).reshape(-1, np.asarray(values).shape[-1])
        arr = arr[~np.isnan(arr).any(axis=1)]
        self.mean = arr.mean(axis=0, keepdims=True)
        self.std = arr.std(axis=0, keepdims=True, ddof=1)
        return self

    def __call__(self, values: Any) -> Any:
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreScaler must be fitted before use.")
        if torch.is_tensor(values):
            mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
            std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
            return ((values - mean) / std.clamp(min=self.eps)).float()
        return (values - self.mean) / np.maximum(self.std, self.eps)


def column_normalizer(dataset: Any, source: str, target: str) -> Any:
    from stable_pretraining.data.transforms import WrapTorchTransform

    scaler = ZScoreScaler().fit(np.asarray(dataset.get_col_data(source)))
    return WrapTorchTransform(scaler, source=source, target=target)


def build_lewm_base_adapter_dataset_transform(
    dataset: Any,
    *,
    pixels_key: str,
    image_size: int,
    keys_to_load: list[Any],
) -> Any:
    from stable_pretraining import data as dt

    transforms = stable_pretraining_image_transforms(pixels_key=str(pixels_key), image_size=int(image_size))
    for col in keys_to_load:
        if str(col).startswith("pixels"):
            continue
        if col in getattr(dataset, "column_names", []):
            transforms.append(column_normalizer(dataset, str(col), str(col)))
    return dt.transforms.Compose(*transforms)


__all__ = [
    "MWMTrainSampleTransform",
    "ZScoreScaler",
    "build_lewm_base_adapter_dataset_transform",
    "column_normalizer",
]
