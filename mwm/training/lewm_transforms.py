from __future__ import annotations

from typing import Any

from mwm.data.transforms import column_normalizer
from mwm.preprocessing.images import stable_pretraining_image_transforms


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


__all__ = ["build_lewm_base_adapter_dataset_transform"]
