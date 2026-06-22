from __future__ import annotations

from typing import Any, Callable

from mwm.data.transforms import MWMTrainSampleTransform


def load_stable_wm_dataset_for_mwm(
    name: str,
    *,
    format: str | None = None,
    frameskip: int = 1,
    num_steps: int = 4,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    from stable_worldmodel.data import load_dataset

    return load_dataset(
        str(name),
        format=format,
        frameskip=int(frameskip),
        num_steps=int(num_steps),
        transform=transform or MWMTrainSampleTransform(),
        **kwargs,
    )


__all__ = ["load_stable_wm_dataset_for_mwm"]
