import copy
import torch
from torch.utils.data import ConcatDataset, Subset
from omegaconf import DictConfig
from .zarr_episodes import ZarrPushTEpisodes


def _assert_zarr_schema(zarr_path: str) -> None:
    try:
        import zarr
    except Exception as exc:
        raise ImportError("zarr not installed. pip install zarr") from exc
    root = zarr.open_group(zarr_path, mode="r")
    data = root["data"]
    required = {"img", "action", "state"}
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{zarr_path} missing required arrays: {missing}")


def _subset_indices(n: int, k: int, seed: int, replace: bool) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if replace:
        return torch.randint(high=n, size=(k,), generator=g)
    else:
        k = min(k, n)
        return torch.randperm(n, generator=g)[:k]


def build_mixed_zarr_episodes(cfg: DictConfig):
    data = cfg.data
    _assert_zarr_schema(data.zarr_path)
    action_mode = str(getattr(data, "action_mode", "relative"))
    # Real datasets
    real_train = ZarrPushTEpisodes(
        data.zarr_path,
        split='train',
        split_ratio=data.split_ratio,
        action_mode=action_mode,
    )
    real_val = ZarrPushTEpisodes(
        data.zarr_path,
        split='valid',
        split_ratio=data.split_ratio,
        action_mode=action_mode,
    )

    s_cfg = getattr(data, 'synthetic', None)
    if not s_cfg or not getattr(s_cfg, 'enable', False):
        return real_train, real_val

    _assert_zarr_schema(s_cfg.zarr_path)
    # Synthetic datasets
    synth_train = ZarrPushTEpisodes(
        s_cfg.zarr_path,
        split='train',
        split_ratio=data.split_ratio,
        action_mode=action_mode,
    )
    synth_val = ZarrPushTEpisodes(
        s_cfg.zarr_path,
        split='valid',
        split_ratio=data.split_ratio,
        action_mode=action_mode,
    )

    total_target = getattr(s_cfg, 'total_train', None)
    frac = float(getattr(s_cfg, 'frac', 0.5))
    seed = int(getattr(s_cfg, 'seed', 0))

    if total_target is None:
        total_target = len(real_train)
    total_target = int(total_target)
    n_synth = int(round(frac * total_target))
    n_real = max(0, total_target - n_synth)

    idx_real = _subset_indices(len(real_train), n_real, seed=seed, replace=(n_real > len(real_train)))
    idx_synth = _subset_indices(len(synth_train), n_synth, seed=seed + 1, replace=(n_synth > len(synth_train)))

    mixed_train = ConcatDataset([
        Subset(real_train, idx_real),
        Subset(synth_train, idx_synth),
    ])

    val_src = str(getattr(s_cfg, 'val_source', 'real')).lower()
    if val_src == 'real':
        mixed_val = real_val
    elif val_src == 'synthetic':
        mixed_val = synth_val
    else:
        n = min(len(real_val), len(synth_val))
        idx_r = _subset_indices(len(real_val), n, seed=seed + 2, replace=False)
        idx_s = _subset_indices(len(synth_val), n, seed=seed + 3, replace=False)
        mixed_val = ConcatDataset([Subset(real_val, idx_r), Subset(synth_val, idx_s)])

    return mixed_train, mixed_val
