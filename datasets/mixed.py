import copy
import torch
from torch.utils.data import ConcatDataset, Subset

from omegaconf import DictConfig
from .pusht_dset import load_pusht_dataset


def _subset_indices(n: int, k: int, seed: int, replace: bool) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if replace:
        return torch.randint(high=n, size=(k,), generator=g)
    else:
        k = min(k, n)
        return torch.randperm(n, generator=g)[:k]


def build_mixed_pusht_datasets(cfg: DictConfig):
    """
    Return (train_dataset, val_dataset) where train is a mixture of real and synthetic
    data according to cfg.data.synthetic, and val is chosen by cfg.data.synthetic.val_source.

    Expected cfg structure:
      data:
        path: <real_dataset_root>
        use_sincos: true|false
        with_velocity: true|false
        synthetic:
          enable: true|false
          path: <synthetic_dataset_root>
          frac: 0.3                # fraction of synthetic samples in train
          total_train: null        # if set, cap total mixed train size
          seed: 0
          val_source: real|synthetic|mixed
    """

    # Load real datasets
    real_train, real_val = load_pusht_dataset(cfg.data)

    s_cfg = getattr(cfg.data, "synthetic", None)
    if not s_cfg or not getattr(s_cfg, "enable", False):
        return real_train, real_val

    # Load synthetic datasets with identical data settings but different path
    data_cfg_syn = copy.deepcopy(cfg.data)
    data_cfg_syn.path = s_cfg.path
    synth_train, synth_val = load_pusht_dataset(data_cfg_syn)

    # Determine sizes
    total_target = getattr(s_cfg, "total_train", None)
    frac = float(getattr(s_cfg, "frac", 0.5))
    seed = int(getattr(s_cfg, "seed", 0))

    if total_target is None:
        total_target = len(real_train)
    total_target = int(total_target)
    n_synth = int(round(frac * total_target))
    n_real = max(0, total_target - n_synth)

    # Sample indices from each pool (with replacement if needed)
    idx_real = _subset_indices(len(real_train), n_real, seed=seed, replace=(n_real > len(real_train)))
    idx_synth = _subset_indices(len(synth_train), n_synth, seed=seed + 1, replace=(n_synth > len(synth_train)))

    train_mixed = ConcatDataset([
        Subset(real_train, idx_real),
        Subset(synth_train, idx_synth),
    ])

    # Validation source
    val_src = str(getattr(s_cfg, "val_source", "real")).lower()
    if val_src == "real":
        val_mixed = real_val
    elif val_src == "synthetic":
        val_mixed = synth_val
    else:  # mixed
        # Take balanced mix the same way as train but without capping size
        n_syn_v = min(len(synth_val), len(real_val))
        n_real_v = n_syn_v
        idx_rv = _subset_indices(len(real_val), n_real_v, seed=seed + 2, replace=False)
        idx_sv = _subset_indices(len(synth_val), n_syn_v, seed=seed + 3, replace=False)
        val_mixed = ConcatDataset([
            Subset(real_val, idx_rv),
            Subset(synth_val, idx_sv),
        ])

    return train_mixed, val_mixed

