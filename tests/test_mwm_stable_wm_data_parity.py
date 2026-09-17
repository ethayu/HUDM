from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from mwm.data.transforms import ZScoreScaler
from mwm.training.stable_wm_data import stable_wm_training_data_source, upstream_split_with_generator
from mwm.training.stable_wm_config import stable_wm_model_init_seed
from mwm.training.stable_wm_lightning import stable_wm_fit_seed


def test_fit_seed_defaults_to_top_level_but_can_match_upstream_manager_seed() -> None:
    assert stable_wm_fit_seed(OmegaConf.create({"seed": 3072, "train": {}})) == 3072
    assert stable_wm_fit_seed(OmegaConf.create({"seed": 3072, "train": {"fit_seed": 0}})) == 0


def test_model_init_seed_is_independent_from_data_and_fit_seeds() -> None:
    assert stable_wm_model_init_seed(OmegaConf.create({"seed": 3072, "train": {}})) == 3072
    cfg = OmegaConf.create(
        {"seed": 3072, "train": {"model_init_seed": 441, "fit_seed": 0}}
    )
    assert stable_wm_model_init_seed(cfg) == 441
    assert stable_wm_fit_seed(cfg) == 0


def test_training_source_is_separate_from_runtime_data_contract() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "path": "runtime/reacher.lance",
                "format": "lance",
                "training_source": {
                    "path": "source/reacher.h5",
                    "format": "hdf5",
                },
            }
        }
    )
    assert stable_wm_training_data_source(cfg) == ("source/reacher.h5", "hdf5")
    del cfg.data.training_source
    assert stable_wm_training_data_source(cfg) == ("runtime/reacher.lance", "lance")


def test_upstream_split_generator_is_reused_for_train_shuffle() -> None:
    dataset = range(101)
    train_set, val_set, actual_generator = upstream_split_with_generator(
        dataset,
        split_ratio=0.9,
        seed=3072,
    )

    expected_generator = torch.Generator().manual_seed(3072)
    expected_permutation = torch.randperm(len(dataset), generator=expected_generator).tolist()
    expected_train_length = 91
    assert list(train_set.indices) == expected_permutation[:expected_train_length]
    assert list(val_set.indices) == expected_permutation[expected_train_length:]

    actual_shuffle = torch.randperm(len(train_set), generator=actual_generator)
    expected_shuffle = torch.randperm(len(train_set), generator=expected_generator)
    torch.testing.assert_close(actual_shuffle, expected_shuffle, rtol=0, atol=0)

    fresh_shuffle = torch.randperm(
        len(train_set),
        generator=torch.Generator().manual_seed(3072),
    )
    assert not torch.equal(actual_shuffle, fresh_shuffle)


def test_zscore_statistics_use_upstream_torch_reduction_exactly() -> None:
    base = np.linspace(-1.0, 1.0, 200_003, dtype=np.float32)
    values = np.stack((base, np.square(base)), axis=1)
    values[17] = np.nan
    scaler = ZScoreScaler().fit(values)

    reference = torch.from_numpy(values)
    reference = reference[~torch.isnan(reference).any(dim=1)]
    expected_mean = reference.mean(0, keepdim=True).numpy()
    expected_std = reference.std(0, keepdim=True).numpy()
    np.testing.assert_array_equal(scaler.mean, expected_mean)
    np.testing.assert_array_equal(scaler.std, expected_std)
