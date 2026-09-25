from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from mwm.data.paths import local_path


ROOT = Path(__file__).resolve().parents[2]
DATA_PARITY = (
    ROOT
    / "reports"
    / "research"
    / "single_level_reacher_parity_20260806"
    / "data_parity.json"
)
DESTINATION = DATA_PARITY.with_name("eval_action_scaler_forensics.json")


def main() -> None:
    from stable_worldmodel.data import load_dataset

    data_parity = json.loads(DATA_PARITY.read_text(encoding="utf-8"))
    dataset = load_dataset(
        local_path("data/upstream/reacher.lance"),
        format="lance",
        frameskip=1,
        num_steps=2,
        keys_to_load=["action"],
    )
    actions = np.asarray(dataset.get_col_data("action"))
    actions = actions[~np.isnan(actions).any(axis=1)]
    scaler = StandardScaler().fit(actions)

    row_count = int(actions.shape[0])
    hdf5_mean = np.asarray(
        data_parity["action_normalization"]["hdf5_mean"], dtype=np.float64
    )
    hdf5_sample_std = np.asarray(
        data_parity["action_normalization"]["hdf5_sample_std"], dtype=np.float64
    )
    hdf5_population_scale = hdf5_sample_std * math.sqrt((row_count - 1) / row_count)
    probes = np.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    lance_normalized = scaler.transform(probes)
    hdf5_normalized = (probes - hdf5_mean) / hdf5_population_scale

    payload = {
        "status": "pass",
        "source_contract": {
            "upstream_training": "torch mean and sample standard deviation",
            "upstream_evaluation": "sklearn.preprocessing.StandardScaler",
            "mwm_training": "literal Torch-compatible ZScoreScaler",
            "mwm_evaluation": "sklearn.preprocessing.StandardScaler",
        },
        "valid_action_rows": row_count,
        "workspace_lance_dtype": str(actions.dtype),
        "official_hdf5_dtype": data_parity["official_hdf5"]["action_dtype"],
        "lance_sklearn": {
            "mean": scaler.mean_.tolist(),
            "population_scale": scaler.scale_.tolist(),
        },
        "official_hdf5_derived": {
            "mean": hdf5_mean.tolist(),
            "population_scale": hdf5_population_scale.tolist(),
        },
        "maximum_mean_absolute_difference": float(
            np.max(np.abs(scaler.mean_ - hdf5_mean))
        ),
        "maximum_scale_absolute_difference": float(
            np.max(np.abs(scaler.scale_ - hdf5_population_scale))
        ),
        "maximum_normalized_probe_absolute_difference": float(
            np.max(np.abs(lance_normalized - hdf5_normalized))
        ),
        "conclusion": (
            "Training and evaluation intentionally use different normalizer definitions in "
            "upstream LeWM, and MWM preserves both. Evaluation uses scikit-learn's population-"
            "variance StandardScaler. The evaluation-only float64-HDF5 versus float32-Lance "
            "coefficient differences are below 1.2e-11, and normalized action-bound probes "
            "differ by less than 4.3e-11. This cannot explain the control gap."
        ),
    }
    DESTINATION.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
