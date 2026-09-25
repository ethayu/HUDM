#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import lance
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "upstream" / "reacher.lance"
OUTPUT = ROOT / "reports" / "research" / "single_level_reacher_parity_20260806" / "action_policy_forensics.json"
QUANTILE_PROBABILITIES = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99], dtype=np.float64)


def _fixed_size_values(column: object, rows: int) -> np.ndarray:
    combined = column.combine_chunks()
    return np.asarray(combined.values).reshape(rows, -1).astype(np.float64)


def main() -> None:
    dataset = lance.dataset(str(DATASET))
    table = dataset.scanner(columns=["action", "step_idx", "qpos", "qvel"]).to_table()
    actions = _fixed_size_values(table["action"], len(table))
    valid = ~np.isnan(actions).any(axis=1)
    actions = actions[valid]
    steps = np.asarray(table["step_idx"]).reshape(-1)[valid]
    qpos = _fixed_size_values(table["qpos"], len(table))[valid]
    qvel = _fixed_size_values(table["qvel"], len(table))[valid]

    same_episode_neighbor = steps[1:] > steps[:-1]
    quantiles = np.quantile(actions, QUANTILE_PROBABILITIES, axis=0)
    uniform_quantiles = 2.0 * QUANTILE_PROBABILITIES[:, None] - 1.0
    uniform_std = 1.0 / math.sqrt(3.0)
    action_state = np.concatenate([actions, qpos, qvel], axis=1)
    correlations = np.corrcoef(action_state.T)
    lag1 = [
        float(np.corrcoef(actions[:-1, dim][same_episode_neighbor], actions[1:, dim][same_episode_neighbor])[0, 1])
        for dim in range(actions.shape[1])
    ]
    diagnostic_max_correlation = max(
        abs(float(np.corrcoef(actions.T)[0, 1])),
        *(abs(value) for value in lag1),
        float(np.abs(correlations[:2, 2:]).max()),
    )
    gates = {
        "absolute_mean_below_0p002": bool(np.abs(actions.mean(axis=0)).max() < 0.002),
        "sample_std_within_0p002_of_uniform": bool(
            np.abs(actions.std(axis=0, ddof=1) - uniform_std).max() < 0.002
        ),
        "quantiles_within_0p002_of_uniform": bool(np.abs(quantiles - uniform_quantiles).max() < 0.002),
        "diagnostic_correlations_below_0p002": bool(diagnostic_max_correlation < 0.002),
    }
    payload = {
        "status": "pass" if all(gates.values()) else "mismatch",
        "source": {
            "lance_path": str(DATASET.relative_to(ROOT)),
            "official_archive_sha256": "4ff2385e49712caa89f21b8e0a246e2614b621d3f22cf2d1224d845e879a1cc2",
            "official_hdf5_sha256": "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06",
            "action_equivalence_evidence": "data_parity.json proves Lance actions equal official HDF5 after float32 cast",
        },
        "valid_action_rows": int(actions.shape[0]),
        "action_dimensions": int(actions.shape[1]),
        "observed": {
            "mean": actions.mean(axis=0).tolist(),
            "sample_std": actions.std(axis=0, ddof=1).tolist(),
            "minimum": actions.min(axis=0).tolist(),
            "maximum": actions.max(axis=0).tolist(),
            "fraction_abs_greater_than_0p99": (np.abs(actions) > 0.99).mean(axis=0).tolist(),
            "quantile_probabilities": QUANTILE_PROBABILITIES.tolist(),
            "quantiles": quantiles.tolist(),
            "action_cross_correlation": float(np.corrcoef(actions.T)[0, 1]),
            "within_episode_lag1_correlation": lag1,
            "action_qpos_correlations": correlations[:2, 2 : 2 + qpos.shape[1]].tolist(),
            "action_qvel_correlations": correlations[:2, 2 + qpos.shape[1] :].tolist(),
            "maximum_absolute_diagnostic_correlation": diagnostic_max_correlation,
        },
        "iid_uniform_reference": {
            "distribution": "Uniform(-1, 1)",
            "mean": 0.0,
            "standard_deviation": uniform_std,
            "quantiles": uniform_quantiles.tolist(),
            "fraction_abs_greater_than_0p99": 0.01,
        },
        "gates": gates,
        "conclusion": {
            "action_policy": "released action stream matches an i.i.d. Uniform(-1, 1) random policy",
            "paper_conflict": "Appendix E describes SAC collection, which is inconsistent with the released action stream",
            "legacy_name_inference": "strongly supports that released reacher.h5 is the legacy dmc/reacher_random data lineage",
            "legacy_byte_identity": "not proven because the legacy dmc/reacher_random.h5 artifact is not publicly resolvable",
            "parity_impact": "canonical MWM uses the official released bytes, so this provenance conflict does not alter paired training parity",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit("Released Reacher action stream did not pass the i.i.d. uniform forensic gates")


if __name__ == "__main__":
    main()
