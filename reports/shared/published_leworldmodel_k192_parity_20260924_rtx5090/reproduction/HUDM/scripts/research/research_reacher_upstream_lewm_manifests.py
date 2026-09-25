#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lance
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mwm.data.manifest import generate_manifest, load_manifest, write_manifest
from mwm.data.sampling import StartGoalPair, sample_start_goal_pairs


DATASET = ROOT / "data" / "upstream" / "reacher.lance"
SOURCE_ROOT = ROOT / "rollouts" / "manifests" / "release20260728_dense_k192_eval_size"
OUTPUT_ROOT = ROOT / "rollouts" / "manifests" / "single_level_upstream_lewm_historical_20260806"
REPORT = ROOT / "reports" / "research" / "single_level_reacher_parity_20260806" / "upstream_eval_sampling_parity.json"
SEEDS = (0, 1, 2, 42, 100)
COUNT = 500
GOAL_OFFSET = 25
EFFECTIVE_GOAL_OFFSET = 24


def _dataset_geometry() -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    table = lance.dataset(str(DATASET)).scanner(columns=["episode_idx", "step_idx"]).to_table()
    episodes = np.asarray(table["episode_idx"]).reshape(-1).astype(np.int64)
    steps = np.asarray(table["step_idx"]).reshape(-1).astype(np.int64)
    unique, offsets, lengths = np.unique(episodes, return_index=True, return_counts=True)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise RuntimeError("Reacher episodes are not contiguous zero-based identifiers")
    if not np.all(steps[offsets] == 0):
        raise RuntimeError("Reacher episode offsets do not point to step zero")
    return SimpleNamespace(lengths=lengths, offsets=offsets), episodes, steps


def main() -> None:
    dataset, episodes, steps = _dataset_geometry()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        source_path = SOURCE_ROOT / f"reacher_seed{seed}_n{COUNT}.json"
        source = load_manifest(source_path)
        pairs = sample_start_goal_pairs(
            dataset,
            count=COUNT,
            goal_offset_steps=GOAL_OFFSET,
            seed=seed,
            mode="upstream_lewm",
        )
        pairs = [
            StartGoalPair(
                episode=pair.episode,
                start_step=pair.start_step,
                goal_step=pair.start_step + EFFECTIVE_GOAL_OFFSET,
                start_row=pair.start_row,
                goal_row=pair.start_row + EFFECTIVE_GOAL_OFFSET,
            )
            for pair in pairs
        ]
        manifest = generate_manifest(
            env_id=str(source["env_id"]),
            dataset_path=str(source["dataset_path"]),
            pairs=pairs,
            goal_offset=int(source["goal_offset"]),
            goal_indexing="upstream_lewm_end_exclusive",
            effective_goal_offset=EFFECTIVE_GOAL_OFFSET,
            eval_budget=int(source["eval_budget"]),
            seed=seed,
            restore_spec=str(source["restore_spec"]),
            dataset_metadata=dict(source["dataset_metadata"]),
            dependency_shas=dict(source["dependency_shas"]),
        )
        output_path = OUTPUT_ROOT / f"reacher_seed{seed}_n{COUNT}.json"
        write_manifest(output_path, manifest)
        reloaded = load_manifest(output_path)
        selected_rows = np.asarray([row["start_row"] for row in reloaded["pairs"]], dtype=np.int64)

        max_start_by_episode = np.asarray(dataset.lengths, dtype=np.int64) - GOAL_OFFSET - 1
        valid = np.nonzero(steps <= max_start_by_episode[episodes])[0]
        direct_rows = np.sort(
            valid[np.random.default_rng(seed).choice(len(valid) - 1, size=COUNT, replace=False)]
        )
        if not np.array_equal(selected_rows, direct_rows):
            raise RuntimeError(f"Generated seed {seed} manifest does not reproduce upstream eval.py")
        source_rows = np.asarray([row["start_row"] for row in source["pairs"]], dtype=np.int64)
        results.append(
            {
                "seed": seed,
                "pairs": COUNT,
                "exact_upstream_rows": True,
                "exact_upstream_goal_rows": all(
                    row["goal_row"] - row["start_row"] == EFFECTIVE_GOAL_OFFSET
                    for row in reloaded["pairs"]
                ),
                "positions_changed_from_corrected_stable_worldmodel_sampler": int(
                    np.count_nonzero(selected_rows != source_rows)
                ),
                "source_manifest": str(source_path.relative_to(ROOT)),
                "source_manifest_sha256": str(source["manifest_sha256"]),
                "output_manifest": str(output_path.relative_to(ROOT)),
                "output_manifest_sha256": str(reloaded["manifest_sha256"]),
            }
        )
    payload = {
        "status": "pass",
        "dataset": str(DATASET.relative_to(ROOT)),
        "rows": int(len(episodes)),
        "episodes": int(len(dataset.lengths)),
        "episode_length_min": int(np.min(dataset.lengths)),
        "episode_length_max": int(np.max(dataset.lengths)),
        "valid_start_rows": int(np.sum(np.asarray(dataset.lengths) - GOAL_OFFSET)),
        "upstream_semantics": {
            "sampling": "np.random.default_rng(seed).choice(len(valid_indices) - 1, replace=False)",
            "requested_goal_offset": GOAL_OFFSET,
            "effective_goal_offset": EFFECTIVE_GOAL_OFFSET,
            "cause": "evaluate_from_dataset passes end=start+offset to an end-exclusive HDF5 slice and selects slice[-1]",
        },
        "seeds": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
