from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the TwoRoom exact-indexed evaluation manifest contract.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {
        "env_id": "swm/TwoRoom-v1",
        "restore_spec": "point_state_goal_state",
        "seed": args.seed,
        "goal_offset": 25,
        "goal_indexing": "exact",
        "effective_goal_offset": 25,
        "eval_budget": 50,
        "dataset_path": "data/upstream/tworoom.lance",
    }
    for key, value in expected.items():
        fallback = "exact" if key == "goal_indexing" else 25 if key == "effective_goal_offset" else None
        observed = payload.get(key, fallback)
        if observed != value:
            raise RuntimeError(f"{args.manifest}: {key}={observed!r}, expected {value!r}")
    pairs = payload.get("pairs", [])
    if not isinstance(pairs, list) or len(pairs) != args.episodes:
        raise RuntimeError(f"{args.manifest}: expected {args.episodes} pairs, found {len(pairs)}")
    for index, pair in enumerate(pairs):
        step_delta = int(pair["goal_step"]) - int(pair["start_step"])
        row_delta = int(pair["goal_row"]) - int(pair["start_row"])
        if step_delta != 25 or row_delta != 25:
            raise RuntimeError(
                f"{args.manifest}: pair {index} has step/row deltas {step_delta}/{row_delta}, expected 25/25"
            )
    print(
        f"TWOROOM_MANIFEST_CONTRACT_PASSED path={args.manifest} "
        f"episodes={args.episodes} seed={args.seed} goal_indexing=exact"
    )


if __name__ == "__main__":
    main()
