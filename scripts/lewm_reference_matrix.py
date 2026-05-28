from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mwm.eval.reference import SampleExpandedGoalCostModel


CALLABLES = [
    {"method": "_set_state", "args": {"state": {"value": "state"}}},
    {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    return value


def _img_transform() -> Any:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def _load_pairs(manifest_eval_json: Path) -> tuple[list[int], list[int], list[dict[str, int]]]:
    payload = json.loads(manifest_eval_json.read_text(encoding="utf-8"))
    pairs = payload["batches"][0]["pairs"]
    return [int(p["episode"]) for p in pairs], [int(p["start_step"]) for p in pairs], pairs


def _build_process(dataset: Any, keys: tuple[str, ...] = ("action", "proprio", "state")) -> dict[str, Any]:
    process: dict[str, Any] = {}
    available = {str(x) for x in dataset.column_names}
    for key in keys:
        if key not in available:
            continue
        values = np.asarray(dataset.get_col_data(key))
        values = values.reshape(values.shape[0], -1)
        if np.issubdtype(values.dtype, np.number):
            values = values[~np.isnan(values).any(axis=1)]
        scaler = preprocessing.StandardScaler()
        scaler.fit(values)
        process[key] = scaler
        if key != "action":
            process[f"goal_{key}"] = scaler
    return process


def _load_hdf5_dataset(path: Path) -> Any:
    return swm.data.HDF5Dataset(
        path=path,
        keys_to_cache=["action", "proprio", "state"],
    )


def _load_lance_dataset(path: Path) -> Any:
    return swm.data.load_dataset(
        str(path),
        format="lance",
        frameskip=1,
        # Stable-WM dataset eval calls load_chunk(start, start + goal_offset + 1)
        # and then keeps only the first and last frames. LanceDataset's
        # _load_slice uses its configured span, so configure the same window.
        num_steps=26,
        keys_to_load=["pixels", "action", "proprio", "state"],
    )


def _evaluate_variant(
    *,
    dataset_name: str,
    dataset: Any,
    model: Any,
    episodes: list[int],
    start_steps: list[int],
    wrapped_goal_emb: bool,
) -> dict[str, Any]:
    cost_model = SampleExpandedGoalCostModel(model) if wrapped_goal_emb else model
    solver = swm.solver.CEMSolver(
        model=cost_model,
        batch_size=1,
        num_samples=300,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device="cuda",
        seed=42,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process=_build_process(dataset),
        transform={"pixels": _img_transform(), "goal": _img_transform()},
    )
    world = swm.World(env_name="swm/PushT-v1", num_envs=len(episodes), max_episode_steps=100, image_shape=(224, 224))
    try:
        world.set_policy(policy)
        start = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=episodes,
            start_steps=start_steps,
            goal_offset=25,
            eval_budget=50,
            callables=CALLABLES,
            video=None,
        )
        elapsed = time.time() - start
    finally:
        world.close()
    successes = [bool(x) for x in metrics["episode_successes"]]
    return {
        "dataset": dataset_name,
        "wrapped_goal_emb": bool(wrapped_goal_emb),
        "success_rate": float(metrics["success_rate"]),
        "failed_indices": [idx for idx, ok in enumerate(successes) if not ok],
        "episode_successes": successes,
        "evaluation_time_sec": float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw-HDF5 vs Lance Stable-WM Le-WM reference diagnostics.")
    parser.add_argument("--manifest-eval-json", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--lance", type=Path, required=True)
    parser.add_argument("--policy", type=str, default="local-lewm-pusht")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes, start_steps, pairs = _load_pairs(args.manifest_eval_json)
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)

    datasets = {
        "hdf5": _load_hdf5_dataset(args.hdf5),
        "lance": _load_lance_dataset(args.lance),
    }
    results: list[dict[str, Any]] = []
    for dataset_name, dataset in datasets.items():
        for wrapped in (False, True):
            results.append(
                _evaluate_variant(
                    dataset_name=dataset_name,
                    dataset=dataset,
                    model=model,
                    episodes=episodes,
                    start_steps=start_steps,
                    wrapped_goal_emb=wrapped,
                )
            )
    payload = {
        "policy": args.policy,
        "manifest_eval_json": str(args.manifest_eval_json),
        "hdf5": str(args.hdf5),
        "lance": str(args.lance),
        "pairs": pairs,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable({"output": str(args.output), "results": results}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
