from __future__ import annotations

from typing import Sequence

import numpy as np

from hudm.specs import PlanSpec


def resolve_dataset_path(plan: PlanSpec) -> str:
    init_goal = plan.task["init_goal"]
    zarr_path = init_goal["dataset"].get("zarr_path", None)
    if zarr_path is None:
        kind = plan.active_backend_kind()
        if kind == "wm":
            zarr_path = plan.backend["wm"]["world_model"].get("zarr_path", None)
        if zarr_path is None:
            raise ValueError("Dataset-backed tasks require task.init_goal.dataset.zarr_path.")
    return str(zarr_path)


def rollout_id(selection: dict) -> str:
    return (
        f"ep{int(selection['episode_index']):04d}_"
        f"s{int(selection['start_index']):05d}_"
        f"g{int(selection['goal_index']):05d}"
    )


def enumerate_rollout_candidates(plan: PlanSpec) -> list[dict]:
    init_goal = plan.task["init_goal"]
    if str(init_goal["source"]).lower() != "dataset":
        raise ValueError("Experiments require task.init_goal.source='dataset'.")
    try:
        import zarr
    except Exception as exc:
        raise ImportError("zarr must be installed to enumerate rollout candidates.") from exc

    zarr_path = resolve_dataset_path(plan)
    root = zarr.open_group(zarr_path, mode="r")
    state_arr = root["data"]["state"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.zeros_like(ends)
    starts[0] = 0
    for idx in range(1, len(ends)):
        starts[idx] = ends[idx - 1] + 1

    dataset_cfg = init_goal["dataset"]
    trajectory_len = int(dataset_cfg["trajectory_len"])
    split_ratio = 0.8 if dataset_cfg.get("split_ratio", None) is None else float(dataset_cfg["split_ratio"])
    split_l = str(dataset_cfg["split"]).lower()
    n_ep = len(ends)
    n_train = int(split_ratio * n_ep)
    episode_ids = np.arange(0, n_train) if split_l == "train" else np.arange(n_train, n_ep)

    pos_thresh = 10.0
    ang_thresh = float(np.pi / 9.0)
    candidates: list[dict] = []
    for episode_index in episode_ids:
        s = int(starts[episode_index])
        e = int(ends[episode_index])
        if e - s < trajectory_len:
            continue
        for start_idx in range(s, e - trajectory_len + 1):
            goal_idx = int(start_idx + trajectory_len)
            init_state = np.asarray(state_arr[start_idx], dtype=np.float32)
            goal_state = np.asarray(state_arr[goal_idx], dtype=np.float32)
            init_agent = bool(np.all(init_state[:2] >= 0.0) and np.all(init_state[:2] <= 512.0))
            goal_agent = bool(np.all(goal_state[:2] >= 0.0) and np.all(goal_state[:2] <= 512.0))
            if not (init_agent and goal_agent):
                continue
            pos_diff = float(np.linalg.norm(goal_state[2:4] - init_state[2:4]))
            ang_diff = float(np.abs(goal_state[4] - init_state[4]))
            ang_diff = float(np.minimum(ang_diff, 2.0 * np.pi - ang_diff))
            if pos_diff < pos_thresh and ang_diff < ang_thresh:
                continue
            candidates.append(
                {
                    "episode_index": int(episode_index),
                    "start_index": int(start_idx),
                    "goal_index": int(goal_idx),
                    "trajectory_len": int(trajectory_len),
                    "split": split_l,
                    "pos_diff": pos_diff,
                    "angle_diff": ang_diff,
                }
            )
    if len(candidates) <= 0:
        raise ValueError("No valid dataset rollout candidates were found.")
    return candidates


def select_rollouts(
    rollouts_cfg: dict,
    candidates: Sequence[dict],
) -> list[dict]:
    n = int(rollouts_cfg["num_rollouts"])
    without_replacement = bool(rollouts_cfg["sample_without_replacement"])
    if without_replacement and n > len(candidates):
        raise ValueError(
            f"experiment.rollouts.num_rollouts={n} exceeds available candidates={len(candidates)} "
            "with sample_without_replacement=true."
        )
    rng = np.random.default_rng(int(rollouts_cfg["seed"]))
    idxs = rng.choice(len(candidates), size=n, replace=not without_replacement)
    selected = [dict(candidates[int(idx)]) for idx in idxs]
    for order_idx, item in enumerate(selected):
        item["rollout_index"] = int(order_idx)
        item["rollout_id"] = rollout_id(item)
    return selected
