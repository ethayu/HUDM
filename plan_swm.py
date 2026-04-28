from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.swm_hdf5 import SWMHDF5Episodes, SWMStartGoalPair, load_swm_dataset_metadata
from hudm.swm_artifacts import write_json
from hudm.swm_envs import make_swm_world, parse_env_kwargs, parse_image_shape, validate_continuous_box_action_space
from hudm.swm_policy import HUDMLatentCEMPolicy, SWMPlannerConfig
from hudm.swm_restore import eval_callables_for_env
from hudm.world_io import load_world_model_from_checkpoint


DEFAULTS = {
    "checkpoint": {"run_dir": "checkpoints_swm/run", "epoch": None},
    "data": {
        "path": "data/swm_dataset.h5",
        "split_ratio": 0.8,
        "pixels_key": "pixels",
        "action_key": "action",
    },
    "eval": {
        "episodes": 4,
        "goal_offset": 30,
        "seed": 0,
        "budget": 50,
        "num_envs": 4,
        "output_path": "rollouts/swm_eval.json",
        "save_video": False,
        "video_path": "rollouts/swm_eval_videos",
    },
    "env": {
        "max_episode_steps": 100,
        "goal_conditioned": True,
        "kwargs": {},
    },
    "restore": {"import_path": None},
    "planner": {
        "horizon": 16,
        "receding_horizon": 1,
        "pop_size": 256,
        "elite_frac": 0.1,
        "n_iter": 5,
        "init_std": 1.0,
        "warm_start": True,
        "fidelity": {
            "enabled": True,
            "mpc": {"mode": "fixed", "level": "finest"},
            "cem": {"mode": "fixed", "level": "base"},
            "rollout": {"mode": "fixed", "level": "base"},
        },
    },
    "device": "auto",
}


def _device(raw: str) -> torch.device:
    if str(raw) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(raw))


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


def _validate_dataset_metadata(dataset: SWMHDF5Episodes, checkpoint_metadata: dict[str, Any]) -> None:
    dataset_meta = load_swm_dataset_metadata(dataset.path, required=True)
    if str(checkpoint_metadata.get("format")) != "swm_hdf5":
        raise ValueError(f"Checkpoint format must be 'swm_hdf5', got {checkpoint_metadata.get('format')!r}.")
    model_meta = checkpoint_metadata.get("model")
    if not isinstance(model_meta, dict) or str(model_meta.get("input")) != "images":
        raise ValueError("Checkpoint model metadata must declare model.input='images'.")
    if str(dataset_meta.get("format")) != "swm_hdf5":
        raise ValueError(f"Dataset format must be 'swm_hdf5', got {dataset_meta.get('format')!r}.")
    checks = (
        ("env_id", str),
        ("restore_spec", str),
        ("action_dim", int),
    )
    for key, caster in checks:
        if key not in dataset_meta:
            raise ValueError(f"Dataset metadata {dataset.path} is missing required key {key!r}.")
        if caster(dataset_meta[key]) != caster(checkpoint_metadata[key]):
            raise ValueError(
                f"Dataset metadata {key}={dataset_meta[key]!r} does not match checkpoint {key}={checkpoint_metadata[key]!r}."
            )

    dataset_shape = tuple(int(x) for x in dataset_meta.get("image_shape", ()))
    checkpoint_shape = tuple(int(x) for x in checkpoint_metadata["image_shape"])
    if dataset_shape != checkpoint_shape:
        raise ValueError(
            f"Dataset image_shape={dataset_meta.get('image_shape')!r} does not match checkpoint image_shape={checkpoint_metadata['image_shape']!r}."
        )
    for key in ("action_low", "action_high"):
        if key not in dataset_meta:
            raise ValueError(f"Dataset metadata {dataset.path} is missing required key {key!r}.")
        ds_bound = np.asarray(dataset_meta[key], dtype=np.float32).reshape(-1)
        ckpt_bound = np.asarray(checkpoint_metadata[key], dtype=np.float32).reshape(-1)
        if ds_bound.shape != ckpt_bound.shape or not np.allclose(ds_bound, ckpt_bound):
            raise ValueError(
                f"Dataset {key}={ds_bound.tolist()} does not match checkpoint {key}={ckpt_bound.tolist()}."
            )
    ckpt_dataset = checkpoint_metadata.get("dataset")
    if not isinstance(ckpt_dataset, dict):
        raise ValueError("Checkpoint metadata is missing required dataset key mapping.")
    dataset_key_meta = dataset_meta.get("dataset", {})
    for meta_key, actual in (("pixels_key", dataset.pixels_key), ("action_key", dataset.action_key)):
        if meta_key not in ckpt_dataset:
            raise ValueError(f"Checkpoint metadata dataset mapping is missing {meta_key!r}.")
        if str(ckpt_dataset[meta_key]) != str(actual):
            raise ValueError(
                f"Checkpoint dataset {meta_key}={ckpt_dataset[meta_key]!r} does not match configured {actual!r}."
            )
        if isinstance(dataset_key_meta, dict) and meta_key in dataset_key_meta and str(dataset_key_meta[meta_key]) != str(actual):
            raise ValueError(
                f"Dataset metadata {meta_key}={dataset_key_meta[meta_key]!r} does not match configured {actual!r}."
            )


def _run_batch(
    *,
    env_id: str,
    image_shape: tuple[int, int],
    model,
    metadata: dict[str, Any],
    dataset: SWMHDF5Episodes,
    pairs: list[SWMStartGoalPair],
    cfg,
    device: torch.device,
    eval_callables: list[dict[str, Any]],
    batch_index: int,
) -> dict[str, Any]:
    env_kwargs = parse_env_kwargs(OmegaConf.to_container(cfg.env.get("kwargs", {}), resolve=True))
    world = make_swm_world(
        env_id,
        num_envs=len(pairs),
        image_shape=image_shape,
        max_episode_steps=int(cfg.env.max_episode_steps),
        goal_conditioned=bool(cfg.env.goal_conditioned),
        env_kwargs=env_kwargs,
    )
    try:
        low, high = validate_continuous_box_action_space(world.envs.single_action_space, env_id)
        if low.shape[0] != int(metadata["action_dim"]):
            raise ValueError(f"Env action_dim={low.shape[0]} does not match checkpoint action_dim={metadata['action_dim']}.")
        policy = HUDMLatentCEMPolicy(
            model,
            SWMPlannerConfig(
                horizon=int(cfg.planner.horizon),
                receding_horizon=int(cfg.planner.receding_horizon),
                pop_size=int(cfg.planner.pop_size),
                elite_frac=float(cfg.planner.elite_frac),
                n_iter=int(cfg.planner.n_iter),
                init_std=float(cfg.planner.init_std),
                warm_start=bool(cfg.planner.warm_start),
                fidelity=OmegaConf.to_container(cfg.planner.fidelity, resolve=True),
            ),
            action_low=np.asarray(metadata["action_low"], dtype=np.float32),
            action_high=np.asarray(metadata["action_high"], dtype=np.float32),
            device=device,
            seed=int(cfg.eval.seed),
        )
        world.set_policy(policy)
        policy.reset_trace()
        batch_video_path = Path(str(cfg.eval.video_path)) / f"batch_{int(batch_index):04d}"
        swm_results = world.evaluate_from_dataset(
            dataset=dataset,
            episodes_idx=[p.episode for p in pairs],
            start_steps=[p.start_step for p in pairs],
            goal_offset_steps=int(cfg.eval.goal_offset),
            eval_budget=int(cfg.eval.budget),
            callables=eval_callables,
            save_video=bool(cfg.eval.save_video),
            video_path=str(batch_video_path),
        )
        videos = sorted(str(p) for p in batch_video_path.glob("rollout_*.mp4")) if bool(cfg.eval.save_video) else []
        return {
            "pairs": [
                {
                    "episode": p.episode,
                    "start_step": p.start_step,
                    "goal_step": p.goal_step,
                    "start_row": p.start_row,
                    "goal_row": p.goal_row,
                }
                for p in pairs
            ],
            "swm_results": _jsonable(swm_results),
            "planning_diagnostics": policy.diagnostics(),
            "videos": videos,
        }
    finally:
        world.close()


def _combine_swm_results(batches: list[dict[str, Any]]) -> dict[str, Any]:
    successes: list[bool] = []
    seeds: list[Any] = []
    for batch in batches:
        results = batch.get("swm_results", {})
        successes.extend(bool(x) for x in results.get("episode_successes", []))
        batch_seeds = results.get("seeds")
        if batch_seeds is not None:
            seeds.extend(batch_seeds if isinstance(batch_seeds, list) else [batch_seeds])
    return {
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "episode_successes": successes,
        "seeds": seeds or None,
    }


def _combine_hudm_diagnostics(batches: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [batch["planning_diagnostics"]["summary"] for batch in batches]
    total_replans = int(sum(s["replans"] for s in summaries))
    total_actions = int(sum(s["actions_recorded"] for s in summaries))
    total_time = float(sum(s["total_plan_time_sec"] for s in summaries))
    total_bits = int(sum(s["total_bits_used_estimate"] for s in summaries))
    return {
        "summary": {
            "actions_recorded": total_actions,
            "replans": total_replans,
            "total_plan_time_sec": total_time,
            "mean_plan_time_sec": total_time / total_replans if total_replans else 0.0,
            "total_bits_used_estimate": total_bits,
        },
        "plans": total_replans,
        "steps": total_actions,
        "bits_used_total": total_bits,
        "plan_time_total_sec": total_time,
    }


def main(cfg_path: str) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    device = _device(str(cfg.device))
    model, metadata, epoch = load_world_model_from_checkpoint(
        str(cfg.checkpoint.run_dir),
        None if cfg.checkpoint.epoch is None else int(cfg.checkpoint.epoch),
        device=device,
    )
    env_id = str(metadata["env_id"])
    image_shape = parse_image_shape(metadata["image_shape"])
    dataset = SWMHDF5Episodes(
        cfg.data.path,
        horizon=2,
        split="valid",
        split_ratio=float(cfg.data.split_ratio),
        seed=int(cfg.eval.seed),
        pixels_key=str(cfg.data.pixels_key),
        action_key=str(cfg.data.action_key),
    )
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec_id, eval_callables = eval_callables_for_env(
        env_id,
        dataset.column_names,
        import_path=restore_import_path,
    )
    _validate_dataset_metadata(dataset, metadata)
    if str(metadata["restore_spec"]) != restore_spec_id:
        raise ValueError(
            f"Runtime restore spec {restore_spec_id!r} does not match checkpoint restore_spec={metadata['restore_spec']!r}."
        )

    pairs = dataset.sample_eval_start_goal_pairs(
        count=int(cfg.eval.episodes),
        goal_offset_steps=int(cfg.eval.goal_offset),
        seed=int(cfg.eval.seed),
    )
    all_results = []
    requested_batch_size = max(1, int(cfg.eval.num_envs))
    batch_size = requested_batch_size
    for batch_index, offset in enumerate(range(0, len(pairs), batch_size)):
        all_results.append(
            _run_batch(
                env_id=env_id,
                image_shape=image_shape,
                model=model,
                metadata=metadata,
                dataset=dataset,
                pairs=pairs[offset : offset + batch_size],
                cfg=cfg,
                device=device,
                eval_callables=eval_callables,
                batch_index=batch_index,
            )
        )
    videos = [video for batch in all_results for video in batch.get("videos", [])]

    output = {
        "env_id": env_id,
        "checkpoint_run_dir": str(cfg.checkpoint.run_dir),
        "checkpoint_epoch": int(epoch),
        "dataset": str(dataset.path),
        "episodes": int(cfg.eval.episodes),
        "goal_offset": int(cfg.eval.goal_offset),
        "eval_budget": int(cfg.eval.budget),
        "restore_spec": restore_spec_id,
        "swm_results": _combine_swm_results(all_results),
        "planning_diagnostics": _combine_hudm_diagnostics(all_results),
        "batches": all_results,
        "videos": videos,
    }
    output_path = Path(str(cfg.eval.output_path))
    write_json(output_path, output)
    print(f"Wrote SWM planning results to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python plan_swm.py configs/plan_swm.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
