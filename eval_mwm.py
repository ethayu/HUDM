from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from stable_worldmodel.policy import PlanConfig

from mwm.dependency_refs import dependency_refs
from mwm.benchmark.artifacts import write_json
from mwm.checkpoints import CHECKPOINT_FORMAT, load_world_model_from_checkpoint
from mwm.data.stable_wm import (
    StartGoalPair,
    load_dataset_metadata,
    sample_start_goal_pairs,
)
from mwm.data.manifest import (
    generate_manifest,
    load_manifest,
    manifest_file_sha256,
    manifest_sha256,
    write_manifest,
)
from mwm.planning.scheduled_cem import MWMScheduledCEMSolver
from mwm.eval.policy import (
    MWMWorldModelPolicy,
    imagenet_image_input_transform,
    model_accounting,
    mwm_image_input_transform,
)
from mwm.eval.reference import build_stable_wm_reference_policy
from mwm.swm.envs import make_swm_world, parse_env_kwargs, parse_image_shape, validate_continuous_box_action_space
from mwm.swm.restore import eval_callables_for_env


DEFAULTS = {
    "checkpoint": {"run_dir": "checkpoints_mwm/run", "epoch": None},
    "data": {
        "path": "data/swm_dataset.lance",
        "format": "lance",
        "split_ratio": 0.8,
        "pixels_key": "pixels",
        "action_key": "action",
        "keys_to_cache": ["action", "proprio", "state"],
        "action_preprocessing": "auto",
    },
    "eval": {
        "episodes": 4,
        "goal_offset": 30,
        "seed": 0,
        "budget": 50,
        "num_envs": 4,
        "output_path": "rollouts/mwm_eval.json",
        "manifest_path": None,
        "write_manifest_path": None,
        "sampling": "mwm",
        "reference_policy": False,
        "save_video": False,
        "video_path": "rollouts/mwm_eval_videos",
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
        "action_block": 1,
        "batch_size": "auto",
        "pop_size": 256,
        "topk": None,
        "elite_frac": 0.1,
        "n_iter": 5,
        "init_std": 1.0,
        "seed": None,
        "warm_start": True,
        "clamp_actions": False,
        "std_unbiased": True,
        "scheduler": {"policy": "fixed", "level": "finest", "rollout_level": "base"},
    },
    "device": "auto",
}


def _device(raw: str) -> torch.device:
    if str(raw) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(raw))


def _local_path(path: str | Path) -> str:
    p = Path(str(path))
    return str(p.resolve()) if p.exists() else str(path)


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


def _metadata_model_target(metadata: dict[str, Any]) -> str:
    model_meta = metadata.get("model", {})
    return str(model_meta.get("target", "")) if isinstance(model_meta, dict) else ""


def _uses_standardized_action_space(model: Any, metadata: dict[str, Any], cfg: Any) -> bool:
    raw = cfg.eval.get("action_preprocessing", cfg.data.get("action_preprocessing", "auto"))
    if isinstance(raw, bool):
        return raw
    mode = str(raw).lower()
    if mode in {"none", "identity", "raw", "false", "0"}:
        return False
    if mode in {"standard_scaler", "standard-scaler", "zscore", "normalized", "true", "1"}:
        return True
    if mode != "auto":
        raise ValueError(f"Unknown action_preprocessing mode {raw!r}.")

    preprocessing = metadata.get("action_preprocessing", metadata.get("action_normalization", None))
    if str(preprocessing).lower() in {"standard_scaler", "standard-scaler", "zscore", "normalized"}:
        return True
    dataset_meta = metadata.get("dataset", {})
    normalized = dataset_meta.get("normalized_columns", []) if isinstance(dataset_meta, dict) else []
    if "action" in {str(x) for x in normalized}:
        return True
    if hasattr(model, "source_model"):
        return True
    if str(metadata.get("role", "")) == "upstream_lewm_converted":
        return True
    return _metadata_model_target(metadata).endswith("build_mwm_lewm_from_object")


def _stat_keys_for_action_process(cfg: Any) -> list[str]:
    raw = cfg.data.get("keys_to_cache", cfg.data.get("process_keys", ["action", "proprio", "state"]))
    keys = [str(k) for k in (list(raw) if raw is not None else [])]
    action_key = str(cfg.data.get("action_key", "action"))
    if action_key not in keys:
        keys.insert(0, action_key)
    return [k for k in keys if k != str(cfg.data.get("pixels_key", "pixels"))]


def _fit_standard_scaler(dataset: Any, key: str) -> Any:
    from sklearn import preprocessing

    values = np.asarray(dataset.get_col_data(key))
    values = values.reshape(values.shape[0], -1)
    if np.issubdtype(values.dtype, np.number):
        values = values[~np.isnan(values).any(axis=1)]
    if values.size == 0:
        raise ValueError(f"Cannot fit action preprocessing scaler for empty dataset column {key!r}.")
    scaler = preprocessing.StandardScaler()
    scaler.fit(values)
    return scaler


def _build_eval_process(dataset: Any, model: Any, metadata: dict[str, Any], cfg: Any) -> dict[str, Any]:
    if not _uses_standardized_action_space(model, metadata, cfg):
        return {}
    columns = {str(c) for c in getattr(dataset, "column_names", [])}
    action_key = str(cfg.data.get("action_key", "action"))
    if action_key not in columns:
        raise ValueError(
            f"Checkpoint expects standardized actions, but dataset column {action_key!r} is not loaded."
        )
    process: dict[str, Any] = {}
    for key in _stat_keys_for_action_process(cfg):
        if key not in columns:
            continue
        scaler = _fit_standard_scaler(dataset, key)
        process[key] = scaler
        if key != action_key:
            process[f"goal_{key}"] = scaler
    return process


def _close_dataset(dataset: Any) -> None:
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _eval_keys_to_load(cfg: Any, model: Any, metadata: dict[str, Any]) -> list[str] | None:
    raw = list(cfg.data.get("keys_to_load", []))
    if not raw:
        return None
    keys = [str(k) for k in raw]
    if _uses_standardized_action_space(model, metadata, cfg):
        for key in _stat_keys_for_action_process(cfg):
            if key not in keys:
                keys.append(key)
    return keys


def _dataset_path(dataset: Any, cfg: Any) -> str:
    return str(cfg.data.path or getattr(dataset, "path", getattr(dataset, "uri", "")))


def _validate_dataset_metadata(dataset: Any, checkpoint_metadata: dict[str, Any], cfg: Any) -> None:
    dataset_path = _dataset_path(dataset, cfg)
    dataset_meta = load_dataset_metadata(dataset_path, required=False)
    if not dataset_meta:
        return
    ckpt_format = str(checkpoint_metadata.get("format"))
    if ckpt_format != CHECKPOINT_FORMAT:
        raise ValueError(f"Checkpoint format must be {CHECKPOINT_FORMAT!r}, got {checkpoint_metadata.get('format')!r}.")
    if str(dataset_meta.get("format", "")) != "swm_lance":
        raise ValueError(f"Dataset format must be swm_lance, got {dataset_meta.get('format')!r}.")
    checks = (
        ("env_id", str),
        ("restore_spec", str),
        ("action_dim", int),
    )
    for key, caster in checks:
        if key not in dataset_meta:
            raise ValueError(f"Dataset metadata {dataset_path} is missing required key {key!r}.")
        if key not in checkpoint_metadata:
            continue
        if caster(dataset_meta[key]) != caster(checkpoint_metadata[key]):
            raise ValueError(
                f"Dataset metadata {key}={dataset_meta[key]!r} does not match checkpoint {key}={checkpoint_metadata[key]!r}."
            )

    dataset_shape = tuple(int(x) for x in dataset_meta.get("image_shape", ()))
    checkpoint_shape = tuple(int(x) for x in checkpoint_metadata.get("image_shape", ()))
    if checkpoint_shape and dataset_shape != checkpoint_shape:
        raise ValueError(
            f"Dataset image_shape={dataset_meta.get('image_shape')!r} does not match checkpoint image_shape={checkpoint_metadata['image_shape']!r}."
        )
    for key in ("action_low", "action_high"):
        if key not in dataset_meta:
            raise ValueError(f"Dataset metadata {dataset_path} is missing required key {key!r}.")
        if key not in checkpoint_metadata:
            continue
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
    for meta_key, actual in (
        ("pixels_key", getattr(dataset, "pixels_key", "pixels")),
        ("action_key", getattr(dataset, "action_key", "action")),
    ):
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


def _same_dataset_ref(left: str, right: str) -> bool:
    if not left or not right:
        return False
    lp = Path(left)
    rp = Path(right)
    if lp.exists() and rp.exists():
        return lp.resolve() == rp.resolve()
    return str(left) == str(right)


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    path: str,
    dataset: Any,
    cfg: Any,
    env_id: str,
    restore_spec_id: str,
) -> None:
    expected = {
        "env_id": str(env_id),
        "restore_spec": str(restore_spec_id),
        "seed": int(cfg.eval.seed),
        "goal_offset": int(cfg.eval.goal_offset),
        "eval_budget": int(cfg.eval.budget),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest {path} has {key}={manifest.get(key)!r}, expected {value!r}.")
    dataset_path = _dataset_path(dataset, cfg)
    if not _same_dataset_ref(str(manifest.get("dataset_path", "")), dataset_path):
        raise ValueError(f"Manifest {path} was generated for dataset {manifest.get('dataset_path')!r}, not {dataset_path!r}.")
    if len(manifest.get("pairs", [])) != int(cfg.eval.episodes):
        raise ValueError(f"Manifest {path} has {len(manifest.get('pairs', []))} pairs, expected {int(cfg.eval.episodes)}.")


def _dataset_runtime_metadata(dataset: Any, cfg: Any) -> dict[str, Any]:
    sidecar = load_dataset_metadata(_dataset_path(dataset, cfg), required=False)
    return {
        "path": _dataset_path(dataset, cfg),
        "split_ratio": float(getattr(dataset, "split_ratio", 1.0)),
        "pixels_key": str(getattr(dataset, "pixels_key", "pixels")),
        "action_key": str(getattr(dataset, "action_key", "action")),
        "column_names": [str(x) for x in getattr(dataset, "column_names", [])],
        "sidecar": sidecar,
    }


def _build_mwm_policy(
    model: Any,
    metadata: dict[str, Any],
    cfg: Any,
    device: torch.device,
    action_space: Any,
    process: dict[str, Any],
) -> Any:
    if not hasattr(model, "get_cost_with_fidelity"):
        raise TypeError("MWM evaluator requires models with get_cost_with_fidelity(...).")
    configured_action_block = int(metadata.get("action_block", metadata.get("model", {}).get("action_block", 1)))
    action_block = int(cfg.planner.get("action_block", configured_action_block))
    raw_batch_size = cfg.planner.get("batch_size", "auto")
    planner_batch_size = (
        int(cfg.eval.num_envs)
        if raw_batch_size is None or str(raw_batch_size).lower() == "auto"
        else int(raw_batch_size)
    )
    raw_topk = cfg.planner.get("topk", None)
    topk = (
        max(1, int(raw_topk))
        if raw_topk is not None
        else max(1, int(round(int(cfg.planner.pop_size) * float(cfg.planner.elite_frac))))
    )
    solver = MWMScheduledCEMSolver(
        model,
        batch_size=max(1, planner_batch_size),
        num_samples=int(cfg.planner.pop_size),
        var_scale=float(cfg.planner.init_std),
        n_steps=int(cfg.planner.n_iter),
        topk=topk,
        scheduler=OmegaConf.to_container(cfg.planner.scheduler, resolve=True),
        device=device,
        seed=int(cfg.planner.seed if cfg.planner.get("seed", None) is not None else cfg.eval.seed),
        clamp_actions=bool(cfg.planner.get("clamp_actions", False)),
        std_unbiased=bool(cfg.planner.get("std_unbiased", True)),
    )
    plan_cfg = PlanConfig(
        horizon=int(cfg.planner.horizon),
        receding_horizon=int(cfg.planner.receding_horizon),
        action_block=action_block,
        warm_start=bool(cfg.planner.warm_start),
    )
    image_transform = {"pixels": mwm_image_input_transform, "goal": mwm_image_input_transform}
    return MWMWorldModelPolicy(model=model, solver=solver, config=plan_cfg, process=process, transform=image_transform)


def _build_stable_wm_reference_policy(
    model: Any,
    metadata: dict[str, Any],
    cfg: Any,
    device: torch.device,
    process: dict[str, Any],
) -> Any:
    source_model = getattr(model, "source_model", None)
    if source_model is None:
        raise TypeError("Stable-WM reference evaluation requires an imported upstream model with `source_model`.")
    configured_action_block = int(metadata.get("action_block", metadata.get("model", {}).get("action_block", 1)))
    action_block = int(cfg.planner.get("action_block", configured_action_block))
    raw_batch_size = cfg.planner.get("batch_size", "auto")
    planner_batch_size = (
        int(cfg.eval.num_envs)
        if raw_batch_size is None or str(raw_batch_size).lower() == "auto"
        else int(raw_batch_size)
    )
    raw_topk = cfg.planner.get("topk", None)
    topk = (
        max(1, int(raw_topk))
        if raw_topk is not None
        else max(1, int(round(int(cfg.planner.pop_size) * float(cfg.planner.elite_frac))))
    )
    plan_cfg = PlanConfig(
        horizon=int(cfg.planner.horizon),
        receding_horizon=int(cfg.planner.receding_horizon),
        action_block=action_block,
        warm_start=bool(cfg.planner.warm_start),
    )
    cem_kwargs = {
        "batch_size": max(1, planner_batch_size),
        "num_samples": int(cfg.planner.pop_size),
        "var_scale": float(cfg.planner.init_std),
        "n_steps": int(cfg.planner.n_iter),
        "topk": topk,
        "device": device,
        "seed": int(cfg.planner.seed if cfg.planner.get("seed", None) is not None else cfg.eval.seed),
    }
    preprocessing_spec = metadata.get("preprocessing_spec", {})
    use_imagenet = bool(metadata.get("normalize_imagenet", False)) or (
        isinstance(preprocessing_spec, dict) and preprocessing_spec.get("image") == "imagenet"
    )
    image_fn = imagenet_image_input_transform if use_imagenet else mwm_image_input_transform
    image_transform = {"pixels": image_fn, "goal": image_fn}
    return build_stable_wm_reference_policy(source_model, plan_cfg, cem_kwargs=cem_kwargs, process=process, transform=image_transform)


def _manifest_row_to_pair(row: dict[str, Any]) -> StartGoalPair:
    return StartGoalPair(
        episode=int(row["episode"]),
        start_step=int(row["start_step"]),
        goal_step=int(row["goal_step"]),
        start_row=int(row["start_row"]),
        goal_row=int(row["goal_row"]),
    )


def _pairs_for_eval(
    *,
    dataset: Any,
    cfg: Any,
    env_id: str,
    restore_spec_id: str,
) -> tuple[list[StartGoalPair], dict[str, Any] | None]:
    manifest_path = cfg.eval.get("manifest_path", None)
    if manifest_path:
        manifest = load_manifest(str(manifest_path))
        _validate_manifest(
            manifest,
            path=str(manifest_path),
            dataset=dataset,
            cfg=cfg,
            env_id=env_id,
            restore_spec_id=restore_spec_id,
        )
        pairs = [_manifest_row_to_pair(row) for row in manifest.get("pairs", [])]
        return pairs, {
            "path": str(manifest_path),
            "sha256": manifest_file_sha256(str(manifest_path)),
            "manifest_sha256": manifest.get("manifest_sha256", manifest_sha256(manifest)),
        }
    pairs = sample_start_goal_pairs(
        dataset,
        count=int(cfg.eval.episodes),
        goal_offset_steps=int(cfg.eval.goal_offset),
        seed=int(cfg.eval.seed),
        mode=str(cfg.eval.get("sampling", "mwm")),
    )
    write_path = cfg.eval.get("write_manifest_path", None)
    if not write_path:
        return pairs, None
    manifest = generate_manifest(
        env_id=env_id,
        dataset_path=_dataset_path(dataset, cfg),
        pairs=pairs,
        goal_offset=int(cfg.eval.goal_offset),
        eval_budget=int(cfg.eval.budget),
        seed=int(cfg.eval.seed),
        restore_spec=restore_spec_id,
        dataset_metadata=_dataset_runtime_metadata(dataset, cfg),
        dependency_shas=dependency_refs(Path(__file__).resolve().parent),
    )
    write_manifest(str(write_path), manifest)
    return pairs, {
        "path": str(write_path),
        "sha256": manifest_file_sha256(str(write_path)),
        "manifest_sha256": manifest.get("manifest_sha256", manifest_sha256(manifest)),
    }


def _run_batch(
    *,
    env_id: str,
    image_shape: tuple[int, int],
    model,
    metadata: dict[str, Any],
    dataset: Any,
    pairs: list[StartGoalPair],
    cfg,
    device: torch.device,
    eval_callables: list[dict[str, Any]],
    batch_index: int,
    process: dict[str, Any],
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
        if "action_dim" not in metadata:
            metadata["action_dim"] = int(low.shape[0])
        if low.shape[0] != int(metadata["action_dim"]):
            raise ValueError(f"Env action_dim={low.shape[0]} does not match checkpoint action_dim={metadata['action_dim']}.")
        if bool(cfg.eval.get("reference_policy", False)):
            policy = _build_stable_wm_reference_policy(model, metadata, cfg, device, process)
        else:
            policy = _build_mwm_policy(model, metadata, cfg, device, world.envs.single_action_space, process)
        world.set_policy(policy)
        if hasattr(policy, "reset_trace"):
            policy.reset_trace()
        batch_video_path = Path(str(cfg.eval.video_path)) / f"batch_{int(batch_index):04d}"
        eval_kwargs = dict(
            dataset=dataset,
            episodes_idx=[p.episode for p in pairs],
            start_steps=[p.start_step for p in pairs],
            eval_budget=int(cfg.eval.budget),
            callables=eval_callables,
        )
        swm_results = world.evaluate(
            **eval_kwargs,
            goal_offset=int(cfg.eval.goal_offset),
            video=str(batch_video_path) if bool(cfg.eval.save_video) else None,
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
            "planning_diagnostics": policy.diagnostics() if hasattr(policy, "diagnostics") else {},
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


def _combine_mwm_diagnostics(batches: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [dict(batch.get("planning_diagnostics", {}).get("summary", {})) for batch in batches]
    traces = [diag for batch in batches for diag in batch.get("planning_diagnostics", {}).get("trace", [])]
    total_replans = int(sum(int(s.get("replans", 0)) for s in summaries))
    total_actions = int(sum(int(s.get("actions_recorded", s.get("action_calls", 0))) for s in summaries))
    total_time = float(sum(float(s.get("total_plan_time_sec", 0.0)) for s in summaries))
    total_policy_time = float(sum(float(s.get("total_policy_time_sec", 0.0)) for s in summaries))
    total_bits = int(sum(int(s.get("total_bits_used_estimate", 0)) for s in summaries))
    total_cem_cost_calls = int(sum(int(s.get("cem_cost_calls", 0)) for s in summaries))
    total_candidate_action_values = int(sum(int(s.get("candidate_action_values", 0)) for s in summaries))
    level_counts: dict[str, int] = {}
    for diag in traces:
        key = str(diag.get("base_level_idx", "unknown"))
        level_counts[key] = level_counts.get(key, 0) + 1
    return {
        "summary": {
            "actions_recorded": total_actions,
            "replans": total_replans,
            "total_plan_time_sec": total_time,
            "mean_plan_time_sec": total_time / total_replans if total_replans else 0.0,
            "total_policy_time_sec": total_policy_time,
            "mean_policy_time_sec": total_policy_time / total_actions if total_actions else 0.0,
            "total_bits_used_estimate": total_bits,
            "cem_cost_calls": total_cem_cost_calls,
            "candidate_action_values": total_candidate_action_values,
        },
        "trace": traces,
        "schedule_level_counts": level_counts,
        "plans": total_replans,
        "steps": total_actions,
        "bits_used_total": total_bits,
        "cem_cost_calls": total_cem_cost_calls,
        "candidate_action_values": total_candidate_action_values,
        "plan_time_total_sec": total_time,
        "policy_time_total_sec": total_policy_time,
    }


def _combine_policy_diagnostics(batches: list[dict[str, Any]]) -> dict[str, Any]:
    return _combine_mwm_diagnostics(batches)


def main(cfg_path: str) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    device = _device(str(cfg.device))
    data_format = str(cfg.data.get("format", "lance"))
    if data_format != "lance":
        raise ValueError(f"MWM v1 evaluation only supports Lance datasets, got format={data_format!r}.")
    from stable_worldmodel.data import load_dataset

    model, metadata, epoch = load_world_model_from_checkpoint(
        str(cfg.checkpoint.run_dir),
        None if cfg.checkpoint.epoch is None else int(cfg.checkpoint.epoch),
        device=device,
    )
    dataset = load_dataset(
        _local_path(cfg.data.path),
        format="lance",
        frameskip=1,
        num_steps=2,
        keys_to_load=_eval_keys_to_load(cfg, model, metadata),
    )
    process = {}
    if _uses_standardized_action_space(model, metadata, cfg):
        stats_dataset = load_dataset(
            _local_path(cfg.data.path),
            format="lance",
            frameskip=1,
            num_steps=2,
            keys_to_load=_stat_keys_for_action_process(cfg),
        )
        try:
            process = _build_eval_process(stats_dataset, model, metadata, cfg)
        finally:
            _close_dataset(stats_dataset)
    env_id = str(metadata["env_id"])
    image_shape = parse_image_shape(metadata["image_shape"])
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec_id, eval_callables = eval_callables_for_env(
        env_id,
        dataset.column_names,
        import_path=restore_import_path,
    )
    _validate_dataset_metadata(dataset, metadata, cfg)
    if str(metadata["restore_spec"]) != restore_spec_id:
        raise ValueError(
            f"Runtime restore spec {restore_spec_id!r} does not match checkpoint restore_spec={metadata['restore_spec']!r}."
        )

    pairs, manifest_info = _pairs_for_eval(dataset=dataset, cfg=cfg, env_id=env_id, restore_spec_id=restore_spec_id)
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
                process=process,
            )
        )
    videos = [video for batch in all_results for video in batch.get("videos", [])]
    checkpoint_ref = str(cfg.checkpoint.run_dir)

    output = {
        "env_id": env_id,
        "checkpoint_run_dir": checkpoint_ref,
        "checkpoint_epoch": int(epoch),
        "dataset": _dataset_path(dataset, cfg),
        "episodes": int(cfg.eval.episodes),
        "goal_offset": int(cfg.eval.goal_offset),
        "eval_budget": int(cfg.eval.budget),
        "restore_spec": restore_spec_id,
        "swm_results": _combine_swm_results(all_results),
        "planning_diagnostics": _combine_mwm_diagnostics(all_results),
        "policy_diagnostics": _combine_policy_diagnostics(all_results),
        "model_accounting": model_accounting(model),
        "dataset_metadata": _dataset_runtime_metadata(dataset, cfg),
        "manifest": manifest_info,
        "batches": all_results,
        "videos": videos,
        "dependencies": dependency_refs(Path(__file__).resolve().parent),
    }
    output["schedule"] = _jsonable(OmegaConf.to_container(cfg.planner.scheduler, resolve=True))
    output["seed"] = int(cfg.eval.seed)
    if cfg.get("config", None):
        output["config"] = _jsonable(OmegaConf.to_container(cfg.config, resolve=True))
    output_path = Path(str(cfg.eval.output_path))
    write_json(output_path, output)
    print(f"Wrote MWM planning results to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python eval_mwm.py configs/eval_mwm.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
