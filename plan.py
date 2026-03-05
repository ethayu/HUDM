"""plan.py — Closed-loop latent-space MPC-CEM planning with a world model.

Usage
-----
python plan.py configs/plan.yaml
"""

from __future__ import annotations

import inspect
import os
import random
import sys
import json
from datetime import datetime
from typing import Any, Dict, Tuple

import cv2
import gym
import numpy as np
import torch
from gym.envs.registration import register
from omegaconf import DictConfig, OmegaConf

# Ensure local imports work even when launched via an absolute path.
sys.path.append(os.path.dirname(__file__))

from models.world.ensemble import WorldModelEnsemble
from models.world.model import HierWorldModel
from planning.gt_env_cem import GTEnvCEMPlanner
from planning.latent_cem import LatentCEMPlanner
from planning.particle_cem import ParticleCEMPlanner
from pusht.pusht_particle_backend import PushTParticleBackend
from pusht.pusht_wrapper import PushTWrapper
from validate_cfg import validate_plan_cfg


def _unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _resolve_dataset_seed(seed_cfg: Any) -> int:
    if isinstance(seed_cfg, str):
        seed_s = seed_cfg.strip().lower()
        if seed_s == "random":
            return int(random.randrange(1_000_000))
        raise ValueError(
            f"plan.init_goal.dataset.seed must be an int or 'random', got {seed_cfg!r}."
        )
    try:
        return int(seed_cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"plan.init_goal.dataset.seed must be an int or 'random', got {seed_cfg!r}."
        ) from exc


def _gym_make_versioned(env_id: str, env_cfg: DictConfig):
    make_kwargs = dict(OmegaConf.to_container(env_cfg, resolve=True))
    make_kwargs["render_action"] = True

    try:
        make_params = inspect.signature(gym.make).parameters
    except (TypeError, ValueError):
        make_params = {}
    if "disable_env_checker" in make_params:
        make_kwargs["disable_env_checker"] = True
    if "apply_api_compatibility" in make_params:
        make_kwargs["apply_api_compatibility"] = False
    return gym.make(str(env_id), **make_kwargs)


def _image_to_model_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(img, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)  # (1,H,W,C)
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)  # (1,C,H,W)
    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x * 2.0 - 1.0
    return x


def _resolve_device(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg).lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("plan.world_model.device=cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if device_cfg != "auto":
        raise ValueError(
            f"plan.world_model.device must be one of auto|cpu|cuda, got '{device_cfg}'."
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_bits_human(bits: int) -> str:
    value = float(max(0, int(bits)))
    units = ["b", "Kb", "Mb", "Gb", "Tb"]
    unit_idx = 0
    while value >= 1000.0 and unit_idx < len(units) - 1:
        value /= 1000.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def _bits_to_flops_estimate(bits: int) -> int:
    # Current planners track compute volume in transferred/processed bit-counts.
    # Convert to an operation-count proxy via 32-bit scalar granularity.
    b = max(0, int(bits))
    return (b + 31) // 32


def _format_flops_human(flops: int) -> str:
    value = float(max(0, int(flops)))
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
    unit_idx = 0
    while value >= 1000.0 and unit_idx < len(units) - 1:
        value /= 1000.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def _latest_run_dir(root: str) -> str:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    run_dirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(run_dirs, key=os.path.getmtime)


def _load_world_model_member(wm_cfg: DictConfig, run_dir: str, device: torch.device) -> HierWorldModel:
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"World-model run directory not found: {run_dir}")

    wm = HierWorldModel(
        K=list(wm_cfg.model.K),
        D=int(wm_cfg.model.D),
        action_dim=int(wm_cfg.data.action_dim),
        decoder_mode=str(getattr(wm_cfg.model, "decoder_mode", "per_level")),
        dynamics_mode=str(getattr(wm_cfg.model, "dynamics_mode", "per_level")),
    ).to(device)

    enc_path = os.path.join(run_dir, "encoder.pt")
    if not os.path.isfile(enc_path):
        raise FileNotFoundError(f"Missing encoder checkpoint: {enc_path}")
    wm.encoder.load_state_dict(torch.load(enc_path, map_location=device))

    dynamics_mode = str(getattr(wm_cfg.model, "dynamics_mode", "per_level"))
    if dynamics_mode == "per_level":
        for li in range(len(wm.K)):
            dyn_path = os.path.join(run_dir, f"dyn_l{li}.pt")
            if not os.path.isfile(dyn_path):
                raise FileNotFoundError(f"Missing dynamics checkpoint: {dyn_path}")
            wm.dynamics[li].load_state_dict(torch.load(dyn_path, map_location=device))
    else:
        dyn_path = os.path.join(run_dir, "dyn.pt")
        if not os.path.isfile(dyn_path):
            raise FileNotFoundError(f"Missing dynamics checkpoint: {dyn_path}")
        wm.dynamics.load_state_dict(torch.load(dyn_path, map_location=device))

    decoder_mode = str(getattr(wm_cfg.model, "decoder_mode", "per_level"))
    if decoder_mode == "per_level":
        for li in range(len(wm.K)):
            dec_path = os.path.join(run_dir, f"decoder_l{li}.pt")
            if not os.path.isfile(dec_path):
                raise FileNotFoundError(f"Missing decoder checkpoint: {dec_path}")
            wm.decoders[li].load_state_dict(torch.load(dec_path, map_location=device))
    else:
        dec_path = os.path.join(run_dir, "decoder.pt")
        if not os.path.isfile(dec_path):
            raise FileNotFoundError(f"Missing decoder checkpoint: {dec_path}")
        wm.decoder.load_state_dict(torch.load(dec_path, map_location=device))

    wm.eval()
    return wm


def _wm_signature(wm_cfg: DictConfig) -> dict:
    return {
        "K": list(wm_cfg.model.K),
        "D": int(wm_cfg.model.D),
        "action_dim": int(wm_cfg.data.action_dim),
        "decoder_mode": str(getattr(wm_cfg.model, "decoder_mode", "per_level")),
        "dynamics_mode": str(getattr(wm_cfg.model, "dynamics_mode", "per_level")),
    }


def _load_run_world_cfg(run_dir: str) -> tuple[DictConfig, str]:
    cfg_path = os.path.join(run_dir, "world.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Missing world config in run directory: {cfg_path}. "
            "Set plan.world_model.config_path explicitly or provide run dirs with world.yaml."
        )
    return OmegaConf.load(cfg_path), cfg_path


def _assert_world_cfg_compatible(base_sig: dict, other_cfg: DictConfig, label: str) -> None:
    other_sig = _wm_signature(other_cfg)
    if other_sig != base_sig:
        raise ValueError(
            f"World-model config mismatch for {label}.\n"
            f"Expected: {base_sig}\n"
            f"Found:    {other_sig}"
        )


def _load_world_model(plan_cfg: DictConfig) -> tuple[object, DictConfig, str, torch.device]:
    device = _resolve_device(str(plan_cfg.world_model.device))
    config_path = plan_cfg.world_model.config_path
    if config_path is not None:
        config_path = str(config_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"World-model config not found: {config_path}")
        cfg_override = OmegaConf.load(config_path)
    else:
        cfg_override = None

    ens_cfg = plan_cfg.world_model.get("ensemble", {})
    ens_enabled = bool(getattr(ens_cfg, "enabled", False))
    if ens_enabled:
        run_dirs = list(getattr(ens_cfg, "run_dirs", []))
        run_dirs = [str(rd) for rd in run_dirs]
        if cfg_override is None:
            wm_cfg, _ = _load_run_world_cfg(run_dirs[0])
        else:
            wm_cfg = cfg_override
        base_sig = _wm_signature(wm_cfg)
        members = []
        for rd in run_dirs:
            if not os.path.isdir(rd):
                raise FileNotFoundError(f"World-model run directory not found: {rd}")
            run_cfg_path = os.path.join(rd, "world.yaml")
            if os.path.isfile(run_cfg_path):
                run_cfg = OmegaConf.load(run_cfg_path)
                _assert_world_cfg_compatible(base_sig, run_cfg, label=rd)
            members.append(_load_world_model_member(wm_cfg, rd, device))
        wm_backend = WorldModelEnsemble(members).to(device)
        run_desc = ", ".join(run_dirs)
        return wm_backend, wm_cfg, run_desc, device

    run_dir = plan_cfg.world_model.run_dir
    if run_dir is None:
        checkpoint_root = str(getattr(plan_cfg.world_model, "checkpoint_root", "checkpoints_world"))
        run_dir = _latest_run_dir(checkpoint_root)
    run_dir = str(run_dir)
    if cfg_override is None:
        wm_cfg, _ = _load_run_world_cfg(run_dir)
    else:
        wm_cfg = cfg_override
        run_cfg_path = os.path.join(run_dir, "world.yaml")
        if os.path.isfile(run_cfg_path):
            run_cfg = OmegaConf.load(run_cfg_path)
            _assert_world_cfg_compatible(_wm_signature(wm_cfg), run_cfg, label=run_dir)
    wm = _load_world_model_member(wm_cfg, run_dir, device)
    return wm, wm_cfg, run_dir, device


@torch.no_grad()
def _encode_visual(wm: object, img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = _image_to_model_tensor(img, device=device)
    return wm.encode(x)


def _sample_init_goal_states(
    env: PushTWrapper,
    cfg: DictConfig,
    wm_cfg: DictConfig | None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    src = str(cfg.init_goal.source).lower()
    if src == "random":
        init_state, goal_state = env.sample_random_init_goal_states(seed=0)
        meta = {"source": "random"}
        return init_state, goal_state, meta

    ds_cfg = cfg.init_goal.dataset
    zarr_path = ds_cfg.zarr_path
    if zarr_path is None:
        if wm_cfg is not None and getattr(wm_cfg, "data", None) is not None:
            zarr_path = wm_cfg.data.zarr_path
        else:
            raise ValueError(
                "plan.init_goal.dataset.zarr_path must be set when backend is gt_env or particle_sim "
                "and no world config is available."
            )
    split_ratio = ds_cfg.split_ratio
    if split_ratio is None:
        split_ratio = 0.8 if wm_cfg is None else float(getattr(wm_cfg.data, "split_ratio", 0.8))
    sample_seed = _resolve_dataset_seed(getattr(ds_cfg, "seed", 0))

    init_state, goal_state, meta = env.sample_dataset_init_goal_states(
        dataset=str(zarr_path),
        trajectory_len=int(ds_cfg.trajectory_len),
        split=str(ds_cfg.split),
        split_ratio=float(split_ratio),
        seed=sample_seed,
    )
    print(
        "[init_goal] source=dataset "
        f"episode={meta['episode_index']} start={meta['start_index']} "
        f"goal={meta['goal_index']} len={meta['trajectory_len']} split={meta['split']} "
        f"seed={sample_seed}"
    )
    meta = dict(meta)
    meta["source"] = "dataset"
    meta["zarr_path"] = str(zarr_path)
    meta["split_ratio"] = float(split_ratio)
    meta["seed"] = int(sample_seed)
    return init_state, goal_state, meta


def _set_goal_pose(env: PushTWrapper, goal_state: np.ndarray) -> None:
    goal_state = np.asarray(goal_state, dtype=np.float32)
    if goal_state.shape[0] < 5:
        raise ValueError(f"goal_state must have at least 5 dims, got {goal_state.shape}")
    
    env.set_task_goal(goal_state[2:5])


def _set_start_pose(env: PushTWrapper, init_state: np.ndarray) -> None:
    init_state = np.asarray(init_state, dtype=np.float32)
    if init_state.shape[0] < 5:
        raise ValueError(f"init_state must have at least 5 dims, got {init_state.shape}")
    if hasattr(env, "set_task_start"):
        env.set_task_start(init_state[2:5])


def _set_execution_fidelity_finest(env: PushTWrapper) -> None:
    """
    Reset planning-time fidelity to finest so execution/rendering is comparable and
    unaffected by the last planner rollout's coarse-level settings.
    """
    if hasattr(env, "_planning_fidelity_num_levels") and hasattr(env, "set_planning_fidelity_level"):
        n_levels = int(getattr(env, "_planning_fidelity_num_levels", 1))
        env.set_planning_fidelity_level(max(0, n_levels - 1))


def _resize_image_hw(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(img)
    h_t, w_t = int(target_hw[0]), int(target_hw[1])
    if out.shape[0] != h_t or out.shape[1] != w_t:
        out = cv2.resize(out, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
    return out


def _overlay_start_pose(
    planner_img: np.ndarray,
    exec_with_start: np.ndarray,
    exec_without_start: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    out = _resize_image_hw(np.asarray(planner_img), target_hw).copy()
    with_start = _resize_image_hw(np.asarray(exec_with_start), target_hw)
    without_start = _resize_image_hw(np.asarray(exec_without_start), target_hw)
    mask = np.any(with_start != without_start, axis=-1)
    out[mask] = with_start[mask]
    return out


def _rollout_level_for_exec_step(info: object, exec_step_in_replan: int) -> int:
    levels = list(getattr(info, "rollout_level_indices", []))
    if exec_step_in_replan < len(levels):
        return int(levels[exec_step_in_replan])
    base = getattr(info, "base_level_idx", None)
    if base is None:
        return 0
    return int(base)


def _planner_view_frame(
    env: PushTWrapper,
    base_visual: np.ndarray,
    level_idx: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """
    Approximate "what planner saw" at a chosen fidelity level, while keeping
    saved GIF frame size fixed for readability.
    """
    img = np.asarray(base_visual)
    out = img

    # Reuse env fidelity transform at requested level without leaving env in
    # a different execution fidelity state.
    if (
        hasattr(env, "set_planning_fidelity_level")
        and hasattr(env, "_apply_planning_fidelity_visual")
        and hasattr(env, "_planning_fidelity_level_idx")
    ):
        prev_idx = int(getattr(env, "_planning_fidelity_level_idx", 0))
        try:
            env.set_planning_fidelity_level(int(level_idx))
            out = env._apply_planning_fidelity_visual(img)
        finally:
            env.set_planning_fidelity_level(prev_idx)
    return _resize_image_hw(out, target_hw)


def _wm_decode_frame(
    wm: object,
    z: torch.Tensor,
    level_idx: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    decoder = wm if hasattr(wm, "decode") else getattr(wm, "primary", None)
    if decoder is None or (not hasattr(decoder, "decode")):
        raise ValueError("World-model backend does not expose a decode(level, z) API.")

    n_levels = len(getattr(decoder, "K", []))
    li = int(level_idx)
    if n_levels > 0:
        li = max(0, min(li, n_levels - 1))

    z_in = z
    if z_in.ndim == 1:
        z_in = z_in.unsqueeze(0)
    x = decoder.decode(li, z_in)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Decoded frame must be rank-3, got shape {tuple(x.shape)}")

    # Decoder outputs are in [-1, 1].
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0).detach().cpu()
    if int(x.shape[0]) in {1, 3} and int(x.shape[-1]) not in {1, 3}:
        x = x.permute(1, 2, 0)
    img = (x.numpy() * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return _resize_image_hw(img, target_hw)


def _particle_planner_view_frame(
    particle_backend: object,
    start_state: np.ndarray,
    cur_state: np.ndarray,
    goal_state: np.ndarray,
    level_idx: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    prev_idx = int(getattr(particle_backend, "_planning_fidelity_level_idx", 0))
    try:
        particle_backend.set_planning_fidelity_level(int(level_idx))
        particle_backend.prepare(seed=0, init_state=cur_state, goal_state=goal_state, with_visual=False)
        # Keep planner-view start overlay anchored to the true episode start.
        if hasattr(particle_backend, "_start_state"):
            particle_backend._start_state = np.asarray(start_state, dtype=np.float32).copy()
        img = particle_backend.render("rgb_array", include_start_pose=True)
    finally:
        particle_backend.set_planning_fidelity_level(prev_idx)
    return _resize_image_hw(np.asarray(img), target_hw)


def _render_dataset_gt_frames(
    env: PushTWrapper,
    sample_meta: Dict[str, Any],
    init_state: np.ndarray,
    goal_state: np.ndarray,
) -> list[np.ndarray]:
    """
    Render GT reference from the sampled dataset state trajectory, with the same
    goal overlay as planned rollouts for apples-to-apples visual comparison.
    """
    if str(sample_meta.get("source", "")).lower() != "dataset":
        return []
    zarr_path = sample_meta.get("zarr_path")
    if zarr_path is None:
        print("[save][gt] skipped: missing dataset zarr path in sample metadata.")
        return []
    start_idx = int(sample_meta.get("start_index", -1))
    goal_idx = int(sample_meta.get("goal_index", -1))
    if start_idx < 0 or goal_idx <= start_idx:
        print("[save][gt] skipped: invalid dataset index range in sample metadata.")
        return []

    try:
        import zarr
    except Exception as exc:
        print(f"[save][gt] skipped: zarr import failed ({exc}).")
        return []

    root = zarr.open_group(str(zarr_path), mode="r")
    state_arr = np.asarray(root["data"]["state"][start_idx:goal_idx + 1], dtype=np.float32)
    if int(state_arr.shape[0]) <= 0:
        print("[save][gt] skipped: empty state segment.")
        return []
    frames: list[np.ndarray] = []
    _set_execution_fidelity_finest(env)
    _set_start_pose(env, init_state)
    for s in state_arr:
        s = np.asarray(s, dtype=np.float32)
        if s.shape[0] < int(getattr(env, "state_dim", s.shape[0])):
            pad = int(getattr(env, "state_dim", s.shape[0])) - s.shape[0]
            s = np.concatenate([s, np.zeros((pad,), dtype=np.float32)], axis=0)
        env.prepare(seed=0, init_state=s[: int(getattr(env, "state_dim", s.shape[0]))])
        _set_goal_pose(env, goal_state)
        frames.append(env.render("rgb_array", include_start_pose=True))
    return frames


def _run_closed_loop(
    env: PushTWrapper,
    wm: object | None,
    planner: object,
    backend: str,
    cfg: DictConfig,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    device: torch.device,
) -> Tuple[bool, list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    _set_start_pose(env, init_state)
    particle_backend = None
    if backend == "particle_sim":
        particle_backend = getattr(planner, "backend", None)
        if particle_backend is None:
            raise ValueError("backend='particle_sim' requires planner.backend for planner-view rendering.")
    z_goal = None
    if backend == "wm":
        if wm is None:
            raise ValueError("backend='wm' requires a loaded world model.")
        # prepare() resets env internals (including goal pose), so apply sampled
        # goal afterward and refresh the rendered observation.
        goal_obs, _ = env.prepare(seed=0, init_state=goal_state)
        _set_goal_pose(env, goal_state)
        goal_obs["visual"] = env.render("rgb_array", include_start_pose=False)
        z_goal = _encode_visual(wm, goal_obs["visual"], device)
    # Reset env to initial state for actual execution.
    obs, cur_state = env.prepare(seed=0, init_state=init_state, goal_state=goal_state)
    _set_execution_fidelity_finest(env)
    obs["visual"] = env.render("rgb_array", include_start_pose=False)

    trajectory = [cur_state.copy()]
    frames: list[np.ndarray] = []
    planner_frames: list[np.ndarray] = []
    if bool(cfg.save):
        frames.append(env.render("rgb_array", include_start_pose=True))

    # Early exit if the reset state already satisfies the goal condition.
    initial_metrics = env.eval_state(goal_state, cur_state)
    if bool(initial_metrics["success"]):
        stats = {
            "plans": 0,
            "bits_used_total": 0,
            "flops_used_total": 0,
            "plan_time_total_sec": 0.0,
        }
        return True, trajectory, frames, planner_frames, stats

    steps = int(cfg.mpc.steps)
    horizon = int(cfg.mpc.horizon)
    replan_every = int(cfg.mpc.replan_every)
    n_replans = max(1, int(np.ceil(steps / replan_every)))

    render_enabled = bool(cfg.render)
    t = 0
    replan_idx = 0
    total_plan_bits = 0
    total_plan_flops = 0
    total_plan_time = 0.0
    n_plans = 0
    prev_exec_steps = 0
    while t < steps:
        mpc_progress = 0.0 if n_replans <= 1 else replan_idx / (n_replans - 1)
        z_cur_for_plan = None
        if backend == "wm":
            z_cur_for_plan = _encode_visual(wm, obs["visual"], device)
            action_seq, info = planner.plan(
                z_cur_for_plan,
                z_goal,
                mpc_progress=mpc_progress,
                warm_start_steps=int(prev_exec_steps),
            )
        elif backend == "gt_env":
            action_seq, info = planner.plan(
                init_state=cur_state,
                goal_state=goal_state,
                mpc_progress=mpc_progress,
                seed=int(1009 * replan_idx + 7919 * t),
                warm_start_steps=int(prev_exec_steps),
            )
            # GT-env planner rollouts reuse env and mutate/reset it internally.
            # Restore the true execution state/goal before applying planned actions.
            obs, cur_state = env.prepare(seed=0, init_state=cur_state, goal_state=goal_state)
            _set_execution_fidelity_finest(env)
            obs["visual"] = env.render("rgb_array", include_start_pose=False)
        else:
            action_seq, info = planner.plan(
                init_state=cur_state,
                goal_state=goal_state,
                mpc_progress=mpc_progress,
                seed=int(1009 * replan_idx + 7919 * t),
                warm_start_steps=int(prev_exec_steps),
            )

        bits_used = int(getattr(info, "bits_used_estimate", 0))
        flops_used = _bits_to_flops_estimate(bits_used)
        plan_time = float(getattr(info, "plan_time_sec", 0.0))
        total_plan_bits += bits_used
        total_plan_flops += flops_used
        total_plan_time += plan_time
        n_plans += 1
        if bits_used > 0 or plan_time > 0:
            print(
                f"[plan] replan {replan_idx:03d}  backend {backend}  "
                f"bits {bits_used} ({_format_bits_human(bits_used)})  "
                f"flops {flops_used} ({_format_flops_human(flops_used)})  "
                f"time {plan_time:.3f}s"
            )

        # Add first planner-view frame (t=0) after the first plan is available.
        if bool(cfg.save) and len(planner_frames) == 0:
            init_level = _rollout_level_for_exec_step(info, exec_step_in_replan=0)
            target_hw = frames[0].shape[:2] if len(frames) > 0 else np.asarray(obs["visual"]).shape[:2]
            if backend == "gt_env":
                frame = _planner_view_frame(
                    env=env,
                    base_visual=np.asarray(obs["visual"]),
                    level_idx=init_level,
                    target_hw=target_hw,
                )
                if len(frames) > 0:
                    frame = _overlay_start_pose(
                        planner_img=frame,
                        exec_with_start=np.asarray(frames[0]),
                        exec_without_start=np.asarray(obs["visual"]),
                        target_hw=target_hw,
                    )
                planner_frames.append(frame)
            elif backend == "wm":
                if z_cur_for_plan is None:
                    z_cur_for_plan = _encode_visual(wm, obs["visual"], device)
                frame = _wm_decode_frame(
                    wm=wm,
                    z=z_cur_for_plan,
                    level_idx=init_level,
                    target_hw=target_hw,
                )
                if len(frames) > 0:
                    frame = _overlay_start_pose(
                        planner_img=frame,
                        exec_with_start=np.asarray(frames[0]),
                        exec_without_start=np.asarray(obs["visual"]),
                        target_hw=target_hw,
                    )
                planner_frames.append(frame)
            else:
                planner_frames.append(
                    _particle_planner_view_frame(
                        particle_backend=particle_backend,
                        start_state=init_state,
                        cur_state=cur_state,
                        goal_state=goal_state,
                        level_idx=init_level,
                        target_hw=target_hw,
                    )
                )

        n_exec = min(replan_every, horizon, steps - t)
        for i in range(n_exec):
            action = np.asarray(action_seq[i].numpy(), dtype=np.float32)
            obs, _, done, step_info = env.step(action)
            cur_state = step_info["state"]
            trajectory.append(cur_state.copy())

            metrics = env.eval_state(goal_state, cur_state)
            dist = metrics["state_dist"]
            base_k = getattr(info, "base_k", None)
            base_spacing = getattr(info, "base_spacing", None)
            base_np = getattr(info, "base_num_particles", None)
            level_idx = int(getattr(info, "base_level_idx", -1))
            k_str = "-" if base_k is None else str(base_k)
            spacing_str = "-" if base_spacing is None else f"{float(base_spacing):.4f}"
            np_str = "-" if base_np is None else str(int(base_np))
            print(
                f"step {t + 1:03d}  dist {dist:6.1f}  "
                f"level_idx {level_idx}  k {k_str}  spacing {spacing_str}  n_particles {np_str}"
            )

            if render_enabled:
                try:
                    env.render("human", include_start_pose=True)
                except Exception as exc:
                    print(f"[render disabled] {exc}")
                    render_enabled = False

            if bool(cfg.save):
                frame_with_start = env.render("rgb_array", include_start_pose=True)
                frames.append(frame_with_start)
                li_exec = _rollout_level_for_exec_step(info, exec_step_in_replan=i)
                if backend == "gt_env":
                    frame = _planner_view_frame(
                        env=env,
                        base_visual=np.asarray(obs["visual"]),
                        level_idx=li_exec,
                        target_hw=frames[-1].shape[:2],
                    )
                    planner_frames.append(
                        _overlay_start_pose(
                            planner_img=frame,
                            exec_with_start=np.asarray(frame_with_start),
                            exec_without_start=np.asarray(obs["visual"]),
                            target_hw=frames[-1].shape[:2],
                        )
                    )
                elif backend == "wm":
                    z_exec = _encode_visual(wm, obs["visual"], device)
                    frame = _wm_decode_frame(
                        wm=wm,
                        z=z_exec,
                        level_idx=li_exec,
                        target_hw=frames[-1].shape[:2],
                    )
                    planner_frames.append(
                        _overlay_start_pose(
                            planner_img=frame,
                            exec_with_start=np.asarray(frame_with_start),
                            exec_without_start=np.asarray(obs["visual"]),
                            target_hw=frames[-1].shape[:2],
                        )
                    )
                else:
                    planner_frames.append(
                        _particle_planner_view_frame(
                            particle_backend=particle_backend,
                            start_state=init_state,
                            cur_state=cur_state,
                            goal_state=goal_state,
                            level_idx=li_exec,
                            target_hw=frames[-1].shape[:2],
                        )
                    )

            # Use either geometric success check or environment terminal signal.
            if bool(metrics["success"]) or bool(done):
                stats = {
                    "plans": n_plans,
                    "bits_used_total": total_plan_bits,
                    "flops_used_total": total_plan_flops,
                    "plan_time_total_sec": total_plan_time,
                }
                return True, trajectory, frames, planner_frames, stats

            t += 1

        prev_exec_steps = int(n_exec)
        replan_idx += 1

    stats = {
        "plans": n_plans,
        "bits_used_total": total_plan_bits,
        "flops_used_total": total_plan_flops,
        "plan_time_total_sec": total_plan_time,
    }
    return False, trajectory, frames, planner_frames, stats


def main(cfg_path: str) -> None:
    cfg_root = OmegaConf.load(cfg_path)
    defaults = {
        "backend": "wm",  # wm | gt_env | particle_sim
        "env_id": "pusht",
        "env": {
            "with_velocity": True,
            "with_target": True,
            "add_noise": 0,
            "noise_std": 0.0,
        },
        "world_model": {
            # If config_path is null, world config is loaded from run_dir/world.yaml
            # (or first ensemble member's world.yaml).
            "config_path": None,
            "run_dir": None,   # used in single-model mode
            "checkpoint_root": "checkpoints_world",  # used if run_dir is null
            "device": "auto",  # auto | cpu | cuda
            "ensemble": {
                "enabled": False,
                "run_dirs": [],
            },
        },
        "mpc": {
            "steps": 50,
            "horizon": 20,
            "replan_every": 1,
        },
        "cem": {
            "pop_size": 256,
            "elite_frac": 0.1,
            "n_iter": 5,
            "init_std": 1.0,
            "warm_start": True,
            "action_low": None,
            "action_high": None,
        },
        "objective": {
            "latent_metric": "l2",  # l1 | l2
            "terminal_weight": 1.0,
            "running_weight": 0.0,
            "action_l2_weight": 0.0,
        },
        "fidelity": {
            "enabled": True,
            "num_levels": 4,  # used by gt_env / particle_sim; wm backend derives count from model.K
            # Level indices are over model.K (0=coarsest, len(K)-1=finest).
            # Use integer indices or tokens: "coarsest", "finest", "base".
            # "base" is valid for CEM and rollout fields; it refers to the
            # current MPC-stage level in CEM fields and current CEM level in rollout fields.
            "mpc": {
                "mode": "linear",      # fixed | linear
                "level": "finest",     # used when mode=fixed
                "start_level": "coarsest",
                "end_level": "finest",
            },
            "cem": {
                "mode": "linear",      # fixed | linear
                "level": "base",       # used when mode=fixed
                "start_level": "base",
                "end_level": "finest",
            },
            "rollout": {
                "mode": "fixed",     # fixed | linear | uncertainty_downshift
                "level": "base",     # used when mode=fixed
                "start_level": "base",
                "end_level": "coarsest",
                "uncertainty": {
                    "criterion": "mean",   # mean | percentile
                    "threshold": 0.05,
                    "percentile": 0.8,
                    "min_level": "coarsest",
                    "max_downshifts_per_step": 1,
                },
            },
        },
        "gt_env": {
            "rollout_samples": 1,
            "objective_space": "image",  # image | state
            "progress": True,
            "progress_leave": False,
            "fidelity_env": {
                "mode": "blur_avgpool",      # blur_avgpool | blur_quantize
                "blur_sigma_max": 2.0,
                "pool_scale_max": 4,
                "quantize_levels_min": 8,
                "quantize_levels_max": 256,
                "action_noise_std_max": 0.0,
                "downsample_output": False,
                "min_downsample_size": 12,
            },
        },
        "particle_env": {
            "rollout_samples": 1,
            "objective_space": "state",  # image | state
            "progress": True,
            "progress_leave": False,
            "fidelity_env": {
                # Coarsest -> finest. Larger spacing => fewer particles.
                "spacings": [0.020, 0.016, 0.013, 0.010],
                "device": "auto",  # auto | cpu | cuda | cuda:0
                "xmin": -0.25,
                "xmax": 0.25,
                "ymin": -0.25,
                "ymax": 0.25,
                "min_particles": 1,
                "coarsest_single_particle": True,
                "particle_radius": None,
                "radius_scale": 1.0,
                "radius_clip_spacing": False,
                "stem_w": 0.05,
                "stem_h": 0.10,
                "bar_w": 0.12,
                "bar_h": 0.04,
                "pusher_radius": 0.015,
                "pusher_speed": 0.6,
                "pusher_interp_substeps": True,
                "frame_dt": 1.0 / 60.0,
                "substeps": 16,
                "iters": 8,
                "mu": 0.6,
                "contact_alpha": 0.35,
                "ground_friction_accel": 2.0,
                "rest_speed_eps": 0.01,
                "lin_damp": 0.995,
                "vel_damp": 0.999,
                "alpha_rigid": 1.0,
            },
        },
        "init_goal": {
            "source": "random",  # random | dataset
            "dataset": {
                "zarr_path": None,      # if None, falls back to world config data.zarr_path
                "split": "valid",
                "split_ratio": None,    # if None, falls back to world config split_ratio (or 0.8)
                "trajectory_len": 20,
                "seed": 0,
            },
        },
        "render": False,
        "save": False,
    }
    cfg = OmegaConf.merge(defaults, cfg_root.get("plan", cfg_root))
    validate_plan_cfg(cfg)
    cfg.init_goal.dataset.seed = _resolve_dataset_seed(getattr(cfg.init_goal.dataset, "seed", 0))

    # Register env id.
    try:
        register(
            id=str(cfg.env_id),
            entry_point="pusht.pusht_wrapper:PushTWrapper",
            max_episode_steps=300,
            reward_threshold=1.0,
        )
    except gym.error.Error:
        pass

    env_wrapped = _gym_make_versioned(str(cfg.env_id), cfg.env)
    env: PushTWrapper = _unwrap_env(env_wrapped)
    backend = str(cfg.backend).lower()
    device = _resolve_device(str(cfg.world_model.device))
    wm = None
    wm_cfg = None
    run_desc = backend

    if backend == "wm":
        wm, wm_cfg, run_desc, device = _load_world_model(cfg)

        # Validate environment/model interface.
        model_action_dim = int(wm_cfg.data.action_dim)
        if int(env.action_dim) != model_action_dim:
            raise ValueError(
                f"Action-dim mismatch: env.action_dim={env.action_dim}, world-model action_dim={model_action_dim}"
            )
        if not bool(cfg.env.with_velocity):
            # World config default uses velocity-aware state/image trajectories for PushT in this repo.
            print("[warn] plan.env.with_velocity=false may mismatch the world model's training distribution.")
        # Align fidelity num_levels to the loaded model levels for consistency.
        cfg.fidelity.num_levels = len(wm.K)

        planner = LatentCEMPlanner(
            world_model=wm,
            horizon=int(cfg.mpc.horizon),
            action_dim=int(env.action_dim),
            pop_size=int(cfg.cem.pop_size),
            elite_frac=float(cfg.cem.elite_frac),
            n_iter=int(cfg.cem.n_iter),
            init_std=float(cfg.cem.init_std),
            warm_start=bool(cfg.cem.warm_start),
            action_low=cfg.cem.action_low,
            action_high=cfg.cem.action_high,
            objective_cfg=OmegaConf.to_container(cfg.objective, resolve=True),
            fidelity_cfg=OmegaConf.to_container(cfg.fidelity, resolve=True),
            drop_tail_on_coarsen=True,
            device=device,
        )
    elif backend == "gt_env":
        planner = GTEnvCEMPlanner(
            env=env,
            horizon=int(cfg.mpc.horizon),
            action_dim=int(env.action_dim),
            pop_size=int(cfg.cem.pop_size),
            elite_frac=float(cfg.cem.elite_frac),
            n_iter=int(cfg.cem.n_iter),
            init_std=float(cfg.cem.init_std),
            warm_start=bool(cfg.cem.warm_start),
            action_low=cfg.cem.action_low,
            action_high=cfg.cem.action_high,
            objective_cfg=OmegaConf.to_container(cfg.objective, resolve=True),
            fidelity_cfg=OmegaConf.to_container(cfg.fidelity, resolve=True),
            gt_env_cfg=OmegaConf.to_container(cfg.gt_env, resolve=True),
            device=device,
        )
    else:
        spacings = list(cfg.particle_env.fidelity_env.spacings)
        cfg.fidelity.num_levels = len(spacings)
        particle_backend = PushTParticleBackend(
            with_velocity=bool(cfg.env.with_velocity),
            with_target=bool(cfg.env.with_target),
            render_size=int(getattr(env, "render_size", 96)),
            relative=bool(getattr(env, "relative", True)),
            action_scale=float(getattr(env, "action_scale", 100.0)),
            device=str(cfg.particle_env.fidelity_env.device),
            fidelity_spacings=[float(s) for s in spacings],
            warp_cfg=OmegaConf.to_container(cfg.particle_env.fidelity_env, resolve=True),
            seed=int(getattr(cfg.init_goal.dataset, "seed", 0)),
        )
        planner = ParticleCEMPlanner(
            particle_backend=particle_backend,
            horizon=int(cfg.mpc.horizon),
            action_dim=int(env.action_dim),
            pop_size=int(cfg.cem.pop_size),
            elite_frac=float(cfg.cem.elite_frac),
            n_iter=int(cfg.cem.n_iter),
            init_std=float(cfg.cem.init_std),
            warm_start=bool(cfg.cem.warm_start),
            action_low=cfg.cem.action_low,
            action_high=cfg.cem.action_high,
            objective_cfg=OmegaConf.to_container(cfg.objective, resolve=True),
            fidelity_cfg=OmegaConf.to_container(cfg.fidelity, resolve=True),
            particle_env_cfg=OmegaConf.to_container(cfg.particle_env, resolve=True),
            device=device,
        )

    init_state, goal_state, sample_meta = _sample_init_goal_states(env, cfg, wm_cfg=wm_cfg)

    if str(sample_meta.get("source", "")).lower() == "dataset":
        gt_len = int(sample_meta.get("trajectory_len", -1))
        plan_steps = int(cfg.mpc.steps)
        if gt_len > 0 and gt_len != plan_steps:
            print(
                "[warn] Dataset GT trajectory length differs from plan.mpc.steps: "
                f"gt_len={gt_len}, planned_steps={plan_steps}. "
                "Set them equal for length-matched visual comparison."
            )
    print(f"[backend] {backend}")
    if wm is not None:
        print(f"[world_model] loaded from {run_desc}")
    else:
        print("[world_model] not used")
    if wm is not None and hasattr(wm, "num_members"):
        print(f"[ensemble] members={wm.num_members}")
    if wm is not None:
        print(f"[levels] K={wm.K}")
    elif backend == "gt_env":
        print(f"[levels] num_levels={int(cfg.fidelity.num_levels)} (gt_env)")
        print(f"[gt_env] objective_space={str(cfg.gt_env.objective_space).lower()}")
        print(f"[gt_env] progress={bool(cfg.gt_env.progress)}")
    else:
        print(f"[levels] num_levels={int(cfg.fidelity.num_levels)} (particle_sim)")
        print(f"[particle] objective_space={str(cfg.particle_env.objective_space).lower()}")
        print(f"[particle] spacings={list(cfg.particle_env.fidelity_env.spacings)}")
        print(f"[particle] progress={bool(cfg.particle_env.progress)}")

    success, traj, frames, planner_frames, run_stats = _run_closed_loop(
        env=env,
        wm=wm,
        planner=planner,
        backend=backend,
        cfg=cfg,
        init_state=init_state,
        goal_state=goal_state,
        device=device,
    )

    if bool(cfg.save):
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError:
            import imageio

        rollout_root = "rollouts"
        os.makedirs(rollout_root, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(rollout_root, f"plan_{backend}_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        source = str(sample_meta.get("source", "unknown"))
        out_path = os.path.join(run_dir, "planned.gif")
        print(f"[save] Writing rollout GIF to {out_path}")
        imageio.mimwrite(out_path, frames, fps=15)
        print(f"[save] Video saved ({len(frames)} frames)")

        meta = {
            "created_at": run_ts,
            "backend": backend,
            "source": source,
            "success": bool(success),
            "planned_steps": int(len(traj) - 1),
            "plans": int(run_stats["plans"]),
            "bits_used_total": int(run_stats["bits_used_total"]),
            "bits_used_total_human": _format_bits_human(int(run_stats["bits_used_total"])),
            "flops_used_total": int(run_stats["flops_used_total"]),
            "flops_used_total_human": _format_flops_human(int(run_stats["flops_used_total"])),
            "plan_time_total_sec": float(run_stats["plan_time_total_sec"]),
            "init_state": np.asarray(init_state, dtype=np.float32).tolist(),
            "goal_state": np.asarray(goal_state, dtype=np.float32).tolist(),
            "sample": sample_meta,
            "plan_config": OmegaConf.to_container(cfg, resolve=True),
        }
        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Metadata written to {meta_path}")

        if source == "dataset":
            gt_frames = _render_dataset_gt_frames(
                env=env,
                sample_meta=sample_meta,
                init_state=init_state,
                goal_state=goal_state,
            )
            if len(gt_frames) > 0:
                gt_path = os.path.join(run_dir, "gt.gif")
                print(f"[save] Writing dataset GT GIF to {gt_path}")
                imageio.mimwrite(gt_path, gt_frames, fps=15)
                print(f"[save] Dataset GT video saved ({len(gt_frames)} frames)")

        if len(planner_frames) > 0:
            planner_path = os.path.join(run_dir, "planner_view.gif")
            print(f"[save] Writing planner-view GIF to {planner_path}")
            imageio.mimwrite(planner_path, planner_frames, fps=15)
            print(f"[save] Planner-view video saved ({len(planner_frames)} frames)")

    print(
        f"[planning_stats] plans={run_stats['plans']}  "
        f"bits_used_total={run_stats['bits_used_total']} "
        f"({ _format_bits_human(int(run_stats['bits_used_total'])) })  "
        f"flops_used_total={run_stats['flops_used_total']} "
        f"({ _format_flops_human(int(run_stats['flops_used_total'])) })  "
        f"plan_time_total_sec={run_stats['plan_time_total_sec']:.3f}"
    )
    print("Reached goal:", success)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plan.py <path/to/config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
