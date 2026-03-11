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
import copy
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

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
    make_kwargs.setdefault("render_action", False)

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


def _load_world_model_member(wm_cfg: DictConfig, run_dir: str, device: torch.device, epoch: int) -> HierWorldModel:
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"World-model run directory not found: {run_dir}")

    wm = HierWorldModel(
        K=list[int](wm_cfg.model.K),
        D=int(wm_cfg.model.D),
        action_dim=int(wm_cfg.data.action_dim),
        decoder_mode=str(getattr(wm_cfg.model, "decoder_mode", "per_level")),
        dynamics_mode=str(getattr(wm_cfg.model, "dynamics_mode", "per_level")),
    ).to(device)

    enc_path = os.path.join(run_dir, f"encoder_epoch{epoch}.pt")
    if not os.path.isfile(enc_path):
        raise FileNotFoundError(f"Missing encoder checkpoint: {enc_path}")
    wm.encoder.load_state_dict(torch.load(enc_path, map_location=device))

    dynamics_mode = str(getattr(wm_cfg.model, "dynamics_mode", "per_level"))

    if dynamics_mode == "per_level":
        for li in range(len(wm.K)):
            dyn_path = os.path.join(run_dir, f"dyn_l{li}_epoch{epoch}.pt")
            if not os.path.isfile(dyn_path):
                raise FileNotFoundError(f"Missing dynamics checkpoint: {dyn_path}")
            wm.dynamics[li].load_state_dict(torch.load(dyn_path, map_location=device))
    else:
        dyn_path = os.path.join(run_dir, f"dyn_epoch{epoch}.pt")
        if not os.path.isfile(dyn_path):
            raise FileNotFoundError(f"Missing dynamics checkpoint: {dyn_path}")
        wm.dynamics.load_state_dict(torch.load(dyn_path, map_location=device))

    decoder_mode = str(getattr(wm_cfg.model, "decoder_mode", "per_level"))
    if decoder_mode == "per_level":
        for li in range(len(wm.K)):
            dec_path = os.path.join(run_dir, f"decoder_l{li}_epoch{epoch}.pt")
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
    wm = _load_world_model_member(wm_cfg, run_dir, device, epoch=plan_cfg.world_model.epoch)
    return wm, wm_cfg, run_dir, device


@torch.no_grad()
def _encode_visual(wm: object, img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = _image_to_model_tensor(img, device=device)
    return wm.encode(x)


def _sample_init_goal_states(
    env: PushTWrapper,
    cfg: DictConfig,
    wm_cfg: DictConfig | None,
    selection: Optional[dict] = None,
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
        reconstruct_goal_state=int(getattr(ds_cfg, "reconstruct_goal_state", 0)),
        selection=selection,
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


def _write_video_mp4(
    path: str,
    frames: list[np.ndarray],
    fps: int = 15,
    output_size: int = 256,
) -> None:
    if len(frames) == 0:
        raise ValueError(f"No frames to write for {path}")

    out_hw = int(output_size)
    if out_hw <= 0:
        raise ValueError(f"output_size must be > 0, got {output_size}")
    h = w = out_hw
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"avc1"),
        float(max(1, int(fps))),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open H.264/AVC MP4 writer for {path}. "
            "Ensure your OpenCV build has FFmpeg/x264 encoding support."
        )

    try:
        for fr in frames:
            x = np.asarray(fr)
            if x.dtype == object:
                x = x.astype(np.float32)
            if x.ndim == 2:
                x = x[:, :, None]
            if x.ndim != 3:
                raise ValueError(f"Video frame must have rank 3 (H,W,C), got shape {x.shape}")
            if x.shape[2] == 1:
                x = np.repeat(x, 3, axis=2)
            elif x.shape[2] != 3:
                raise ValueError(f"Expected 1 or 3 channels, got {x.shape[2]}")
            if x.shape[0] != h or x.shape[1] != w:
                if x.dtype != np.uint8:
                    x = np.clip(x, 0, 255).astype(np.uint8)
                x = cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST)
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


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
    #import pdb; pdb.set_trace()
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
    saved video frame size fixed for readability.
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
    state_dim = int(getattr(env, "state_dim", np.asarray(init_state).shape[0]))

    def _state_to_env_dim(x: np.ndarray) -> np.ndarray:
        s = np.asarray(x, dtype=np.float32)
        if s.shape[0] < state_dim:
            pad = state_dim - s.shape[0]
            s = np.concatenate([s, np.zeros((pad,), dtype=np.float32)], axis=0)
        return s[:state_dim]

    force_replay = bool(sample_meta.get("force_gt_action_replay", False)) or int(
        sample_meta.get("reconstruct_goal_state", 0)
    ) == 3
    actions = sample_meta.get("actions", None)
    gt_state_traj = sample_meta.get("gt_state_trajectory", None)
    if gt_state_traj is not None:
        state_seq = np.asarray(gt_state_traj, dtype=np.float32)
    elif actions is not None:
        try:
            init_roll = _state_to_env_dim(np.asarray(init_state, dtype=np.float32))
            _, states = env.rollout(
                seed=0,
                init_state=init_roll,
                actions=np.asarray(actions, dtype=np.float32),
            )
            state_seq = states
        except Exception as exc:
            if force_replay:
                raise RuntimeError(
                    f"Forced GT action replay failed while rendering gt.mp4: {exc}"
                ) from exc
            print(f"[save][gt] action-rollout fallback to dataset states ({exc}).")
            state_seq = state_arr
    else:
        if force_replay:
            raise ValueError(
                "Forced GT action replay requested but no replay actions/trajectory found in sample metadata."
            )
        state_seq = state_arr

    for s in state_seq:
        s = _state_to_env_dim(np.asarray(s, dtype=np.float32))
        env.prepare(seed=0, init_state=s)
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
) -> tuple[bool, list[np.ndarray], list[np.ndarray], list[np.ndarray], dict, dict]:
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
    executed_actions: list[np.ndarray] = []
    pos_diffs: list[float] = []
    angle_diffs: list[float] = []
    eef_diffs: list[float] = []
    coverages: list[float] = []
    metric_success_flags: list[bool] = []
    done_flags: list[bool] = []
    state_dists: list[float] = []
    replan_traces: list[dict[str, Any]] = []
    last_term: dict | None = None
    if bool(cfg.save):
        frames.append(env.render("rgb_array", include_start_pose=True))

    # Early exit only if reset state satisfies BOTH metric success and env-done coverage.
    initial_term = env.eval_termination(goal_state, cur_state, done=None, info=None)
    if bool(initial_term["success_and_done"]):
        cov_s = "n/a" if initial_term["coverage"] is None else f"{float(initial_term['coverage']):.4f}"
        print(
            "[terminate] reason=initial_metric_and_env_done "
            f"step=0 pos_diff={float(initial_term['pos_diff']):.3f} "
            f"angle_diff={float(initial_term['angle_diff']):.3f} "
            f"eef_diff={float(initial_term['eef_diff']):.3f} "
            f"coverage={cov_s}"
        )
        stats = {
            "plans": 0,
            "bits_used_total": 0,
            "flops_used_total": 0,
            "plan_time_total_sec": 0.0,
            "termination_reason": "initial_metric_and_env_done",
            "termination_step": 0,
            "termination_metric_success": bool(initial_term["success"]),
            "termination_done": bool(initial_term["done"]),
            "termination_pos_diff": float(initial_term["pos_diff"]),
            "termination_angle_diff": float(initial_term["angle_diff"]),
            "termination_eef_diff": float(initial_term["eef_diff"]),
            "termination_coverage": initial_term["coverage"],
        }
        trace = {
            "executed_actions": [],
            "trajectory": np.asarray(trajectory, dtype=np.float32).tolist(),
            "pos_diffs": [],
            "angle_diffs": [],
            "eef_diffs": [],
            "coverages": [],
            "metric_success_flags": [],
            "done_flags": [],
            "state_dists": [],
            "replans": [],
        }
        return True, trajectory, frames, planner_frames, stats, trace

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
        plan_seed = int(1009 * replan_idx + 7919 * t)
        if backend == "wm":
            z_cur_for_plan = _encode_visual(wm, obs["visual"], device)
            action_seq, info = planner.plan(
                z_cur_for_plan,
                z_goal,
                mpc_progress=mpc_progress,
                warm_start_steps=int(prev_exec_steps),
                seed=plan_seed,
            )
        elif backend == "gt_env":
            action_seq, info = planner.plan(
                init_state=cur_state,
                goal_state=goal_state,
                mpc_progress=mpc_progress,
                seed=plan_seed,
                warm_start_steps=int(prev_exec_steps),
                rng_seed=plan_seed,
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
                seed=plan_seed,
                warm_start_steps=int(prev_exec_steps),
                rng_seed=plan_seed,
            )

        planned_actions_np = np.asarray(action_seq.detach().cpu().numpy(), dtype=np.float32)
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

        replan_traces.append(
            {
                "replan_idx": int(replan_idx),
                "step_start": int(t),
                "mpc_progress": float(mpc_progress),
                "seed": int(plan_seed),
                "action_seq": planned_actions_np.tolist(),
                "base_level_idx": int(getattr(info, "base_level_idx", -1)),
                "rollout_level_indices": [int(x) for x in list(getattr(info, "rollout_level_indices", []))],
                "bits_used_estimate": bits_used,
                "flops_used_estimate": flops_used,
                "plan_time_sec": plan_time,
                "base_k": None if getattr(info, "base_k", None) is None else int(getattr(info, "base_k")),
                "base_spacing": None if getattr(info, "base_spacing", None) is None else float(getattr(info, "base_spacing")),
                "base_num_particles": None if getattr(info, "base_num_particles", None) is None else int(getattr(info, "base_num_particles")),
                "start_state": np.asarray(cur_state, dtype=np.float32).tolist(),
            }
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
                import pdb; pdb.set_trace()
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
            action = np.asarray(planned_actions_np[i], dtype=np.float32)
            executed_actions.append(action.copy())
            obs, _, done, step_info = env.step(action)
            cur_state = step_info["state"]
            trajectory.append(cur_state.copy())

            term = env.eval_termination(goal_state, cur_state, done=done, info=step_info)
            last_term = term
            pd = term["pos_diff"]
            ad = term["angle_diff"] 
            ed = term["eef_diff"]
            pos_diffs.append(pd)
            angle_diffs.append(ad)
            eef_diffs.append(ed)
            coverages.append(float(term["coverage"]) if term["coverage"] is not None else float("nan"))
            metric_success_flags.append(bool(term["success"]))
            done_flags.append(bool(term["done"]))
            dist = term["state_dist"]
            state_dists.append(float(dist))
            base_k = getattr(info, "base_k", None)
            base_spacing = getattr(info, "base_spacing", None)
            base_np = getattr(info, "base_num_particles", None)
            level_idx = int(getattr(info, "base_level_idx", -1))
            k_str = "-" if base_k is None else str(base_k)
            spacing_str = "-" if base_spacing is None else f"{float(base_spacing):.4f}"
            np_str = "-" if base_np is None else str(int(base_np))
            print(
                f"step {t + 1:03d}  dist {dist:6.1f}  "
                f"level_idx {level_idx}  pos_diff {pd:6.1f} angle_diff {ad:6.1f} eef_diff {ed:6.1f} k {k_str}  spacing {spacing_str}  n_particles {np_str}"
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

            # Require both geometric success and env terminal signal.
            if bool(term["success_and_done"]):
                term_reason = "metric_success_and_env_done"
                cov_s = "n/a" if term["coverage"] is None else f"{float(term['coverage']):.4f}"
                print(
                    f"[terminate] reason={term_reason} step={t + 1} "
                    f"metric_success={bool(term['success'])} done={bool(term['done'])} "
                    f"pos_diff={float(pd):.3f} angle_diff={float(ad):.3f} "
                    f"eef_diff={float(ed):.3f} coverage={cov_s}"
                )
                stats = {
                    "plans": n_plans,
                    "bits_used_total": total_plan_bits,
                    "flops_used_total": total_plan_flops,
                    "plan_time_total_sec": total_plan_time,
                    "termination_reason": str(term_reason),
                    "termination_step": int(t + 1),
                    "termination_metric_success": bool(term["success"]),
                    "termination_done": bool(term["done"]),
                    "termination_pos_diff": float(term["pos_diff"]),
                    "termination_angle_diff": float(term["angle_diff"]),
                    "termination_eef_diff": float(term["eef_diff"]),
                    "termination_coverage": term["coverage"],
                }
                trace = {
                    "executed_actions": np.asarray(executed_actions, dtype=np.float32).tolist(),
                    "trajectory": np.asarray(trajectory, dtype=np.float32).tolist(),
                    "pos_diffs": [float(x) for x in pos_diffs],
                    "angle_diffs": [float(x) for x in angle_diffs],
                    "eef_diffs": [float(x) for x in eef_diffs],
                    "coverages": [float(x) for x in coverages],
                    "metric_success_flags": [bool(x) for x in metric_success_flags],
                    "done_flags": [bool(x) for x in done_flags],
                    "state_dists": [float(x) for x in state_dists],
                    "replans": replan_traces,
                }
                return True, trajectory, frames, planner_frames, stats, trace

            t += 1

        prev_exec_steps = int(n_exec)
        replan_idx += 1

    cov_s = "n/a" if (last_term is None or last_term["coverage"] is None) else f"{float(last_term['coverage']):.4f}"
    if last_term is not None:
        print(
            "[terminate] reason=max_steps "
            f"step={int(steps)} metric_success={bool(last_term['success'])} "
            f"done={bool(last_term['done'])} pos_diff={float(last_term['pos_diff']):.3f} "
            f"angle_diff={float(last_term['angle_diff']):.3f} "
            f"eef_diff={float(last_term['eef_diff']):.3f} coverage={cov_s}"
        )
    else:
        print("[terminate] reason=max_steps step=0 metric_success=False done=False coverage=n/a")
    stats = {
        "plans": n_plans,
        "bits_used_total": total_plan_bits,
        "flops_used_total": total_plan_flops,
        "plan_time_total_sec": total_plan_time,
        "termination_reason": "max_steps",
        "termination_step": int(steps),
        "termination_metric_success": False if last_term is None else bool(last_term["success"]),
        "termination_done": False if last_term is None else bool(last_term["done"]),
        "termination_pos_diff": None if last_term is None else float(last_term["pos_diff"]),
        "termination_angle_diff": None if last_term is None else float(last_term["angle_diff"]),
        "termination_eef_diff": None if last_term is None else float(last_term["eef_diff"]),
        "termination_coverage": None if last_term is None else last_term["coverage"],
    }
    trace = {
        "executed_actions": np.asarray(executed_actions, dtype=np.float32).tolist(),
        "trajectory": np.asarray(trajectory, dtype=np.float32).tolist(),
        "pos_diffs": [float(x) for x in pos_diffs],
        "angle_diffs": [float(x) for x in angle_diffs],
        "eef_diffs": [float(x) for x in eef_diffs],
        "coverages": [float(x) for x in coverages],
        "metric_success_flags": [bool(x) for x in metric_success_flags],
        "done_flags": [bool(x) for x in done_flags],
        "state_dists": [float(x) for x in state_dists],
        "replans": replan_traces,
    }
    return False, trajectory, frames, planner_frames, stats, trace


def _plan_defaults() -> dict:
    return {
        "backend": "wm",  # wm | gt_env | particle_sim
        "env_id": "pusht",
        "env": {
            "with_velocity": True,
            "with_target": True,
            "add_noise": 0,
            "noise_std": 0.0,
            "render_size": 512,
        },
        "world_model": {
            "config_path": None,
            "run_dir": None,
            "checkpoint_root": "checkpoints_world",
            "device": "auto",
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
            "latent_metric": "l2",
            "terminal_weight": 1.0,
            "running_weight": 0.0,
            "action_l2_weight": 0.0,
        },
        "fidelity": {
            "enabled": True,
            "num_levels": 4,
            "mpc": {
                "mode": "linear",
                "level": "finest",
                "start_level": "coarsest",
                "end_level": "finest",
            },
            "cem": {
                "mode": "linear",
                "level": "base",
                "start_level": "base",
                "end_level": "finest",
            },
            "rollout": {
                "mode": "fixed",
                "level": "base",
                "start_level": "base",
                "end_level": "coarsest",
                "uncertainty": {
                    "criterion": "mean",
                    "threshold": 0.05,
                    "percentile": 0.8,
                    "min_level": "coarsest",
                    "max_downshifts_per_step": 1,
                },
            },
        },
        "gt_env": {
            "rollout_samples": 1,
            "objective_space": "image",
            "progress": True,
            "progress_leave": False,
            "fidelity_env": {
                "mode": "blur_avgpool",
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
            "objective_space": "state",
            "progress": True,
            "progress_leave": False,
            "fidelity_env": {
                "spacings": [0.020, 0.016, 0.013, 0.010],
                "device": "auto",
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
            "source": "random",
            "dataset": {
                "zarr_path": None,
                "split": "valid",
                "split_ratio": None,
                "trajectory_len": 20,
                "seed": 0,
                "reconstruct_goal_state": 0,
            },
        },
        "render": False,
        "save": False,
    }


def load_plan_cfg(cfg_path: str) -> DictConfig:
    cfg_root = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(_plan_defaults(), cfg_root.get("plan", cfg_root))
    validate_plan_cfg(cfg)
    cfg.init_goal.dataset.seed = _resolve_dataset_seed(getattr(cfg.init_goal.dataset, "seed", 0))
    return cfg


def _register_plan_env(cfg: DictConfig) -> None:
    try:
        register(
            id=str(cfg.env_id),
            entry_point="pusht.pusht_wrapper:PushTWrapper",
            max_episode_steps=300,
            reward_threshold=1.0,
        )
    except gym.error.Error:
        pass


def build_plan_runtime(cfg: DictConfig) -> dict:
    _register_plan_env(cfg)
    env_wrapped = _gym_make_versioned(str(cfg.env_id), cfg.env)
    env: PushTWrapper = _unwrap_env(env_wrapped)
    backend = str(cfg.backend).lower()
    device = _resolve_device(str(cfg.world_model.device))
    wm = None
    wm_cfg = None
    run_desc = backend

    if backend == "wm":
        wm, wm_cfg, run_desc, device = _load_world_model(cfg)
        model_action_dim = int(wm_cfg.data.action_dim)
        if int(env.action_dim) != model_action_dim:
            raise ValueError(
                f"Action-dim mismatch: env.action_dim={env.action_dim}, world-model action_dim={model_action_dim}"
            )
        if not bool(cfg.env.with_velocity):
            print("[warn] plan.env.with_velocity=false may mismatch the world model's training distribution.")
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
            render_size=int(getattr(env, "render_size", 512)),
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
    return {
        "env": env,
        "planner": planner,
        "wm": wm,
        "wm_cfg": wm_cfg,
        "backend": backend,
        "device": device,
        "run_desc": run_desc,
    }


def print_plan_runtime_summary(runtime: dict, cfg: DictConfig) -> None:
    backend = str(runtime["backend"])
    wm = runtime.get("wm", None)
    print(f"[backend] {backend}")
    if wm is not None:
        print(f"[world_model] loaded from {runtime['run_desc']}")
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


def load_selected_rollout(
    env: PushTWrapper,
    cfg: DictConfig,
    wm_cfg: DictConfig | None,
    selection: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if str(cfg.init_goal.source).lower() != "dataset":
        raise ValueError("Explicit rollout selection requires init_goal.source=dataset.")
    return _sample_init_goal_states(env, cfg, wm_cfg=wm_cfg, selection=selection)


def run_plan_session(
    cfg: DictConfig,
    rollout_selection: Optional[dict] = None,
    schedule_name: Optional[str] = None,
    print_summary: bool = True,
) -> dict:
    runtime = build_plan_runtime(cfg)
    env = runtime["env"]
    wm_cfg = runtime["wm_cfg"]
    if rollout_selection is None:
        init_state, goal_state, sample_meta = _sample_init_goal_states(env, cfg, wm_cfg=wm_cfg)
    else:
        init_state, goal_state, sample_meta = load_selected_rollout(env, cfg, wm_cfg=wm_cfg, selection=rollout_selection)
    if str(sample_meta.get("source", "")).lower() == "dataset":
        gt_len = int(sample_meta.get("trajectory_len", -1))
        plan_steps = int(cfg.mpc.steps)
        if gt_len > 0 and gt_len != plan_steps:
            print(
                "[warn] Dataset GT trajectory length differs from plan.mpc.steps: "
                f"gt_len={gt_len}, planned_steps={plan_steps}. "
                "Set them equal for length-matched visual comparison."
            )
    if bool(print_summary):
        print_plan_runtime_summary(runtime, cfg)
    planner = runtime["planner"]
    if hasattr(planner, "reset"):
        planner.reset()
    success, traj, frames, planner_frames, run_stats, trace = _run_closed_loop(
        env=runtime["env"],
        wm=runtime["wm"],
        planner=runtime["planner"],
        backend=runtime["backend"],
        cfg=cfg,
        init_state=init_state,
        goal_state=goal_state,
        device=runtime["device"],
    )
    return {
        "cfg": cfg,
        "runtime": runtime,
        "success": bool(success),
        "trajectory": traj,
        "frames": frames,
        "planner_frames": planner_frames,
        "run_stats": run_stats,
        "trace": trace,
        "init_state": np.asarray(init_state, dtype=np.float32),
        "goal_state": np.asarray(goal_state, dtype=np.float32),
        "sample_meta": sample_meta,
        "schedule_name": schedule_name,
    }


def _trace_arrays_from_trace(trace: dict) -> dict:
    replans = list(trace.get("replans", []))
    n_replans = len(replans)
    action_dim = 0
    horizon = 0
    state_dim = 0
    if len(trace.get("executed_actions", [])) > 0:
        action_dim = int(np.asarray(trace["executed_actions"], dtype=np.float32).shape[-1])
    if len(trace.get("trajectory", [])) > 0:
        state_dim = int(np.asarray(trace["trajectory"], dtype=np.float32).shape[-1])
    if n_replans > 0:
        horizon = int(np.asarray(replans[0]["action_seq"], dtype=np.float32).shape[0])
        if action_dim == 0:
            action_dim = int(np.asarray(replans[0]["action_seq"], dtype=np.float32).shape[-1])
        if state_dim == 0:
            state_dim = int(np.asarray(replans[0]["start_state"], dtype=np.float32).shape[-1])
    rollout_len_max = max((len(r.get("rollout_level_indices", [])) for r in replans), default=0)
    replan_action_seqs = np.zeros((n_replans, horizon, action_dim), dtype=np.float32)
    replan_start_states = np.zeros((n_replans, state_dim), dtype=np.float32)
    replan_rollout_levels = np.full((n_replans, rollout_len_max), -1, dtype=np.int32)
    replan_rollout_lengths = np.zeros((n_replans,), dtype=np.int32)
    replan_step_starts = np.zeros((n_replans,), dtype=np.int32)
    replan_mpc_progress = np.zeros((n_replans,), dtype=np.float32)
    replan_seeds = np.zeros((n_replans,), dtype=np.int64)
    replan_base_levels = np.full((n_replans,), -1, dtype=np.int32)
    replan_bits = np.zeros((n_replans,), dtype=np.int64)
    replan_flops = np.zeros((n_replans,), dtype=np.int64)
    replan_plan_times = np.zeros((n_replans,), dtype=np.float32)
    replan_base_ks = np.full((n_replans,), -1, dtype=np.int32)
    replan_base_num_particles = np.full((n_replans,), -1, dtype=np.int32)
    replan_base_spacings = np.full((n_replans,), np.nan, dtype=np.float32)
    for idx, replan in enumerate(replans):
        replan_action_seqs[idx] = np.asarray(replan["action_seq"], dtype=np.float32)
        replan_start_states[idx] = np.asarray(replan["start_state"], dtype=np.float32)
        rl = np.asarray(replan.get("rollout_level_indices", []), dtype=np.int32)
        replan_rollout_lengths[idx] = int(rl.shape[0])
        if rl.shape[0] > 0:
            replan_rollout_levels[idx, : rl.shape[0]] = rl
        replan_step_starts[idx] = int(replan.get("step_start", 0))
        replan_mpc_progress[idx] = float(replan.get("mpc_progress", 0.0))
        replan_seeds[idx] = int(replan.get("seed", 0))
        replan_base_levels[idx] = int(replan.get("base_level_idx", -1))
        replan_bits[idx] = int(replan.get("bits_used_estimate", 0))
        replan_flops[idx] = int(replan.get("flops_used_estimate", 0))
        replan_plan_times[idx] = float(replan.get("plan_time_sec", 0.0))
        if replan.get("base_k", None) is not None:
            replan_base_ks[idx] = int(replan["base_k"])
        if replan.get("base_num_particles", None) is not None:
            replan_base_num_particles[idx] = int(replan["base_num_particles"])
        if replan.get("base_spacing", None) is not None:
            replan_base_spacings[idx] = float(replan["base_spacing"])
    return {
        "executed_actions": np.asarray(trace.get("executed_actions", []), dtype=np.float32),
        "trajectory": np.asarray(trace.get("trajectory", []), dtype=np.float32),
        "pos_diffs": np.asarray(trace.get("pos_diffs", []), dtype=np.float32),
        "angle_diffs": np.asarray(trace.get("angle_diffs", []), dtype=np.float32),
        "eef_diffs": np.asarray(trace.get("eef_diffs", []), dtype=np.float32),
        "coverages": np.asarray(trace.get("coverages", []), dtype=np.float32),
        "metric_success_flags": np.asarray(trace.get("metric_success_flags", []), dtype=np.bool_),
        "done_flags": np.asarray(trace.get("done_flags", []), dtype=np.bool_),
        "state_dists": np.asarray(trace.get("state_dists", []), dtype=np.float32),
        "replan_action_seqs": replan_action_seqs,
        "replan_start_states": replan_start_states,
        "replan_rollout_levels": replan_rollout_levels,
        "replan_rollout_lengths": replan_rollout_lengths,
        "replan_step_starts": replan_step_starts,
        "replan_mpc_progress": replan_mpc_progress,
        "replan_seeds": replan_seeds,
        "replan_base_levels": replan_base_levels,
        "replan_bits": replan_bits,
        "replan_flops": replan_flops,
        "replan_plan_times": replan_plan_times,
        "replan_base_ks": replan_base_ks,
        "replan_base_num_particles": replan_base_num_particles,
        "replan_base_spacings": replan_base_spacings,
    }


def _save_error_curves(path: str, trace: dict) -> None:
    import matplotlib.pyplot as plt

    pos_diffs = list(trace.get("pos_diffs", []))
    angle_diffs = list(trace.get("angle_diffs", []))
    eef_diffs = list(trace.get("eef_diffs", []))
    if len(pos_diffs) == 0 and len(angle_diffs) == 0 and len(eef_diffs) == 0:
        return
    plt.xlabel("Step")
    plt.ylabel("Distance")
    plt.title("Positional, Angular and End-effector Distance vs Step")
    if len(pos_diffs) > 0:
        plt.plot(range(len(pos_diffs)), pos_diffs, label="pos_diffs")
    if len(angle_diffs) > 0:
        plt.plot(range(len(angle_diffs)), angle_diffs, label="angle_diffs")
    if len(eef_diffs) > 0:
        plt.plot(range(len(eef_diffs)), eef_diffs, label="eef_diffs")
    plt.legend()
    plt.savefig(path)
    plt.close()


def save_trace_bundle(run_dir: str, result: dict) -> tuple[str, str]:
    arrays = _trace_arrays_from_trace(result["trace"])
    arrays_path = os.path.join(run_dir, "trace.npz")
    np.savez_compressed(arrays_path, **arrays)
    meta = {
        "trace_version": 1,
        "backend": result["runtime"]["backend"],
        "success": bool(result["success"]),
        "schedule_name": result.get("schedule_name", None),
        "sample": result["sample_meta"],
        "run_stats": result["run_stats"],
        "init_state": np.asarray(result["init_state"], dtype=np.float32).tolist(),
        "goal_state": np.asarray(result["goal_state"], dtype=np.float32).tolist(),
        "plan_config": OmegaConf.to_container(result["cfg"], resolve=True),
        "arrays_file": "trace.npz",
        "replans": [
            {
                "replan_idx": int(replan.get("replan_idx", idx)),
                "step_start": int(replan.get("step_start", 0)),
                "mpc_progress": float(replan.get("mpc_progress", 0.0)),
                "seed": int(replan.get("seed", 0)),
                "base_level_idx": int(replan.get("base_level_idx", -1)),
                "rollout_level_indices": [int(x) for x in list(replan.get("rollout_level_indices", []))],
                "bits_used_estimate": int(replan.get("bits_used_estimate", 0)),
                "flops_used_estimate": int(replan.get("flops_used_estimate", 0)),
                "plan_time_sec": float(replan.get("plan_time_sec", 0.0)),
                "base_k": replan.get("base_k", None),
                "base_spacing": replan.get("base_spacing", None),
                "base_num_particles": replan.get("base_num_particles", None),
            }
            for idx, replan in enumerate(result["trace"].get("replans", []))
        ],
    }
    meta_path = os.path.join(run_dir, "trace.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path, arrays_path


def save_plan_result(
    result: dict,
    run_dir: str,
    save_media: bool = True,
) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    cfg = result["cfg"]
    backend = result["runtime"]["backend"]
    source = str(result["sample_meta"].get("source", "unknown"))
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_json_path, trace_npz_path = save_trace_bundle(run_dir, result)
    _save_error_curves(os.path.join(run_dir, "pos_diffs_angle_diffs_eef_diffs.png"), result["trace"])
    meta = {
        "created_at": created_at,
        "backend": backend,
        "source": source,
        "success": bool(result["success"]),
        "planned_steps": int(len(result["trajectory"]) - 1),
        "plans": int(result["run_stats"]["plans"]),
        "bits_used_total": int(result["run_stats"]["bits_used_total"]),
        "bits_used_total_human": _format_bits_human(int(result["run_stats"]["bits_used_total"])),
        "flops_used_total": int(result["run_stats"]["flops_used_total"]),
        "flops_used_total_human": _format_flops_human(int(result["run_stats"]["flops_used_total"])),
        "plan_time_total_sec": float(result["run_stats"]["plan_time_total_sec"]),
        "termination_reason": str(result["run_stats"].get("termination_reason", "unknown")),
        "termination_step": int(result["run_stats"].get("termination_step", -1)),
        "termination_metric_success": bool(result["run_stats"].get("termination_metric_success", False)),
        "termination_done": bool(result["run_stats"].get("termination_done", False)),
        "termination_pos_diff": result["run_stats"].get("termination_pos_diff", None),
        "termination_angle_diff": result["run_stats"].get("termination_angle_diff", None),
        "termination_eef_diff": result["run_stats"].get("termination_eef_diff", None),
        "termination_coverage": result["run_stats"].get("termination_coverage", None),
        "init_state": np.asarray(result["init_state"], dtype=np.float32).tolist(),
        "goal_state": np.asarray(result["goal_state"], dtype=np.float32).tolist(),
        "sample": result["sample_meta"],
        "schedule_name": result.get("schedule_name", None),
        "plan_config": OmegaConf.to_container(cfg, resolve=True),
        "trace_json": os.path.basename(trace_json_path),
        "trace_npz": os.path.basename(trace_npz_path),
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not save_media:
        return meta

    if len(result["frames"]) > 0:
        out_path = os.path.join(run_dir, "planned.mp4")
        print(f"[save] Writing rollout MP4 to {out_path}")
        _write_video_mp4(out_path, result["frames"], fps=15)
        print(f"[save] Video saved ({len(result['frames'])} frames)")

    if source == "dataset":
        gt_frames = _render_dataset_gt_frames(
            env=result["runtime"]["env"],
            sample_meta=result["sample_meta"],
            init_state=result["init_state"],
            goal_state=result["goal_state"],
        )
        if len(gt_frames) > 0:
            gt_path = os.path.join(run_dir, "gt.mp4")
            print(f"[save] Writing dataset GT MP4 to {gt_path}")
            _write_video_mp4(gt_path, gt_frames, fps=15)
            print(f"[save] Dataset GT video saved ({len(gt_frames)} frames)")

    if len(result["planner_frames"]) > 0:
        planner_path = os.path.join(run_dir, "planner_view.mp4")
        print(f"[save] Writing planner-view MP4 to {planner_path}")
        _write_video_mp4(planner_path, result["planner_frames"], fps=15)
        print(f"[save] Planner-view video saved ({len(result['planner_frames'])} frames)")
    return meta


def main(cfg_path: str) -> None:
    cfg = load_plan_cfg(cfg_path)
    result = run_plan_session(cfg)

    if bool(cfg.save):
        rollout_root = "rollouts"
        os.makedirs(rollout_root, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(rollout_root, f"plan_{result['runtime']['backend']}_{run_ts}")
        save_plan_result(result, run_dir, save_media=True)

    run_stats = result["run_stats"]
    print(
        f"[planning_stats] plans={run_stats['plans']}  "
        f"bits_used_total={run_stats['bits_used_total']} "
        f"({ _format_bits_human(int(run_stats['bits_used_total'])) })  "
        f"flops_used_total={run_stats['flops_used_total']} "
        f"({ _format_flops_human(int(run_stats['flops_used_total'])) })  "
        f"plan_time_total_sec={run_stats['plan_time_total_sec']:.3f}"
    )
    print("Reached goal:", result["success"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plan.py <path/to/config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
