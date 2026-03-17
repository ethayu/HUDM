from __future__ import annotations

import json
import os
import csv
from datetime import datetime
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from hudm.runtime import format_bits_human, format_flops_human
from hudm.session_helpers import (
    set_execution_fidelity_finest,
    set_goal_pose,
    set_start_pose,
)
from pusht.pusht_wrapper import PushTWrapper

DEFAULT_ACTION_REFERENCE_SIZE = 512.0


def resize_image_hw(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(img)
    h_t, w_t = int(target_hw[0]), int(target_hw[1])
    if out.shape[0] != h_t or out.shape[1] != w_t:
        out = cv2.resize(out, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
    return out


def write_video_mp4(
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
    try:
        writer = imageio.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=float(max(1, int(fps))),
            codec="libx264",
            pixelformat="yuv420p",
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
                writer.append_data(x)
        finally:
            writer.close()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write MP4 via imageio for {path}. "
            "Install project dependencies from requirements.txt so imageio ffmpeg support is available."
        ) from exc


def action_overlay_spec_from_env(
    env: object | None,
    *,
    action_format: str = "env_input",
) -> dict[str, Any]:
    if env is None:
        return {
            "action_format": str(action_format),
            "action_relative": True,
            "action_scale": 1.0,
            "reference_size": float(DEFAULT_ACTION_REFERENCE_SIZE),
        }
    return {
        "action_format": str(action_format),
        "action_relative": bool(getattr(env, "relative", True)),
        "action_scale": float(getattr(env, "action_scale", 1.0)),
        "reference_size": float(getattr(env, "window_size", DEFAULT_ACTION_REFERENCE_SIZE)),
    }


def infer_action_overlay_spec(
    trace_meta: dict[str, Any] | None,
    actions: np.ndarray | list,
    *,
    env: object | None = None,
) -> dict[str, Any]:
    spec = action_overlay_spec_from_env(env)
    meta = dict(trace_meta or {})
    if ("action_relative" in meta) and ("action_scale" in meta):
        spec["action_format"] = str(meta.get("action_format", "env_input"))
        spec["action_relative"] = bool(meta["action_relative"])
        spec["action_scale"] = float(meta["action_scale"])
        return spec

    action_arr = np.asarray(actions, dtype=np.float32)
    finite = action_arr[np.isfinite(action_arr)]
    if finite.size <= 0:
        return spec
    if np.any(finite < 0.0):
        return spec
    ref_size = float(spec["reference_size"])
    if np.all((finite >= 0.0) & (finite <= ref_size)):
        return {
            "action_format": "absolute_target",
            "action_relative": False,
            "action_scale": 1.0,
            "reference_size": ref_size,
        }
    return spec


def resolve_action_target(
    agent_pos: np.ndarray | list[float],
    action: np.ndarray | list[float],
    *,
    action_format: str,
    action_relative: bool,
    action_scale: float,
) -> np.ndarray:
    agent = np.asarray(agent_pos, dtype=np.float32).reshape(-1)
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    if agent.shape[0] < 2 or act.shape[0] < 2:
        raise ValueError(
            f"Action target resolution requires 2D agent/action, got agent={agent.shape}, action={act.shape}"
        )
    if str(action_format).strip().lower() == "absolute_target":
        return act[:2].astype(np.float32)
    if bool(action_relative):
        return (agent[:2] + act[:2] * float(action_scale)).astype(np.float32)
    return (act[:2] * float(action_scale)).astype(np.float32)


def _project_target_to_frame(
    target_xy: np.ndarray | list[float],
    frame_shape: tuple[int, ...],
    *,
    reference_size: float,
) -> tuple[int, int]:
    h = max(1, int(frame_shape[0]))
    w = max(1, int(frame_shape[1]))
    ref = max(1.0, float(reference_size))
    target = np.asarray(target_xy, dtype=np.float32).reshape(-1)
    if target.shape[0] < 2:
        raise ValueError(f"Target must be 2D, got shape {target.shape}")
    x = float(target[0]) / ref * float(max(1, w - 1))
    y = float(target[1]) / ref * float(max(1, h - 1))
    return int(np.clip(np.rint(x), 0, w - 1)), int(np.clip(np.rint(y), 0, h - 1))


def draw_action_target_cross(
    frame: np.ndarray,
    target_xy: np.ndarray | list[float],
    *,
    reference_size: float = DEFAULT_ACTION_REFERENCE_SIZE,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    out = np.asarray(frame).copy()
    if out.dtype == object:
        out = out.astype(np.float32)
    if out.ndim == 2:
        out = np.repeat(out[:, :, None], 3, axis=2)
    if out.ndim != 3:
        raise ValueError(f"Expected frame with rank 3, got shape {out.shape}")
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    if out.shape[2] != 3:
        raise ValueError(f"Expected RGB frame, got shape {out.shape}")
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    x, y = _project_target_to_frame(target_xy, out.shape, reference_size=reference_size)
    min_side = max(1, min(int(out.shape[0]), int(out.shape[1])))
    marker_size = max(6, int(round(min_side * (8.0 / 96.0))))
    thickness = max(1, int(round(min_side * (1.0 / 96.0))))
    cv2.drawMarker(
        out,
        (x, y),
        color=color,
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )
    return out


def overlay_action_targets_on_frames(
    frames: list[np.ndarray] | np.ndarray,
    states: list[np.ndarray] | np.ndarray,
    actions: list[np.ndarray] | np.ndarray,
    overlay_spec: dict[str, Any],
) -> list[np.ndarray]:
    out_frames = [np.asarray(frame).copy() for frame in list(frames)]
    if len(out_frames) <= 0:
        return out_frames

    state_arr = np.asarray(states, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    if state_arr.ndim < 2:
        return out_frames

    if action_arr.size <= 0 or action_arr.ndim < 2:
        out_frames[0] = draw_action_target_cross(
            out_frames[0],
            state_arr[0, :2],
            reference_size=float(overlay_spec.get("reference_size", DEFAULT_ACTION_REFERENCE_SIZE)),
        )
        return out_frames

    overlay_count = min(len(out_frames), int(state_arr.shape[0]), int(action_arr.shape[0]))
    for idx in range(overlay_count):
        agent_pos = state_arr[idx, :2]
        target_xy = resolve_action_target(
            agent_pos,
            action_arr[idx],
            action_format=str(overlay_spec.get("action_format", "env_input")),
            action_relative=bool(overlay_spec.get("action_relative", True)),
            action_scale=float(overlay_spec.get("action_scale", 1.0)),
        )
        out_frames[idx] = draw_action_target_cross(
            out_frames[idx],
            target_xy,
            reference_size=float(overlay_spec.get("reference_size", DEFAULT_ACTION_REFERENCE_SIZE)),
        )
    return out_frames


def overlay_start_pose(
    planner_img: np.ndarray,
    exec_with_start: np.ndarray,
    exec_without_start: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    out = resize_image_hw(np.asarray(planner_img), target_hw).copy()
    with_start = resize_image_hw(np.asarray(exec_with_start), target_hw)
    without_start = resize_image_hw(np.asarray(exec_without_start), target_hw)
    mask = np.any(with_start != without_start, axis=-1)
    out[mask] = with_start[mask]
    return out


def rollout_level_for_exec_step(info: object, exec_step_in_replan: int) -> int:
    levels = list(getattr(info, "rollout_level_indices", []))
    if exec_step_in_replan < len(levels):
        return int(levels[exec_step_in_replan])
    base = getattr(info, "base_level_idx", None)
    if base is None:
        return 0
    return int(base)


def planner_view_frame(
    env: PushTWrapper,
    base_visual: np.ndarray,
    level_idx: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    img = np.asarray(base_visual)
    out = img
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
    return resize_image_hw(out, target_hw)


def wm_decode_frame(
    wm: object,
    z: torch.Tensor,
    level_idx: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    decoder = wm if hasattr(wm, "decode") else getattr(wm, "primary", None)
    if decoder is None or not hasattr(decoder, "decode"):
        raise ValueError("World-model backend does not expose a decode(level, z) API.")

    n_levels = len(getattr(decoder, "K", []))
    li = int(level_idx)
    if n_levels > 0:
        li = max(0, min(li, n_levels - 1))

    z_in = z.unsqueeze(0) if z.ndim == 1 else z
    x = decoder.decode(li, z_in)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Decoded frame must be rank-3, got shape {tuple(x.shape)}")

    x = (x * 0.5 + 0.5).clamp(0.0, 1.0).detach().cpu()
    if int(x.shape[0]) in {1, 3} and int(x.shape[-1]) not in {1, 3}:
        x = x.permute(1, 2, 0)
    img = (x.numpy() * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return resize_image_hw(img, target_hw)


def particle_planner_view_frame(
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
        if hasattr(particle_backend, "_start_state"):
            particle_backend._start_state = np.asarray(start_state, dtype=np.float32).copy()
        img = particle_backend.render("rgb_array", include_start_pose=True)
    finally:
        particle_backend.set_planning_fidelity_level(prev_idx)
    return resize_image_hw(np.asarray(img), target_hw)


def render_dataset_gt_frames(
    env: PushTWrapper,
    sample_meta: dict[str, Any],
    init_state: np.ndarray,
    goal_state: np.ndarray,
) -> list[np.ndarray]:
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
    set_execution_fidelity_finest(env)
    set_start_pose(env, init_state)
    state_dim = int(getattr(env, "state_dim", np.asarray(init_state).shape[0]))

    def _state_to_env_dim(x: np.ndarray) -> np.ndarray:
        s = np.asarray(x, dtype=np.float32)
        if s.shape[0] < state_dim:
            s = np.concatenate([s, np.zeros((state_dim - s.shape[0],), dtype=np.float32)], axis=0)
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
        set_goal_pose(env, goal_state)
        frames.append(env.render("rgb_array", include_start_pose=True))
    if actions is not None:
        action_arr = np.asarray(actions, dtype=np.float32)
        if action_arr.ndim >= 2 and state_seq.shape[0] >= action_arr.shape[0]:
            frames = overlay_action_targets_on_frames(
                frames=frames,
                states=np.asarray(state_seq[: action_arr.shape[0]], dtype=np.float32),
                actions=action_arr,
                overlay_spec=action_overlay_spec_from_env(env),
            )
    return frames


def trace_arrays_from_trace(trace: dict) -> dict:
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
    latent_len_max = max((len(r.get("rollout_latent_losses", [])) for r in replans), default=0)
    replan_action_seqs = np.zeros((n_replans, horizon, action_dim), dtype=np.float32)
    replan_start_states = np.zeros((n_replans, state_dim), dtype=np.float32)
    replan_rollout_levels = np.full((n_replans, rollout_len_max), -1, dtype=np.int32)
    replan_rollout_lengths = np.zeros((n_replans,), dtype=np.int32)
    replan_rollout_latent_losses = np.full((n_replans, latent_len_max), np.nan, dtype=np.float32)
    replan_rollout_latent_lengths = np.zeros((n_replans,), dtype=np.int32)
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
        latent_losses = np.asarray(replan.get("rollout_latent_losses", []), dtype=np.float32)
        replan_rollout_latent_lengths[idx] = int(latent_losses.shape[0])
        if latent_losses.shape[0] > 0:
            replan_rollout_latent_losses[idx, : latent_losses.shape[0]] = latent_losses
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
        "replan_rollout_latent_losses": replan_rollout_latent_losses,
        "replan_rollout_latent_lengths": replan_rollout_latent_lengths,
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


def save_error_curves(path: str, trace: dict) -> None:
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
        angle_diffs_scaled = [100.0 * float(x) for x in angle_diffs]
        plt.plot(range(len(angle_diffs_scaled)), angle_diffs_scaled, label="angle_diffs_x100")
    if len(eef_diffs) > 0:
        plt.plot(range(len(eef_diffs)), eef_diffs, label="eef_diffs")
    plt.legend()
    plt.savefig(path)
    plt.close()


def save_termination_loss_curve(path: str, trace: dict, backend: str) -> None:
    import matplotlib.pyplot as plt

    losses = list(trace.get("state_dists", []))
    if len(losses) == 0:
        return
    plt.xlabel("Step")
    plt.ylabel("Loss")
    if str(backend).lower() == "wm":
        plt.title("Termination Latent Loss (z_cur vs z_goal) vs Step")
        label = "latent_loss"
    else:
        plt.title("Termination State Distance vs Step")
        label = "state_dist"
    plt.plot(range(len(losses)), losses, label=label)
    plt.legend()
    plt.savefig(path)
    plt.close()


def save_cem_rollout_latent_loss_curves(path: str, trace: dict, backend: str) -> None:
    import matplotlib.pyplot as plt

    if str(backend).lower() != "wm":
        return
    replans = list(trace.get("replans", []))
    if len(replans) == 0:
        return
    plotted = 0
    for replan in replans:
        per_iter = list(replan.get("iter_best_rollout_latent_losses", []))
        if len(per_iter) > 0:
            for it_idx, losses in enumerate(per_iter):
                if len(losses) <= 0:
                    continue
                plt.plot(
                    range(len(losses)),
                    losses,
                    alpha=0.35,
                    label=f"replan:it{it_idx}",
                )
                plotted += 1
            continue
        losses = list(replan.get("rollout_latent_losses", []))
        if len(losses) <= 0:
            continue
        plt.plot(range(len(losses)), losses, alpha=0.5, label=f"replan")
        plotted += 1
    if plotted <= 0:
        return
    plt.xlabel("Horizon Step")
    plt.ylabel("Latent Loss")
    plt.title("CEM Rollout Latent Loss (z_t vs z_goal) per Replan")
    if plotted <= 12:
        plt.legend()
    plt.savefig(path)
    plt.close()


def save_coverage_curve(path: str, trace: dict) -> None:
    import matplotlib.pyplot as plt

    cov = np.asarray(trace.get("coverages", []), dtype=np.float32)
    if cov.size <= 0:
        return
    mask = np.isfinite(cov)
    if not bool(np.any(mask)):
        return
    x = np.arange(cov.shape[0], dtype=np.int32)
    plt.xlabel("Step")
    plt.ylabel("Coverage")
    plt.title("Coverage vs Step")
    plt.plot(x[mask], cov[mask], label="coverage")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.savefig(path)
    plt.close()


def save_step_metrics_csv(path: str, trace: dict) -> None:
    pos_diffs = list(trace.get("pos_diffs", []))
    angle_diffs = list(trace.get("angle_diffs", []))
    eef_diffs = list(trace.get("eef_diffs", []))
    n = max(len(pos_diffs), len(angle_diffs), len(eef_diffs))
    if n <= 0:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "pos_diff", "angle_diff", "eef_diff"])
        for i in range(n):
            pos_v = float(pos_diffs[i]) if i < len(pos_diffs) else float("nan")
            ang_v = float(angle_diffs[i]) if i < len(angle_diffs) else float("nan")
            eef_v = float(eef_diffs[i]) if i < len(eef_diffs) else float("nan")
            writer.writerow([i, pos_v, ang_v, eef_v])


def save_trace_bundle(run_dir: str, result: dict) -> tuple[str, str]:
    arrays = trace_arrays_from_trace(result["trace"])
    arrays_path = os.path.join(run_dir, "trace.npz")
    np.savez_compressed(arrays_path, **arrays)
    action_overlay = action_overlay_spec_from_env(result["runtime"].get("env", None))
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
        "action_format": str(action_overlay["action_format"]),
        "action_relative": bool(action_overlay["action_relative"]),
        "action_scale": float(action_overlay["action_scale"]),
        "replans": [
            {
                "replan_idx": int(replan.get("replan_idx", idx)),
                "step_start": int(replan.get("step_start", 0)),
                "mpc_progress": float(replan.get("mpc_progress", 0.0)),
                "seed": int(replan.get("seed", 0)),
                "action_horizon": int(len(replan.get("action_seq", []))),
                "base_level_idx": int(replan.get("base_level_idx", -1)),
                "rollout_level_indices": [int(x) for x in list(replan.get("rollout_level_indices", []))],
                "rollout_latent_losses": [float(x) for x in list(replan.get("rollout_latent_losses", []))],
                "iter_best_rollout_latent_losses": [
                    [float(y) for y in list(x)]
                    for x in list(replan.get("iter_best_rollout_latent_losses", []))
                ],
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
    save_error_curves(os.path.join(run_dir, "pos_diffs_angle_diffs_eef_diffs.png"), result["trace"])
    save_termination_loss_curve(
        os.path.join(run_dir, "termination_loss.png"),
        result["trace"],
        backend=str(backend),
    )
    save_cem_rollout_latent_loss_curves(
        os.path.join(run_dir, "cem_rollout_latent_losses.png"),
        result["trace"],
        backend=str(backend),
    )
    save_coverage_curve(
        os.path.join(run_dir, "coverage_vs_step.png"),
        result["trace"],
    )
    save_step_metrics_csv(
        os.path.join(run_dir, "step_metrics.csv"),
        result["trace"],
    )
    action_overlay = action_overlay_spec_from_env(result["runtime"].get("env", None))
    meta = {
        "created_at": created_at,
        "backend": backend,
        "source": source,
        "success": bool(result["success"]),
        "planned_steps": int(len(result["trajectory"]) - 1),
        "plans": int(result["run_stats"]["plans"]),
        "bits_used_total": int(result["run_stats"]["bits_used_total"]),
        "bits_used_total_human": format_bits_human(int(result["run_stats"]["bits_used_total"])),
        "flops_used_total": int(result["run_stats"]["flops_used_total"]),
        "flops_used_total_human": format_flops_human(int(result["run_stats"]["flops_used_total"])),
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
        "action_format": str(action_overlay["action_format"]),
        "action_relative": bool(action_overlay["action_relative"]),
        "action_scale": float(action_overlay["action_scale"]),
        "trace_json": os.path.basename(trace_json_path),
        "trace_npz": os.path.basename(trace_npz_path),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not save_media:
        return meta

    if len(result["frames"]) > 0:
        rollout_frames = list(result["frames"])
        executed_actions = np.asarray(result["trace"].get("executed_actions", []), dtype=np.float32)
        trajectory = np.asarray(result["trace"].get("trajectory", []), dtype=np.float32)
        if executed_actions.ndim >= 2 and trajectory.ndim >= 2:
            rollout_frames = overlay_action_targets_on_frames(
                frames=rollout_frames,
                states=trajectory,
                actions=executed_actions,
                overlay_spec=action_overlay,
            )
        out_path = os.path.join(run_dir, "planned.mp4")
        print(f"[save] Writing rollout MP4 to {out_path}")
        write_video_mp4(out_path, rollout_frames, fps=15)
        print(f"[save] Video saved ({len(rollout_frames)} frames)")

    if source == "dataset":
        gt_frames = render_dataset_gt_frames(
            env=result["runtime"]["env"],
            sample_meta=result["sample_meta"],
            init_state=result["init_state"],
            goal_state=result["goal_state"],
        )
        if len(gt_frames) > 0:
            gt_path = os.path.join(run_dir, "gt.mp4")
            print(f"[save] Writing dataset GT MP4 to {gt_path}")
            write_video_mp4(gt_path, gt_frames, fps=15)
            print(f"[save] Dataset GT video saved ({len(gt_frames)} frames)")

    if len(result["planner_frames"]) > 0:
        planner_frames = list(result["planner_frames"])
        executed_actions = np.asarray(result["trace"].get("executed_actions", []), dtype=np.float32)
        trajectory = np.asarray(result["trace"].get("trajectory", []), dtype=np.float32)
        if executed_actions.ndim >= 2 and trajectory.ndim >= 2:
            planner_frames = overlay_action_targets_on_frames(
                frames=planner_frames,
                states=trajectory,
                actions=executed_actions,
                overlay_spec=action_overlay,
            )
        planner_path = os.path.join(run_dir, "planner_view.mp4")
        print(f"[save] Writing planner-view MP4 to {planner_path}")
        write_video_mp4(planner_path, planner_frames, fps=15)
        print(f"[save] Planner-view video saved ({len(planner_frames)} frames)")
    return meta
