from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import plan as single_plan
from hudm.artifacts import infer_action_overlay_spec, overlay_action_targets_on_frames


MEDIA_ALIASES = {
    "gt_replay": ["gt_replay.mp4", "gt.mp4"],
    "closed_loop_replay": ["closed_loop_replay.mp4", "planned.mp4"],
    "planner_view_replay": ["planner_view_replay.mp4", "planner_view.mp4"],
    "predicted_backend_replay": ["predicted_backend_replay.mp4"],
}


def _load_trace_dir(trace_dir: str) -> tuple[dict, dict]:
    trace_json = os.path.join(trace_dir, "trace.json")
    trace_npz = os.path.join(trace_dir, "trace.npz")
    if not os.path.isfile(trace_json) or not os.path.isfile(trace_npz):
        raise FileNotFoundError(f"Trace bundle not found in {trace_dir}")
    with open(trace_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with np.load(trace_npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return meta, arrays


def _discover_trace_dirs(run_dir: str) -> list[str]:
    trace_dirs = []
    if os.path.isfile(os.path.join(run_dir, "trace.json")):
        trace_dirs.append(run_dir)
    traces_root = os.path.join(run_dir, "traces")
    if os.path.isdir(traces_root):
        for schedule_name in sorted(os.listdir(traces_root)):
            schedule_dir = os.path.join(traces_root, schedule_name)
            if not os.path.isdir(schedule_dir):
                continue
            for rollout_id in sorted(os.listdir(schedule_dir)):
                trace_dir = os.path.join(schedule_dir, rollout_id)
                if os.path.isfile(os.path.join(trace_dir, "trace.json")):
                    trace_dirs.append(trace_dir)
    return trace_dirs


def _resolve_trace_dir(run_dir: str, schedule: str | None, rollout_id: str | None) -> str:
    if os.path.isfile(os.path.join(run_dir, "trace.json")):
        return run_dir
    if schedule is None or rollout_id is None:
        raise ValueError("Trace bundles under planner_eval runs require both --schedule and --rollout-id.")
    trace_dir = os.path.join(run_dir, "traces", schedule, rollout_id)
    if not os.path.isfile(os.path.join(trace_dir, "trace.json")):
        raise FileNotFoundError(f"Trace bundle not found: {trace_dir}")
    return trace_dir


def _existing_media_path(trace_dir: str, media: str) -> str | None:
    for filename in MEDIA_ALIASES[media]:
        path = os.path.join(trace_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def _build_runtime(meta: dict):
    cfg = OmegaConf.create(meta["plan_config"])
    runtime = single_plan.build_plan_runtime(cfg)
    return cfg, runtime


def _render_closed_loop_replay(trace_dir: str, meta: dict, arrays: dict, *, output_dir: str | None = None) -> str:
    media_dir = output_dir or trace_dir
    existing = _existing_media_path(media_dir, "closed_loop_replay")
    if existing is not None:
        return existing
    cfg, runtime = _build_runtime(meta)
    env = runtime["env"]
    init_state = np.asarray(meta["init_state"], dtype=np.float32)
    goal_state = np.asarray(meta["goal_state"], dtype=np.float32)
    actions = np.asarray(arrays["executed_actions"], dtype=np.float32)
    single_plan._set_start_pose(env, init_state)
    env.prepare(seed=0, init_state=init_state, goal_state=goal_state)
    single_plan._set_execution_fidelity_finest(env)
    frames = [env.render("rgb_array", include_start_pose=True)]
    for action in actions:
        env.step(action)
        frames.append(env.render("rgb_array", include_start_pose=True))
    trajectory = np.asarray(arrays.get("trajectory", []), dtype=np.float32)
    overlay_spec = infer_action_overlay_spec(meta, actions, env=env)
    frames = overlay_action_targets_on_frames(frames, trajectory, actions, overlay_spec)
    os.makedirs(media_dir, exist_ok=True)
    out_path = os.path.join(media_dir, "closed_loop_replay.mp4")
    single_plan._write_video_mp4(out_path, frames, fps=15)
    return out_path


def _render_gt_replay(trace_dir: str, meta: dict, arrays: dict, *, output_dir: str | None = None) -> str:
    del arrays
    media_dir = output_dir or trace_dir
    existing = _existing_media_path(media_dir, "gt_replay")
    if existing is not None:
        return existing
    _, runtime = _build_runtime(meta)
    frames = single_plan._render_dataset_gt_frames(
        env=runtime["env"],
        sample_meta=meta["sample"],
        init_state=np.asarray(meta["init_state"], dtype=np.float32),
        goal_state=np.asarray(meta["goal_state"], dtype=np.float32),
    )
    if len(frames) <= 0:
        raise ValueError("Unable to render gt_replay; trace does not correspond to a dataset rollout.")
    os.makedirs(media_dir, exist_ok=True)
    out_path = os.path.join(media_dir, "gt_replay.mp4")
    single_plan._write_video_mp4(out_path, frames, fps=15)
    return out_path


def _levels_for_steps(arrays: dict, total_steps: int) -> list[int]:
    levels = []
    replan_levels = np.asarray(arrays["replan_rollout_levels"], dtype=np.int32)
    replan_lengths = np.asarray(arrays["replan_rollout_lengths"], dtype=np.int32)
    replan_starts = np.asarray(arrays["replan_step_starts"], dtype=np.int32)
    for step_idx in range(total_steps):
        level_idx = 0
        for r_idx in range(replan_starts.shape[0]):
            start = int(replan_starts[r_idx])
            length = int(replan_lengths[r_idx])
            if start <= step_idx < start + max(1, length):
                exec_step = min(length - 1, max(0, step_idx - start))
                level_idx = int(replan_levels[r_idx, exec_step]) if length > 0 else 0
        levels.append(level_idx)
    return levels


def _render_gt_env_planner_frames(
    env,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    actions: np.ndarray,
    levels: list[int],
) -> tuple[list[np.ndarray], np.ndarray]:
    obs, cur_state = env.prepare(seed=0, init_state=start_state, goal_state=goal_state)
    single_plan._set_execution_fidelity_finest(env)
    obs["visual"] = env.render("rgb_array", include_start_pose=False)
    frame_with_start = env.render("rgb_array", include_start_pose=True)

    level_idx = levels[0] if len(levels) > 0 else 0
    frames = [
        single_plan._overlay_start_pose(
            planner_img=single_plan._planner_view_frame(
                env=env,
                base_visual=np.asarray(obs["visual"]),
                level_idx=level_idx,
                target_hw=frame_with_start.shape[:2],
            ),
            exec_with_start=np.asarray(frame_with_start),
            exec_without_start=np.asarray(obs["visual"]),
            target_hw=frame_with_start.shape[:2],
        )
    ]
    replay_states = [np.asarray(cur_state, dtype=np.float32).copy()]

    for step_idx, action in enumerate(np.asarray(actions, dtype=np.float32)):
        obs, _, _, step_info = env.step(action)
        cur_state = np.asarray(step_info["state"], dtype=np.float32).copy()
        replay_states.append(cur_state)
        frame_with_start = env.render("rgb_array", include_start_pose=True)
        level_idx = levels[min(step_idx, max(0, len(levels) - 1))] if len(levels) > 0 else 0
        frames.append(
            single_plan._overlay_start_pose(
                planner_img=single_plan._planner_view_frame(
                    env=env,
                    base_visual=np.asarray(obs["visual"]),
                    level_idx=level_idx,
                    target_hw=frame_with_start.shape[:2],
                ),
                exec_with_start=np.asarray(frame_with_start),
                exec_without_start=np.asarray(obs["visual"]),
                target_hw=frame_with_start.shape[:2],
            )
        )

    return frames, np.asarray(replay_states, dtype=np.float32)


def _render_planner_view_replay(trace_dir: str, meta: dict, arrays: dict, *, output_dir: str | None = None) -> str:
    media_dir = output_dir or trace_dir
    existing = _existing_media_path(media_dir, "planner_view_replay")
    if existing is not None:
        return existing
    cfg, runtime = _build_runtime(meta)
    backend = str(runtime["backend"])
    env = runtime["env"]
    wm = runtime.get("wm", None)
    trajectory = np.asarray(arrays["trajectory"], dtype=np.float32)
    if trajectory.shape[0] <= 0:
        raise ValueError("Trace trajectory is empty.")
    goal_state = np.asarray(meta["goal_state"], dtype=np.float32)
    init_state = np.asarray(meta["init_state"], dtype=np.float32)
    executed_actions = np.asarray(arrays.get("executed_actions", []), dtype=np.float32)
    levels = _levels_for_steps(arrays, total_steps=max(1, trajectory.shape[0] - 1))
    frames = []
    overlay_states = trajectory
    particle_backend = None
    if backend == "particle_sim":
        particle_backend = getattr(runtime["planner"], "backend", None)
    if backend == "gt_env":
        frames, overlay_states = _render_gt_env_planner_frames(
            env=env,
            start_state=np.asarray(trajectory[0], dtype=np.float32),
            goal_state=goal_state,
            actions=executed_actions,
            levels=levels,
        )
    else:
        for state_idx, state in enumerate(trajectory):
            level_idx = levels[min(max(0, state_idx - 1), max(0, len(levels) - 1))]
            if backend == "wm":
                obs, _ = env.prepare(seed=0, init_state=state, goal_state=goal_state)
                single_plan._set_execution_fidelity_finest(env)
                obs["visual"] = env.render("rgb_array", include_start_pose=False)
                z = single_plan._encode_visual(wm, obs["visual"], runtime["device"])
                frame = single_plan._wm_decode_frame(
                    wm=wm,
                    z=z,
                    level_idx=level_idx,
                    target_hw=np.asarray(obs["visual"]).shape[:2],
                )
                frames.append(frame)
            else:
                frame = single_plan._particle_planner_view_frame(
                    particle_backend=particle_backend,
                    start_state=init_state,
                    cur_state=state,
                    goal_state=goal_state,
                    level_idx=level_idx,
                    target_hw=(int(cfg.env.render_size), int(cfg.env.render_size)),
                )
                frames.append(frame)
    overlay_spec = infer_action_overlay_spec(meta, executed_actions, env=env)
    frames = overlay_action_targets_on_frames(frames, overlay_states, executed_actions, overlay_spec)
    os.makedirs(media_dir, exist_ok=True)
    out_path = os.path.join(media_dir, "planner_view_replay.mp4")
    single_plan._write_video_mp4(out_path, frames, fps=15)
    return out_path


def _render_predicted_backend_replay(trace_dir: str, meta: dict, arrays: dict, *, output_dir: str | None = None) -> str:
    media_dir = output_dir or trace_dir
    existing = _existing_media_path(media_dir, "predicted_backend_replay")
    if existing is not None:
        return existing
    cfg, runtime = _build_runtime(meta)
    backend = str(runtime["backend"])
    env = runtime["env"]
    goal_state = np.asarray(meta["goal_state"], dtype=np.float32)
    frames = []
    replan_action_seqs = np.asarray(arrays["replan_action_seqs"], dtype=np.float32)
    replan_start_states = np.asarray(arrays["replan_start_states"], dtype=np.float32)
    trajectory = np.asarray(arrays.get("trajectory", []), dtype=np.float32)
    replan_rollout_levels = np.asarray(arrays["replan_rollout_levels"], dtype=np.int32)
    replan_rollout_lengths = np.asarray(arrays["replan_rollout_lengths"], dtype=np.int32)
    replan_step_starts = np.asarray(arrays.get("replan_step_starts", []), dtype=np.int32)
    overlay_spec = infer_action_overlay_spec(meta, replan_action_seqs, env=env)

    if backend == "wm":
        wm = runtime["wm"]
        planner = runtime["planner"]
        for r_idx in range(replan_action_seqs.shape[0]):
            start_state = replan_start_states[r_idx]
            obs, _ = env.prepare(seed=0, init_state=start_state, goal_state=goal_state)
            obs["visual"] = env.render("rgb_array", include_start_pose=False)
            z = single_plan._encode_visual(wm, obs["visual"], runtime["device"])
            z_cur = z.clone()
            horizon = int(replan_rollout_lengths[r_idx])
            if horizon <= 0:
                continue
            seg_frames = []
            seg_states = [np.asarray(start_state, dtype=np.float32).copy()]
            seg_actions = np.asarray(replan_action_seqs[r_idx, :horizon], dtype=np.float32)
            for t in range(horizon):
                level_idx = int(replan_rollout_levels[r_idx, t])
                seg_frames.append(
                    single_plan._wm_decode_frame(
                        wm=wm,
                        z=z_cur,
                        level_idx=level_idx,
                        target_hw=(int(cfg.env.render_size), int(cfg.env.render_size)),
                    )
                )
                action_t = torch.as_tensor(replan_action_seqs[r_idx, t : t + 1], dtype=torch.float32, device=runtime["device"])
                z_next_k, _ = planner._predict_next_stats(level_idx, z_cur, action_t)
                k = int(planner.K[level_idx])
                z_next = z_cur.clone()
                z_next[:, :k] = z_next_k
                if k < planner.D:
                    z_next[:, k:] = 0.0
                z_cur = z_next
                _, _, _, step_info = env.step(replan_action_seqs[r_idx, t])
                seg_states.append(np.asarray(step_info["state"], dtype=np.float32).copy())
            last_level = int(replan_rollout_levels[r_idx, max(0, horizon - 1)])
            seg_frames.append(
                single_plan._wm_decode_frame(
                    wm=wm,
                    z=z_cur,
                    level_idx=last_level,
                    target_hw=(int(cfg.env.render_size), int(cfg.env.render_size)),
                )
            )
            frames.extend(overlay_action_targets_on_frames(seg_frames, seg_states, seg_actions, overlay_spec))
    elif backend == "gt_env":
        for r_idx in range(replan_action_seqs.shape[0]):
            start_state = replan_start_states[r_idx]
            step_start = int(replan_step_starts[r_idx]) if r_idx < replan_step_starts.shape[0] else 0
            if trajectory.ndim >= 2 and 0 <= step_start < trajectory.shape[0]:
                start_state = np.asarray(trajectory[step_start], dtype=np.float32)
            _, cur_state = env.prepare(seed=0, init_state=start_state, goal_state=goal_state)
            horizon = int(replan_rollout_lengths[r_idx])
            if horizon <= 0:
                continue
            seg_frames = []
            seg_states = [np.asarray(cur_state, dtype=np.float32).copy()]
            seg_actions = np.asarray(replan_action_seqs[r_idx, :horizon], dtype=np.float32)
            first_level = int(replan_rollout_levels[r_idx, 0])
            env.set_planning_fidelity_level(first_level)
            seg_frames.append(env.render("rgb_array", include_start_pose=True))
            for t in range(horizon):
                level_idx = int(replan_rollout_levels[r_idx, t])
                env.set_planning_fidelity_level(level_idx)
                _, _, _, step_info = env.step(replan_action_seqs[r_idx, t])
                seg_states.append(np.asarray(step_info["state"], dtype=np.float32).copy())
                next_level = int(replan_rollout_levels[r_idx, min(t + 1, horizon - 1)])
                env.set_planning_fidelity_level(next_level)
                seg_frames.append(env.render("rgb_array", include_start_pose=True))
            frames.extend(overlay_action_targets_on_frames(seg_frames, seg_states, seg_actions, overlay_spec))
    else:
        particle_backend = getattr(runtime["planner"], "backend", None)
        for r_idx in range(replan_action_seqs.shape[0]):
            start_state = replan_start_states[r_idx]
            horizon = int(replan_rollout_lengths[r_idx])
            if horizon <= 0:
                continue
            level_idx = int(replan_rollout_levels[r_idx, 0])
            particle_backend.set_planning_fidelity_level(level_idx)
            obs, cur_state = particle_backend.prepare(
                seed=0,
                init_state=start_state,
                goal_state=goal_state,
                with_visual=True,
            )
            seg_frames = [np.asarray(obs["visual"])]
            seg_states = [np.asarray(cur_state, dtype=np.float32).copy()]
            seg_actions = np.asarray(replan_action_seqs[r_idx, :horizon], dtype=np.float32)
            for t in range(horizon):
                particle_backend.set_planning_fidelity_level(int(replan_rollout_levels[r_idx, t]))
                obs, cur_state, _, _ = particle_backend.step(replan_action_seqs[r_idx, t], with_visual=True)
                seg_states.append(np.asarray(cur_state, dtype=np.float32).copy())
                seg_frames.append(np.asarray(obs["visual"]))
            frames.extend(overlay_action_targets_on_frames(seg_frames, seg_states, seg_actions, overlay_spec))

    if len(frames) <= 0:
        raise ValueError("No predicted backend frames were generated for this trace.")
    os.makedirs(media_dir, exist_ok=True)
    out_path = os.path.join(media_dir, "predicted_backend_replay.mp4")
    single_plan._write_video_mp4(out_path, frames, fps=15)
    return out_path


def list_artifacts(run_dir: str) -> None:
    trace_dirs = _discover_trace_dirs(run_dir)
    if len(trace_dirs) <= 0:
        raise FileNotFoundError(f"No trace bundles or planning artifacts found under {run_dir}")
    for trace_dir in trace_dirs:
        print(trace_dir)
        for media_name in MEDIA_ALIASES:
            path = _existing_media_path(trace_dir, media_name)
            if path is not None:
                print(f"  {media_name}: {path}")
        if os.path.isfile(os.path.join(trace_dir, "trace.json")):
            print(f"  trace: {os.path.join(trace_dir, 'trace.json')}")


def render_media(
    run_dir: str,
    schedule: str | None,
    rollout_id: str | None,
    media: Sequence[str],
    *,
    output_dir: str | None = None,
) -> list[str]:
    trace_dir = _resolve_trace_dir(run_dir, schedule=schedule, rollout_id=rollout_id)
    meta, arrays = _load_trace_dir(trace_dir)
    outputs = []
    for media_name in media:
        if media_name == "gt_replay":
            outputs.append(_render_gt_replay(trace_dir, meta, arrays, output_dir=output_dir))
        elif media_name == "closed_loop_replay":
            outputs.append(_render_closed_loop_replay(trace_dir, meta, arrays, output_dir=output_dir))
        elif media_name == "planner_view_replay":
            outputs.append(_render_planner_view_replay(trace_dir, meta, arrays, output_dir=output_dir))
        elif media_name == "predicted_backend_replay":
            outputs.append(_render_predicted_backend_replay(trace_dir, meta, arrays, output_dir=output_dir))
        else:
            raise ValueError(f"Unknown media type: {media_name}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse and render planner media artifacts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_ap = sub.add_parser("list", help="List trace bundles and available MP4 artifacts.")
    list_ap.add_argument("--run-dir", required=True, help="Planner run dir or planner_eval run dir.")

    render_ap = sub.add_parser("render", help="Render missing media from a saved trace bundle.")
    render_ap.add_argument("--run-dir", required=True, help="Planner run dir or planner_eval run dir.")
    render_ap.add_argument("--schedule", default=None, help="Schedule name for planner_eval traces.")
    render_ap.add_argument("--rollout-id", default=None, help="Rollout id for planner_eval traces.")
    render_ap.add_argument(
        "--media",
        required=True,
        nargs="+",
        help="Media names, space- or comma-separated: gt_replay closed_loop_replay planner_view_replay predicted_backend_replay",
    )

    args = parser.parse_args()
    if args.cmd == "list":
        list_artifacts(args.run_dir)
        return
    media = []
    for raw_item in args.media:
        media.extend(item.strip() for item in str(raw_item).split(",") if item.strip())
    outputs = render_media(args.run_dir, schedule=args.schedule, rollout_id=args.rollout_id, media=media)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
