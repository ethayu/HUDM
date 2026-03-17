from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from hudm.artifacts import (
    action_overlay_spec_from_env,
    draw_action_target_cross,
    overlay_start_pose,
    particle_planner_view_frame,
    planner_view_frame,
    resolve_action_target,
    rollout_level_for_exec_step,
    wm_decode_frame,
)
from hudm.runtime import (
    bits_to_flops_estimate,
    encode_visual,
    format_bits_human,
    format_flops_human,
)
from hudm.session_helpers import (
    set_execution_fidelity_finest,
    set_goal_pose,
    set_start_pose,
)
from pusht.pusht_wrapper import PushTWrapper


def run_closed_loop(
    env: PushTWrapper,
    wm: object | None,
    planner: object,
    backend: str,
    cfg: DictConfig,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    device: torch.device,
) -> tuple[bool, list[np.ndarray], list[np.ndarray], list[np.ndarray], dict, dict]:
    set_start_pose(env, init_state)
    particle_backend = None
    if backend == "particle_sim":
        particle_backend = getattr(planner, "backend", None)
        if particle_backend is None:
            raise ValueError("backend='particle_sim' requires planner.backend for planner-view rendering.")
    z_goal = None
    if backend == "wm":
        if wm is None:
            raise ValueError("backend='wm' requires a loaded world model.")
        goal_obs, _ = env.prepare(seed=0, init_state=goal_state)
        set_goal_pose(env, goal_state)
        goal_obs["visual"] = env.render("rgb_array", include_start_pose=False)
        z_goal = encode_visual(wm, goal_obs["visual"], device)
    obs, cur_state = env.prepare(seed=0, init_state=init_state, goal_state=goal_state)
    set_execution_fidelity_finest(env)
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

    initial_term = env.eval_termination(goal_state, cur_state, done=None, info=None)
    if bool(initial_term["done"]):
        cov_s = "n/a" if initial_term["coverage"] is None else f"{float(initial_term['coverage']):.4f}"
        print(
            "[terminate] reason=initial_env_done "
            f"step=0 metric_success={bool(initial_term['success'])} "
            f"done={bool(initial_term['done'])} pos_diff={float(initial_term['pos_diff']):.3f} "
            f"angle_diff={float(initial_term['angle_diff']):.3f} "
            f"eef_diff={float(initial_term['eef_diff']):.3f} "
            f"coverage={cov_s}"
        )
        stats = {
            "plans": 0,
            "bits_used_total": 0,
            "flops_used_total": 0,
            "plan_time_total_sec": 0.0,
            "termination_reason": "initial_env_done",
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
    action_overlay = action_overlay_spec_from_env(env) if bool(cfg.save) else None

    def _overlay_target_on_last_frames(state: np.ndarray, action: np.ndarray) -> None:
        if not bool(cfg.save) or action_overlay is None:
            return
        target_xy = resolve_action_target(
            np.asarray(state, dtype=np.float32)[:2],
            np.asarray(action, dtype=np.float32),
            action_format=str(action_overlay["action_format"]),
            action_relative=bool(action_overlay["action_relative"]),
            action_scale=float(action_overlay["action_scale"]),
        )
        if len(frames) > 0:
            frames[-1] = draw_action_target_cross(
                frames[-1],
                target_xy,
                reference_size=float(action_overlay["reference_size"]),
            )
        if len(planner_frames) > 0:
            planner_frames[-1] = draw_action_target_cross(
                planner_frames[-1],
                target_xy,
                reference_size=float(action_overlay["reference_size"]),
            )

    def _append_saved_frames_for_step(cur_obs: dict[str, Any], cur_state_after: np.ndarray, exec_step_in_replan: int) -> None:
        if not bool(cfg.save):
            return
        frame_with_start = env.render("rgb_array", include_start_pose=True)
        frames.append(frame_with_start)
        li_exec = rollout_level_for_exec_step(info, exec_step_in_replan=exec_step_in_replan)
        if backend == "gt_env":
            frame = planner_view_frame(
                env=env,
                base_visual=np.asarray(cur_obs["visual"]),
                level_idx=li_exec,
                target_hw=frames[-1].shape[:2],
            )
            planner_frames.append(
                overlay_start_pose(
                    planner_img=frame,
                    exec_with_start=np.asarray(frame_with_start),
                    exec_without_start=np.asarray(cur_obs["visual"]),
                    target_hw=frames[-1].shape[:2],
                )
            )
        elif backend == "wm":
            z_exec = encode_visual(wm, cur_obs["visual"], device)
            frame = wm_decode_frame(
                wm=wm,
                z=z_exec,
                level_idx=li_exec,
                target_hw=frames[-1].shape[:2],
            )
            planner_frames.append(
                overlay_start_pose(
                    planner_img=frame,
                    exec_with_start=np.asarray(frame_with_start),
                    exec_without_start=np.asarray(cur_obs["visual"]),
                    target_hw=frames[-1].shape[:2],
                )
            )
        else:
            planner_frames.append(
                particle_planner_view_frame(
                    particle_backend=particle_backend,
                    start_state=init_state,
                    cur_state=cur_state_after,
                    goal_state=goal_state,
                    level_idx=li_exec,
                    target_hw=frames[-1].shape[:2],
                )
            )

    while t < steps:
        mpc_progress = 0.0 if n_replans <= 1 else replan_idx / (n_replans - 1)
        z_cur_for_plan = None
        plan_seed = int(1009 * replan_idx + 7919 * t)
        if backend == "wm":
            z_cur_for_plan = encode_visual(wm, obs["visual"], device)
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
            obs, cur_state = env.prepare(seed=0, init_state=cur_state, goal_state=goal_state)
            set_execution_fidelity_finest(env)
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
        flops_used = bits_to_flops_estimate(bits_used)
        plan_time = float(getattr(info, "plan_time_sec", 0.0))
        total_plan_bits += bits_used
        total_plan_flops += flops_used
        total_plan_time += plan_time
        n_plans += 1
        if bits_used > 0 or plan_time > 0:
            print(
                f"[plan] replan {replan_idx:03d}  backend {backend}  "
                f"bits {bits_used} ({format_bits_human(bits_used)})  "
                f"flops {flops_used} ({format_flops_human(flops_used)})  "
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

        if bool(cfg.save) and len(planner_frames) == 0:
            init_level = rollout_level_for_exec_step(info, exec_step_in_replan=0)
            target_hw = frames[0].shape[:2] if len(frames) > 0 else np.asarray(obs["visual"]).shape[:2]
            if backend == "gt_env":
                frame = planner_view_frame(
                    env=env,
                    base_visual=np.asarray(obs["visual"]),
                    level_idx=init_level,
                    target_hw=target_hw,
                )
                if len(frames) > 0:
                    frame = overlay_start_pose(
                        planner_img=frame,
                        exec_with_start=np.asarray(frames[0]),
                        exec_without_start=np.asarray(obs["visual"]),
                        target_hw=target_hw,
                    )
                planner_frames.append(frame)
            elif backend == "wm":
                if z_cur_for_plan is None:
                    z_cur_for_plan = encode_visual(wm, obs["visual"], device)
                frame = wm_decode_frame(
                    wm=wm,
                    z=z_cur_for_plan,
                    level_idx=init_level,
                    target_hw=target_hw,
                )
                if len(frames) > 0:
                    frame = overlay_start_pose(
                        planner_img=frame,
                        exec_with_start=np.asarray(frames[0]),
                        exec_without_start=np.asarray(obs["visual"]),
                        target_hw=target_hw,
                    )
                planner_frames.append(frame)
            else:
                planner_frames.append(
                    particle_planner_view_frame(
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
            _overlay_target_on_last_frames(cur_state, action)
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

            _append_saved_frames_for_step(obs, cur_state, i)

            if bool(term["done"]):
                if bool(cfg.save):
                    for j in range(i + 1, n_exec):
                        cont_action = np.asarray(planned_actions_np[j], dtype=np.float32)
                        _overlay_target_on_last_frames(cur_state, cont_action)
                        obs, _, _, step_info = env.step(cont_action)
                        cur_state = step_info["state"]
                        _append_saved_frames_for_step(obs, cur_state, j)
                term_reason = "env_done"
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
