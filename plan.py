"""Thin CLI facade for single-run planning."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure local imports work even when launched via an absolute path.
sys.path.append(os.path.dirname(__file__))

from hudm.artifacts import (
    overlay_start_pose as _overlay_start_pose,
    particle_planner_view_frame as _particle_planner_view_frame,
    planner_view_frame as _planner_view_frame,
    render_dataset_gt_frames as _render_dataset_gt_frames,
    rollout_level_for_exec_step as _rollout_level_for_exec_step,
    save_error_curves as _save_error_curves,
    save_plan_result,
    save_trace_bundle,
    trace_arrays_from_trace as _trace_arrays_from_trace,
    wm_decode_frame as _wm_decode_frame,
    write_video_mp4 as _write_video_mp4,
)
from hudm.runtime import (
    bits_to_flops_estimate as _bits_to_flops_estimate,
    build_plan_runtime,
    encode_visual as _encode_visual,
    format_bits_human as _format_bits_human,
    format_flops_human as _format_flops_human,
    gym_make_versioned as _gym_make_versioned,
    register_plan_env as _register_plan_env,
    unwrap_env as _unwrap_env,
)
from hudm.session import load_plan_cfg, run_plan_session
from hudm.session_exec import run_closed_loop as _run_closed_loop
from hudm.session_helpers import (
    load_selected_rollout,
    sample_init_goal_states as _sample_init_goal_states,
    set_execution_fidelity_finest as _set_execution_fidelity_finest,
    set_goal_pose as _set_goal_pose,
    set_start_pose as _set_start_pose,
)


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
        f"({_format_bits_human(int(run_stats['bits_used_total']))})  "
        f"flops_used_total={run_stats['flops_used_total']} "
        f"({_format_flops_human(int(run_stats['flops_used_total']))})  "
        f"plan_time_total_sec={run_stats['plan_time_total_sec']:.3f}"
    )
    print("Reached goal:", result["success"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plan.py <path/to/config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
