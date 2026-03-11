from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import plan as single_plan
from planning.latent_cem_batch import BatchedLatentCEMPlanner
from validate_cfg import validate_plan_cfg, validate_planner_eval_cfg


def _planner_eval_defaults() -> dict:
    return {
        "plan_config": "configs/plan.yaml",
        "seed": 0,
        "num_rollouts": 4,
        "sample_without_replacement": True,
        "output_root": "rollouts",
        "parallel": {
            "mode": "auto",
            "max_workers": 2,
            "wm_schedule_batch_size": 0,
        },
        "schedules": [],
    }


def load_planner_eval_cfg(cfg_path: str) -> tuple[DictConfig, DictConfig]:
    cfg_root = OmegaConf.load(cfg_path)
    eval_cfg = OmegaConf.merge(_planner_eval_defaults(), cfg_root.get("planner_eval", cfg_root))
    base_plan_cfg = single_plan.load_plan_cfg(str(eval_cfg.plan_config))
    if int(eval_cfg.parallel.wm_schedule_batch_size) <= 0:
        eval_cfg.parallel.wm_schedule_batch_size = max(1, len(eval_cfg.schedules))
    validate_planner_eval_cfg(eval_cfg, base_plan_cfg=base_plan_cfg)
    return eval_cfg, base_plan_cfg


def _resolve_dataset_path(base_cfg: DictConfig) -> str:
    zarr_path = getattr(base_cfg.init_goal.dataset, "zarr_path", None)
    if zarr_path is None:
        raise ValueError("planner_eval currently requires plan.init_goal.dataset.zarr_path to be set.")
    return str(zarr_path)


def enumerate_rollout_candidates(base_cfg: DictConfig) -> list[dict]:
    if str(base_cfg.init_goal.source).lower() != "dataset":
        raise ValueError("planner_eval requires plan.init_goal.source=dataset.")
    try:
        import zarr
    except Exception as exc:
        raise ImportError("zarr must be installed to enumerate rollout candidates.") from exc

    zarr_path = _resolve_dataset_path(base_cfg)
    root = zarr.open_group(zarr_path, mode="r")
    state_arr = root["data"]["state"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.zeros_like(ends)
    starts[0] = 0
    for idx in range(1, len(ends)):
        starts[idx] = ends[idx - 1] + 1

    trajectory_len = int(base_cfg.init_goal.dataset.trajectory_len)
    split_ratio = (
        float(base_cfg.init_goal.dataset.split_ratio)
        if getattr(base_cfg.init_goal.dataset, "split_ratio", None) is not None
        else 0.8
    )
    split_l = str(base_cfg.init_goal.dataset.split).lower()
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
        raise ValueError("No valid dataset rollout candidates were found for planner_eval.")
    return candidates


def select_rollouts(eval_cfg: DictConfig, candidates: Sequence[dict]) -> list[dict]:
    n = int(eval_cfg.num_rollouts)
    if bool(eval_cfg.sample_without_replacement) and n > len(candidates):
        raise ValueError(
            f"planner_eval.num_rollouts={n} exceeds available candidates={len(candidates)} "
            "with sample_without_replacement=true."
        )
    rng = np.random.default_rng(int(eval_cfg.seed))
    if bool(eval_cfg.sample_without_replacement):
        idxs = rng.choice(len(candidates), size=n, replace=False)
    else:
        idxs = rng.choice(len(candidates), size=n, replace=True)
    selected = [dict(candidates[int(idx)]) for idx in idxs]
    for order_idx, item in enumerate(selected):
        item["rollout_index"] = int(order_idx)
        item["rollout_id"] = rollout_id(item)
    return selected


def rollout_id(selection: dict) -> str:
    return f"ep{int(selection['episode_index']):04d}_s{int(selection['start_index']):05d}_g{int(selection['goal_index']):05d}"


def schedule_cfg(base_cfg: DictConfig, fidelity_override, trace_only: bool = True) -> DictConfig:
    merged = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True)),
        OmegaConf.create({"fidelity": OmegaConf.to_container(fidelity_override, resolve=True)}),
    )
    if trace_only:
        merged.render = False
        merged.save = False
        merged.gt_env.progress = False
        merged.gt_env.progress_leave = False
        merged.particle_env.progress = False
        merged.particle_env.progress_leave = False
    validate_plan_cfg(merged)
    return merged


def _trace_final_value(values: Sequence[float], fallback: Any) -> float:
    if len(values) > 0:
        return float(values[-1])
    return float(fallback) if fallback is not None else float("nan")


def _trace_best_value(values: Sequence[float], fallback: Any) -> float:
    if len(values) > 0:
        return float(np.min(np.asarray(values, dtype=np.float32)))
    return float(fallback) if fallback is not None else float("nan")


def _coverage_final(values: Sequence[float], fallback: Any) -> float:
    if len(values) > 0:
        arr = np.asarray(values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            return float(finite[-1])
    return float(fallback) if fallback is not None else float("nan")


def _auc(values: Sequence[float]) -> float:
    if len(values) <= 0:
        return float("nan")
    return float(np.sum(np.asarray(values, dtype=np.float32)))


def result_row(result: dict, run_dir: str) -> dict:
    trace = result["trace"]
    run_stats = result["run_stats"]
    sample = result["sample_meta"]
    executed_steps = int(len(trace.get("executed_actions", [])))
    plans = int(run_stats["plans"])
    bits_total = int(run_stats["bits_used_total"])
    flops_total = int(run_stats["flops_used_total"])
    plan_time_total = float(run_stats["plan_time_total_sec"])
    return {
        "schedule_name": str(result.get("schedule_name", "")),
        "rollout_id": str(sample.get("rollout_id", rollout_id(sample))),
        "rollout_index": int(sample.get("rollout_index", -1)),
        "episode_index": int(sample.get("episode_index", -1)),
        "start_index": int(sample.get("start_index", -1)),
        "goal_index": int(sample.get("goal_index", -1)),
        "success": int(bool(result["success"])),
        "termination_reason": str(run_stats.get("termination_reason", "unknown")),
        "termination_step": int(run_stats.get("termination_step", -1)),
        "executed_steps": executed_steps,
        "plans": plans,
        "success_and_done": int(bool(run_stats.get("termination_metric_success", False) and run_stats.get("termination_done", False))),
        "final_pos_diff": _trace_final_value(trace.get("pos_diffs", []), run_stats.get("termination_pos_diff")),
        "final_angle_diff": _trace_final_value(trace.get("angle_diffs", []), run_stats.get("termination_angle_diff")),
        "final_eef_diff": _trace_final_value(trace.get("eef_diffs", []), run_stats.get("termination_eef_diff")),
        "best_pos_diff": _trace_best_value(trace.get("pos_diffs", []), run_stats.get("termination_pos_diff")),
        "best_angle_diff": _trace_best_value(trace.get("angle_diffs", []), run_stats.get("termination_angle_diff")),
        "best_eef_diff": _trace_best_value(trace.get("eef_diffs", []), run_stats.get("termination_eef_diff")),
        "final_coverage": _coverage_final(trace.get("coverages", []), run_stats.get("termination_coverage")),
        "auc_pos_diff": _auc(trace.get("pos_diffs", [])),
        "auc_angle_diff": _auc(trace.get("angle_diffs", [])),
        "auc_eef_diff": _auc(trace.get("eef_diffs", [])),
        "bits_used_total": bits_total,
        "bits_used_per_step": float(bits_total / max(1, executed_steps)),
        "flops_used_total": flops_total,
        "flops_used_per_step": float(flops_total / max(1, executed_steps)),
        "plan_time_total_sec": plan_time_total,
        "plan_time_per_replan_sec": float(plan_time_total / max(1, plans)),
        "run_dir": run_dir,
        "trace_json": os.path.join(run_dir, "trace.json"),
        "trace_npz": os.path.join(run_dir, "trace.npz"),
    }


def _write_rows_csv(path: str, rows: Sequence[dict]) -> None:
    if len(rows) <= 0:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_trace_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _stepwise_curve(rows: Sequence[dict], key: str, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    stacked = []
    for row in rows:
        trace = _load_trace_npz(row["trace_npz"])
        arr = np.asarray(trace[key], dtype=np.float32)
        if arr.size <= 0:
            arr = np.full((max_steps,), np.nan, dtype=np.float32)
        elif arr.shape[0] < max_steps:
            pad_value = arr[-1]
            arr = np.concatenate([arr, np.full((max_steps - arr.shape[0],), pad_value, dtype=np.float32)], axis=0)
        else:
            arr = arr[:max_steps]
        stacked.append(arr)
    mat = np.stack(stacked, axis=0)
    return np.nanmedian(mat, axis=0), np.nanmean(mat, axis=0)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _metric_array(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def aggregate_summary(
    rows: Sequence[dict],
    base_cfg: DictConfig,
    baseline_schedule: str,
    schedule_order: Sequence[str],
) -> tuple[list[dict], list[dict]]:
    by_schedule: dict[str, list[dict]] = {}
    for row in rows:
        by_schedule.setdefault(str(row["schedule_name"]), []).append(row)
    baseline = str(baseline_schedule) if rows else ""
    summary_rows: list[dict] = []
    paired_rows: list[dict] = []
    baseline_rows = {str(row["rollout_id"]): row for row in by_schedule.get(baseline, [])}
    ordered_schedule_names = [name for name in schedule_order if name in by_schedule]
    ordered_schedule_names.extend(sorted(name for name in by_schedule if name not in ordered_schedule_names))
    for schedule_name in ordered_schedule_names:
        schedule_rows = by_schedule[schedule_name]
        success_vals = _metric_array(schedule_rows, "success")
        success_and_done = _metric_array(schedule_rows, "success_and_done")
        executed_steps = _metric_array(schedule_rows, "executed_steps")
        plans = _metric_array(schedule_rows, "plans")
        final_pos = _metric_array(schedule_rows, "final_pos_diff")
        final_angle = _metric_array(schedule_rows, "final_angle_diff")
        final_eef = _metric_array(schedule_rows, "final_eef_diff")
        best_pos = _metric_array(schedule_rows, "best_pos_diff")
        best_angle = _metric_array(schedule_rows, "best_angle_diff")
        best_eef = _metric_array(schedule_rows, "best_eef_diff")
        final_cov = _metric_array(schedule_rows, "final_coverage")
        auc_pos = _metric_array(schedule_rows, "auc_pos_diff")
        auc_angle = _metric_array(schedule_rows, "auc_angle_diff")
        auc_eef = _metric_array(schedule_rows, "auc_eef_diff")
        bits = _metric_array(schedule_rows, "bits_used_total")
        bits_per_step = _metric_array(schedule_rows, "bits_used_per_step")
        flops = _metric_array(schedule_rows, "flops_used_total")
        flops_per_step = _metric_array(schedule_rows, "flops_used_per_step")
        plan_time = _metric_array(schedule_rows, "plan_time_total_sec")
        plan_time_per_replan = _metric_array(schedule_rows, "plan_time_per_replan_sec")
        term_reasons = [str(row["termination_reason"]) for row in schedule_rows]
        reason_counts = {reason: term_reasons.count(reason) for reason in sorted(set(term_reasons))}
        summary_row = {
            "schedule_name": schedule_name,
            "n_rollouts": int(len(schedule_rows)),
            "success_rate": float(np.mean(success_vals)) if success_vals.size > 0 else float("nan"),
            "success_and_done_rate": float(np.mean(success_and_done)) if success_and_done.size > 0 else float("nan"),
            "mean_executed_steps": float(np.nanmean(executed_steps)) if executed_steps.size > 0 else float("nan"),
            "median_executed_steps": float(np.nanmedian(executed_steps)) if executed_steps.size > 0 else float("nan"),
            "mean_plans": float(np.nanmean(plans)) if plans.size > 0 else float("nan"),
            "mean_final_pos_diff": float(np.nanmean(final_pos)) if final_pos.size > 0 else float("nan"),
            "median_final_pos_diff": float(np.nanmedian(final_pos)) if final_pos.size > 0 else float("nan"),
            "mean_final_angle_diff": float(np.nanmean(final_angle)) if final_angle.size > 0 else float("nan"),
            "median_final_angle_diff": float(np.nanmedian(final_angle)) if final_angle.size > 0 else float("nan"),
            "mean_final_eef_diff": float(np.nanmean(final_eef)) if final_eef.size > 0 else float("nan"),
            "median_final_eef_diff": float(np.nanmedian(final_eef)) if final_eef.size > 0 else float("nan"),
            "mean_best_pos_diff": float(np.nanmean(best_pos)) if best_pos.size > 0 else float("nan"),
            "mean_best_angle_diff": float(np.nanmean(best_angle)) if best_angle.size > 0 else float("nan"),
            "mean_best_eef_diff": float(np.nanmean(best_eef)) if best_eef.size > 0 else float("nan"),
            "mean_final_coverage": float(np.nanmean(final_cov)) if final_cov.size > 0 else float("nan"),
            "median_final_coverage": float(np.nanmedian(final_cov)) if final_cov.size > 0 else float("nan"),
            "mean_auc_pos_diff": float(np.nanmean(auc_pos)) if auc_pos.size > 0 else float("nan"),
            "mean_auc_angle_diff": float(np.nanmean(auc_angle)) if auc_angle.size > 0 else float("nan"),
            "mean_auc_eef_diff": float(np.nanmean(auc_eef)) if auc_eef.size > 0 else float("nan"),
            "mean_bits_used_total": float(np.nanmean(bits)) if bits.size > 0 else float("nan"),
            "mean_bits_used_per_step": float(np.nanmean(bits_per_step)) if bits_per_step.size > 0 else float("nan"),
            "mean_flops_used_total": float(np.nanmean(flops)) if flops.size > 0 else float("nan"),
            "mean_flops_used_per_step": float(np.nanmean(flops_per_step)) if flops_per_step.size > 0 else float("nan"),
            "mean_plan_time_total_sec": float(np.nanmean(plan_time)) if plan_time.size > 0 else float("nan"),
            "median_plan_time_total_sec": float(np.nanmedian(plan_time)) if plan_time.size > 0 else float("nan"),
            "mean_plan_time_per_replan_sec": float(np.nanmean(plan_time_per_replan)) if plan_time_per_replan.size > 0 else float("nan"),
        }
        for reason, count in reason_counts.items():
            summary_row[f"termination_reason__{_safe_name(reason)}"] = int(count)
        summary_rows.append(summary_row)
        if schedule_name == baseline:
            continue
        success_win_count = 0
        success_loss_count = 0
        success_tie_count = 0
        pos_better_count = 0
        pos_worse_count = 0
        pos_tie_count = 0
        for row in schedule_rows:
            base_row = baseline_rows.get(str(row["rollout_id"]))
            if base_row is None:
                continue
            success_delta = int(row["success"]) - int(base_row["success"])
            final_pos_diff_delta = float(row["final_pos_diff"]) - float(base_row["final_pos_diff"])
            if success_delta > 0:
                success_win_count += 1
            elif success_delta < 0:
                success_loss_count += 1
            else:
                success_tie_count += 1
            if final_pos_diff_delta < 0:
                pos_better_count += 1
            elif final_pos_diff_delta > 0:
                pos_worse_count += 1
            else:
                pos_tie_count += 1
            paired_rows.append(
                {
                    "baseline_schedule": baseline,
                    "schedule_name": schedule_name,
                    "rollout_id": row["rollout_id"],
                    "success_delta": success_delta,
                    "final_pos_diff_delta": final_pos_diff_delta,
                    "final_angle_diff_delta": float(row["final_angle_diff"]) - float(base_row["final_angle_diff"]),
                    "final_eef_diff_delta": float(row["final_eef_diff"]) - float(base_row["final_eef_diff"]),
                    "final_coverage_delta": float(row["final_coverage"]) - float(base_row["final_coverage"]),
                    "bits_used_total_delta": float(row["bits_used_total"]) - float(base_row["bits_used_total"]),
                    "plan_time_total_sec_delta": float(row["plan_time_total_sec"]) - float(base_row["plan_time_total_sec"]),
                }
            )
        summary_rows[-1]["paired_success_wins_vs_baseline"] = int(success_win_count)
        summary_rows[-1]["paired_success_losses_vs_baseline"] = int(success_loss_count)
        summary_rows[-1]["paired_success_ties_vs_baseline"] = int(success_tie_count)
        summary_rows[-1]["paired_final_pos_better_vs_baseline"] = int(pos_better_count)
        summary_rows[-1]["paired_final_pos_worse_vs_baseline"] = int(pos_worse_count)
        summary_rows[-1]["paired_final_pos_ties_vs_baseline"] = int(pos_tie_count)
    return summary_rows, paired_rows


def save_summary_plots(rows: Sequence[dict], summary_rows: Sequence[dict], out_dir: str, max_steps: int) -> None:
    import matplotlib.pyplot as plt

    by_schedule: dict[str, list[dict]] = {}
    for row in rows:
        by_schedule.setdefault(str(row["schedule_name"]), []).append(row)
    schedule_names = [row["schedule_name"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(schedule_names, [row["success_rate"] for row in summary_rows])
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Schedule")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "success_rates.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].boxplot([[row["final_pos_diff"] for row in by_schedule[name]] for name in schedule_names], tick_labels=schedule_names)
    axes[0].set_title("Final Pos Diff")
    axes[1].boxplot([[row["final_angle_diff"] for row in by_schedule[name]] for name in schedule_names], tick_labels=schedule_names)
    axes[1].set_title("Final Angle Diff")
    axes[2].boxplot([[row["final_coverage"] for row in by_schedule[name]] for name in schedule_names], tick_labels=schedule_names)
    axes[2].set_title("Final Coverage")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_metrics_boxplots.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [row["mean_bits_used_total"] for row in summary_rows],
        [row["success_rate"] for row in summary_rows],
    )
    for row in summary_rows:
        ax.annotate(row["schedule_name"], (row["mean_bits_used_total"], row["success_rate"]))
    ax.set_xlabel("Mean Bits Used Total")
    ax.set_ylabel("Success Rate")
    ax.set_title("Compute vs Success")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compute_vs_success.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for schedule_name in schedule_names:
        schedule_rows = by_schedule[schedule_name]
        med, _ = _stepwise_curve(schedule_rows, "pos_diffs", max_steps=max_steps)
        axes[0].plot(med, label=schedule_name)
        med, _ = _stepwise_curve(schedule_rows, "angle_diffs", max_steps=max_steps)
        axes[1].plot(med, label=schedule_name)
        med, _ = _stepwise_curve(schedule_rows, "eef_diffs", max_steps=max_steps)
        axes[2].plot(med, label=schedule_name)
    axes[0].set_title("Median Pos Diff")
    axes[1].set_title("Median Angle Diff")
    axes[2].set_title("Median EEF Diff")
    for ax in axes:
        ax.set_xlabel("Step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "median_stepwise_curves.png"))
    plt.close(fig)


def _non_wm_task(task: dict) -> dict:
    cfg = OmegaConf.create(task["cfg"])
    result = single_plan.run_plan_session(
        cfg,
        rollout_selection=task["selection"],
        schedule_name=task["schedule_name"],
        print_summary=False,
    )
    single_plan.save_plan_result(result, task["run_dir"], save_media=False)
    return result_row(result, task["run_dir"])


def _make_exec_env(cfg: DictConfig):
    single_plan._register_plan_env(cfg)
    env_wrapped = single_plan._gym_make_versioned(str(cfg.env_id), cfg.env)
    return single_plan._unwrap_env(env_wrapped)


def _finalize_wm_state(state: dict, term_reason: str, term_step: int) -> dict:
    last_term = state["last_term"]
    run_stats = {
        "plans": int(state["n_plans"]),
        "bits_used_total": int(state["bits_total"]),
        "flops_used_total": int(state["flops_total"]),
        "plan_time_total_sec": float(state["plan_time_total"]),
        "termination_reason": str(term_reason),
        "termination_step": int(term_step),
        "termination_metric_success": False if last_term is None else bool(last_term["success"]),
        "termination_done": False if last_term is None else bool(last_term["done"]),
        "termination_pos_diff": None if last_term is None else float(last_term["pos_diff"]),
        "termination_angle_diff": None if last_term is None else float(last_term["angle_diff"]),
        "termination_eef_diff": None if last_term is None else float(last_term["eef_diff"]),
        "termination_coverage": None if last_term is None else last_term["coverage"],
    }
    return {
        "executed_actions": [np.asarray(x, dtype=np.float32).tolist() for x in state["executed_actions"]],
        "trajectory": [np.asarray(x, dtype=np.float32).tolist() for x in state["trajectory"]],
        "pos_diffs": [float(x) for x in state["pos_diffs"]],
        "angle_diffs": [float(x) for x in state["angle_diffs"]],
        "eef_diffs": [float(x) for x in state["eef_diffs"]],
        "coverages": [float(x) for x in state["coverages"]],
        "metric_success_flags": [bool(x) for x in state["metric_success_flags"]],
        "done_flags": [bool(x) for x in state["done_flags"]],
        "state_dists": [float(x) for x in state["state_dists"]],
        "replans": state["replans"],
    }, run_stats


def run_wm_batched_chunk(
    base_cfg: DictConfig,
    schedules: Sequence[dict],
    selection: dict,
    run_root: str,
    seed_base: int,
) -> list[dict]:
    chunk_cfgs = [schedule_cfg(base_cfg, schedule["fidelity"], trace_only=True) for schedule in schedules]
    runtime = single_plan.build_plan_runtime(chunk_cfgs[0])
    wm = runtime["wm"]
    device = runtime["device"]
    envs = [runtime["env"]]
    for _ in range(1, len(chunk_cfgs)):
        envs.append(_make_exec_env(chunk_cfgs[0]))

    fidelity_cfgs = [OmegaConf.to_container(cfg.fidelity, resolve=True) for cfg in chunk_cfgs]
    batch_planner = BatchedLatentCEMPlanner(
        world_model=wm,
        fidelity_cfgs=fidelity_cfgs,
        horizon=int(chunk_cfgs[0].mpc.horizon),
        action_dim=int(envs[0].action_dim),
        pop_size=int(chunk_cfgs[0].cem.pop_size),
        elite_frac=float(chunk_cfgs[0].cem.elite_frac),
        n_iter=int(chunk_cfgs[0].cem.n_iter),
        init_std=float(chunk_cfgs[0].cem.init_std),
        action_low=chunk_cfgs[0].cem.action_low,
        action_high=chunk_cfgs[0].cem.action_high,
        objective_cfg=OmegaConf.to_container(chunk_cfgs[0].objective, resolve=True),
        drop_tail_on_coarsen=True,
        warm_start=bool(chunk_cfgs[0].cem.warm_start),
        device=device,
    )
    init_state, goal_state, sample_meta = single_plan.load_selected_rollout(
        envs[0],
        chunk_cfgs[0],
        runtime["wm_cfg"],
        selection,
    )
    goal_obs, _ = envs[0].prepare(seed=0, init_state=goal_state)
    single_plan._set_goal_pose(envs[0], goal_state)
    goal_obs["visual"] = envs[0].render("rgb_array", include_start_pose=False)
    z_goal = single_plan._encode_visual(wm, goal_obs["visual"], device)

    schedule_states: list[dict] = []
    for env, cfg, schedule in zip(envs, chunk_cfgs, schedules):
        single_plan._set_start_pose(env, init_state)
        obs, cur_state = env.prepare(seed=0, init_state=init_state, goal_state=goal_state)
        single_plan._set_execution_fidelity_finest(env)
        obs["visual"] = env.render("rgb_array", include_start_pose=False)
        schedule_states.append(
            {
                "env": env,
                "cfg": cfg,
                "schedule_name": str(schedule["name"]),
                "obs": obs,
                "cur_state": np.asarray(cur_state, dtype=np.float32),
                "executed_actions": [],
                "trajectory": [np.asarray(cur_state, dtype=np.float32).copy()],
                "pos_diffs": [],
                "angle_diffs": [],
                "eef_diffs": [],
                "coverages": [],
                "metric_success_flags": [],
                "done_flags": [],
                "state_dists": [],
                "replans": [],
                "bits_total": 0,
                "flops_total": 0,
                "plan_time_total": 0.0,
                "n_plans": 0,
                "last_term": None,
                "done": False,
            }
        )

    initial_term = envs[0].eval_termination(goal_state, schedule_states[0]["cur_state"], done=None, info=None)
    if bool(initial_term["success_and_done"]):
        rows = []
        for state in schedule_states:
            state["last_term"] = initial_term
            trace, run_stats = _finalize_wm_state(state, "initial_metric_and_env_done", 0)
            result = {
                "cfg": state["cfg"],
                "runtime": {"backend": "wm"},
                "success": True,
                "trajectory": state["trajectory"],
                "frames": [],
                "planner_frames": [],
                "run_stats": run_stats,
                "trace": trace,
                "init_state": np.asarray(init_state, dtype=np.float32),
                "goal_state": np.asarray(goal_state, dtype=np.float32),
                "sample_meta": {**sample_meta, "rollout_id": selection["rollout_id"], "rollout_index": selection["rollout_index"]},
                "schedule_name": state["schedule_name"],
            }
            run_dir = os.path.join(run_root, "traces", state["schedule_name"], selection["rollout_id"])
            single_plan.save_plan_result(result, run_dir, save_media=False)
            rows.append(result_row(result, run_dir))
        return rows

    steps = int(chunk_cfgs[0].mpc.steps)
    horizon = int(chunk_cfgs[0].mpc.horizon)
    replan_every = int(chunk_cfgs[0].mpc.replan_every)
    n_replans = max(1, int(np.ceil(steps / replan_every)))
    t = 0
    replan_idx = 0
    prev_exec_steps = 0

    while t < steps and any(not state["done"] for state in schedule_states):
        mpc_progress = 0.0 if n_replans <= 1 else replan_idx / (n_replans - 1)
        z_batch = torch.cat(
            [single_plan._encode_visual(wm, state["obs"]["visual"], device) for state in schedule_states],
            dim=0,
        )
        plan_seeds = [int(seed_base + 1000003 * idx + 7919 * replan_idx + 101 * t) for idx in range(len(schedule_states))]
        batch_results = batch_planner.plan_batch(
            z_batch,
            z_goal.expand(len(schedule_states), -1),
            mpc_progress=mpc_progress,
            warm_start_steps=int(prev_exec_steps),
            seeds=plan_seeds,
        )

        for sched_idx, (state, batch_result) in enumerate(zip(schedule_states, batch_results)):
            info = batch_result.info
            action_seq = np.asarray(batch_result.action_seq.detach().cpu().numpy(), dtype=np.float32)
            bits_used = int(getattr(info, "bits_used_estimate", 0))
            flops_used = int(single_plan._bits_to_flops_estimate(bits_used))
            plan_time = float(getattr(info, "plan_time_sec", 0.0))
            state["bits_total"] += bits_used
            state["flops_total"] += flops_used
            state["plan_time_total"] += plan_time
            state["n_plans"] += 1
            state["replans"].append(
                {
                    "replan_idx": int(replan_idx),
                    "step_start": int(t),
                    "mpc_progress": float(mpc_progress),
                    "seed": int(plan_seeds[sched_idx]),
                    "action_seq": action_seq.tolist(),
                    "base_level_idx": int(getattr(info, "base_level_idx", -1)),
                    "rollout_level_indices": [int(x) for x in list(getattr(info, "rollout_level_indices", []))],
                    "bits_used_estimate": bits_used,
                    "flops_used_estimate": flops_used,
                    "plan_time_sec": plan_time,
                    "base_k": None if getattr(info, "base_k", None) is None else int(getattr(info, "base_k")),
                    "base_spacing": None,
                    "base_num_particles": None,
                    "start_state": np.asarray(state["cur_state"], dtype=np.float32).tolist(),
                }
            )

        n_exec = min(replan_every, horizon, steps - t)
        for exec_idx in range(n_exec):
            for sched_idx, state in enumerate(schedule_states):
                if state["done"]:
                    continue
                action_seq = np.asarray(state["replans"][-1]["action_seq"], dtype=np.float32)
                action = np.asarray(action_seq[exec_idx], dtype=np.float32)
                state["executed_actions"].append(action.copy())
                obs, _, done, step_info = state["env"].step(action)
                cur_state = np.asarray(step_info["state"], dtype=np.float32)
                state["obs"] = obs
                state["cur_state"] = cur_state
                state["trajectory"].append(cur_state.copy())
                term = state["env"].eval_termination(goal_state, cur_state, done=done, info=step_info)
                state["last_term"] = term
                state["pos_diffs"].append(float(term["pos_diff"]))
                state["angle_diffs"].append(float(term["angle_diff"]))
                state["eef_diffs"].append(float(term["eef_diff"]))
                state["coverages"].append(float(term["coverage"]) if term["coverage"] is not None else float("nan"))
                state["metric_success_flags"].append(bool(term["success"]))
                state["done_flags"].append(bool(term["done"]))
                state["state_dists"].append(float(term["state_dist"]))
                if bool(term["success_and_done"]):
                    state["done"] = True
            t += 1
            if t >= steps or all(state["done"] for state in schedule_states):
                break

        prev_exec_steps = int(n_exec)
        replan_idx += 1

    rows = []
    for state in schedule_states:
        trace, run_stats = _finalize_wm_state(
            state,
            "metric_success_and_env_done" if state["done"] else "max_steps",
            run_stats_term_step(state["last_term"], state["trajectory"]),
        )
        result = {
            "cfg": state["cfg"],
            "runtime": {"backend": "wm"},
            "success": bool(state["done"]),
            "trajectory": state["trajectory"],
            "frames": [],
            "planner_frames": [],
            "run_stats": run_stats,
            "trace": trace,
            "init_state": np.asarray(init_state, dtype=np.float32),
            "goal_state": np.asarray(goal_state, dtype=np.float32),
            "sample_meta": {**sample_meta, "rollout_id": selection["rollout_id"], "rollout_index": selection["rollout_index"]},
            "schedule_name": state["schedule_name"],
        }
        run_dir = os.path.join(run_root, "traces", state["schedule_name"], selection["rollout_id"])
        single_plan.save_plan_result(result, run_dir, save_media=False)
        rows.append(result_row(result, run_dir))
    return rows


def run_stats_term_step(last_term: dict | None, trajectory: Sequence[np.ndarray]) -> int:
    if last_term is None:
        return max(0, len(trajectory) - 1)
    return max(0, len(trajectory) - 1)


def run_eval(cfg_path: str) -> str:
    eval_cfg, base_plan_cfg = load_planner_eval_cfg(cfg_path)
    candidates = enumerate_rollout_candidates(base_plan_cfg)
    selected = select_rollouts(eval_cfg, candidates)
    schedule_order = [str(schedule.name) for schedule in eval_cfg.schedules]
    baseline_schedule = "" if len(schedule_order) <= 0 else schedule_order[0]
    backend = str(base_plan_cfg.backend).lower()
    parallel_mode = str(eval_cfg.parallel.mode).lower()
    if parallel_mode == "auto":
        parallel_mode = "wm_batch" if backend == "wm" else "process"

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(str(eval_cfg.output_root), f"planner_eval_{backend}_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    combined_cfg = OmegaConf.create(
        {
            "planner_eval": OmegaConf.to_container(eval_cfg, resolve=True),
            "plan": OmegaConf.to_container(base_plan_cfg, resolve=True),
        }
    )
    OmegaConf.save(config=combined_cfg, f=os.path.join(run_dir, "config_resolved.yaml"))
    OmegaConf.save(config=OmegaConf.create({"planner_eval": OmegaConf.to_container(eval_cfg, resolve=True)}), f=os.path.join(run_dir, "planner_eval_resolved.yaml"))
    OmegaConf.save(config=OmegaConf.create({"plan": OmegaConf.to_container(base_plan_cfg, resolve=True)}), f=os.path.join(run_dir, "plan_resolved.yaml"))
    with open(os.path.join(run_dir, "selected_rollouts.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    rows: list[dict] = []
    if backend == "wm" and parallel_mode == "wm_batch":
        batch_size = int(eval_cfg.parallel.wm_schedule_batch_size)
        schedule_chunks = [
            eval_cfg.schedules[idx : idx + batch_size]
            for idx in range(0, len(eval_cfg.schedules), batch_size)
        ]
        for selection in selected:
            for chunk_idx, schedule_chunk in enumerate(schedule_chunks):
                seed_base = int(eval_cfg.seed + 100003 * selection["rollout_index"] + 10007 * chunk_idx)
                rows.extend(
                    run_wm_batched_chunk(
                        base_cfg=base_plan_cfg,
                        schedules=schedule_chunk,
                        selection=selection,
                        run_root=run_dir,
                        seed_base=seed_base,
                    )
                )
    else:
        tasks = []
        for selection in selected:
            for schedule in eval_cfg.schedules:
                cfg = schedule_cfg(base_plan_cfg, schedule.fidelity, trace_only=True)
                task_run_dir = os.path.join(run_dir, "traces", str(schedule.name), selection["rollout_id"])
                tasks.append(
                    {
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                        "selection": selection,
                        "schedule_name": str(schedule.name),
                        "run_dir": task_run_dir,
                    }
                )
        if parallel_mode == "process":
            max_workers = min(int(eval_cfg.parallel.max_workers), max(1, len(tasks)))
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_non_wm_task, task) for task in tasks]
                    for fut in as_completed(futures):
                        rows.append(fut.result())
            except Exception as exc:
                print(f"[planner_eval][warn] process parallelism unavailable ({exc}); falling back to serial.")
                for task in tasks:
                    rows.append(_non_wm_task(task))
        else:
            for task in tasks:
                rows.append(_non_wm_task(task))

    rows = sorted(rows, key=lambda row: (row["rollout_index"], row["schedule_name"]))
    summary_rows, paired_rows = aggregate_summary(
        rows,
        base_plan_cfg,
        baseline_schedule=baseline_schedule,
        schedule_order=schedule_order,
    )
    _write_rows_csv(os.path.join(run_dir, "per_rollout.csv"), rows)
    _write_rows_csv(os.path.join(run_dir, "summary.csv"), summary_rows)
    baseline_filename = f"paired_deltas_vs_{_safe_name(baseline_schedule or 'baseline')}.csv"
    _write_rows_csv(os.path.join(run_dir, baseline_filename), paired_rows)
    _write_rows_csv(os.path.join(run_dir, "paired_deltas_vs_baseline.csv"), paired_rows)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": backend,
                "created_at": run_ts,
                "num_rollouts": int(eval_cfg.num_rollouts),
                "parallel_mode": parallel_mode,
                "baseline_schedule": baseline_schedule,
                "summary": summary_rows,
            },
            f,
            indent=2,
        )
    save_summary_plots(rows, summary_rows, run_dir, max_steps=int(base_plan_cfg.mpc.steps))
    print(f"[planner_eval] wrote results to {run_dir}")
    for summary in summary_rows:
        print(
            f"[planner_eval][summary] schedule={summary['schedule_name']} "
            f"success_rate={summary['success_rate']:.3f} "
            f"mean_final_pos_diff={summary['mean_final_pos_diff']:.3f} "
            f"mean_bits_used_total={summary['mean_bits_used_total']:.1f} "
            f"mean_plan_time_total_sec={summary['mean_plan_time_total_sec']:.3f}"
        )
    return run_dir


def main(cfg_path: str) -> None:
    run_eval(cfg_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python planner_eval.py <path/to/config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
