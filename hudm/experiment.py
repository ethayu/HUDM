from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hudm.artifacts import save_plan_result
from hudm.config import resolve_experiment_spec
from hudm.experiment_report import write_experiment_report
from hudm.runtime import (
    bits_to_flops_estimate,
    build_plan_runtime,
    encode_visual,
    gym_make_versioned,
    register_plan_env,
    unwrap_env,
)
from hudm.session import run_plan_session
from hudm.session_helpers import (
    load_selected_rollout,
    set_execution_fidelity_finest,
    set_goal_pose,
    set_start_pose,
)
from hudm.specs import ExperimentSpec, ExperimentVariant
from hudm.task_sampling import enumerate_rollout_candidates, rollout_id, select_rollouts
from planning.latent_cem_batch import BatchedLatentCEMPlanner


def load_experiment_spec(cfg_path: str) -> ExperimentSpec:
    return resolve_experiment_spec(cfg_path)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


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


def _wm_termination_latent_loss(
    wm: object,
    z_goal: torch.Tensor,
    visual_obs: np.ndarray,
    device: torch.device,
) -> float:
    z_cur = encode_visual(wm, visual_obs, device)
    diff = z_cur - z_goal
    return float(torch.sqrt(diff.pow(2).mean() + 1e-8).item())


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
        "variant_name": str(result.get("variant_name", "")),
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
        "success_and_done": int(
            bool(run_stats.get("termination_metric_success", False) and run_stats.get("termination_done", False))
        ),
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
        "run_log": os.path.join(run_dir, "run.log"),
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


class _TeeStream(io.TextIOBase):
    def __init__(self, *targets: io.TextIOBase | None):
        self._targets = [target for target in targets if target is not None]

    def write(self, s: str) -> int:
        for target in self._targets:
            target.write(s)
        return len(s)

    def flush(self) -> None:
        for target in self._targets:
            target.flush()


class _ExperimentProgress:
    def __init__(self, *, total_runs: int, mode: str):
        self.total_runs = max(0, int(total_runs))
        self.mode = str(mode).lower()
        self.completed = 0
        self.successes = 0
        self.start_time = time.time()
        self._interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def _status_line(self, row: dict | None = None) -> str:
        elapsed = max(0.0, time.time() - self.start_time)
        rate = 0.0 if self.completed <= 0 else elapsed / float(self.completed)
        remaining = max(0, self.total_runs - self.completed)
        eta = rate * float(remaining)
        parts = [
            f"[experiment] {self.completed}/{self.total_runs}",
            f"success={self.successes}",
            f"elapsed={elapsed:.1f}s",
            f"eta={eta:.1f}s",
        ]
        if row is not None:
            parts.extend(
                [
                    f"variant={row.get('variant_name', '')}",
                    f"rollout={row.get('rollout_id', '')}",
                    f"term={row.get('termination_reason', '')}",
                ]
            )
        return "  ".join(parts)

    def advance(self, row: dict) -> None:
        self.completed += 1
        self.successes += int(bool(row.get("success", 0)))
        if self.mode == "quiet":
            return
        line = self._status_line(row)
        if self.mode == "compact" and self._interactive:
            print("\r" + line + " " * 8, end="", flush=True)
        else:
            print(line)

    def finish(self) -> None:
        if self.mode == "compact" and self._interactive:
            print()


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _capture_output(mode: str, fn):
    buffer = io.StringIO()
    stdout_target = sys.stdout if str(mode).lower() == "verbose" else None
    stderr_target = sys.stderr if str(mode).lower() == "verbose" else None
    tee_out = _TeeStream(buffer, stdout_target)
    tee_err = _TeeStream(buffer, stderr_target)
    with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
        result = fn()
    return result, buffer.getvalue()


def _write_run_log(run_dir: str, log_text: str) -> None:
    path = os.path.join(run_dir, "run.log")
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(log_text)


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
            arr = np.concatenate(
                [arr, np.full((max_steps - arr.shape[0],), pad_value, dtype=np.float32)],
                axis=0,
            )
        else:
            arr = arr[:max_steps]
        stacked.append(arr)
    mat = np.stack(stacked, axis=0)
    return np.nanmedian(mat, axis=0), np.nanmean(mat, axis=0)


def _metric_array(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def aggregate_summary(
    rows: Sequence[dict],
    baseline_variant: str,
    variant_order: Sequence[str],
) -> tuple[list[dict], list[dict]]:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant_name"]), []).append(row)
    baseline_rows = {str(row["rollout_id"]): row for row in by_variant.get(str(baseline_variant), [])}
    summary_rows: list[dict] = []
    paired_rows: list[dict] = []
    ordered_variant_names = [name for name in variant_order if name in by_variant]
    ordered_variant_names.extend(sorted(name for name in by_variant if name not in ordered_variant_names))

    for variant_name in ordered_variant_names:
        variant_rows = by_variant[variant_name]
        success_vals = _metric_array(variant_rows, "success")
        success_and_done = _metric_array(variant_rows, "success_and_done")
        executed_steps = _metric_array(variant_rows, "executed_steps")
        plans = _metric_array(variant_rows, "plans")
        final_pos = _metric_array(variant_rows, "final_pos_diff")
        final_angle = _metric_array(variant_rows, "final_angle_diff")
        final_eef = _metric_array(variant_rows, "final_eef_diff")
        best_pos = _metric_array(variant_rows, "best_pos_diff")
        best_angle = _metric_array(variant_rows, "best_angle_diff")
        best_eef = _metric_array(variant_rows, "best_eef_diff")
        final_cov = _metric_array(variant_rows, "final_coverage")
        auc_pos = _metric_array(variant_rows, "auc_pos_diff")
        auc_angle = _metric_array(variant_rows, "auc_angle_diff")
        auc_eef = _metric_array(variant_rows, "auc_eef_diff")
        bits = _metric_array(variant_rows, "bits_used_total")
        bits_per_step = _metric_array(variant_rows, "bits_used_per_step")
        flops = _metric_array(variant_rows, "flops_used_total")
        flops_per_step = _metric_array(variant_rows, "flops_used_per_step")
        plan_time = _metric_array(variant_rows, "plan_time_total_sec")
        plan_time_per_replan = _metric_array(variant_rows, "plan_time_per_replan_sec")
        term_reasons = [str(row["termination_reason"]) for row in variant_rows]
        reason_counts = {reason: term_reasons.count(reason) for reason in sorted(set(term_reasons))}

        summary_row = {
            "variant_name": variant_name,
            "n_rollouts": int(len(variant_rows)),
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

        if variant_name == baseline_variant:
            continue

        success_win_count = 0
        success_loss_count = 0
        success_tie_count = 0
        pos_better_count = 0
        pos_worse_count = 0
        pos_tie_count = 0
        for row in variant_rows:
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
                    "baseline_variant": baseline_variant,
                    "variant_name": variant_name,
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

    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant_name"]), []).append(row)
    variant_names = [row["variant_name"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(variant_names, [row["success_rate"] for row in summary_rows])
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Variant")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "success_rates.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].boxplot([[row["final_pos_diff"] for row in by_variant[name]] for name in variant_names], tick_labels=variant_names)
    axes[0].set_title("Final Pos Diff")
    axes[1].boxplot([[row["final_angle_diff"] for row in by_variant[name]] for name in variant_names], tick_labels=variant_names)
    axes[1].set_title("Final Angle Diff")
    axes[2].boxplot([[row["final_coverage"] for row in by_variant[name]] for name in variant_names], tick_labels=variant_names)
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
        ax.annotate(row["variant_name"], (row["mean_bits_used_total"], row["success_rate"]))
    ax.set_xlabel("Mean Bits Used Total")
    ax.set_ylabel("Success Rate")
    ax.set_title("Compute vs Success")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compute_vs_success.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for variant_name in variant_names:
        variant_rows = by_variant[variant_name]
        med, _ = _stepwise_curve(variant_rows, "pos_diffs", max_steps=max_steps)
        axes[0].plot(med, label=variant_name)
        med, _ = _stepwise_curve(variant_rows, "angle_diffs", max_steps=max_steps)
        axes[1].plot(med, label=variant_name)
        med, _ = _stepwise_curve(variant_rows, "eef_diffs", max_steps=max_steps)
        axes[2].plot(med, label=variant_name)
    axes[0].set_title("Median Pos Diff")
    axes[1].set_title("Median Angle Diff")
    axes[2].set_title("Median EEF Diff")
    for ax in axes:
        ax.set_xlabel("Step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "median_stepwise_curves.png"))
    plt.close(fig)


def _run_variant_task(task: dict) -> dict:
    cfg = OmegaConf.create(task["cfg"])
    terminal_mode = str(task.get("terminal_mode", "compact")).lower()
    run_dir = str(task["run_dir"])

    def _execute():
        result = run_plan_session(
            cfg,
            rollout_selection=task["selection"],
            schedule_name=task["variant_name"],
            print_summary=False,
        )
        result["variant_name"] = task["variant_name"]
        save_plan_result(result, run_dir, save_media=bool(result["cfg"].save))
        return result

    try:
        result, log_text = _capture_output(terminal_mode, _execute)
    except Exception as exc:
        _write_run_log(run_dir, f"{type(exc).__name__}: {exc}\n")
        raise

    _write_run_log(run_dir, log_text)
    return result_row(result, run_dir)


def _make_exec_env(cfg: DictConfig):
    register_plan_env(cfg)
    env_wrapped = gym_make_versioned(str(cfg.env_id), cfg.env)
    return unwrap_env(env_wrapped)


def _finalize_wm_state(state: dict, term_reason: str, term_step: int) -> tuple[dict, dict]:
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
    trace = {
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
    }
    return trace, run_stats


def _run_wm_batched_variants(
    variants: Sequence[ExperimentVariant],
    selection: dict,
    run_root: str,
    seed_base: int,
    terminal_mode: str,
) -> list[dict]:
    runtime_cfgs = [variant.plan.runtime_cfg for variant in variants]
    run_dirs = [
        os.path.join(run_root, "traces", variant.name, selection["rollout_id"])
        for variant in variants
    ]

    def _execute() -> list[dict]:
        runtime = build_plan_runtime(runtime_cfgs[0])
        wm = runtime["wm"]
        device = runtime["device"]
        envs = [runtime["env"]]
        for _ in range(1, len(runtime_cfgs)):
            envs.append(_make_exec_env(runtime_cfgs[0]))

        fidelity_cfgs = [OmegaConf.to_container(cfg.fidelity, resolve=True) for cfg in runtime_cfgs]
        batch_planner = BatchedLatentCEMPlanner(
            world_model=wm,
            fidelity_cfgs=fidelity_cfgs,
            horizon=int(runtime_cfgs[0].mpc.horizon),
            action_dim=int(envs[0].action_dim),
            pop_size=int(runtime_cfgs[0].cem.pop_size),
            elite_frac=float(runtime_cfgs[0].cem.elite_frac),
            n_iter=int(runtime_cfgs[0].cem.n_iter),
            init_std=float(runtime_cfgs[0].cem.init_std),
            action_low=runtime_cfgs[0].cem.action_low,
            action_high=runtime_cfgs[0].cem.action_high,
            objective_cfg=OmegaConf.to_container(runtime_cfgs[0].objective, resolve=True),
            drop_tail_on_coarsen=True,
            warm_start=bool(runtime_cfgs[0].cem.warm_start),
            device=device,
        )
        init_state, goal_state, sample_meta = load_selected_rollout(
            envs[0],
            runtime_cfgs[0],
            runtime["wm_cfg"],
            selection,
        )
        goal_obs, _ = envs[0].prepare(seed=0, init_state=goal_state)
        set_goal_pose(envs[0], goal_state)
        goal_obs["visual"] = envs[0].render("rgb_array", include_start_pose=False)
        z_goal = encode_visual(wm, goal_obs["visual"], device)

        variant_states: list[dict] = []
        for env, cfg, variant in zip(envs, runtime_cfgs, variants):
            set_start_pose(env, init_state)
            obs, cur_state = env.prepare(seed=0, init_state=init_state, goal_state=goal_state)
            set_execution_fidelity_finest(env)
            obs["visual"] = env.render("rgb_array", include_start_pose=False)
            variant_states.append(
                {
                    "env": env,
                    "cfg": cfg,
                    "variant_name": variant.name,
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

        initial_term = envs[0].eval_termination(goal_state, variant_states[0]["cur_state"], done=None, info=None)
        initial_latent = _wm_termination_latent_loss(
            wm,
            z_goal,
            np.asarray(variant_states[0]["obs"]["visual"]),
            device,
        )
        initial_term["latent_loss"] = float(initial_latent)
        initial_term["state_dist"] = float(initial_latent)
        if bool(initial_term["done"]):
            rows = []
            for state, run_dir in zip(variant_states, run_dirs):
                state["last_term"] = initial_term
                trace, run_stats = _finalize_wm_state(state, "initial_env_done", 0)
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
                    "variant_name": state["variant_name"],
                }
                save_plan_result(result, run_dir, save_media=bool(state["cfg"].save))
                rows.append(result_row(result, run_dir))
            return rows

        steps = int(runtime_cfgs[0].mpc.steps)
        horizon = int(runtime_cfgs[0].mpc.horizon)
        replan_every = int(runtime_cfgs[0].mpc.replan_every)
        n_replans = max(1, int(np.ceil(steps / replan_every)))
        t = 0
        replan_idx = 0
        prev_exec_steps = 0

        while t < steps and any(not state["done"] for state in variant_states):
            mpc_progress = 0.0 if n_replans <= 1 else replan_idx / (n_replans - 1)
            z_batch = torch.cat(
                [encode_visual(wm, state["obs"]["visual"], device) for state in variant_states],
                dim=0,
            )
            plan_seeds = [
                int(seed_base + 1000003 * idx + 7919 * replan_idx + 101 * t)
                for idx in range(len(variant_states))
            ]
            batch_results = batch_planner.plan_batch(
                z0=z_batch,
                z_goal=z_goal.expand(len(variant_states), -1),
                mpc_progress=mpc_progress,
                warm_start_steps=int(prev_exec_steps),
                seeds=plan_seeds,
            )

            for variant_idx, (state, batch_result) in enumerate(zip(variant_states, batch_results)):
                info = batch_result.info
                action_seq = np.asarray(batch_result.action_seq.detach().cpu().numpy(), dtype=np.float32)
                bits_used = int(getattr(info, "bits_used_estimate", 0))
                flops_used = int(bits_to_flops_estimate(bits_used))
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
                        "seed": int(plan_seeds[variant_idx]),
                        "action_seq": action_seq.tolist(),
                        "base_level_idx": int(getattr(info, "base_level_idx", -1)),
                        "rollout_level_indices": [int(x) for x in list(getattr(info, "rollout_level_indices", []))],
                        "rollout_latent_losses": [float(x) for x in list(getattr(info, "rollout_latent_losses", []))],
                        "iter_best_rollout_latent_losses": [
                            [float(y) for y in list(x)]
                            for x in list(getattr(info, "iter_best_rollout_latent_losses", []))
                        ],
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
                for state in variant_states:
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
                    latent_loss = _wm_termination_latent_loss(
                        wm,
                        z_goal,
                        np.asarray(obs["visual"]),
                        device,
                    )
                    term["latent_loss"] = float(latent_loss)
                    term["state_dist"] = float(latent_loss)
                    state["last_term"] = term
                    state["pos_diffs"].append(float(term["pos_diff"]))
                    state["angle_diffs"].append(float(term["angle_diff"]))
                    state["eef_diffs"].append(float(term["eef_diff"]))
                    state["coverages"].append(float(term["coverage"]) if term["coverage"] is not None else float("nan"))
                    state["metric_success_flags"].append(bool(term["success"]))
                    state["done_flags"].append(bool(term["done"]))
                    state["state_dists"].append(float(term["state_dist"]))
                    if bool(term["done"]):
                        state["done"] = True
                t += 1
                if t >= steps or all(state["done"] for state in variant_states):
                    break

            prev_exec_steps = int(n_exec)
            replan_idx += 1

        rows = []
        for state, run_dir in zip(variant_states, run_dirs):
            trace, run_stats = _finalize_wm_state(
                state,
                "env_done" if state["done"] else "max_steps",
                max(0, len(state["trajectory"]) - 1),
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
                "variant_name": state["variant_name"],
            }
            save_plan_result(result, run_dir, save_media=bool(state["cfg"].save))
            rows.append(result_row(result, run_dir))
        return rows

    try:
        rows, log_text = _capture_output(terminal_mode, _execute)
    except Exception as exc:
        for run_dir in run_dirs:
            _write_run_log(run_dir, f"{type(exc).__name__}: {exc}\n")
        raise

    for run_dir in run_dirs:
        _write_run_log(run_dir, log_text)
    return rows


def _group_wm_variants(variants: Sequence[ExperimentVariant]) -> tuple[list[list[ExperimentVariant]], list[ExperimentVariant]]:
    wm_groups: dict[str, list[ExperimentVariant]] = defaultdict(list)
    singles: list[ExperimentVariant] = []
    for variant in variants:
        if variant.plan.active_backend_kind() != "wm":
            singles.append(variant)
            continue
        wm_groups[variant.plan.variant_compatibility_signature()].append(variant)
    batched = []
    for group in wm_groups.values():
        if len(group) > 1:
            batched.append(group)
        else:
            singles.extend(group)
    return batched, singles


def _resolved_experiment_dict(spec: ExperimentSpec) -> dict:
    return {
        "name": spec.name,
        "baseline": spec.baseline,
        "rollouts": spec.rollouts,
        "execution": spec.execution,
        "terminal": spec.terminal,
        "reporting": spec.reporting,
        "shared_plan": OmegaConf.to_container(spec.shared_plan.clean_cfg, resolve=True),
        "variants": [
            {
                "name": variant.name,
                "plan": OmegaConf.to_container(variant.plan.clean_cfg, resolve=True),
            }
            for variant in spec.variants
        ],
    }


def run_experiment(spec_or_path: str | ExperimentSpec, *, output_root: str | None = None) -> str:
    spec = load_experiment_spec(spec_or_path) if isinstance(spec_or_path, str) else spec_or_path
    candidates = enumerate_rollout_candidates(spec.shared_plan)
    selected = select_rollouts(spec.rollouts, candidates)
    variant_order = spec.variant_names()
    terminal_mode = str(spec.terminal.get("mode", "compact")).lower()
    progress = _ExperimentProgress(total_runs=len(selected) * len(spec.variants), mode=terminal_mode)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root or spec.reporting.get("output_root", "rollouts")
    run_dir = os.path.join(str(run_root), f"experiment_{_safe_name(spec.name)}_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "experiment_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(_resolved_experiment_dict(spec), f, indent=2)
    with open(os.path.join(run_dir, "selected_rollouts.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    rows: list[dict] = []
    batched_groups, single_variants = _group_wm_variants(spec.variants)
    for selection in selected:
        for group_idx, variants in enumerate(batched_groups):
            seed_base = int(spec.rollouts["seed"] + 100003 * selection["rollout_index"] + 10007 * group_idx)
            group_rows = _run_wm_batched_variants(
                variants=variants,
                selection=selection,
                run_root=run_dir,
                seed_base=seed_base,
                terminal_mode=terminal_mode,
            )
            rows.extend(group_rows)
            for row in group_rows:
                progress.advance(row)

    tasks = []
    for selection in selected:
        for variant in single_variants:
            tasks.append(
                {
                    "cfg": OmegaConf.to_container(variant.plan.runtime_cfg, resolve=True),
                    "selection": selection,
                    "variant_name": variant.name,
                    "run_dir": os.path.join(run_dir, "traces", variant.name, selection["rollout_id"]),
                    "terminal_mode": terminal_mode,
                }
            )

    exec_mode = str(spec.execution.get("mode", "auto")).lower()
    if exec_mode == "auto":
        exec_mode = "process"
    if exec_mode == "process" and len(tasks) > 0:
        max_workers = min(int(spec.execution.get("max_workers", 1)), max(1, len(tasks)))
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_run_variant_task, task) for task in tasks]
                for fut in as_completed(futures):
                    row = fut.result()
                    rows.append(row)
                    progress.advance(row)
        except Exception as exc:
            print(f"[experiment][warn] process parallelism unavailable ({exc}); falling back to serial.")
            for task in tasks:
                row = _run_variant_task(task)
                rows.append(row)
                progress.advance(row)
    else:
        for task in tasks:
            row = _run_variant_task(task)
            rows.append(row)
            progress.advance(row)

    progress.finish()

    rows = sorted(rows, key=lambda row: (row["rollout_index"], row["variant_name"]))
    summary_rows, paired_rows = aggregate_summary(
        rows,
        baseline_variant=spec.baseline,
        variant_order=variant_order,
    )
    _write_rows_csv(os.path.join(run_dir, "per_rollout.csv"), rows)
    _write_rows_csv(os.path.join(run_dir, "summary.csv"), summary_rows)
    baseline_filename = f"paired_deltas_vs_{_safe_name(spec.baseline)}.csv"
    _write_rows_csv(os.path.join(run_dir, baseline_filename), paired_rows)
    _write_rows_csv(os.path.join(run_dir, "paired_deltas_vs_baseline.csv"), paired_rows)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": spec.name,
                "created_at": run_ts,
                "num_rollouts": int(spec.rollouts["num_rollouts"]),
                "baseline_variant": spec.baseline,
                "summary": summary_rows,
            },
            f,
            indent=2,
        )
    save_summary_plots(rows, summary_rows, run_dir, max_steps=int(spec.shared_plan.budget["max_env_steps"]))
    report_path = write_experiment_report(
        run_dir,
        summary_rows,
        rows,
        experiment_name=spec.name,
        baseline_variant=spec.baseline,
        summary_plot_files=(
            "success_rates.png",
            "final_metrics_boxplots.png",
            "compute_vs_success.png",
            "median_stepwise_curves.png",
        ),
    )
    if terminal_mode != "quiet":
        print(f"[experiment] wrote results to {run_dir}")
        print(f"[experiment] report: {report_path}")
        print(f"[experiment] review: python3 scripts/experiment_review.py --run-dir {run_dir}")
        for summary in summary_rows:
            print(
                f"[experiment][summary] variant={summary['variant_name']} "
                f"success_rate={summary['success_rate']:.3f} "
                f"mean_final_pos_diff={summary['mean_final_pos_diff']:.3f} "
                f"mean_bits_used_total={summary['mean_bits_used_total']:.1f} "
                f"mean_plan_time_total_sec={summary['mean_plan_time_total_sec']:.3f}"
            )
    return run_dir
