from __future__ import annotations

import copy
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hudm.artifacts import save_plan_result
from hudm.experiment_bundle import (
    EXPERIMENT_SCHEMA_VERSION,
    REVIEWER_SCHEMA_VERSION,
    trace_dir,
    write_experiment_bundle,
)
from hudm.config import resolve_experiment_spec
from hudm.metrics import termination_success
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
    shared_plan_time_total = float(run_stats.get("shared_plan_time_total_sec", plan_time_total))
    success = bool(
        run_stats.get("termination_metric_success", result.get("success", False))
        and run_stats.get("termination_done", result.get("success", False))
    )
    return {
        "variant_name": str(result.get("variant_name", "")),
        "rollout_id": str(sample.get("rollout_id", rollout_id(sample))),
        "rollout_index": int(sample.get("rollout_index", -1)),
        "episode_index": int(sample.get("episode_index", -1)),
        "start_index": int(sample.get("start_index", -1)),
        "goal_index": int(sample.get("goal_index", -1)),
        "success": int(success),
        "termination_reason": str(run_stats.get("termination_reason", "unknown")),
        "termination_step": int(run_stats.get("termination_step", -1)),
        "executed_steps": executed_steps,
        "plans": plans,
        "success_and_done": int(success),
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
        "shared_plan_time_total_sec": shared_plan_time_total,
        "plan_time_per_replan_sec": float(plan_time_total / max(1, plans)),
        "run_dir": run_dir,
        "trace_json": os.path.join(run_dir, "trace.json"),
        "trace_npz": os.path.join(run_dir, "trace.npz"),
        "run_log": os.path.join(run_dir, "run.log"),
    }


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


def _metric_array(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def _paired_rows_vs_reference(
    rows: Sequence[dict],
    *,
    reference_variant: str,
    variant_order: Sequence[str],
) -> list[dict]:
    reference_name = str(reference_variant or "").strip()
    if not reference_name:
        return []

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_variant.setdefault(str(row["variant_name"]), []).append(row)
    reference_rows = {
        str(row["rollout_id"]): row
        for row in by_variant.get(reference_name, [])
    }
    if len(reference_rows) <= 0:
        return []

    ordered_variant_names = [name for name in variant_order if name in by_variant]
    ordered_variant_names.extend(sorted(name for name in by_variant if name not in ordered_variant_names))

    paired_rows: list[dict] = []
    for variant_name in ordered_variant_names:
        if variant_name == reference_name:
            continue
        for row in by_variant.get(variant_name, []):
            rollout_id_value = str(row["rollout_id"])
            reference_row = reference_rows.get(rollout_id_value)
            if reference_row is None:
                continue
            paired_rows.append(
                {
                    "reference_variant": reference_name,
                    "variant_name": variant_name,
                    "rollout_id": rollout_id_value,
                    "success_delta": int(row["success"]) - int(reference_row["success"]),
                    "final_pos_diff_delta": float(row["final_pos_diff"]) - float(reference_row["final_pos_diff"]),
                    "final_angle_diff_delta": float(row["final_angle_diff"]) - float(reference_row["final_angle_diff"]),
                    "final_eef_diff_delta": float(row["final_eef_diff"]) - float(reference_row["final_eef_diff"]),
                    "final_coverage_delta": float(row["final_coverage"]) - float(reference_row["final_coverage"]),
                    "bits_used_total_delta": float(row["bits_used_total"]) - float(reference_row["bits_used_total"]),
                    "plan_time_total_sec_delta": float(row["plan_time_total_sec"]) - float(reference_row["plan_time_total_sec"]),
                }
            )
    return paired_rows


def aggregate_summary(
    rows: Sequence[dict],
    variant_order: Sequence[str],
    *,
    baseline_variant: str | None = None,
) -> tuple[list[dict], list[dict]]:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant_name"]), []).append(row)
    summary_rows: list[dict] = []
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
    paired_rows = _paired_rows_vs_reference(
        rows,
        reference_variant=str(baseline_variant or ""),
        variant_order=ordered_variant_names,
    )
    return summary_rows, paired_rows


def _experiment_payload(
    spec: ExperimentSpec,
    *,
    run_ts: str,
    variant_order: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "reviewer_version": REVIEWER_SCHEMA_VERSION,
        "experiment_name": spec.name,
        "created_at": run_ts,
        "baseline_variant": spec.baseline or "",
        "variant_order": list(variant_order),
        "num_rollouts": int(spec.rollouts["num_rollouts"]),
        "rollouts": dict(spec.rollouts),
        "execution": dict(spec.execution),
        "terminal": dict(spec.terminal),
        "reporting": dict(spec.reporting),
        "shared_plan": OmegaConf.to_container(spec.shared_plan.clean_cfg, resolve=True),
        "variants": [
            {
                "name": variant.name,
                "plan": OmegaConf.to_container(variant.plan.clean_cfg, resolve=True),
            }
            for variant in spec.variants
        ],
    }


def _write_experiment_bundle_snapshot(
    run_dir: str,
    *,
    experiment_payload: dict[str, Any],
    selected_rollouts: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    variant_order: Sequence[str],
    total_runs: int,
) -> None:
    run_rows = sorted(
        list(rows),
        key=lambda row: (
            int(row.get("rollout_index", -1)),
            str(row.get("variant_name", "")),
            str(row.get("rollout_id", "")),
        ),
    )
    variant_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    if len(run_rows) > 0:
        variant_rows, paired_rows = aggregate_summary(
            run_rows,
            variant_order=variant_order,
            baseline_variant=str(experiment_payload.get("baseline_variant", "")),
        )

    completed_runs = int(len(run_rows))
    if total_runs <= 0:
        bundle_status = "complete"
    elif completed_runs <= 0:
        bundle_status = "pending"
    elif completed_runs < int(total_runs):
        bundle_status = "running"
    else:
        bundle_status = "complete"

    payload = dict(experiment_payload)
    payload["completed_runs"] = completed_runs
    payload["total_runs"] = int(total_runs)
    payload["bundle_status"] = bundle_status
    payload["partial_bundle"] = bundle_status != "complete"

    write_experiment_bundle(
        run_dir,
        experiment_payload=payload,
        selected_rollouts=selected_rollouts,
        run_rows=run_rows,
        variant_rows=variant_rows,
        paired_rows=paired_rows,
    )


def _run_variant_task(task: dict) -> dict:
    cfg = OmegaConf.create(task["cfg"])
    execution_cfg = None
    if task.get("execution_cfg", None) is not None:
        execution_cfg = OmegaConf.create(task["execution_cfg"])
    terminal_mode = str(task.get("terminal_mode", "compact")).lower()
    run_dir = str(task["run_dir"])

    def _execute():
        result = run_plan_session(
            cfg,
            rollout_selection=task["selection"],
            schedule_name=task["variant_name"],
            print_summary=False,
            execution_cfg=execution_cfg,
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
        "shared_plan_time_total_sec": float(state.get("shared_plan_time_total", 0.0)),
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
        trace_dir(run_root, variant.name, selection["rollout_id"])
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
                    "shared_plan_time_total": 0.0,
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
                    "success": termination_success(initial_term),
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
                shared_plan_time = float(getattr(info, "plan_time_sec", 0.0))
                plan_time = shared_plan_time / max(1, len(variant_states))
                state["bits_total"] += bits_used
                state["flops_total"] += flops_used
                state["plan_time_total"] += plan_time
                state["shared_plan_time_total"] += shared_plan_time
                state["n_plans"] += 1
                state["replans"].append(
                    {
                        "replan_idx": int(replan_idx),
                        "step_start": int(t),
                        "mpc_progress": float(mpc_progress),
                        "seed": int(plan_seeds[variant_idx]),
                        "action_seq": action_seq.tolist(),
                        "start_level_idx": int(getattr(info, "start_level_idx", getattr(info, "base_level_idx", -1))),
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
                        "shared_plan_time_sec": shared_plan_time,
                        "plan_time_allocation": "equal_split",
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
                "success": termination_success(state["last_term"]),
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


def _normalize_execution_mode(mode: str | None) -> str:
    exec_mode = str(mode or "auto").lower()
    if exec_mode == "auto":
        return "process"
    return exec_mode


def _build_variant_task(
    *,
    variant: ExperimentVariant,
    selection: dict,
    run_root: str,
    terminal_mode: str,
    backend_kind: str,
    lane_key: str,
    execution_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cfg": OmegaConf.to_container(variant.plan.runtime_cfg, resolve=True),
        "execution_cfg": execution_cfg,
        "selection": selection,
        "variant_name": variant.name,
        "run_dir": trace_dir(run_root, variant.name, selection["rollout_id"]),
        "terminal_mode": terminal_mode,
        "backend_kind": backend_kind,
        "lane_key": lane_key,
    }


def _build_execution_lanes(
    spec: ExperimentSpec,
    *,
    selected: Sequence[dict],
    run_root: str,
    terminal_mode: str,
) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    batched_groups, single_variants = _group_wm_variants(spec.variants)
    baseline_execution_cfg = None
    baseline_backend_kind = None
    if spec.baseline:
        for variant in spec.variants:
            if variant.name == spec.baseline:
                baseline_execution_cfg = OmegaConf.to_container(variant.plan.runtime_cfg, resolve=True)
                baseline_backend_kind = variant.plan.active_backend_kind()
                break
    for selection in selected:
        for group_idx, variants in enumerate(batched_groups):
            lanes.append(
                {
                    "lane_type": "wm_batch",
                    "backend_kind": "wm",
                    "lane_key": f"wm_batch:{selection['rollout_id']}:{group_idx}",
                    "selection": selection,
                    "variants": list(variants),
                    "seed_base": int(spec.rollouts["seed"] + 100003 * selection["rollout_index"] + 10007 * group_idx),
                }
            )

    tasks_by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for selection in selected:
        for variant in single_variants:
            backend_kind = variant.plan.active_backend_kind()
            lane_key = f"task_pool:{backend_kind}"
            execution_cfg = (
                baseline_execution_cfg
                if baseline_execution_cfg is not None and backend_kind == baseline_backend_kind
                else None
            )
            tasks_by_backend[backend_kind].append(
                _build_variant_task(
                    variant=variant,
                    selection=selection,
                    run_root=run_root,
                    terminal_mode=terminal_mode,
                    backend_kind=backend_kind,
                    lane_key=lane_key,
                    execution_cfg=execution_cfg,
                )
            )

    for backend_kind in ("wm", "particle_sim", "gt_env"):
        tasks = tasks_by_backend.get(backend_kind, [])
        if len(tasks) <= 0:
            continue
        lanes.append(
            {
                "lane_type": "task_pool",
                "backend_kind": backend_kind,
                "lane_key": f"task_pool:{backend_kind}",
                "tasks": tasks,
            }
        )
    return lanes


def _normalize_device_slots(device_slots: Any) -> list[str]:
    if device_slots is None:
        return []
    if isinstance(device_slots, str):
        slots = [device_slots]
    else:
        slots = list(device_slots)
    normalized: list[str] = []
    seen: set[str] = set()
    for slot in slots:
        slot_name = str(slot).strip()
        if not slot_name or slot_name in seen:
            continue
        seen.add(slot_name)
        normalized.append(slot_name)
    return normalized


def _particle_task_requested_device(task: dict[str, Any]) -> str:
    cfg = task.get("cfg", {})
    particle_env = cfg.get("particle_env", {}) if isinstance(cfg, dict) else {}
    fidelity_env = particle_env.get("fidelity_env", {}) if isinstance(particle_env, dict) else {}
    requested = str(fidelity_env.get("device", "auto")).strip().lower()
    if requested in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        return "cuda:0"
    return requested


def _task_with_particle_device(task: dict[str, Any], device_slot: str) -> dict[str, Any]:
    patched = dict(task)
    cfg = copy.deepcopy(task.get("cfg", {}))
    particle_env = cfg.setdefault("particle_env", {})
    fidelity_env = particle_env.setdefault("fidelity_env", {})
    fidelity_env["device"] = str(device_slot)
    patched["cfg"] = cfg
    patched["device_slot"] = str(device_slot)
    return patched


def _prepare_task_lane(
    tasks: Sequence[dict[str, Any]],
    *,
    backend_kind: str,
    max_workers: int,
    device_slots: Any = None,
) -> tuple[list[dict[str, Any]], int]:
    lane_tasks = list(tasks)
    worker_count = min(int(max_workers), max(1, len(lane_tasks)))
    if len(lane_tasks) <= 0 or str(backend_kind) != "particle_sim":
        return lane_tasks, worker_count

    normalized_slots = _normalize_device_slots(device_slots)
    if len(normalized_slots) > 0:
        slotted_tasks = [
            _task_with_particle_device(task, normalized_slots[idx % len(normalized_slots)])
            for idx, task in enumerate(lane_tasks)
        ]
        return slotted_tasks, min(worker_count, len(normalized_slots))

    resolved_device = _particle_task_requested_device(lane_tasks[0])
    if not resolved_device.startswith("cuda"):
        return lane_tasks, worker_count

    pinned_tasks = [
        _task_with_particle_device(task, resolved_device)
        for task in lane_tasks
    ]
    return pinned_tasks, 1


def _run_task_lane(
    tasks: Sequence[dict[str, Any]],
    *,
    exec_mode: str,
    max_workers: int,
    lane_key: str,
    backend_kind: str,
    device_slots: Any = None,
    row_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict]:
    lane_rows: list[dict] = []
    if len(tasks) <= 0:
        return lane_rows
    lane_tasks, worker_count = _prepare_task_lane(
        tasks,
        backend_kind=backend_kind,
        max_workers=max_workers,
        device_slots=device_slots,
    )
    if exec_mode == "process":
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as ex:
                futures = [ex.submit(_run_variant_task, task) for task in lane_tasks]
                for fut in as_completed(futures):
                    row = fut.result()
                    lane_rows.append(row)
                    if row_callback is not None:
                        row_callback(row)
            return lane_rows
        except Exception as exc:
            print(
                f"[experiment][warn] process parallelism unavailable ({exc}); "
                f"falling back to serial. lane={lane_key}"
            )
    for task in lane_tasks:
        row = _run_variant_task(task)
        lane_rows.append(row)
        if row_callback is not None:
            row_callback(row)
    return lane_rows


def run_experiment(spec_or_path: str | ExperimentSpec, *, output_root: str | None = None) -> str:
    spec = load_experiment_spec(spec_or_path) if isinstance(spec_or_path, str) else spec_or_path
    candidates = enumerate_rollout_candidates(spec.shared_plan)
    selected = select_rollouts(spec.rollouts, candidates)
    variant_order = spec.variant_names()
    total_runs = len(selected) * len(spec.variants)
    terminal_mode = str(spec.terminal.get("mode", "compact")).lower()
    progress = _ExperimentProgress(total_runs=total_runs, mode=terminal_mode)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root or spec.reporting.get("output_root", "rollouts")
    run_dir = os.path.join(str(run_root), f"experiment_{_safe_name(spec.name)}_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    experiment_payload = _experiment_payload(spec, run_ts=run_ts, variant_order=variant_order)
    rows: list[dict] = []
    _write_experiment_bundle_snapshot(
        run_dir,
        experiment_payload=experiment_payload,
        selected_rollouts=selected,
        rows=rows,
        variant_order=variant_order,
        total_runs=total_runs,
    )

    def _record_row(row: dict[str, Any]) -> None:
        rows.append(row)
        progress.advance(row)
        _write_experiment_bundle_snapshot(
            run_dir,
            experiment_payload=experiment_payload,
            selected_rollouts=selected,
            rows=rows,
            variant_order=variant_order,
            total_runs=total_runs,
        )

    exec_mode = _normalize_execution_mode(spec.execution.get("mode", "auto"))
    max_workers = int(spec.execution.get("max_workers", 1))
    lanes = _build_execution_lanes(
        spec,
        selected=selected,
        run_root=run_dir,
        terminal_mode=terminal_mode,
    )
    for lane in lanes:
        if lane["lane_type"] == "wm_batch":
            lane_rows = _run_wm_batched_variants(
                variants=lane["variants"],
                selection=lane["selection"],
                run_root=run_dir,
                seed_base=int(lane["seed_base"]),
                terminal_mode=terminal_mode,
            )
            for row in lane_rows:
                _record_row(row)
        else:
            _run_task_lane(
                lane["tasks"],
                exec_mode=exec_mode,
                max_workers=max_workers,
                lane_key=str(lane["lane_key"]),
                backend_kind=str(lane["backend_kind"]),
                device_slots=(
                    spec.execution.get("particle", {}).get("device_slots")
                    if str(lane["backend_kind"]) == "particle_sim"
                    else None
                ),
                row_callback=_record_row,
            )

    progress.finish()
    rows_sorted = sorted(rows, key=lambda row: (row["rollout_index"], row["variant_name"]))
    _write_experiment_bundle_snapshot(
        run_dir,
        experiment_payload=experiment_payload,
        selected_rollouts=selected,
        rows=rows_sorted,
        variant_order=variant_order,
        total_runs=total_runs,
    )
    if terminal_mode != "quiet":
        print(f"[experiment] wrote results to {run_dir}")
        print(f"[experiment] review: python3 scripts/experiment_review.py --run-dir {run_dir}")
    return run_dir
