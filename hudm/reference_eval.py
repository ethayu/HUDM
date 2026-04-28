from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from hudm.artifacts import (
    infer_action_overlay_spec,
    overlay_action_targets_on_frames,
    wm_decode_frame,
    write_video_mp4,
)
from hudm.experiment_bundle import review_derived_dir, trace_dir
from hudm.metrics import tee_pose_coverage_px
from hudm.runtime import build_plan_runtime, encode_visual
from hudm.session_helpers import set_execution_fidelity_finest, set_start_pose


REFERENCE_MEDIA_ALIASES: dict[str, list[str]] = {
    "reference_env_replay": ["reference_env_replay.mp4", "baseline_env_replay.mp4"],
    "reference_goal_state": ["reference_goal_state.png", "baseline_goal_state.png"],
}

SUPPORTED_REFERENCE_METRIC_BACKENDS = {"gt_env", "particle_sim"}

@dataclass(frozen=True)
class ReferenceContext:
    baseline_variant: str
    backend: str
    backend_label: str
    plan_cfg: dict[str, Any]
    gt_backend_present: bool

    @property
    def runtime_ready(self) -> bool:
        return "env_id" in self.plan_cfg

    @property
    def metrics_supported(self) -> bool:
        return self.backend in SUPPORTED_REFERENCE_METRIC_BACKENDS

    @property
    def gt_media_allowed(self) -> bool:
        return self.runtime_ready and (self.backend == "gt_env" or self.gt_backend_present)


@dataclass(frozen=True)
class ReferenceEvalResult:
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _variant_backend(plan_cfg: Mapping[str, Any] | None) -> str:
    if not isinstance(plan_cfg, Mapping):
        return ""
    backend = plan_cfg.get("backend", "")
    if isinstance(backend, str):
        return backend.strip().lower()
    if isinstance(backend, Mapping):
        kind = backend.get("kind", "")
        return str(kind).strip().lower()
    return ""


def _backend_label(plan_cfg: Mapping[str, Any] | None) -> str:
    backend = _variant_backend(plan_cfg)
    if backend == "wm":
        wm_cfg = None
        if isinstance(plan_cfg, Mapping):
            wm_cfg = plan_cfg.get("world_model", None)
            if wm_cfg is None:
                wm_cfg = plan_cfg.get("wm", None)
        if isinstance(wm_cfg, Mapping):
            run_dir = str(wm_cfg.get("run_dir", "")).strip()
            if run_dir:
                return f"wm ({os.path.basename(run_dir)})"
        return "wm"
    return backend or "baseline backend"


def _normalize_reference_plan_cfg(plan_cfg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(plan_cfg, Mapping):
        return None
    payload = dict(plan_cfg)
    if "env_id" in payload:
        return payload
    if not {"task", "budget", "planner", "backend", "artifacts"}.issubset(payload.keys()):
        return payload
    try:
        from hudm.config import plan_spec_to_runtime_cfg

        runtime_cfg = plan_spec_to_runtime_cfg(OmegaConf.create(payload))
        return dict(OmegaConf.to_container(runtime_cfg, resolve=True))
    except Exception:
        return payload


def resolve_reference_context(meta: Mapping[str, Any] | None) -> ReferenceContext | None:
    payload = dict(meta or {})
    baseline_variant = str(payload.get("baseline_variant", "")).strip()
    variants = list(payload.get("variants", []))
    if not baseline_variant or len(variants) <= 0:
        return None

    baseline_plan_cfg: dict[str, Any] | None = None
    gt_backend_present = False
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        variant_name = str(variant.get("name", "")).strip()
        plan_cfg = variant.get("plan", None)
        backend = _variant_backend(plan_cfg)
        if backend == "gt_env":
            gt_backend_present = True
        if variant_name == baseline_variant and isinstance(plan_cfg, Mapping):
            baseline_plan_cfg = _normalize_reference_plan_cfg(plan_cfg)

    if baseline_plan_cfg is None:
        return None

    return ReferenceContext(
        baseline_variant=baseline_variant,
        backend=_variant_backend(baseline_plan_cfg),
        backend_label=_backend_label(baseline_plan_cfg),
        plan_cfg=baseline_plan_cfg,
        gt_backend_present=gt_backend_present,
    )


def _reference_cache_dir(run_dir: str, context: ReferenceContext, variant_name: str, rollout_id: str) -> str:
    return os.path.join(
        review_derived_dir(run_dir),
        "reference_eval",
        _safe_name(context.baseline_variant),
        _safe_name(variant_name),
        _safe_name(rollout_id),
    )


def _reference_cache_paths(
    run_dir: str,
    context: ReferenceContext,
    variant_name: str,
    rollout_id: str,
) -> tuple[str, str]:
    root = _reference_cache_dir(run_dir, context, variant_name, rollout_id)
    return os.path.join(root, "summary.json"), os.path.join(root, "arrays.npz")


def _source_mtime(paths: list[str]) -> float:
    mtimes = [os.path.getmtime(path) for path in paths if path and os.path.isfile(path)]
    return max(mtimes) if mtimes else 0.0


def load_cached_reference_eval(
    run_dir: str,
    *,
    context: ReferenceContext,
    variant_name: str,
    rollout_id: str,
    source_paths: list[str] | None = None,
) -> ReferenceEvalResult | None:
    summary_path, arrays_path = _reference_cache_paths(run_dir, context, variant_name, rollout_id)
    if not (os.path.isfile(summary_path) and os.path.isfile(arrays_path)):
        return None
    cache_mtime = min(os.path.getmtime(summary_path), os.path.getmtime(arrays_path))
    if source_paths is not None and cache_mtime < _source_mtime(source_paths):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return ReferenceEvalResult(summary=summary, arrays=arrays)


def load_cached_reference_arrays(
    run_dir: str,
    *,
    context: ReferenceContext,
    variant_name: str,
    rollout_id: str,
) -> dict[str, np.ndarray] | None:
    summary_path, arrays_path = _reference_cache_paths(run_dir, context, variant_name, rollout_id)
    if not (os.path.isfile(summary_path) and os.path.isfile(arrays_path)):
        return None
    with np.load(arrays_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _write_reference_eval_cache(
    run_dir: str,
    *,
    context: ReferenceContext,
    variant_name: str,
    rollout_id: str,
    result: ReferenceEvalResult,
) -> None:
    summary_path, arrays_path = _reference_cache_paths(run_dir, context, variant_name, rollout_id)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result.summary, f, indent=2)
    np.savez_compressed(arrays_path, **result.arrays)


class ReferenceEvaluator:
    def __init__(self, context: ReferenceContext):
        self.context = context
        self.runtime = build_plan_runtime(OmegaConf.create(context.plan_cfg))
        self.backend = str(self.runtime.get("backend", "")).strip().lower()
        self.env = self.runtime["env"]
        self.wm = self.runtime.get("wm", None)
        self.device = self.runtime["device"]
        self.planner = self.runtime["planner"]
        self.particle_backend = getattr(self.planner, "backend", None) if self.backend == "particle_sim" else None

    def evaluate_from_trace(self, trace_meta: Mapping[str, Any], trace_arrays: Mapping[str, np.ndarray]) -> ReferenceEvalResult | None:
        if not self.context.metrics_supported:
            return None
        init_state = np.asarray(trace_meta["init_state"], dtype=np.float32)
        goal_state = np.asarray(trace_meta["goal_state"], dtype=np.float32)
        actions = np.asarray(trace_arrays.get("executed_actions", []), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        trajectory: list[np.ndarray] = []
        pos_diffs: list[float] = []
        angle_diffs: list[float] = []
        eef_diffs: list[float] = []
        coverages: list[float] = []
        state_dists: list[float] = []
        metric_success_flags: list[bool] = []
        done_flags: list[bool] = []

        cur_state = self._prepare_execution(init_state=init_state, goal_state=goal_state, with_visual=False)
        trajectory.append(cur_state.copy())
        initial_term = self._initial_reference_term(goal_state=goal_state, cur_state=cur_state)
        last_success = bool(initial_term["success"])
        last_done = bool(initial_term["done"])
        for action in actions:
            term, cur_state = self._step_reference(
                action=np.asarray(action, dtype=np.float32),
                goal_state=goal_state,
                with_visual=False,
            )
            trajectory.append(cur_state.copy())
            pos_diffs.append(float(term["pos_diff"]))
            angle_diffs.append(float(term["angle_diff"]))
            eef_diffs.append(float(term["eef_diff"]))
            coverages.append(float(term["coverage"]) if term["coverage"] is not None else float("nan"))
            state_dists.append(float(term["state_dist"]))
            metric_success_flags.append(bool(term["success"]))
            done_flags.append(bool(term["done"]))
            last_success = bool(term["success"])
            last_done = bool(term["done"])

        executed_steps = int(actions.shape[0]) if actions.ndim >= 2 else 0
        if executed_steps <= 0 and last_done:
            termination_reason = "reference_initial_done"
            termination_step = 0
        else:
            termination_reason = "reference_env_done" if last_done else "max_steps"
            termination_step = executed_steps if executed_steps > 0 else 0
        final_pos = float(pos_diffs[-1]) if pos_diffs else float("nan")
        final_angle = float(angle_diffs[-1]) if angle_diffs else float("nan")
        final_eef = float(eef_diffs[-1]) if eef_diffs else float("nan")
        finite_cov = [float(x) for x in coverages if np.isfinite(x)]
        final_cov = float(finite_cov[-1]) if finite_cov else float("nan")

        summary = {
            "reference_backend": self.context.backend,
            "reference_backend_label": self.context.backend_label,
            "success": int(last_success and last_done),
            "success_and_done": int(last_success and last_done),
            "termination_reason": termination_reason,
            "termination_step": termination_step,
            "final_pos_diff": final_pos,
            "final_angle_diff": final_angle,
            "final_eef_diff": final_eef,
            "best_pos_diff": float(np.min(np.asarray(pos_diffs, dtype=np.float32))) if pos_diffs else float("nan"),
            "best_angle_diff": float(np.min(np.asarray(angle_diffs, dtype=np.float32))) if angle_diffs else float("nan"),
            "best_eef_diff": float(np.min(np.asarray(eef_diffs, dtype=np.float32))) if eef_diffs else float("nan"),
            "final_coverage": final_cov,
            "auc_pos_diff": float(np.sum(np.asarray(pos_diffs, dtype=np.float32))) if pos_diffs else float("nan"),
            "auc_angle_diff": float(np.sum(np.asarray(angle_diffs, dtype=np.float32))) if angle_diffs else float("nan"),
            "auc_eef_diff": float(np.sum(np.asarray(eef_diffs, dtype=np.float32))) if eef_diffs else float("nan"),
        }
        arrays = {
            "executed_actions": np.asarray(actions, dtype=np.float32),
            "trajectory": np.asarray(trajectory, dtype=np.float32),
            "pos_diffs": np.asarray(pos_diffs, dtype=np.float32),
            "angle_diffs": np.asarray(angle_diffs, dtype=np.float32),
            "eef_diffs": np.asarray(eef_diffs, dtype=np.float32),
            "coverages": np.asarray(coverages, dtype=np.float32),
            "state_dists": np.asarray(state_dists, dtype=np.float32),
            "metric_success_flags": np.asarray(metric_success_flags, dtype=np.bool_),
            "done_flags": np.asarray(done_flags, dtype=np.bool_),
        }
        return ReferenceEvalResult(summary=summary, arrays=arrays)

    def render_reference_env_replay(
        self,
        trace_meta: Mapping[str, Any],
        trace_arrays: Mapping[str, np.ndarray],
        *,
        output_path: str,
    ) -> str:
        init_state = np.asarray(trace_meta["init_state"], dtype=np.float32)
        goal_state = np.asarray(trace_meta["goal_state"], dtype=np.float32)
        actions = np.asarray(trace_arrays.get("executed_actions", []), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        frames = self._render_reference_frames(
            init_state=init_state,
            goal_state=goal_state,
            actions=actions,
        )
        overlay_spec = infer_action_overlay_spec(dict(trace_meta), actions, env=self._overlay_env())
        trajectory = self._render_overlay_states(init_state=init_state, goal_state=goal_state, actions=actions)
        frames = overlay_action_targets_on_frames(frames, trajectory, actions, overlay_spec)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_video_mp4(output_path, frames, fps=15)
        return output_path

    def render_reference_goal_state(
        self,
        trace_meta: Mapping[str, Any],
        *,
        output_path: str,
    ) -> str:
        goal_state = np.asarray(trace_meta["goal_state"], dtype=np.float32)
        frame = self._render_goal_frame(goal_state=goal_state)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.imwrite(output_path, np.asarray(frame, dtype=np.uint8))
        return output_path

    def _overlay_env(self) -> object | None:
        if self.backend == "particle_sim" and self.particle_backend is not None:
            return self.particle_backend
        return self.env

    def _prepare_execution(self, *, init_state: np.ndarray, goal_state: np.ndarray, with_visual: bool) -> np.ndarray:
        if self.backend == "particle_sim":
            if self.particle_backend is None:
                raise ValueError("particle_sim reference evaluation requires planner.backend.")
            obs, cur_state = self.particle_backend.prepare(
                seed=0,
                init_state=init_state,
                goal_state=goal_state,
                with_visual=with_visual,
            )
            del obs
            return np.asarray(cur_state, dtype=np.float32)
        set_start_pose(self.env, init_state)
        set_execution_fidelity_finest(self.env)
        obs, cur_state = self.env.prepare(seed=0, init_state=init_state, goal_state=goal_state)
        del obs
        return np.asarray(cur_state, dtype=np.float32)

    def _initial_reference_term(self, *, goal_state: np.ndarray, cur_state: np.ndarray) -> dict[str, Any]:
        if self.backend == "particle_sim":
            if self.particle_backend is None:
                raise ValueError("particle_sim reference evaluation requires planner.backend.")
            metrics = dict(self.particle_backend.eval_state(goal_state, cur_state))
            coverage = tee_pose_coverage_px(goal_state, cur_state)
            coverage_success = (
                coverage is not None
                and coverage > float(getattr(self.particle_backend, "success_threshold", np.inf))
            )
            success = bool(coverage_success) if coverage is not None else bool(metrics.get("success", False))
            return {
                "success": success,
                "pos_diff": float(metrics.get("pos_diff", float("nan"))),
                "angle_diff": float(metrics.get("angle_diff", float("nan"))),
                "eef_diff": float(metrics.get("eef_diff", float("nan"))),
                "state_dist": float(metrics.get("state_dist", float("nan"))),
                "coverage": coverage,
                "done": success,
            }
        return dict(self.env.eval_termination(goal_state, cur_state, done=None, info=None))

    def _step_reference(
        self,
        *,
        action: np.ndarray,
        goal_state: np.ndarray,
        with_visual: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        if self.backend == "particle_sim":
            if self.particle_backend is None:
                raise ValueError("particle_sim reference evaluation requires planner.backend.")
            obs, _, done, info = self.particle_backend.step(action, with_visual=with_visual)
            del obs
            cur_state = np.asarray(info["state"], dtype=np.float32)
            metrics = dict(info.get("metrics", {}))
            coverage = tee_pose_coverage_px(goal_state, cur_state)
            coverage_success = (
                coverage is not None
                and coverage > float(getattr(self.particle_backend, "success_threshold", np.inf))
            )
            success = bool(coverage_success) if coverage is not None else bool(metrics.get("success", False))
            done_flag = success if coverage is not None else bool(done)
            term = {
                "success": success,
                "pos_diff": float(metrics.get("pos_diff", float("nan"))),
                "angle_diff": float(metrics.get("angle_diff", float("nan"))),
                "eef_diff": float(metrics.get("eef_diff", float("nan"))),
                "state_dist": float(metrics.get("state_dist", float("nan"))),
                "coverage": coverage,
                "done": bool(done_flag),
            }
            return term, cur_state

        obs, _, done, info = self.env.step(action)
        del obs
        cur_state = np.asarray(info["state"], dtype=np.float32)
        term = dict(self.env.eval_termination(goal_state, cur_state, done=done, info=info))
        return term, cur_state

    def _render_reference_frames(
        self,
        *,
        init_state: np.ndarray,
        goal_state: np.ndarray,
        actions: np.ndarray,
    ) -> list[np.ndarray]:
        if self.backend == "wm":
            return self._render_wm_reference_frames(
                init_state=init_state,
                goal_state=goal_state,
                actions=actions,
            )

        frames: list[np.ndarray] = []
        if self.backend == "particle_sim":
            if self.particle_backend is None:
                raise ValueError("particle_sim reference rendering requires planner.backend.")
            self.particle_backend.prepare(seed=0, init_state=init_state, goal_state=goal_state, with_visual=True)
            frames.append(np.asarray(self.particle_backend.render("rgb_array", include_start_pose=True)))
        else:
            set_start_pose(self.env, init_state)
            self.env.prepare(seed=0, init_state=init_state, goal_state=goal_state)
            set_execution_fidelity_finest(self.env)
            frames.append(np.asarray(self.env.render("rgb_array", include_start_pose=True)))
        for action in np.asarray(actions, dtype=np.float32):
            if self.backend == "particle_sim":
                self.particle_backend.step(action, with_visual=True)
                frames.append(np.asarray(self.particle_backend.render("rgb_array", include_start_pose=True)))
            else:
                self.env.step(action)
                frames.append(np.asarray(self.env.render("rgb_array", include_start_pose=True)))
        return frames

    def _render_wm_reference_frames(
        self,
        *,
        init_state: np.ndarray,
        goal_state: np.ndarray,
        actions: np.ndarray,
    ) -> list[np.ndarray]:
        if self.wm is None:
            raise ValueError("wm reference rendering requires a loaded world model.")
        goal_obs, _ = self.env.prepare(seed=0, init_state=goal_state, goal_state=goal_state)
        goal_obs["visual"] = self.env.render("rgb_array", include_start_pose=False)
        del goal_obs
        obs, _ = self.env.prepare(seed=0, init_state=init_state, goal_state=goal_state)
        obs["visual"] = self.env.render("rgb_array", include_start_pose=False)
        z = encode_visual(self.wm, obs["visual"], self.device)
        finest_level = max(0, len(getattr(self.wm, "K", [])) - 1)
        target_hw = np.asarray(obs["visual"]).shape[:2]
        frames = [wm_decode_frame(self.wm, z=z, level_idx=finest_level, target_hw=target_hw)]
        for action in np.asarray(actions, dtype=np.float32):
            action_torch = torch.as_tensor(
                np.asarray(action, dtype=np.float32).reshape(1, -1),
                dtype=torch.float32,
                device=self.device,
            )
            planner = self.planner
            z_next_k, _ = planner._predict_next_stats(finest_level, z, action_torch)
            k = int(planner.K[finest_level])
            z_next = z.clone()
            z_next[:, :k] = z_next_k
            if k < planner.D:
                z_next[:, k:] = 0.0
            z = z_next
            frames.append(wm_decode_frame(self.wm, z=z, level_idx=finest_level, target_hw=target_hw))
        return frames

    def _render_overlay_states(
        self,
        *,
        init_state: np.ndarray,
        goal_state: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        if self.backend == "wm":
            return np.asarray([init_state for _ in range(int(actions.shape[0]) + 1)], dtype=np.float32)
        cur_state = self._prepare_execution(init_state=init_state, goal_state=goal_state, with_visual=False)
        states = [cur_state.copy()]
        for action in np.asarray(actions, dtype=np.float32):
            _, cur_state = self._step_reference(action=action, goal_state=goal_state, with_visual=False)
            states.append(cur_state.copy())
        return np.asarray(states, dtype=np.float32)

    def _render_goal_frame(self, *, goal_state: np.ndarray) -> np.ndarray:
        if self.backend == "wm":
            if self.wm is None:
                raise ValueError("wm reference rendering requires a loaded world model.")
            goal_obs, _ = self.env.prepare(seed=0, init_state=goal_state, goal_state=goal_state)
            goal_obs["visual"] = self.env.render("rgb_array", include_start_pose=False)
            z_goal = encode_visual(self.wm, goal_obs["visual"], self.device)
            finest_level = max(0, len(getattr(self.wm, "K", [])) - 1)
            return wm_decode_frame(
                self.wm,
                z=z_goal,
                level_idx=finest_level,
                target_hw=np.asarray(goal_obs["visual"]).shape[:2],
            )

        if self.backend == "particle_sim":
            if self.particle_backend is None:
                raise ValueError("particle_sim reference rendering requires planner.backend.")
            self.particle_backend.prepare(seed=0, init_state=goal_state, goal_state=goal_state, with_visual=True)
            frame = self.particle_backend.render("rgb_array", include_start_pose=False)
            return np.asarray(frame)

        self.env.prepare(seed=0, init_state=goal_state, goal_state=goal_state)
        set_execution_fidelity_finest(self.env)
        frame = self.env.render("rgb_array", include_start_pose=False)
        return np.asarray(frame)


def ensure_reference_eval(
    run_dir: str,
    *,
    meta_path: str | None,
    context: ReferenceContext,
    variant_name: str,
    rollout_id: str,
    evaluator: ReferenceEvaluator | None = None,
) -> ReferenceEvalResult | None:
    trace_root = trace_dir(run_dir, variant_name, rollout_id)
    trace_json_path = os.path.join(trace_root, "trace.json")
    trace_npz_path = os.path.join(trace_root, "trace.npz")
    source_paths = [trace_json_path, trace_npz_path]
    if meta_path:
        source_paths.append(meta_path)
    cached = load_cached_reference_eval(
        run_dir,
        context=context,
        variant_name=variant_name,
        rollout_id=rollout_id,
        source_paths=source_paths,
    )
    if cached is not None:
        return cached
    if not context.metrics_supported:
        return None
    if not (os.path.isfile(trace_json_path) and os.path.isfile(trace_npz_path)):
        return None
    with open(trace_json_path, "r", encoding="utf-8") as f:
        trace_meta = json.load(f)
    with np.load(trace_npz_path, allow_pickle=False) as data:
        trace_arrays = {key: data[key] for key in data.files}
    active_evaluator = evaluator or ReferenceEvaluator(context)
    result = active_evaluator.evaluate_from_trace(trace_meta, trace_arrays)
    if result is None:
        return None
    _write_reference_eval_cache(
        run_dir,
        context=context,
        variant_name=variant_name,
        rollout_id=rollout_id,
        result=result,
    )
    return result


def render_reference_media(
    run_dir: str,
    *,
    meta_path: str | None,
    context: ReferenceContext,
    variant_name: str,
    rollout_id: str,
    media_name: str,
    output_dir: str,
) -> str:
    if media_name not in REFERENCE_MEDIA_ALIASES:
        raise ValueError(f"Unknown reference media type: {media_name}")
    trace_root = trace_dir(run_dir, variant_name, rollout_id)
    trace_json_path = os.path.join(trace_root, "trace.json")
    trace_npz_path = os.path.join(trace_root, "trace.npz")
    if not (os.path.isfile(trace_json_path) and os.path.isfile(trace_npz_path)):
        raise FileNotFoundError(f"Trace bundle missing for {variant_name}/{rollout_id}")
    with open(trace_json_path, "r", encoding="utf-8") as f:
        trace_meta = json.load(f)
    with np.load(trace_npz_path, allow_pickle=False) as data:
        trace_arrays = {key: data[key] for key in data.files}

    os.makedirs(output_dir, exist_ok=True)
    active_evaluator = ReferenceEvaluator(context)
    if media_name == "reference_env_replay":
        output_path = os.path.join(output_dir, REFERENCE_MEDIA_ALIASES[media_name][0])
        return active_evaluator.render_reference_env_replay(trace_meta, trace_arrays, output_path=output_path)
    if media_name == "reference_goal_state":
        output_path = os.path.join(output_dir, REFERENCE_MEDIA_ALIASES[media_name][0])
        return active_evaluator.render_reference_goal_state(trace_meta, output_path=output_path)
    raise ValueError(f"Unhandled reference media type: {media_name}")
