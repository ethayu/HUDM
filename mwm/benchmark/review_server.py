from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.parse import parse_qs
import uuid

from mwm.benchmark.review_media import (
    latent_decoder_metadata_supported,
    render_rollout_media,
    rollout_by_index,
    rollout_checkpoint_metadata,
    rollout_key,
)
from mwm.benchmark.eval_artifacts import load_eval_artifact, load_eval_capsule
from mwm.io import load_json


def warm_review_assets(root: str | Path, progress: Any = None) -> dict[str, Any]:
    """Eagerly load each review checkpoint on CUDA and initialize its env."""

    from mwm.benchmark.replay_runtime import warm_review_runtime

    base = Path(root).resolve()
    eval_paths = sorted(base.glob("*/eval.json"))
    seen_checkpoints: set[str] = set()
    warmed: list[dict[str, Any]] = []
    warnings: list[str] = []
    candidates: list[tuple[Path, Path, dict[str, Any], str]] = []
    for eval_path in eval_paths:
        try:
            capsule = load_eval_capsule(eval_path, verify="metadata")
            cfg_path = eval_path.parent / "resolved_config.yaml"
            if not cfg_path.is_file():
                raise FileNotFoundError(f"missing {cfg_path.name}")
            checkpoint = Path(str(capsule.get("checkpoint_run_dir") or cfg_path))
            if not checkpoint.is_absolute():
                checkpoint = (Path.cwd() / checkpoint).resolve()
            checkpoint_key = str(checkpoint)
            if checkpoint_key in seen_checkpoints:
                continue
            payload = load_eval_artifact(eval_path)
            rollout = next(
                item
                for item in payload.get("review_rollouts", [])
                if isinstance(item, dict)
                and item.get("start_row") is not None
                and item.get("goal_row") is not None
            )
            seen_checkpoints.add(checkpoint_key)
            candidates.append((eval_path, cfg_path, rollout, checkpoint_key))
        except (OSError, RuntimeError, StopIteration, TypeError, ValueError) as exc:
            warnings.append(f"{eval_path.parent.name}: {exc}")

    total = len(candidates)
    for index, (eval_path, cfg_path, rollout, _checkpoint) in enumerate(candidates, start=1):
        prefix = f"Warming {index}/{total} ({eval_path.parent.name})"
        try:
            if progress is not None:
                progress(prefix)
            report = warm_review_runtime(
                cfg_path,
                start_row=int(rollout["start_row"]),
                goal_row=int(rollout["goal_row"]),
                progress=(
                    None
                    if progress is None
                    else lambda phase, prefix=prefix: progress(f"{prefix}: {phase}")
                ),
            )
            report["run"] = eval_path.parent.name
            warmed.append(report)
        except Exception as exc:
            warnings.append(f"{eval_path.parent.name}: {exc}")
    return {"warmed": warmed, "warnings": warnings, "checkpoint_count": len(warmed)}


class ReviewRenderManager:
    """Serialize heavyweight renders and expose pollable progress."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-render")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active: dict[tuple[Any, ...], str] = {}
        self._warmup: dict[str, Any] = {
            "status": "idle",
            "phase": "Warm-up has not started",
            "result": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
        }

    def start_warmup(self, root: str | Path) -> dict[str, Any]:
        with self._lock:
            if self._warmup["status"] != "idle":
                return dict(self._warmup)
            self._warmup.update(
                status="queued",
                phase="Waiting to warm environments and CUDA checkpoints",
            )
        self._executor.submit(self._run_warmup, Path(root).resolve())
        return self.warmup()

    def _run_warmup(self, root: Path) -> None:
        with self._lock:
            self._warmup.update(
                status="running",
                phase="Discovering review checkpoints",
                started_at=time.time(),
            )
        try:
            result = warm_review_assets(root, progress=self._update_warmup_phase)
            with self._lock:
                self._warmup.update(
                    status="complete",
                    phase="Warm-up complete",
                    result=result,
                    finished_at=time.time(),
                )
        except Exception as exc:
            with self._lock:
                self._warmup.update(
                    status="failed",
                    phase="Warm-up failed; on-demand rendering remains available",
                    error=str(exc),
                    finished_at=time.time(),
                )

    def _update_warmup_phase(self, phase: str) -> None:
        with self._lock:
            self._warmup["phase"] = str(phase)

    def warmup(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self._warmup)
        started = snapshot.get("started_at")
        if started is not None:
            finished = snapshot.get("finished_at") or time.time()
            snapshot["elapsed_seconds"] = round(max(0.0, float(finished) - float(started)), 1)
        else:
            snapshot["elapsed_seconds"] = 0.0
        return snapshot

    def _snapshot(self, job: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            key: value
            for key, value in job.items()
            if key not in {"key"}
        }
        started = float(job.get("started_at") or job["created_at"])
        finished = float(job.get("finished_at") or time.time())
        snapshot["elapsed_seconds"] = round(max(0.0, finished - started), 1)
        return snapshot

    def submit(
        self,
        eval_path: Path,
        *,
        episode_index: int,
        sources: list[str],
        force: bool,
    ) -> dict[str, Any]:
        key = (
            str(eval_path.resolve()),
            int(episode_index),
            tuple(str(source) for source in sources),
            bool(force),
        )
        with self._lock:
            active_id = self._active.get(key)
            if active_id is not None:
                snapshot = self._snapshot(self._jobs[active_id])
                snapshot["deduplicated"] = True
                return snapshot
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "key": key,
                "status": "queued",
                "phase": "Waiting for render worker",
                "episode_index": int(episode_index),
                "sources": list(sources),
                "result": None,
                "error": None,
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
            }
            self._jobs[job_id] = job
            self._active[key] = job_id
            self._executor.submit(
                self._run,
                job_id,
                eval_path,
                int(episode_index),
                list(sources),
                bool(force),
            )
            return self._snapshot(job)

    def _update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(updates)

    def _run(
        self,
        job_id: str,
        eval_path: Path,
        episode_index: int,
        sources: list[str],
        force: bool,
    ) -> None:
        self._update(job_id, status="running", phase="Preparing rollout", started_at=time.time())
        try:
            result = render_rollout_media(
                eval_path,
                episode_index=episode_index,
                sources=sources,
                force=force,
                progress=lambda phase: self._update(job_id, phase=str(phase)),
            )
            if result.get("warnings") and not result.get("rendered"):
                self._update(
                    job_id,
                    status="failed",
                    phase="Render unavailable",
                    result=result,
                    error="\n".join(str(item) for item in result["warnings"]),
                    finished_at=time.time(),
                )
            else:
                self._update(
                    job_id,
                    status="complete",
                    phase="Complete",
                    result=result,
                    finished_at=time.time(),
                )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                phase="Render failed",
                error=str(exc),
                finished_at=time.time(),
            )
        finally:
            with self._lock:
                job = self._jobs[job_id]
                self._active.pop(job["key"], None)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown render job {job_id!r}")
            return self._snapshot(self._jobs[job_id])

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def resolve_eval_path(root: str | Path, path_text: str) -> Path:
    base = Path(root).resolve()
    raw = Path(str(path_text))
    candidate = raw if raw.is_absolute() else base / raw
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"eval path is outside review root: {path_text}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing eval json: {resolved}")
    if resolved.name != "eval.json":
        raise ValueError(f"expected eval.json path, got {resolved.name!r}")
    return resolved


def _href(path_text: Any, base_dir: Path) -> str:
    path = Path(str(path_text or ""))
    try:
        if path.is_absolute():
            return path.resolve().relative_to(base_dir.resolve()).as_posix()
        return (Path.cwd() / path).resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _existing_media_href(entry: Any, base_dir: Path) -> str | None:
    if not isinstance(entry, dict) or not entry.get("path"):
        return None
    href = _href(entry["path"], base_dir)
    candidate = (base_dir / href).resolve()
    try:
        candidate.relative_to(base_dir.resolve())
    except ValueError:
        return None
    return href if candidate.is_file() else None


def _media_html(payload: dict[str, Any], episode_index: int, base_dir: Path) -> str:
    entries = payload.get("review_media", {}).get("rollouts", {}).get(rollout_key(episode_index), {})
    if not isinstance(entries, dict) or not entries:
        return "<p class='muted'>No media rendered yet.</p>"
    cards: list[str] = []
    for kind, entry in sorted(entries.items()):
        href = _existing_media_href(entry, base_dir)
        if href is None:
            continue
        label = str(kind).replace("_", " ")
        descriptions = {
            "env": "Simulator replay from the stored action trace. Check task completion, contacts, collisions, and whether the success label matches the motion.",
            "latent_reconstruction": "Dataset observation (left) versus decoded latent reconstruction (right). Check missing objects, geometry drift, blur, and fidelity changes; this is not a planner prediction.",
            "latent_predictive_rollout": "Actual policy replay (left) versus piecewise open-loop model prediction (right). Blue reset frames are MPC replan anchors; predictions appear only at complete action-block endpoints.",
        }
        description = descriptions.get(str(kind), "Rendered rollout review media.")
        media_warnings = entry.get("warnings", []) if isinstance(entry, dict) else []
        warning_html = "".join(
            f"<span class='media-warning'>Note: {html.escape(str(warning))}</span>"
            for warning in media_warnings
        )
        cards.append(
            f"<figure data-media-kind='{html.escape(str(kind))}'>"
            f"<video controls preload='auto' src='/{html.escape(href)}'></video>"
            f"<figcaption><strong>{html.escape(label)}</strong><span>{html.escape(description)}</span>{warning_html}</figcaption>"
            "</figure>"
        )
    return "".join(cards) or "<p class='muted'>No media rendered yet.</p>"


def _related_episode_links(root: Path, current_eval: Path, episode_index: int) -> str:
    links: list[str] = []
    for candidate in sorted(root.glob("*/eval.json")):
        try:
            payload = load_eval_capsule(candidate, verify="metadata")
            trace_path = candidate.parent / "episode_traces.jsonl"
            rollout = None
            if trace_path.is_file():
                with trace_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            item = json.loads(line)
                            if isinstance(item, dict) and int(item.get("episode_index", -1)) == int(episode_index):
                                rollout = item
                                break
            if rollout is None:
                rollout = rollout_by_index(load_eval_artifact(candidate), episode_index)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            continue
        label = str(payload.get("benchmark_name") or candidate.parent.name)
        success = rollout.get("success")
        status = "success" if success is True else "failure" if success is False else "unknown"
        current = candidate.resolve() == current_eval.resolve()
        current_text = " · current" if current else ""
        links.append(
            f"<a class='related-link {status}{' current' if current else ''}' "
            f"href='/rollouts/{html.escape(candidate.parent.name)}/episode_{int(episode_index):04d}.html'>"
            f"{html.escape(label)} · {status}{current_text}</a>"
        )
    return "".join(links)


def _render_button(source: str, label: str, *, supported: bool, ready: bool) -> str:
    if source == "all":
        if not supported:
            title = "No review media is available for this rollout"
        elif ready:
            title = "All supported media is already available; enable re-render to replace it"
        else:
            title = "Render every supported media view for this rollout"
        disabled = " disabled" if not supported or ready else ""
        text = "All media ready" if ready else "Render all"
        return (
            f"<button data-source='all' data-supported='{'true' if supported else 'false'}' "
            f"data-ready='{'true' if ready else 'false'}' title='{html.escape(title)}'{disabled}>"
            f"{html.escape(text)}</button>"
        )
    if not supported:
        title = f"{label} is unavailable because this eval lacks its required trace or decoder"
    elif ready:
        title = f"{label} is already available; enable re-render to replace it"
    else:
        title = f"Render missing {label.lower()} media"
    disabled = " disabled" if not supported or ready else ""
    text = f"{label} ready" if ready else f"Render {label}"
    return (
        f"<button data-source='{html.escape(source)}' data-supported='{'true' if supported else 'false'}' "
        f"data-ready='{'true' if ready else 'false'}' title='{html.escape(title)}'{disabled}>{html.escape(text)}</button>"
    )


def _has_usable_action(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_has_usable_action(item) for item in value)
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def rollout_page_html(root: Path, eval_path: Path, episode_index: int) -> str:
    payload = load_eval_artifact(eval_path)
    rollout = rollout_by_index(payload, episode_index)
    rel_eval = eval_path.resolve().relative_to(root.resolve()).as_posix()
    status = "success" if rollout.get("success") else "failure" if rollout.get("success") is False else "unknown"
    media = _media_html(payload, episode_index, root)
    run_label = str(payload.get("benchmark_name") or eval_path.parent.name)
    title = f"{run_label} · episode {int(episode_index)}"
    rollouts = [item for item in payload.get("review_rollouts", []) if isinstance(item, dict)]
    episode_indices = sorted(int(item["episode_index"]) for item in rollouts if "episode_index" in item)
    position = episode_indices.index(int(episode_index))
    previous_index = episode_indices[position - 1] if position > 0 else None
    next_index = episode_indices[position + 1] if position + 1 < len(episode_indices) else None
    previous_link = (
        f"<a href='/rollouts/{html.escape(eval_path.parent.name)}/episode_{previous_index:04d}.html'>← Episode {previous_index}</a>"
        if previous_index is not None else "<span></span>"
    )
    next_link = (
        f"<a href='/rollouts/{html.escape(eval_path.parent.name)}/episode_{next_index:04d}.html'>Episode {next_index} →</a>"
        if next_index is not None else "<span></span>"
    )
    entries = payload.get("review_media", {}).get("rollouts", {}).get(rollout_key(episode_index), {})
    entries = entries if isinstance(entries, dict) else {}
    env_supported = isinstance(rollout.get("action_trace"), list) and bool(rollout.get("action_trace"))
    latent_has_trace = isinstance(rollout.get("fidelity_trace"), list) and bool(rollout.get("fidelity_trace"))
    checkpoint_metadata = rollout_checkpoint_metadata(payload)
    latent_has_decoder = checkpoint_metadata is None or latent_decoder_metadata_supported(checkpoint_metadata)
    latent_supported = latent_has_trace and latent_has_decoder
    predictive_supported = env_supported and latent_has_trace and latent_has_decoder
    env_ready = _existing_media_href(entries.get("env"), root) is not None
    latent_ready = _existing_media_href(entries.get("latent_reconstruction"), root) is not None
    predictive_ready = _existing_media_href(entries.get("latent_predictive_rollout"), root) is not None
    supported_ready = [
        ready
        for supported, ready in (
            (env_supported, env_ready),
            (latent_supported, latent_ready),
            (predictive_supported, predictive_ready),
        )
        if supported
    ]
    all_supported = bool(supported_ready)
    all_ready = all_supported and all(supported_ready)
    action_trace = rollout.get("action_trace") or []
    action_steps = sum(
        1
        for action in action_trace
        if _has_usable_action(action)
    )
    recorded_fidelity_steps = len(rollout.get("fidelity_trace") or [])
    fidelity_steps = (
        min(recorded_fidelity_steps, action_steps)
        if action_steps and rollout.get("success") is True
        else recorded_fidelity_steps
    )
    fidelity_step_text = (
        f"{fidelity_steps} executable"
        if fidelity_steps < recorded_fidelity_steps
        else f"{fidelity_steps} steps"
    )
    evaluation_budget = int(
        rollout.get("evaluation_budget")
        or max(len(action_trace), recorded_fidelity_steps, action_steps)
    )
    action_step_text = f"{action_steps} executable"
    if action_steps < evaluation_budget and rollout.get("success") is True:
        replay_note = (
            f"The original evaluation succeeded after {action_steps} action(s); later vector slots were masked. "
            "Replay stops at the same termination point."
        )
    else:
        replay_note = (
            f"This benchmark allows at most {evaluation_budget} environment action(s). "
            "Replay uses that recorded evaluation horizon, not the environment's separate safety time limit."
        )
    related_links = _related_episode_links(root, eval_path, episode_index)
    buttons = "".join(
        [
            _render_button("all", "all", supported=all_supported, ready=all_ready),
            _render_button("env", "Env", supported=env_supported, ready=env_ready),
            _render_button("latent", "Latent", supported=latent_supported, ready=latent_ready),
            _render_button(
                "predictive",
                "Predictive rollout",
                supported=predictive_supported,
                ready=predictive_ready,
            ),
        ]
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; color: #172026; background: #f6f8fa; }}
    main {{ max-width: 1160px; margin: 0 auto; padding: 28px; }}
    a {{ color: #0b5cad; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    h1 {{ margin-bottom: 6px; }}
    button {{ border: 1px solid #7890a8; background: #fff; border-radius: 6px; padding: 9px 13px; cursor: pointer; }}
    button:hover:not(:disabled) {{ background: #eef3f7; }}
    button:disabled {{ cursor: not-allowed; color: #718096; background: #edf1f5; border-color: #cbd5df; }}
    button[data-source="all"] {{ background: #0b5cad; border-color: #0b5cad; color: #fff; font-weight: 700; }}
    button[data-source="all"]:hover:not(:disabled) {{ background: #084b8a; }}
    button[data-source="all"]:disabled {{ background: #dce5ed; border-color: #c4d0db; color: #607284; }}
    .panel {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; margin-top: 16px; }}
    .muted {{ color: #627282; }}
    .status {{ display: inline-block; border-radius: 999px; padding: 5px 10px; font-weight: 700; text-transform: uppercase; font-size: 12px; letter-spacing: .04em; }}
    .status.success {{ background: #e8f5ed; color: #0f7b3f; }}
    .status.failure {{ background: #fff0ef; color: #b42318; }}
    .status.unknown {{ background: #eef3f7; color: #52616b; }}
    .facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 14px; }}
    .fact {{ border-left: 3px solid #d9e2ec; padding-left: 10px; }}
    .fact span {{ display: block; color: #627282; font-size: 12px; text-transform: uppercase; }}
    .fact strong {{ display: block; margin-top: 3px; }}
    .related {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .related-link {{ border: 1px solid #cbd5df; border-radius: 999px; padding: 6px 10px; }}
    .related-link.failure {{ border-color: #f2b8b5; color: #8f1912; background: #fff5f4; }}
    .related-link.success {{ border-color: #b8dfc7; color: #146c3a; background: #f3fbf6; }}
    .related-link.current {{ box-shadow: 0 0 0 2px #0b5cad; }}
    .guide {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .guide article {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; }}
    .guide h3 {{ margin: 0 0 7px; }}
    .guide p {{ margin: 0; color: #52616b; line-height: 1.45; }}
    .render-controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
    .force {{ margin-left: 6px; }}
    .render-help {{ margin: 10px 0 0; }}
    #result {{ white-space: pre-wrap; background: #111827; color: #f8fafc; padding: 12px; border-radius: 6px; margin: 12px 0 0; }}
    #result.error {{ background: #711c18; }}
    #result.success {{ background: #14532d; }}
    #result[hidden] {{ display: none; }}
    .media {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; }}
    video {{ width: 100%; max-height: 520px; background: #000; }}
    figcaption {{ margin-top: 7px; }}
    figcaption span {{ display: block; color: #52616b; margin-top: 4px; line-height: 1.4; }}
    figcaption .media-warning {{ color: #8a4b08; }}
    code {{ background: #eef3f7; padding: 2px 4px; border-radius: 4px; overflow-wrap: anywhere; }}
    .episode-nav {{ display: flex; justify-content: space-between; margin-top: 18px; }}
    @media (max-width: 700px) {{
      main {{ padding: 18px 12px; }}
      h1 {{ font-size: 26px; }}
      .guide {{ grid-template-columns: 1fr; }}
      .media {{ grid-template-columns: 1fr; }}
      .force {{ width: 100%; margin: 6px 0 0; }}
    }}
  </style>
</head>
<body>
<main>
  <p><a href="/review.html">← Benchmark review</a></p>
  <span class="status {html.escape(status)}">{html.escape(status)}</span>
  <h1>{html.escape(title)}</h1>
  <p class="muted">Episode {position + 1} of {len(episode_indices)} in this run. Compare the same episode across runs because the benchmark uses a shared evaluation manifest.</p>

  <section class="panel">
    <div class="facts">
      <div class="fact"><span>Dataset episode</span><strong>{html.escape(str(rollout.get("dataset_episode", "n/a")))}</strong></div>
      <div class="fact"><span>Start → goal step</span><strong>{html.escape(str(rollout.get("start_step", "n/a")))} → {html.escape(str(rollout.get("goal_step", "n/a")))}</strong></div>
      <div class="fact"><span>Evaluation horizon</span><strong>{evaluation_budget} actions max</strong></div>
      <div class="fact"><span>Action trace</span><strong>{html.escape(action_step_text)}</strong></div>
      <div class="fact"><span>Fidelity trace</span><strong>{html.escape(fidelity_step_text)}</strong></div>
    </div>
    <p>{html.escape(replay_note)}</p>
    <p class="muted"><strong>Source:</strong> <code>{html.escape(rel_eval)}</code></p>
  </section>

  <section class="panel">
    <h2>Compare this episode across runs</h2>
    <div class="related">{related_links}</div>
  </section>

  <section class="panel">
    <h2>What to look for</h2>
    <div class="guide">
      <article><h3>Environment replay</h3><p>Use the actual simulator replay to verify the score. Look for task completion, contact mistakes, collisions, oscillation, late recovery, and whether a reported success or failure is visually credible.</p></article>
      <article><h3>Latent reconstruction</h3><p>Compare the dataset observation on the left with the decoded representation on the right. Look for missing task objects, incorrect pose or geometry, blur, temporal flicker, and quality changes when K changes. This is a reconstruction, not a predicted planner rollout.</p></article>
      <article><h3>Latent predictive rollout</h3><p>Compare the actual policy replay on the left with model predictions on the right. Each blue reset frame re-anchors from the actual history at an MPC boundary; subsequent frames are autoregressive predictions at complete action-block endpoints, not at every primitive action.</p></article>
    </div>
  </section>

  <section class="panel">
    <div class="render-controls">
      {buttons}
      <label class="force"><input id="force" type="checkbox"> Re-render existing media</label>
    </div>
    <p id="backend-status" class="muted render-help">Checking environment and CUDA checkpoint warm-up…</p>
    <p class="muted render-help">Render all creates every supported view: environment replay, latent reconstruction, and predictive rollout. Existing videos load directly. Environment replay reads only this rollout's dataset rows. Reconstruction decodes actual observations; predictive rollout replays stored actions, resets at each real MPC boundary, and predicts only complete action blocks. Checkpoints and legacy action statistics are cached.</p>
    <pre id="result" aria-live="polite" hidden></pre>
  </section>
  <section class="panel media">{media}</section>
  <nav class="episode-nav" aria-label="Episode navigation">{previous_link}{next_link}</nav>
</main>
<script>
const result = document.getElementById('result');
const force = document.getElementById('force');
const backendStatus = document.getElementById('backend-status');
const renderButtons = Array.from(document.querySelectorAll('button[data-source]'));
let busy = false;

function syncButtons() {{
  renderButtons.forEach((button) => {{
    const supported = button.dataset.supported === 'true';
    const ready = button.dataset.ready === 'true';
    button.disabled = busy || !supported || (ready && !force.checked);
  }});
}}

function showResult(message, kind = '') {{
  result.hidden = false;
  result.className = kind;
  result.textContent = message;
}}

async function render(source) {{
  busy = true;
  syncButtons();
  const startedAt = performance.now();
  showResult('Preparing ' + source + ' render…');
  try {{
    const response = await fetch('/api/render-rollout', {{
      method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify({{
        eval_path: {json.dumps(rel_eval)},
        episode_index: {int(episode_index)},
        sources: [source],
        force: force.checked
      }})
    }});
    let payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not queue render');
    if (response.status === 202 && payload.job_id) {{
      while (payload.status === 'queued' || payload.status === 'running') {{
        const elapsed = Math.max(0, Math.round((performance.now() - startedAt) / 1000));
        showResult((payload.status === 'queued' ? 'Queued: ' : '') + payload.phase + ' · ' + elapsed + 's');
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        const statusResponse = await fetch('/api/render-status?job_id=' + encodeURIComponent(payload.job_id), {{cache: 'no-store'}});
        payload = await statusResponse.json();
        if (!statusResponse.ok) throw new Error(payload.error || 'Could not read render status');
      }}
      if (payload.status === 'failed') throw new Error(payload.error || 'Render failed');
      payload = payload.result || {{}};
    }}
    const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];
    const rendered = Array.isArray(payload.rendered) ? payload.rendered : [];
    const names = rendered.map((item) => item.kind.replaceAll('_', ' '));
    const message = names.length ? 'Ready: ' + names.join(', ') + '.' : 'No media changed.';
    showResult(message + (warnings.length ? '\\nWarnings: ' + warnings.join('\\n') : ''), warnings.length ? 'error' : 'success');
    if (rendered.length) window.setTimeout(() => window.location.reload(), 700);
  }} catch (error) {{
    showResult('Render failed: ' + error.message, 'error');
  }} finally {{
    busy = false;
    syncButtons();
  }}
}}

async function refreshBackendStatus() {{
  try {{
    const response = await fetch('/api/status', {{cache: 'no-store'}});
    const payload = await response.json();
    const warmup = payload.warmup || {{}};
    if (warmup.status === 'complete') {{
      const count = warmup.result?.checkpoint_count ?? 0;
      const warnings = warmup.result?.warnings || [];
      backendStatus.textContent = 'Backend ready · ' + count + ' CUDA checkpoint' + (count === 1 ? '' : 's') + ' cached' + (warnings.length ? ' · warm-up warnings available' : '');
      return;
    }}
    if (warmup.status === 'failed') {{
      backendStatus.textContent = 'Background warm-up failed; rendering will load dependencies on demand.';
      return;
    }}
    backendStatus.textContent = 'Backend warm-up: ' + (warmup.phase || 'starting') + ' · ' + Math.round(warmup.elapsed_seconds || 0) + 's';
    window.setTimeout(refreshBackendStatus, 750);
  }} catch (error) {{
    backendStatus.textContent = 'Could not read backend warm-up status.';
  }}
}}

force.addEventListener('change', syncButtons);
renderButtons.forEach((button) => button.addEventListener('click', () => render(button.dataset.source)));
syncButtons();
refreshBackendStatus();
</script>
</body>
</html>
"""


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        root: Path,
        render_manager: ReviewRenderManager | None = None,
        **kwargs: Any,
    ) -> None:
        self.review_root = root.resolve()
        self.render_manager = render_manager or ReviewRenderManager()
        super().__init__(*args, directory=str(self.review_root), **kwargs)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/status":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "review_root": str(self.review_root),
                    "active_renders": self.render_manager.active_count(),
                    "warmup": self.render_manager.warmup(),
                },
            )
            return
        if path == "/api/render-status":
            try:
                query = parse_qs(parsed.query)
                job_id = str(query.get("job_id", [""])[0])
                self._send_json(HTTPStatus.OK, self.render_manager.get(job_id))
            except Exception as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if path.startswith("/rollouts/") and path.endswith(".html"):
            parts = [part for part in path.split("/") if part]
            if len(parts) == 3:
                run_name = parts[1]
                page_name = parts[2]
                try:
                    if not page_name.startswith("episode_"):
                        raise ValueError("not an episode page")
                    episode_index = int(page_name.removeprefix("episode_").removesuffix(".html"))
                    eval_path = resolve_eval_path(self.review_root, f"{run_name}/eval.json")
                    body = rollout_page_html(self.review_root, eval_path, episode_index).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                    return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        if parsed.path != "/api/render-rollout":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            eval_path = resolve_eval_path(self.review_root, str(data.get("eval_path", "")))
            sources = data.get("sources", ["both"])
            if isinstance(sources, str):
                sources = [sources]
            if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
                raise ValueError("sources must be a string or list of strings")
            job = self.render_manager.submit(
                eval_path,
                episode_index=int(data.get("episode_index", 0)),
                sources=sources,
                force=bool(data.get("force", False)),
            )
            self._send_json(HTTPStatus.ACCEPTED, job)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def validate_server_address(host: str, port: int) -> None:
    if str(host) not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--serve is restricted to localhost hosts")
    family = socket.AF_INET6 if str(host) == "::1" else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        probe.bind((str(host), int(port)))
    except OSError as exc:
        raise OSError(
            f"could not start review server on {host}:{int(port)}; the port may already be in use. "
            "Choose another with --port, for example --port 8766"
        ) from exc
    finally:
        probe.close()


def serve_review(output_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    root = Path(output_dir).resolve()
    if str(host) not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--serve is restricted to localhost hosts")

    render_manager = ReviewRenderManager()

    def handler(*args: Any, **kwargs: Any) -> ReviewRequestHandler:
        return ReviewRequestHandler(
            *args,
            root=root,
            render_manager=render_manager,
            **kwargs,
        )

    try:
        server = ThreadingHTTPServer((str(host), int(port)), handler)
    except OSError as exc:
        render_manager.shutdown()
        raise OSError(
            f"could not start review server on {host}:{int(port)}; the port may already be in use. "
            "Choose another with --port, for example --port 8766"
        ) from exc
    render_manager.start_warmup(root)
    print(f"Serving benchmark review at http://{host}:{int(server.server_port)}/review.html", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        render_manager.shutdown()


__all__ = [
    "ReviewRequestHandler",
    "ReviewRenderManager",
    "resolve_eval_path",
    "rollout_page_html",
    "serve_review",
    "validate_server_address",
    "warm_review_assets",
]
