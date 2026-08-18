#!/usr/bin/env python3
"""Profile two independent release20260728 cells on one full B200.

This harness is deliberately narrower than the benchmark launcher.  It runs
exactly four already-completed PushT cells in isolated profiling directories:
one low- and one high-population cell for each of the goal-25 and goal-50
matrices.  Each cell is first measured sequentially, then the matched pair is
run concurrently on the same GPU.  It never writes into a benchmark output
root and it never attempts a three- or four-process packing experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Running this file by path would otherwise prefer an editable install from
    # another checkout.  The pinned profiling worktree must own every `mwm`
    # import used by both preflight and execution.
    sys.path.insert(0, str(REPO_ROOT))
PROFILE_BASE = Path(
    "/ceph/projects/dineshj/lab/ethanyu/HUDM/profiling/packed_b200_release20260728"
)
REFERENCE_BASE = Path("/ceph/projects/dineshj/lab/ethanyu/HUDM/reports/research")
DEFAULT_PYTHON = Path("/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python")
NUM_SHARDS = 3120
SELECTED_INDICES = (0, 105)
MIN_FULL_B200_MEMORY_MIB = 150 * 1024

COMPLETION_FILES = (
    "eval.json",
    "resolved_config.yaml",
    "metrics.jsonl",
    "summary.json",
    "planning_diagnostics.json",
    "episode_traces.jsonl",
)

TIMING_DIAGNOSTIC_KEYS = frozenset(
    {
        "cost_time_sec",
        "total_plan_time_sec",
        "mean_plan_time_sec",
        "total_policy_time_sec",
        "mean_policy_time_sec",
        "plan_time_total_sec",
        "policy_time_total_sec",
    }
)

GPU_QUERY_FIELDS = (
    "timestamp",
    "name",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
)
COMPUTE_QUERY_FIELDS = ("pid", "process_name", "used_gpu_memory")


class ProfileError(RuntimeError):
    """Raised when a fail-closed profiling invariant is not satisfied."""


@dataclass(frozen=True)
class CellSpec:
    key: str
    goal: int
    population: int
    elite_frac: float
    topk: int
    matrix_index: int
    config_relpath: str
    reference_root_name: str
    role: str
    cell_id: str
    budget: int
    horizon: int
    receding_horizon: int

    @property
    def reference_run_dir(self) -> Path:
        return REFERENCE_BASE / self.reference_root_name / f"{self.matrix_index:03d}_{self.cell_id}"


GOAL25_CONFIG = "configs/research/release20260728_dense_pusht_all_fidelity_schedules.yaml"
GOAL50_CONFIG = (
    "configs/research/"
    "release20260728_dense_pusht_goal50_plan50_execute20_all_fidelity_schedules.yaml"
)
GOAL25_REFERENCE = "release20260728_dense_pusht_all_fidelity_schedules"
GOAL50_REFERENCE = (
    "release20260728_dense_pusht_goal50_plan50_execute20_all_fidelity_schedules"
)

CELL_SPECS = (
    CellSpec(
        key="g25_low",
        goal=25,
        population=20,
        elite_frac=0.1,
        topk=2,
        matrix_index=0,
        config_relpath=GOAL25_CONFIG,
        reference_root_name=GOAL25_REFERENCE,
        role="release20260728_dense_schedule_01_rollout_fixed",
        cell_id="01_rollout_fixed__pop20__elite0p1__iter5",
        budget=50,
        horizon=5,
        receding_horizon=2,
    ),
    CellSpec(
        key="g25_high",
        goal=25,
        population=200,
        elite_frac=0.2,
        topk=40,
        matrix_index=105,
        config_relpath=GOAL25_CONFIG,
        reference_root_name=GOAL25_REFERENCE,
        role="release20260728_dense_schedule_01_rollout_fixed",
        cell_id="01_rollout_fixed__pop200__elite0p2__iter5",
        budget=50,
        horizon=5,
        receding_horizon=2,
    ),
    CellSpec(
        key="g50_low",
        goal=50,
        population=20,
        elite_frac=0.1,
        topk=2,
        matrix_index=0,
        config_relpath=GOAL50_CONFIG,
        reference_root_name=GOAL50_REFERENCE,
        role="release20260728_goal50_plan50_execute20_dense_schedule_01_rollout_fixed",
        cell_id="01_rollout_fixed__pop20__elite0p1__iter5",
        budget=100,
        horizon=10,
        receding_horizon=4,
    ),
    CellSpec(
        key="g50_high",
        goal=50,
        population=200,
        elite_frac=0.2,
        topk=40,
        matrix_index=105,
        config_relpath=GOAL50_CONFIG,
        reference_root_name=GOAL50_REFERENCE,
        role="release20260728_goal50_plan50_execute20_dense_schedule_01_rollout_fixed",
        cell_id="01_rollout_fixed__pop200__elite0p2__iter5",
        budget=100,
        horizon=10,
        receding_horizon=4,
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in CELL_SPECS}


@dataclass
class RunningCell:
    spec: CellSpec
    mode: str
    run_root: Path
    command: list[str]
    process: subprocess.Popen[str]
    log_handle: Any
    log_path: Path
    time_path: Path
    cpu_affinity: tuple[int, ...]
    started_monotonic: float
    finished_monotonic: float | None = None
    returncode: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_checked(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise ProfileError(f"Command failed: {' '.join(command)}\n{stderr.strip()}") from exc
    return result.stdout.strip()


def validate_code_snapshot(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    expected = str(expected_commit).strip()
    if len(expected) != 40 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
        raise ProfileError("--expected-commit must be one full 40-character Git commit ID")
    actual = _run_checked(("git", "rev-parse", "HEAD"), cwd=repo_root)
    if actual != expected:
        raise ProfileError(f"Code snapshot mismatch: expected {expected}, got {actual}")
    for command, label in (
        (("git", "diff", "--quiet", "HEAD", "--"), "working tree"),
        (("git", "diff", "--cached", "--quiet", "HEAD", "--"), "index"),
    ):
        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            raise ProfileError(f"Profiling requires a clean tracked {label} at the pinned commit")
    return {"commit_id": actual, "tracked_worktree_clean": True}


def validate_output_root(
    output_root: Path,
    *,
    profile_base: Path = PROFILE_BASE,
    job_id: str,
) -> Path:
    if not str(job_id).isdigit():
        raise ProfileError("A numeric Slurm job ID is required")
    if not output_root.is_absolute():
        raise ProfileError("Profiling output root must be absolute")
    resolved_base = profile_base.resolve(strict=False)
    resolved_output = output_root.resolve(strict=False)
    try:
        relative = resolved_output.relative_to(resolved_base)
    except ValueError as exc:
        raise ProfileError(f"Output root must be beneath {resolved_base}") from exc
    if len(relative.parts) != 1:
        raise ProfileError("Output root must be one unique job directory directly below the profile base")
    if not relative.name.startswith(f"job_{job_id}_"):
        raise ProfileError(f"Output directory name must start with job_{job_id}_")
    if resolved_output.exists() or resolved_output.is_symlink():
        raise ProfileError(f"Refusing to reuse existing profile output root: {resolved_output}")
    return resolved_output


def _expected_config_values(spec: CellSpec) -> dict[str, Any]:
    return {
        "env_id": "swm/PushT-v1",
        "checkpoint": "checkpoints_mwm/mwm_paper10_pusht_k96_120_144_168_192_release20260728",
        "episodes": 250,
        "num_envs": 50,
        "goal_offset": spec.goal,
        "goal_indexing": "exact",
        "budget": spec.budget,
        "horizon": spec.horizon,
        "receding_horizon": spec.receding_horizon,
        "action_block": 5,
        "pop_size": spec.population,
        "elite_frac": spec.elite_frac,
        "n_iter": 5,
        "flop_accounting": "dynamics_audit",
        "save_video": False,
    }


def validate_benchmark_sources(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Resolve the real matrices and prove the four hard-coded indices still match."""

    from mwm.benchmark.config import DEFAULTS, cell_id, merged_run_config, role
    from mwm.benchmark.sweep import expand_benchmark_runs
    from mwm.config_cli import load_config

    records: dict[str, Any] = {}
    for config_relpath in sorted({spec.config_relpath for spec in CELL_SPECS}):
        config_path = repo_root / config_relpath
        if not config_path.is_file():
            raise ProfileError(f"Missing benchmark config: {config_path}")
        cfg = load_config(DEFAULTS, str(config_path), [])
        runs = expand_benchmark_runs(cfg)
        if len(runs) != NUM_SHARDS:
            raise ProfileError(f"Expected {NUM_SHARDS} cells in {config_relpath}, found {len(runs)}")
        records[config_relpath] = {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "cells": len(runs),
        }
        for spec in (item for item in CELL_SPECS if item.config_relpath == config_relpath):
            run = runs[spec.matrix_index]
            if int(run.get("matrix_index", spec.matrix_index)) != spec.matrix_index:
                raise ProfileError(f"Matrix index drift for {spec.key}")
            _, run_cfg = merged_run_config(cfg, run)
            actual = {
                "env_id": str(run_cfg.env_id),
                "checkpoint": str(run_cfg.checkpoint.run_dir),
                "episodes": int(run_cfg.eval.episodes),
                "num_envs": int(run_cfg.eval.num_envs),
                "goal_offset": int(run_cfg.eval.goal_offset),
                "goal_indexing": str(run_cfg.eval.goal_indexing),
                "budget": int(run_cfg.eval.budget),
                "horizon": int(run_cfg.planner.horizon),
                "receding_horizon": int(run_cfg.planner.receding_horizon),
                "action_block": int(run_cfg.planner.action_block),
                "pop_size": int(run_cfg.planner.pop_size),
                "elite_frac": float(run_cfg.planner.elite_frac),
                "n_iter": int(run_cfg.planner.n_iter),
                "flop_accounting": str(run_cfg.planner.flop_accounting),
                "save_video": bool(run_cfg.eval.save_video),
            }
            if actual != _expected_config_values(spec):
                raise ProfileError(
                    f"Resolved configuration drift for {spec.key}: "
                    f"expected {_expected_config_values(spec)!r}, got {actual!r}"
                )
            if str(run.get("name", "")) != spec.cell_id:
                raise ProfileError(f"Cell name drift for {spec.key}: {run.get('name')!r}")
            if cell_id(run, run_cfg) != spec.cell_id or role(run, run_cfg) != spec.role:
                raise ProfileError(f"Cell identity drift for {spec.key}")
            if run_cfg.planner.get("topk", None) is not None:
                raise ProfileError(f"Expected elite-fraction top-k resolution for {spec.key}")
            effective_topk = max(1, int(round(spec.population * spec.elite_frac)))
            if effective_topk != spec.topk:
                raise ProfileError(f"Effective top-k drift for {spec.key}")
            scheduler = run_cfg.planner.scheduler
            scheduler_tuple = (
                str(scheduler.mpc.mode),
                str(scheduler.mpc.level),
                str(scheduler.cem.mode),
                str(scheduler.cem.level),
                str(scheduler.rollout.mode),
                str(scheduler.rollout.level),
            )
            if scheduler_tuple != ("fixed", "finest", "fixed", "base", "fixed", "base"):
                raise ProfileError(f"Scheduler drift for {spec.key}: {scheduler_tuple!r}")
    if tuple(sorted({spec.matrix_index for spec in CELL_SPECS})) != SELECTED_INDICES:
        raise ProfileError("The profile must select exactly matrix indices 0 and 105")
    return records


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Could not load valid JSON from {path}: {exc}") from exc


def validate_reference_cells() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for spec in CELL_SPECS:
        run_dir = spec.reference_run_dir
        missing = [name for name in COMPLETION_FILES if not (run_dir / name).is_file()]
        if missing:
            raise ProfileError(f"Reference cell {spec.key} is incomplete: missing {missing}")
        payload = _load_json(run_dir / "eval.json")
        planner = dict(payload.get("planner_params", {}))
        actual = {
            "cell_id": payload.get("cell_id"),
            "role": payload.get("role"),
            "goal_offset": payload.get("goal_offset"),
            "budget": payload.get("eval_budget"),
            "population": planner.get("pop_size"),
            "elite_frac": planner.get("elite_frac"),
            "topk": planner.get("topk"),
            "n_iter": planner.get("n_iter"),
        }
        expected = {
            "cell_id": spec.cell_id,
            "role": spec.role,
            "goal_offset": spec.goal,
            "budget": spec.budget,
            "population": spec.population,
            "elite_frac": spec.elite_frac,
            "topk": spec.topk,
            "n_iter": 5,
        }
        if actual != expected:
            raise ProfileError(f"Reference identity drift for {spec.key}: {actual!r}")
        records[spec.key] = {
            "run_dir": str(run_dir),
            "eval_sha256": _sha256(run_dir / "eval.json"),
            "config_sha256": _sha256(run_dir / "resolved_config.yaml"),
            "reference_wall_time_sec": float(payload.get("wall_time_sec", 0.0)),
        }
    return records


def _visible_gpu_token() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if len(tokens) != 1 or tokens[0].lower() in {"all", "none", "void"}:
        raise ProfileError(f"Expected exactly one explicit CUDA_VISIBLE_DEVICES token, got {visible!r}")
    if tokens[0].startswith("MIG-"):
        raise ProfileError("The packed profile requires a full B200, not a MIG device")
    return tokens[0]


def _nvidia_query(token: str, *, query: str) -> str:
    return _run_checked(
        (
            "nvidia-smi",
            f"--id={token}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        )
    )


def _nvidia_compute_query(token: str) -> str:
    return _run_checked(
        (
            "nvidia-smi",
            f"--id={token}",
            f"--query-compute-apps={','.join(COMPUTE_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        )
    )


def _parse_csv_rows(text: str, fields: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for values in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(values) != len(fields):
            raise ProfileError(f"Unexpected nvidia-smi row for {fields}: {values!r}")
        rows.append({key: value.strip() for key, value in zip(fields, values)})
    return rows


def assert_full_b200() -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ProfileError("The packed profile must run inside a Slurm allocation")
    token = _visible_gpu_token()
    rows = _parse_csv_rows(
        _nvidia_query(token, query="name,memory.total,mig.mode.current,uuid"),
        ("name", "memory.total", "mig.mode.current", "uuid"),
    )
    if len(rows) != 1:
        raise ProfileError(f"Expected one selected physical GPU, got {len(rows)}")
    row = rows[0]
    name = row["name"]
    memory_mib = int(float(row["memory.total"]))
    mig_mode = row["mig.mode.current"].strip().lower()
    if "b200" not in name.lower():
        raise ProfileError(f"Expected a B200, got {name!r}")
    if memory_mib < MIN_FULL_B200_MEMORY_MIB:
        raise ProfileError(
            f"Expected >= {MIN_FULL_B200_MEMORY_MIB} MiB for a full B200, got {memory_mib} MiB"
        )
    if mig_mode not in {"disabled", "n/a", "[n/a]"}:
        raise ProfileError(f"Expected MIG-disabled full GPU, got MIG mode {row['mig.mode.current']!r}")

    import torch

    if torch.cuda.device_count() != 1:
        raise ProfileError(f"PyTorch must see exactly one CUDA device, got {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    torch_memory = int(properties.total_memory)
    if "b200" not in str(properties.name).lower() or torch_memory < MIN_FULL_B200_MEMORY_MIB * 1024**2:
        raise ProfileError(
            f"PyTorch did not expose a full B200: name={properties.name!r}, bytes={torch_memory}"
        )

    compute_rows = _parse_csv_rows(_nvidia_compute_query(token), COMPUTE_QUERY_FIELDS)
    foreign = []
    for process in compute_rows:
        try:
            pid = int(process["pid"])
        except ValueError:
            foreign.append(process)
            continue
        if pid != os.getpid():
            foreign.append(process)
    if foreign:
        raise ProfileError(f"Selected GPU already has foreign compute processes: {foreign!r}")
    return {
        "visible_token": token,
        "name": name,
        "uuid": row["uuid"],
        "memory_total_mib": memory_mib,
        "mig_mode": row["mig.mode.current"],
        "torch_name": str(properties.name),
        "torch_memory_total_bytes": torch_memory,
    }


class ResourceSampler:
    def __init__(self, path: Path, *, gpu_token: str, interval_sec: float) -> None:
        if interval_sec < 0.25:
            raise ProfileError("Sampling interval must be at least 0.25 seconds")
        self.path = path
        self.gpu_token = gpu_token
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._roots: dict[int, str] = {}
        self._last_cpu: dict[int, tuple[float, float]] = {}
        self._thread: threading.Thread | None = None

    def set_processes(self, roots: Mapping[int, str]) -> None:
        with self._lock:
            self._roots = {int(pid): str(label) for pid, label in roots.items()}

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="packed-b200-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_sec * 5))
            if self._thread.is_alive():
                raise ProfileError("Resource sampler did not stop cleanly")

    def _process_metrics(self, roots: Mapping[int, str], now: float) -> tuple[list[dict[str, Any]], dict[int, int]]:
        import psutil

        records: list[dict[str, Any]] = []
        pid_to_root: dict[int, int] = {}
        for root_pid, label in roots.items():
            try:
                root = psutil.Process(root_pid)
                processes = [root, *root.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            rss = 0
            cpu_percent = 0.0
            live_pids: list[int] = []
            for process in processes:
                try:
                    pid = int(process.pid)
                    times = process.cpu_times()
                    cpu_total = float(times.user + times.system)
                    rss += int(process.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                previous = self._last_cpu.get(pid)
                if previous is not None and now > previous[0]:
                    cpu_percent += max(0.0, (cpu_total - previous[1]) / (now - previous[0]) * 100.0)
                self._last_cpu[pid] = (now, cpu_total)
                live_pids.append(pid)
                pid_to_root[pid] = root_pid
            records.append(
                {
                    "root_pid": root_pid,
                    "label": label,
                    "pids": sorted(set(live_pids)),
                    "rss_bytes": rss,
                    "cpu_percent": cpu_percent,
                }
            )
        return records, pid_to_root

    def _sample(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            roots = dict(self._roots)
        process_records, pid_to_root = self._process_metrics(roots, now)
        record: dict[str, Any] = {
            "timestamp_utc": _utc_now(),
            "monotonic_sec": now,
            "processes": process_records,
        }
        try:
            gpu_rows = _parse_csv_rows(
                _nvidia_query(self.gpu_token, query=",".join(GPU_QUERY_FIELDS)),
                GPU_QUERY_FIELDS,
            )
            if len(gpu_rows) != 1:
                raise ProfileError(f"Expected one GPU sampling row, got {len(gpu_rows)}")
            record["gpu"] = gpu_rows[0]
        except Exception as exc:  # keep the run alive; final validation is fail-closed
            record["gpu_error"] = repr(exc)
        try:
            apps = _parse_csv_rows(_nvidia_compute_query(self.gpu_token), COMPUTE_QUERY_FIELDS)
            for app in apps:
                try:
                    app_pid = int(app["pid"])
                except ValueError:
                    app["root_pid"] = None
                    app["label"] = None
                else:
                    root_pid = pid_to_root.get(app_pid)
                    app["root_pid"] = root_pid
                    app["label"] = roots.get(root_pid) if root_pid is not None else None
            record["compute_apps"] = apps
        except Exception as exc:  # keep the run alive; final validation is fail-closed
            record["compute_apps_error"] = repr(exc)
        return record

    def _loop(self) -> None:
        with self.path.open("a", encoding="utf-8", buffering=1) as handle:
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    record = self._sample()
                except Exception as exc:  # pragma: no cover - last-resort sampler guard
                    record = {
                        "timestamp_utc": _utc_now(),
                        "monotonic_sec": time.monotonic(),
                        "sampler_error": repr(exc),
                    }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                remaining = self.interval_sec - (time.monotonic() - started)
                self._stop.wait(max(0.0, remaining))


def split_cpu_affinity(cpus: Iterable[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = tuple(sorted({int(cpu) for cpu in cpus}))
    if len(ordered) < 2:
        raise ProfileError("Two-way packing requires at least two allocated CPUs")
    midpoint = (len(ordered) + 1) // 2
    first, second = ordered[:midpoint], ordered[midpoint:]
    if not first or not second or set(first) & set(second) or set(first) | set(second) != set(ordered):
        raise ProfileError("Could not split the Slurm CPU affinity into two disjoint sets")
    return first, second


def build_cell_command(
    *,
    python: Path,
    config_path: Path,
    run_root: Path,
    spec: CellSpec,
    cpu_affinity: Sequence[int],
    time_path: Path,
) -> list[str]:
    if spec.matrix_index not in SELECTED_INDICES:
        raise ProfileError(f"Unapproved matrix index {spec.matrix_index}")
    if run_root.exists() or run_root.is_symlink():
        raise ProfileError(f"Refusing to overwrite profiling run root: {run_root}")
    if not cpu_affinity:
        raise ProfileError("Cell process requires a nonempty CPU affinity")
    cpu_list = ",".join(str(int(cpu)) for cpu in cpu_affinity)
    return [
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_path),
        "/usr/bin/taskset",
        "--cpu-list",
        cpu_list,
        str(python),
        "-m",
        "mwm.benchmark.matrix",
        str(config_path),
        "--set",
        f"output_dir={run_root}",
        "--num-shards",
        str(NUM_SHARDS),
        "--shard-index",
        str(spec.matrix_index),
    ]


def _launch_cell(
    *,
    spec: CellSpec,
    mode: str,
    run_root: Path,
    config_path: Path,
    python: Path,
    cpu_affinity: Sequence[int],
    log_dir: Path,
    repo_root: Path,
) -> RunningCell:
    log_path = log_dir / f"{mode}_{spec.key}.log"
    time_path = log_dir / f"{mode}_{spec.key}.time.txt"
    command = build_cell_command(
        python=python,
        config_path=config_path,
        run_root=run_root,
        spec=spec,
        cpu_affinity=cpu_affinity,
        time_path=time_path,
    )
    log_handle = log_path.open("x", encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MPLBACKEND": "Agg",
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "PYOPENGL_PLATFORM": env.get("PYOPENGL_PLATFORM", "egl"),
        }
    )
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        raise
    return RunningCell(
        spec=spec,
        mode=mode,
        run_root=run_root,
        command=command,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
        time_path=time_path,
        cpu_affinity=tuple(int(cpu) for cpu in cpu_affinity),
        started_monotonic=started,
    )


def _wait_for_cells(cells: Sequence[RunningCell], sampler: ResourceSampler) -> list[dict[str, Any]]:
    sampler.set_processes({cell.process.pid: f"{cell.mode}:{cell.spec.key}" for cell in cells})
    unfinished = list(cells)
    try:
        while unfinished:
            for cell in list(unfinished):
                returncode = cell.process.poll()
                if returncode is None:
                    continue
                cell.returncode = int(returncode)
                cell.finished_monotonic = time.monotonic()
                cell.log_handle.close()
                unfinished.remove(cell)
            if unfinished:
                time.sleep(0.2)
    finally:
        sampler.set_processes({})
        for cell in cells:
            if not cell.log_handle.closed:
                cell.log_handle.close()
    results: list[dict[str, Any]] = []
    for cell in cells:
        if cell.returncode is None or cell.finished_monotonic is None:
            raise ProfileError(f"Lost process completion status for {cell.spec.key}")
        result = {
            "key": cell.spec.key,
            "mode": cell.mode,
            "pid": int(cell.process.pid),
            "returncode": cell.returncode,
            "started_monotonic": cell.started_monotonic,
            "finished_monotonic": cell.finished_monotonic,
            "external_wall_time_sec": cell.finished_monotonic - cell.started_monotonic,
            "run_root": str(cell.run_root),
            "log_path": str(cell.log_path),
            "time_path": str(cell.time_path),
            "cpu_affinity": list(cell.cpu_affinity),
            "command": cell.command,
            "time_verbose": _parse_time_verbose(cell.time_path),
        }
        results.append(result)
        if cell.returncode != 0:
            raise ProfileError(
                f"Profile cell {cell.mode}:{cell.spec.key} exited {cell.returncode}; "
                f"see {cell.log_path}"
            )
    return results


def _parse_time_verbose(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProfileError(f"GNU time did not write its resource report: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if ": " in stripped:
            key, value = stripped.split(": ", 1)
            values[key] = value
    required = (
        "User time (seconds)",
        "System time (seconds)",
        "Percent of CPU this job got",
        "Elapsed (wall clock) time (h:mm:ss or m:ss)",
        "Maximum resident set size (kbytes)",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ProfileError(f"GNU time report is incomplete at {path}: missing {missing}")
    return {
        "user_time_sec": float(values[required[0]]),
        "system_time_sec": float(values[required[1]]),
        "cpu_percent": values[required[2]],
        "elapsed": values[required[3]],
        "maximum_rss_kib": int(values[required[4]]),
    }


def _terminate_profile_cells(cells: Sequence[RunningCell]) -> list[str]:
    """Stop only child processes owned by this profiling harness after failure."""

    errors: list[str] = []
    for cell in cells:
        if cell.process.poll() is None:
            try:
                cell.process.terminate()
            except OSError as exc:
                errors.append(f"terminate {cell.spec.key}: {exc}")
    deadline = time.monotonic() + 10.0
    for cell in cells:
        if cell.process.poll() is None:
            try:
                cell.process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    cell.process.kill()
                    cell.process.wait(timeout=5.0)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(f"kill {cell.spec.key}: {exc}")
        if not cell.log_handle.closed:
            cell.log_handle.close()
    return errors


def _strip_diagnostic_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_diagnostic_timing(item)
            for key, item in value.items()
            if key not in TIMING_DIAGNOSTIC_KEYS
        }
    if isinstance(value, list):
        return [_strip_diagnostic_timing(item) for item in value]
    return value


def normalize_scientific_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only declared path, timing, and provenance differences."""

    normalized = copy.deepcopy(dict(payload))
    if "policy_diagnostics" in normalized:
        if normalized.get("policy_diagnostics") != normalized.get("planning_diagnostics"):
            raise ProfileError("Legacy policy_diagnostics differs from planning_diagnostics")
        normalized.pop("policy_diagnostics")
    normalized.pop("wall_time_sec", None)
    normalized.pop("dependencies", None)
    videos = normalized.pop("videos", None)
    if videos not in (None, []):
        raise ProfileError("The pinned save_video=false profile unexpectedly contains top-level videos")
    config = normalized.get("config")
    if isinstance(config, dict):
        config.pop("resolved_path", None)
        config.pop("sha256", None)
        if not config:
            normalized.pop("config", None)
    if "planning_diagnostics" in normalized:
        normalized["planning_diagnostics"] = _strip_diagnostic_timing(
            normalized["planning_diagnostics"]
        )
    batches = normalized.get("batches", [])
    if not isinstance(batches, list):
        raise ProfileError("Evaluation batches must be a list")
    for batch in batches:
        if not isinstance(batch, dict):
            raise ProfileError("Each evaluation batch must be a mapping")
        batch_videos = batch.pop("videos", None)
        if batch_videos not in (None, []):
            raise ProfileError("The pinned save_video=false profile unexpectedly contains batch videos")
        if "planning_diagnostics" in batch:
            batch["planning_diagnostics"] = _strip_diagnostic_timing(
                batch["planning_diagnostics"]
            )
    return normalized


def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "left": repr(left)[:500], "right": repr(right)[:500]}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {
                "path": path,
                "left_only_keys": sorted(set(left) - set(right)),
                "right_only_keys": sorted(set(right) - set(left)),
            }
        for key in sorted(left):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return {"path": path, "left": repr(left)[:500], "right": repr(right)[:500]}
    return None


def _load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Invalid JSONL at {path}: {exc}") from exc
    return rows


def verify_cell_artifacts(spec: CellSpec, run_root: Path) -> dict[str, Any]:
    run_dir = run_root / f"{spec.matrix_index:03d}_{spec.cell_id}"
    missing = [name for name in COMPLETION_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ProfileError(f"Profile output {spec.key} is incomplete: missing {missing}")
    marker = run_root / "shards" / f"shard_{spec.matrix_index:03d}_of_{NUM_SHARDS:03d}.json"
    if not marker.is_file():
        raise ProfileError(f"Missing exact shard completion marker for {spec.key}: {marker}")
    marker_payload = _load_json(marker)
    expected_marker = {
        "shard_index": spec.matrix_index,
        "num_shards": NUM_SHARDS,
        "cells": 1,
        "completed": 1,
    }
    if marker_payload != expected_marker:
        raise ProfileError(f"Unexpected shard marker for {spec.key}: {marker_payload!r}")

    reference_dir = spec.reference_run_dir
    reference = _load_json(reference_dir / "eval.json")
    candidate = _load_json(run_dir / "eval.json")
    if "policy_diagnostics" in candidate:
        raise ProfileError("New profile output unexpectedly emitted policy_diagnostics")
    candidate_sidecar = _load_json(run_dir / "planning_diagnostics.json")
    reference_sidecar = _load_json(reference_dir / "planning_diagnostics.json")
    if candidate_sidecar != candidate.get("planning_diagnostics"):
        raise ProfileError(f"Planning sidecar does not match eval.json for {spec.key}")
    if reference_sidecar != reference.get("planning_diagnostics"):
        raise ProfileError(f"Reference planning sidecar mismatch for {spec.key}")

    normalized_reference = normalize_scientific_payload(reference)
    normalized_candidate = normalize_scientific_payload(candidate)
    difference = first_difference(normalized_reference, normalized_candidate)
    reference_traces = _load_jsonl(reference_dir / "episode_traces.jsonl")
    candidate_traces = _load_jsonl(run_dir / "episode_traces.jsonl")
    trace_difference = first_difference(reference_traces, candidate_traces)
    if difference is not None or trace_difference is not None:
        raise ProfileError(
            f"Scientific parity failed for {spec.key}: "
            f"eval_diff={difference!r}, episode_trace_diff={trace_difference!r}"
        )

    diagnostics = dict(candidate.get("planning_diagnostics", {}))
    return {
        "key": spec.key,
        "run_dir": str(run_dir),
        "reference_run_dir": str(reference_dir),
        "scientific_sha256": _canonical_sha256(normalized_candidate),
        "reference_scientific_sha256": _canonical_sha256(normalized_reference),
        "episode_traces_sha256": _canonical_sha256(candidate_traces),
        "reference_episode_traces_sha256": _canonical_sha256(reference_traces),
        "eval_wall_time_sec": float(candidate.get("wall_time_sec", 0.0)),
        "plan_time_total_sec": float(diagnostics.get("plan_time_total_sec", 0.0)),
        "dynamics_flops_total": int(diagnostics.get("dynamics_flops_total", 0)),
        "gpu_science_parity": True,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ProfileError("Cannot take a quantile of an empty sample")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summarize_resource_rows(rows: Sequence[Any], *, label: str) -> dict[str, Any]:
    if not rows:
        raise ProfileError(f"Resource sampler produced no rows for {label}")
    gpu_rows = [row["gpu"] for row in rows if isinstance(row, dict) and isinstance(row.get("gpu"), dict)]
    compute_valid = sum(1 for row in rows if isinstance(row, dict) and isinstance(row.get("compute_apps"), list))
    if len(gpu_rows) < 3 or len(gpu_rows) < math.ceil(len(rows) * 0.8):
        raise ProfileError(
            f"Insufficient valid GPU samples for {label}: "
            f"{len(gpu_rows)}/{len(rows)} (need >=3 and >=80%)"
        )
    if compute_valid < 3 or compute_valid < math.ceil(len(rows) * 0.8):
        raise ProfileError(
            f"Insufficient valid compute-process samples for {label}: {compute_valid}/{len(rows)}"
        )

    util = [value for row in gpu_rows if (value := _float_or_none(row.get("utilization.gpu"))) is not None]
    memory = [value for row in gpu_rows if (value := _float_or_none(row.get("memory.used"))) is not None]
    power = [value for row in gpu_rows if (value := _float_or_none(row.get("power.draw"))) is not None]
    if not util or not memory:
        raise ProfileError("GPU samples do not contain utilization and memory values")

    per_label: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for process in row.get("processes", []):
            if not isinstance(process, dict) or not process.get("label"):
                continue
            bucket = per_label.setdefault(
                str(process["label"]), {"rss_bytes": [], "cpu_percent": [], "gpu_memory_mib": []}
            )
            bucket["rss_bytes"].append(float(process.get("rss_bytes", 0)))
            bucket["cpu_percent"].append(float(process.get("cpu_percent", 0.0)))
        app_memory: dict[str, float] = {}
        for app in row.get("compute_apps", []):
            if not isinstance(app, dict) or not app.get("label"):
                continue
            value = _float_or_none(app.get("used_gpu_memory"))
            if value is not None:
                app_memory[str(app["label"])] = app_memory.get(str(app["label"]), 0.0) + value
        for label, value in app_memory.items():
            per_label.setdefault(
                label, {"rss_bytes": [], "cpu_percent": [], "gpu_memory_mib": []}
            )["gpu_memory_mib"].append(value)

    label_summary: dict[str, Any] = {}
    for label, values in sorted(per_label.items()):
        label_summary[label] = {
            "peak_rss_bytes": max(values["rss_bytes"], default=0.0),
            "mean_cpu_percent": statistics.fmean(values["cpu_percent"]) if values["cpu_percent"] else 0.0,
            "peak_cpu_percent": max(values["cpu_percent"], default=0.0),
            "peak_gpu_memory_mib": max(values["gpu_memory_mib"], default=0.0),
        }
    return {
        "samples": len(rows),
        "valid_gpu_samples": len(gpu_rows),
        "valid_compute_samples": compute_valid,
        "gpu_utilization_percent": {
            "mean": statistics.fmean(util),
            "p50": _quantile(util, 0.5),
            "p95": _quantile(util, 0.95),
            "max": max(util),
            "busy_fraction_ge_50pct": sum(value >= 50.0 for value in util) / len(util),
        },
        "gpu_memory_used_mib": {"mean": statistics.fmean(memory), "max": max(memory)},
        "gpu_power_watts": {
            "mean": statistics.fmean(power) if power else None,
            "max": max(power) if power else None,
        },
        "processes": label_summary,
    }


def summarize_resource_samples(
    path: Path,
    *,
    windows: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(path)
    summary = {"overall": _summarize_resource_rows(rows, label="overall")}
    window_summaries: dict[str, Any] = {}
    for label, (start, finish) in (windows or {}).items():
        selected = [
            row
            for row in rows
            if isinstance(row, dict)
            and (timestamp := _float_or_none(row.get("monotonic_sec"))) is not None
            and float(start) <= timestamp <= float(finish)
        ]
        window_summaries[label] = _summarize_resource_rows(selected, label=label)
    summary["windows"] = window_summaries
    return summary


def _copy_configs(profile_root: Path, source_records: Mapping[str, Any]) -> dict[str, Path]:
    config_dir = profile_root / "configs"
    config_dir.mkdir()
    copied: dict[str, Path] = {}
    for relpath, source_record in source_records.items():
        source = Path(str(source_record["path"]))
        destination = config_dir / source.name
        shutil.copy2(source, destination)
        if _sha256(destination) != str(source_record["sha256"]):
            raise ProfileError(f"Copied config hash mismatch: {destination}")
        copied[relpath] = destination
    return copied


def _pair_metrics(
    goal: int,
    sequential: Mapping[str, dict[str, Any]],
    packed: Mapping[str, dict[str, Any]],
    pair_window: Mapping[str, float],
    parity: Mapping[str, Mapping[str, dict[str, Any]]],
) -> dict[str, Any]:
    keys = (f"g{goal}_low", f"g{goal}_high")
    sequential_sum = sum(float(sequential[key]["external_wall_time_sec"]) for key in keys)
    makespan = float(pair_window["finished_monotonic"] - pair_window["started_monotonic"])
    if makespan <= 0:
        raise ProfileError(f"Invalid packed makespan for goal {goal}")
    audited_flops = sum(int(parity["pack2"][key]["dynamics_flops_total"]) for key in keys)
    return {
        "goal": goal,
        "sequential_sum_sec": sequential_sum,
        "packed_pair_makespan_sec": makespan,
        "aggregate_speedup": sequential_sum / makespan,
        "packed_cells_per_hour": 2.0 * 3600.0 / makespan,
        "audited_dynamics_teraflops_per_sec": audited_flops / makespan / 1.0e12,
        "packed_individual_external_wall_time_sec": {
            key: float(packed[key]["external_wall_time_sec"]) for key in keys
        },
        "per_cell_slowdown": {
            key: float(parity["pack2"][key]["eval_wall_time_sec"])
            / float(parity["sequential"][key]["eval_wall_time_sec"])
            for key in keys
        },
    }


def run_profile(
    *,
    output_root: Path,
    expected_commit: str,
    python: Path,
    sample_interval_sec: float,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    resolved_output = validate_output_root(output_root, job_id=job_id)
    code = validate_code_snapshot(repo_root, expected_commit)
    sources = validate_benchmark_sources(repo_root)
    references = validate_reference_cells()
    gpu = assert_full_b200()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ProfileError(f"Python executable is unavailable: {python}")
    for executable in (Path("/usr/bin/time"), Path("/usr/bin/taskset")):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ProfileError(f"Required profiling executable is unavailable: {executable}")

    resolved_output.mkdir(parents=True, exist_ok=False)
    copied_configs = _copy_configs(resolved_output, sources)
    report_path = resolved_output / "profile_report.json"
    sample_path = resolved_output / "resource_samples.jsonl"
    log_dir = resolved_output / "logs"
    log_dir.mkdir()
    cpu_affinity = tuple(sorted(os.sched_getaffinity(0)))
    first_half, second_half = split_cpu_affinity(cpu_affinity)
    if len(cpu_affinity) < 4:
        raise ProfileError(f"Expected a substantial Slurm CPU allocation, got {cpu_affinity!r}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "profile_root": str(resolved_output),
        "job": {
            "job_id": job_id,
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "no_array_required": True,
        },
        "code": code,
        "gpu": gpu,
        "cpu_affinity": list(cpu_affinity),
        "pack2_cpu_affinities": [list(first_half), list(second_half)],
        "sources": sources,
        "references": references,
        "selection": [
            {
                "key": spec.key,
                "goal": spec.goal,
                "population": spec.population,
                "elite_frac": spec.elite_frac,
                "topk": spec.topk,
                "n_iter": 5,
                "matrix_index": spec.matrix_index,
                "cell_id": spec.cell_id,
            }
            for spec in CELL_SPECS
        ],
        "automatic_pack_factors_attempted": [1, 2],
        "automatic_pack_factors_not_attempted": [3, 4],
        "execution": {"sequential": {}, "pack2": {}, "pack2_windows": {}},
        "parity": {"sequential": {}, "pack2": {}},
    }
    _write_json_atomic(report_path, report)

    sampler = ResourceSampler(
        sample_path,
        gpu_token=str(gpu["visible_token"]),
        interval_sec=sample_interval_sec,
    )
    sampler_started = False
    active_cells: list[RunningCell] = []
    try:
        sampler.start()
        sampler_started = True
        for spec in CELL_SPECS:
            mode = "sequential"
            run_root = resolved_output / "runs" / mode / spec.key
            run_root.parent.mkdir(parents=True, exist_ok=True)
            running = _launch_cell(
                spec=spec,
                mode=mode,
                run_root=run_root,
                config_path=copied_configs[spec.config_relpath],
                python=python,
                cpu_affinity=cpu_affinity,
                log_dir=log_dir,
                repo_root=repo_root,
            )
            active_cells = [running]
            result = _wait_for_cells(active_cells, sampler)[0]
            active_cells = []
            report["execution"]["sequential"][spec.key] = result
            report["parity"]["sequential"][spec.key] = verify_cell_artifacts(spec, run_root)
            _write_json_atomic(report_path, report)

        for goal in (25, 50):
            specs = (SPEC_BY_KEY[f"g{goal}_low"], SPEC_BY_KEY[f"g{goal}_high"])
            pair_started = time.monotonic()
            active_cells = []
            for spec, affinity in zip(specs, (first_half, second_half)):
                mode = "pack2"
                run_root = resolved_output / "runs" / mode / spec.key
                run_root.parent.mkdir(parents=True, exist_ok=True)
                active_cells.append(
                    _launch_cell(
                        spec=spec,
                        mode=mode,
                        run_root=run_root,
                        config_path=copied_configs[spec.config_relpath],
                        python=python,
                        cpu_affinity=affinity,
                        log_dir=log_dir,
                        repo_root=repo_root,
                    )
                )
            results = _wait_for_cells(active_cells, sampler)
            active_cells = []
            pair_finished = time.monotonic()
            report["execution"]["pack2_windows"][f"goal{goal}"] = {
                "started_monotonic": pair_started,
                "finished_monotonic": pair_finished,
                "makespan_sec": pair_finished - pair_started,
            }
            for spec, result in zip(specs, results):
                report["execution"]["pack2"][spec.key] = result
                report["parity"]["pack2"][spec.key] = verify_cell_artifacts(
                    spec, Path(str(result["run_root"]))
                )
            _write_json_atomic(report_path, report)

        sampler.stop()
        sampler_started = False
        windows: dict[str, tuple[float, float]] = {}
        for mode in ("sequential", "pack2"):
            for key, result in report["execution"][mode].items():
                windows[f"{mode}:{key}"] = (
                    float(result["started_monotonic"]),
                    float(result["finished_monotonic"]),
                )
        for pair, window in report["execution"]["pack2_windows"].items():
            windows[f"pack2:{pair}_pair"] = (
                float(window["started_monotonic"]),
                float(window["finished_monotonic"]),
            )
        report["resources"] = summarize_resource_samples(sample_path, windows=windows)
        expected_process_labels = {
            f"{mode}:{spec.key}" for mode in ("sequential", "pack2") for spec in CELL_SPECS
        }
        process_resources = report["resources"]["overall"]["processes"]
        missing_process_resources = sorted(
            label
            for label in expected_process_labels
            if label not in process_resources
            or float(process_resources[label].get("peak_rss_bytes", 0.0)) <= 0.0
            or float(process_resources[label].get("peak_gpu_memory_mib", 0.0)) <= 0.0
        )
        if missing_process_resources:
            raise ProfileError(
                "Resource instrumentation did not capture positive CPU/GPU memory for: "
                f"{missing_process_resources}"
            )
        sequential = report["execution"]["sequential"]
        packed = report["execution"]["pack2"]
        report["throughput"] = {
            f"goal{goal}": _pair_metrics(
                goal,
                sequential,
                packed,
                report["execution"]["pack2_windows"][f"goal{goal}"],
                report["parity"],
            )
            for goal in (25, 50)
        }
        report["status"] = "pass"
        report["completed_at"] = _utc_now()
        report["deployment_authorized"] = False
        report["next_step"] = (
            "Review parity, speedup, GPU/host headroom, and contention before separately authorizing "
            "any pack-3/pack-4/MPS profile or live packed-worker deployment."
        )
        _write_json_atomic(report_path, report)
        return report
    except Exception as exc:
        cleanup_errors = _terminate_profile_cells(active_cells)
        if sampler_started:
            try:
                sampler.stop()
            except Exception as sampler_exc:  # preserve the primary failure
                cleanup_errors.append(f"sampler stop: {sampler_exc}")
        report["status"] = "failed"
        report["failed_at"] = _utc_now()
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "cleanup_errors": cleanup_errors,
        }
        _write_json_atomic(report_path, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile sequential versus two-way release20260728 evaluation on one full B200."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    args = parser.parse_args()
    try:
        report = run_profile(
            output_root=args.output_root,
            expected_commit=args.expected_commit,
            python=args.python,
            sample_interval_sec=args.sample_interval_sec,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps({"status": report["status"], "profile_root": report["profile_root"]}, sort_keys=True))


if __name__ == "__main__":
    main()
