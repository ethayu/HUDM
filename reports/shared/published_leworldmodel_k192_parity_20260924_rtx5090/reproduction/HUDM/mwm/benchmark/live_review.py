from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import threading
from typing import Any, Iterator


@dataclass(frozen=True)
class LiveBenchmark:
    config_path: Path
    cfg: Any
    output_dir: Path
    resolved: list[tuple[Any, Any]]
    manifest_path: Path
    manifest_info: dict[str, Any]
    validation_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any] | None]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


def _cell_signature(run_dir: Path, manifest_sha: str) -> tuple[Any, ...]:
    from mwm.benchmark.eval_artifacts import eval_artifact_signature

    paths = (
        run_dir / "resolved_config.yaml",
        run_dir / "metrics.jsonl",
        run_dir / "summary.json",
        run_dir / "planning_diagnostics.json",
        run_dir / "episode_traces.jsonl",
    )
    signature: list[Any] = [manifest_sha, eval_artifact_signature(run_dir / "eval.json")]
    for path in paths:
        try:
            stat = path.stat()
            signature.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append(None)
    return tuple(signature)


def load_live_benchmark(
    config_path: str | Path,
    *,
    roles: Any = None,
    overrides: list[str] | None = None,
) -> LiveBenchmark:
    from mwm.benchmark.config import (
        DEFAULTS,
        filter_resolved_by_roles,
        load_manifest_config,
        manifest_path,
        merged_run_config,
        require_no_legacy_fields,
        validate_benchmark_matrix,
    )
    from mwm.benchmark.sweep import expand_benchmark_runs
    from mwm.config_cli import load_config

    path = Path(config_path).resolve()
    cfg = load_config(DEFAULTS, str(path), overrides or [])
    require_no_legacy_fields(cfg)
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = expand_benchmark_runs(cfg)
    if not runs:
        raise ValueError("Benchmark config must include at least one run.")
    for index, run in enumerate(runs):
        if "matrix_index" not in run:
            run.matrix_index = index
    resolved = [(run, merged_run_config(cfg, run)[1]) for run in runs]
    resolved = filter_resolved_by_roles(cfg, resolved, roles)
    validate_benchmark_matrix(cfg, resolved)
    return LiveBenchmark(
        config_path=path,
        cfg=cfg,
        output_dir=output_dir,
        resolved=resolved,
        manifest_path=manifest_path(cfg),
        manifest_info=load_manifest_config(cfg),
    )


def collect_completed_cells(benchmark: LiveBenchmark) -> tuple[list[dict[str, Any]], list[str]]:
    from mwm.benchmark.config import cell_id
    from mwm.benchmark.eval_artifacts import validate_eval_storage_reference
    from mwm.benchmark.matrix import _completed_row, _configure_run_paths, _run_dir
    from mwm.io import file_sha256

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    current_manifest_sha = file_sha256(benchmark.manifest_path) if benchmark.manifest_path.is_file() else ""
    for fallback_index, (run, run_cfg) in enumerate(benchmark.resolved):
        run_dir = _run_dir(benchmark.output_dir, run, fallback_index)
        _configure_run_paths(run_cfg, run_dir, benchmark.manifest_path)
        cache_key = str(run_dir.resolve())
        signature = _cell_signature(run_dir, current_manifest_sha)
        cached = benchmark.validation_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            row = dict(cached[1]) if cached[1] is not None else None
        else:
            try:
                row = _completed_row(run_dir, run_cfg, benchmark.manifest_path)
            except (OSError, TypeError, ValueError):
                row = None
        expected_cell_id = cell_id(run, run_cfg)
        if row is not None:
            expected_output = (run_dir / "eval.json").resolve()
            try:
                actual_output = Path(str(row.get("output_json", ""))).resolve()
            except (OSError, RuntimeError, ValueError):
                actual_output = Path()
            identity_matches = (
                str(row.get("cell_id", "")) == expected_cell_id
                and str(row.get("env_id", "")) == str(run_cfg.env_id)
                and int(row.get("seed", -1)) == int(run_cfg.eval.seed)
                and actual_output == expected_output
            )
            manifest_matches = (
                bool(current_manifest_sha)
                and str(row.get("manifest_file_sha256", "")) == current_manifest_sha
            )
            storage_matches = validate_eval_storage_reference(
                run_dir / "eval.json",
                verify="metadata",
            )
            if not identity_matches or not manifest_matches or not storage_matches:
                row = None
        benchmark.validation_cache[cache_key] = (signature, dict(row) if row is not None else None)
        if row is None:
            missing.append(expected_cell_id)
        else:
            rows.append(row)

    missing_hashes = [str(row.get("name", "run")) for row in rows if not row.get("manifest_sha256")]
    if missing_hashes:
        raise ValueError(f"Completed benchmark cells missing immutable manifest hashes: {missing_hashes[:10]}")
    manifest_hashes = {str(row.get("manifest_sha256", "")) for row in rows}
    if len(manifest_hashes) > 1:
        raise ValueError(f"Completed benchmark cells do not share one manifest: {sorted(manifest_hashes)}")
    return rows, missing


def _snapshot_fingerprint(rows: list[dict[str, Any]], expected_cells: int) -> str:
    raw = json.dumps(
        {"expected_cells": expected_cells, "rows": rows},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _snapshot_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / ".live_review.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def render_live_snapshot(
    benchmark: LiveBenchmark,
    *,
    refresh_seconds: int = 60,
    force: bool = False,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from mwm.benchmark.html import write_review_html
    from mwm.benchmark.pareto import write_pareto_html
    from mwm.benchmark.plots import write_default_plots
    from mwm.benchmark.summary import write_per_env_table, write_summary_csv
    from mwm.io import load_json, write_json, write_metrics_jsonl

    root = benchmark.output_dir
    with _snapshot_lock(root):
        rows, missing = collect_completed_cells(benchmark)
        expected_cells = len(benchmark.resolved)
        completed_cells = len(rows)
        complete = completed_cells == expected_cells
        fingerprint = _snapshot_fingerprint(rows, expected_cells)
        final_summary = root / "summary.live.json"
        if not force and final_summary.is_file():
            try:
                existing = load_json(final_summary)
            except (OSError, ValueError):
                existing = {}
            if str(existing.get("fingerprint", "")) == fingerprint:
                return {
                    "updated": False,
                    "complete": complete,
                    "completed_cells": completed_cells,
                    "expected_cells": expected_cells,
                    "output_dir": str(root),
                    "review": str(root / "review.live.html"),
                }

        generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        temp_root = Path(tempfile.mkdtemp(prefix=".live-review-", dir=root))
        final_plot_dir = root / "plots" / "live"
        temp_plot_dir = temp_root / "plots"
        final_pareto = final_plot_dir / "pareto.html"
        temp_pareto = temp_plot_dir / "pareto.html"
        temp_summary_csv = temp_root / "summary.live.csv"
        temp_metrics = temp_root / "metrics.live.jsonl"
        temp_per_env = temp_root / "per_env_summary.live.csv"
        temp_summary = temp_root / "summary.live.json"
        temp_review = root / f".review.live.{os.getpid()}.tmp"
        try:
            write_summary_csv(temp_summary_csv, rows)
            write_metrics_jsonl(temp_metrics, rows)
            write_per_env_table(temp_per_env, rows)
            temp_plots = write_default_plots(temp_plot_dir, rows, compact=True)
            write_pareto_html(temp_pareto, rows, provisional=not complete)
            final_plots = [str(final_plot_dir / Path(path).name) for path in temp_plots]
            live_status = {
                "complete": complete,
                "completed_cells": completed_cells,
                "expected_cells": expected_cells,
                "generated_at": generated_at,
                "refresh_seconds": int(refresh_seconds),
                "summary_href": "summary.live.json",
                "fingerprint": fingerprint,
            }
            serve_command = (
                "python -m mwm.benchmark.live_review "
                f"{shlex.quote(str(benchmark.config_path))} --serve --refresh-seconds {int(refresh_seconds)}"
            )
            write_review_html(
                temp_review,
                f"{str(benchmark.cfg.title)} — Live",
                rows,
                [],
                plots=final_plots,
                expected_cells=expected_cells,
                pareto_html=str(final_pareto),
                live_status=live_status,
                include_rollouts=False,
                serve_command=serve_command,
            )
            summary = {
                "title": str(benchmark.cfg.title),
                "status": "complete" if complete else "in_progress",
                "complete": complete,
                "generated_at": generated_at,
                "fingerprint": fingerprint,
                "completed_cells": completed_cells,
                "expected_cells": expected_cells,
                "missing_cells": len(missing),
                "missing_preview": missing[:100],
                "output_dir": str(root),
                "review": str(root / "review.live.html"),
                "runs": rows,
                "sweep": OmegaConf.to_container(benchmark.cfg.get("sweep", {}), resolve=True),
                "pareto_cost": "dynamics_flops_total",
                "pareto_html": str(final_pareto),
                "plots": final_plots,
                "per_env_table": str(root / "per_env_summary.live.csv"),
                "manifest": {
                    "group": str(benchmark.manifest_info["group"]),
                    "path": str(benchmark.manifest_path),
                    "seed": int(benchmark.cfg.seed),
                },
            }
            write_json(temp_summary, summary)

            final_plot_dir.mkdir(parents=True, exist_ok=True)
            final_plot_names = {Path(path).name for path in temp_plots}
            for old_plot in final_plot_dir.glob("*.png"):
                if old_plot.name not in final_plot_names:
                    old_plot.unlink()
            for plot in temp_plots:
                _replace(Path(plot), final_plot_dir / Path(plot).name)
            _replace(temp_pareto, final_pareto)
            _replace(temp_summary_csv, root / "summary.live.csv")
            _replace(temp_metrics, root / "metrics.live.jsonl")
            _replace(temp_per_env, root / "per_env_summary.live.csv")
            _replace(temp_summary, final_summary)
            # The user-facing document is the commit marker: publish it only
            # after every linked artifact and the pollable summary are ready.
            _replace(temp_review, root / "review.live.html")
        finally:
            temp_review.unlink(missing_ok=True)
            shutil.rmtree(temp_root, ignore_errors=True)

    print(
        f"[live-review] {completed_cells}/{expected_cells} cells; wrote {root / 'review.live.html'}",
        flush=True,
    )
    return {
        "updated": True,
        "complete": complete,
        "completed_cells": completed_cells,
        "expected_cells": expected_cells,
        "output_dir": str(root),
        "review": str(root / "review.live.html"),
    }


def watch_live_snapshots(
    benchmark: LiveBenchmark,
    *,
    refresh_seconds: int,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            report = render_live_snapshot(benchmark, refresh_seconds=refresh_seconds)
            if bool(report.get("complete", False)):
                return
        except Exception as exc:
            print(f"[live-review] refresh failed: {exc}", flush=True)
        stop_event.wait(refresh_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and optionally serve an in-progress MWM benchmark review.")
    parser.add_argument("config", help="Benchmark YAML config whose resolved cells define the live snapshot.")
    parser.add_argument("--roles", nargs="+", help="Optional role filter matching the benchmark matrix CLI.")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. seed=1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Render one snapshot and exit (the default without --serve).")
    mode.add_argument("--serve", action="store_true", help="Watch for completed cells and serve review.live.html locally.")
    parser.add_argument("--refresh-seconds", type=int, default=60, help="Polling interval for --serve (minimum 5).")
    parser.add_argument("--host", default="127.0.0.1", help="Localhost address for --serve.")
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve.")
    args = parser.parse_args()
    if args.refresh_seconds < 5:
        parser.error("--refresh-seconds must be at least 5")
    if args.serve:
        from mwm.benchmark.review_server import validate_server_address

        try:
            validate_server_address(args.host, args.port)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    benchmark = load_live_benchmark(args.config, roles=args.roles, overrides=args.set)
    report = render_live_snapshot(benchmark, refresh_seconds=args.refresh_seconds, force=True)
    if not args.serve:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return

    from mwm.benchmark.review_server import serve_review
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=watch_live_snapshots,
        kwargs={
            "benchmark": benchmark,
            "refresh_seconds": int(args.refresh_seconds),
            "stop_event": stop_event,
        },
        name="benchmark-live-review",
        daemon=True,
    )
    watcher.start()
    try:
        serve_review(
            benchmark.output_dir,
            host=args.host,
            port=args.port,
            review_page="review.live.html",
            warmup=False,
        )
    finally:
        stop_event.set()
        watcher.join(timeout=max(5, int(args.refresh_seconds) + 1))


if __name__ == "__main__":
    main()


__all__ = [
    "LiveBenchmark",
    "collect_completed_cells",
    "load_live_benchmark",
    "main",
    "render_live_snapshot",
    "watch_live_snapshots",
]
