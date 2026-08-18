from __future__ import annotations

import contextlib
import io
import shutil
import time
import traceback
from pathlib import Path
from typing import Any


AGGREGATE_OUTPUTS = ("summary.json", "summary.csv", "metrics.jsonl", "per_env_summary.csv", "review.html")


def run_eval_mwm(cfg_path: str) -> None:
    from mwm.eval.runner import main as run_eval

    run_eval(cfg_path)


def _run_dir(output_dir: Path, run: Any, fallback_index: int) -> Path:
    from mwm.benchmark.config import safe_name

    idx = int(run.get("matrix_index", fallback_index))
    name = safe_name(str(run.get("name", run.get("role", "run"))))
    return output_dir / f"{idx:03d}_{name}"


def _clear_aggregate_outputs(output_dir: Path) -> None:
    for name in AGGREGATE_OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)
    shutil.rmtree(output_dir / "plots", ignore_errors=True)


def _normalized_config(value: Any, manifest_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if isinstance(value, (str, Path)):
        value = OmegaConf.load(str(value))
    plain = OmegaConf.to_container(value, resolve=True)
    if not isinstance(plain, dict):
        return {}
    eval_cfg = plain.get("eval")
    if isinstance(eval_cfg, dict):
        # The first cell may create the shared manifest while later cells read
        # it. Treat those two equivalent transport configurations as the same.
        eval_cfg["manifest_path"] = str(manifest_path)
        eval_cfg["write_manifest_path"] = None
    return plain


def _completed_run(
    run_dir: Path,
    run_cfg: Any,
    manifest_path: Path,
    *,
    materialize: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    from mwm.benchmark.eval_artifacts import load_eval_artifact, load_eval_capsule
    from mwm.io import load_json

    required = (
        run_dir / "eval.json",
        run_dir / "resolved_config.yaml",
        run_dir / "metrics.jsonl",
        run_dir / "summary.json",
        run_dir / "planning_diagnostics.json",
        run_dir / "episode_traces.jsonl",
    )
    if any(not path.is_file() for path in required):
        return None
    if _normalized_config(run_dir / "resolved_config.yaml", manifest_path) != _normalized_config(run_cfg, manifest_path):
        return None
    summary = load_json(run_dir / "summary.json")
    row = summary.get("run")
    if not isinstance(row, dict):
        return None
    try:
        payload = (
            load_eval_artifact(run_dir / "eval.json", verify="full")
            if materialize
            else load_eval_capsule(run_dir / "eval.json", verify="compressed_hash")
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return row, payload


def _configure_run_paths(run_cfg: Any, run_dir: Path, manifest_path: Path) -> None:
    run_cfg.eval.output_path = str(run_dir / "eval.json")
    if manifest_path.is_file():
        run_cfg.eval.manifest_path = str(manifest_path)
        run_cfg.eval.write_manifest_path = None
    else:
        run_cfg.eval.manifest_path = None
        run_cfg.eval.write_manifest_path = str(manifest_path)
    run_cfg.eval.video_path = str(run_dir / "videos")
    if "save_video" not in run_cfg.eval:
        run_cfg.eval.save_video = True


def _episode_trace_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    successes = list(payload.get("swm_results", {}).get("episode_successes", []))
    rows: list[dict[str, Any]] = []
    index = 0
    for batch_id, batch in enumerate(payload.get("batches", [])):
        for pair in batch.get("pairs", []):
            rows.append(
                {
                    "episode_index": index,
                    "batch": batch_id,
                    "dataset_episode": pair.get("episode"),
                    "start_step": pair.get("start_step"),
                    "goal_step": pair.get("goal_step"),
                    "success": bool(successes[index]) if index < len(successes) else None,
                }
            )
            index += 1
    return rows


def _finalize(
    *,
    cfg: Any,
    output_dir: Path,
    resolved: list[tuple[Any, Any]],
    manifest_path: Path,
    manifest_info: dict[str, Any],
) -> None:
    from omegaconf import OmegaConf

    from mwm.benchmark.html import write_review_html
    from mwm.benchmark.pareto import write_pareto_html
    from mwm.benchmark.plots import write_default_plots
    from mwm.benchmark.summary import write_per_env_table, write_summary_csv
    from mwm.io import load_json, write_json, write_metrics_jsonl

    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    missing: list[str] = []
    for fallback_index, (run, run_cfg) in enumerate(resolved):
        run_dir = _run_dir(output_dir, run, fallback_index)
        _configure_run_paths(run_cfg, run_dir, manifest_path)
        completed = _completed_run(run_dir, run_cfg, manifest_path, materialize=True)
        if completed is None:
            missing.append(str(run.get("cell_id", run.get("name", run_dir.name))))
            continue
        row, payload = completed
        rows.append(row)
        outputs.append(payload)
    if missing:
        preview = ", ".join(missing[:10])
        extra = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(f"Cannot finalize benchmark; {len(missing)} cells are missing or stale: {preview}{extra}")

    summary = {
        "title": str(cfg.title),
        "output_dir": str(output_dir),
        "runs": rows,
        "sweep": OmegaConf.to_container(cfg.get("sweep", {}), resolve=True),
        "pareto_cost": "dynamics_flops_total",
        "manifest": {
            "group": str(manifest_info["group"]),
            "path": str(manifest_path),
            "seed": int(cfg.seed),
        },
    }
    missing_hashes = [row["name"] for row in rows if not row.get("manifest_sha256")]
    if missing_hashes:
        raise ValueError(f"Benchmark runs missing immutable manifest hashes: {missing_hashes}")
    manifest_hashes = {str(row.get("manifest_sha256", "")) for row in rows}
    if len(manifest_hashes) > 1:
        raise ValueError(f"Benchmark runs must share one manifest, got hashes: {sorted(manifest_hashes)}")
    write_summary_csv(output_dir / "summary.csv", rows)
    write_metrics_jsonl(output_dir / "metrics.jsonl", rows)
    summary["per_env_table"] = write_per_env_table(output_dir / "per_env_summary.csv", rows)
    plots = write_default_plots(output_dir / "plots", rows)
    pareto = write_pareto_html(output_dir / "plots" / "pareto.html", rows)
    summary["plots"] = plots
    summary["pareto_html"] = pareto
    write_json(output_dir / "summary.json", summary)
    write_review_html(
        output_dir / "review.html",
        str(cfg.title),
        rows,
        outputs,
        plots=plots,
        expected_cells=len(resolved),
        pareto_html=pareto,
    )
    print(f"[benchmark] wrote {output_dir / 'summary.json'}")
    print(f"[benchmark] wrote {output_dir / 'review.html'}")
    print("[benchmark] inspect rollout videos with: " f"python -m mwm.benchmark.render_review {output_dir} --serve")


def main(
    cfg_path: str,
    *,
    roles: Any = None,
    overrides: list[str] | None = None,
    resume: bool = False,
    shard_index: int | None = None,
    num_shards: int | None = None,
    finalize_only: bool = False,
) -> None:
    from omegaconf import OmegaConf

    from mwm.benchmark.config import (
        DEFAULTS,
        cell_id,
        filter_resolved_by_roles,
        load_manifest_config,
        manifest_path as benchmark_manifest_path,
        merged_run_config,
        require_no_legacy_fields,
        role,
        safe_name,
        validate_benchmark_matrix,
        write_temp_config,
    )
    from mwm.benchmark.io import write_run_sidecars
    from mwm.benchmark.summary import eval_summary_row
    from mwm.benchmark.sweep import expand_benchmark_runs
    from mwm.config_cli import load_config
    from mwm.io import file_sha256, load_json, write_json, write_metrics_jsonl

    cfg = load_config(DEFAULTS, cfg_path, overrides or [])
    require_no_legacy_fields(cfg)
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = benchmark_manifest_path(cfg)
    manifest_info = load_manifest_config(cfg)

    runs = expand_benchmark_runs(cfg)
    if not runs:
        raise ValueError("Benchmark config must include at least one run.")
    for idx, run in enumerate(runs):
        if "matrix_index" not in run:
            run.matrix_index = idx
    resolved = []
    for run in runs:
        _, run_cfg = merged_run_config(cfg, run)
        resolved.append((run, run_cfg))
    resolved = filter_resolved_by_roles(cfg, resolved, roles)
    validate_benchmark_matrix(cfg, resolved)
    if (shard_index is None) != (num_shards is None):
        raise ValueError("--shard-index and --num-shards must be provided together.")
    if num_shards is not None and (num_shards <= 0 or shard_index is None or not 0 <= shard_index < num_shards):
        raise ValueError("Shard index must satisfy 0 <= shard-index < num-shards.")
    if finalize_only:
        if shard_index is not None:
            raise ValueError("--finalize-only cannot be combined with shard selection.")
        _finalize(cfg=cfg, output_dir=output_dir, resolved=resolved, manifest_path=manifest_path, manifest_info=manifest_info)
        return

    sharded = shard_index is not None
    if sharded and not manifest_path.is_file():
        raise ValueError(
            f"Sharded execution requires the immutable manifest to exist first: {manifest_path}. "
            "Generate it once before submitting shards."
        )
    selected = [
        item
        for item in resolved
        if not sharded or int(item[0].matrix_index) % int(num_shards) == int(shard_index)
    ]
    if not resume:
        if not sharded:
            _clear_aggregate_outputs(output_dir)
        for fallback_index, (run, _) in enumerate(selected):
            shutil.rmtree(_run_dir(output_dir, run, fallback_index), ignore_errors=True)

    completed_count = 0
    for selected_index, (run, run_cfg) in enumerate(selected):
        name = safe_name(str(run.get("name", run.get("role", "run"))))
        run_dir = _run_dir(output_dir, run, selected_index)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _configure_run_paths(run_cfg, run_dir, manifest_path)
        resolved_config_path = run_dir / "resolved_config.yaml"
        if resume:
            completed = _completed_run(run_dir, run_cfg, manifest_path)
            if completed is not None:
                completed_count += 1
                print(f"[benchmark] resume skip {name}")
                continue
            shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path.write_text(OmegaConf.to_yaml(run_cfg), encoding="utf-8")
        temp_cfg = write_temp_config(run_cfg)

        print(f"[benchmark] {selected_index + 1}/{len(selected)} {name}")
        buffer = io.StringIO()
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                run_eval_mwm(temp_cfg)
        except Exception as exc:
            buffer.write("\n")
            buffer.write(traceback.format_exc())
            raise RuntimeError(f"Benchmark run {name!r} failed; see {run_dir / 'run.log'}") from exc
        finally:
            Path(temp_cfg).unlink(missing_ok=True)
            (run_dir / "run.log").write_text(buffer.getvalue(), encoding="utf-8")
        wall_time = time.perf_counter() - start

        payload = load_json(run_cfg.eval.output_path)
        payload["benchmark_name"] = name
        payload["role"] = role(run, run_cfg)
        payload["cell_id"] = cell_id(run, run_cfg)
        payload["base_name"] = str(run.get("base_name", run.get("name", name)))
        payload["strategy"] = str(run.get("strategy", role(run, run_cfg)))
        payload["sweep_key"] = str(run.get("sweep_key", "{}"))
        raw_sweep_params = run.get("sweep_params", {})
        payload["sweep_params"] = (
            OmegaConf.to_container(raw_sweep_params, resolve=True)
            if OmegaConf.is_config(raw_sweep_params)
            else dict(raw_sweep_params)
        )
        pop_size = int(run_cfg.planner.get("pop_size", 0))
        elite_frac = float(run_cfg.planner.get("elite_frac", 0.0))
        n_iter = int(run_cfg.planner.get("n_iter", 0))
        raw_topk = run_cfg.planner.get("topk", None)
        effective_topk = max(1, int(raw_topk)) if raw_topk is not None else max(
            1, int(round(pop_size * elite_frac))
        )
        payload["planner_params"] = {
            "pop_size": pop_size,
            "elite_frac": elite_frac,
            "topk": effective_topk,
            "n_iter": n_iter,
        }
        payload["schedule"] = str(run.get("schedule", OmegaConf.to_container(run_cfg.planner.scheduler, resolve=True)))
        payload["seed"] = int(run_cfg.eval.seed)
        payload["wall_time_sec"] = float(wall_time)
        payload.setdefault("config", {})
        payload["config"]["resolved_path"] = str(resolved_config_path)
        payload["config"]["sha256"] = file_sha256(resolved_config_path)
        write_json(run_cfg.eval.output_path, payload)
        row = eval_summary_row(name, run_cfg.eval.output_path, payload)
        write_metrics_jsonl(run_dir / "metrics.jsonl", [row])
        write_run_sidecars(run_dir, row, payload)
        trace_path = run_dir / "episode_traces.jsonl"
        write_metrics_jsonl(trace_path, _episode_trace_rows(payload))
        completed_count += 1

    if sharded:
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            shard_dir / f"shard_{int(shard_index):03d}_of_{int(num_shards):03d}.json",
            {"shard_index": int(shard_index), "num_shards": int(num_shards), "cells": len(selected), "completed": completed_count},
        )
        print(f"[benchmark] shard {shard_index}/{num_shards} complete; run --finalize-only after all shards finish")
        return
    _finalize(cfg=cfg, output_dir=output_dir, resolved=resolved, manifest_path=manifest_path, manifest_info=manifest_info)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an MWM benchmark matrix.")
    parser.add_argument("config", help="Benchmark YAML config")
    parser.add_argument("--roles", nargs="+", help="Optional role filter, e.g. upstream_lewm_converted")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. seed=1")
    parser.add_argument("--resume", action="store_true", help="Skip cells with complete sidecars and an identical resolved config.")
    parser.add_argument("--shard-index", type=int, default=None, help="Zero-based deterministic shard index.")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of deterministic shards.")
    parser.add_argument("--finalize-only", action="store_true", help="Build aggregate outputs from completed cells without evaluating.")
    args = parser.parse_args()
    main(
        args.config,
        roles=args.roles,
        overrides=args.set,
        resume=args.resume,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        finalize_only=args.finalize_only,
    )
