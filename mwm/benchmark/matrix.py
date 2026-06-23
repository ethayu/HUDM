from __future__ import annotations

import contextlib
import io
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.config import (
    DEFAULTS,
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
from mwm.benchmark.html import write_review_html
from mwm.benchmark.io import write_run_sidecars
from mwm.benchmark.plots import write_default_plots
from mwm.benchmark.summary import eval_summary_row, write_per_env_table, write_summary_csv
from mwm.config_cli import load_config
from mwm.eval.runner import main as run_eval_mwm
from mwm.io import file_sha256, load_json, write_json, write_metrics_jsonl


AGGREGATE_OUTPUTS = ("summary.json", "summary.csv", "metrics.jsonl", "per_env_summary.csv", "review.html")


def _clear_stale_outputs(output_dir: Path, resolved: list[tuple[Any, Any]]) -> None:
    for name in AGGREGATE_OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)
    shutil.rmtree(output_dir / "plots", ignore_errors=True)
    for idx, (run, _) in enumerate(resolved):
        name = safe_name(str(run.get("name", run.get("role", "run"))))
        shutil.rmtree(output_dir / f"{idx:03d}_{name}", ignore_errors=True)


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


def main(cfg_path: str, *, roles: Any = None, overrides: list[str] | None = None) -> None:
    cfg = load_config(DEFAULTS, cfg_path, overrides or [])
    require_no_legacy_fields(cfg)
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    manifest_path = benchmark_manifest_path(cfg)
    manifest_info = load_manifest_config(cfg)

    runs = list(cfg.runs)
    if not runs:
        raise ValueError("Benchmark config must include at least one run.")
    resolved = []
    for run in runs:
        _, run_cfg = merged_run_config(cfg, run)
        resolved.append((run, run_cfg))
    resolved = filter_resolved_by_roles(cfg, resolved, roles)
    validate_benchmark_matrix(cfg, resolved)
    _clear_stale_outputs(output_dir, resolved)

    for idx, (run, run_cfg) in enumerate(resolved):
        name = safe_name(str(run.get("name", run.get("role", "run"))))
        run_dir = output_dir / f"{idx:03d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_cfg.eval.output_path = str(run_dir / "eval.json")
        if manifest_path.is_file():
            run_cfg.eval.manifest_path = str(manifest_path)
            run_cfg.eval.write_manifest_path = None
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            run_cfg.eval.manifest_path = None
            run_cfg.eval.write_manifest_path = str(manifest_path)
        run_cfg.eval.video_path = str(run_dir / "videos")
        if "save_video" not in run_cfg.eval:
            run_cfg.eval.save_video = True
        resolved_config_path = run_dir / "resolved_config.yaml"
        resolved_config_path.write_text(OmegaConf.to_yaml(run_cfg), encoding="utf-8")
        temp_cfg = write_temp_config(run_cfg)

        print(f"[benchmark] {idx + 1}/{len(resolved)} {name}")
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
        payload["schedule"] = str(run.get("schedule", OmegaConf.to_container(run_cfg.planner.scheduler, resolve=True)))
        payload["seed"] = int(run_cfg.eval.seed)
        payload["wall_time_sec"] = float(wall_time)
        payload.setdefault("config", {})
        payload["config"]["resolved_path"] = str(resolved_config_path)
        payload["config"]["sha256"] = file_sha256(resolved_config_path)
        write_json(run_cfg.eval.output_path, payload)
        row = eval_summary_row(name, run_cfg.eval.output_path, payload)
        rows.append(row)
        outputs.append(payload)
        write_metrics_jsonl(run_dir / "metrics.jsonl", [row])
        write_run_sidecars(run_dir, row, payload)
        trace_path = run_dir / "episode_traces.jsonl"
        write_metrics_jsonl(trace_path, _episode_trace_rows(payload))

    summary = {
        "title": str(cfg.title),
        "output_dir": str(output_dir),
        "runs": rows,
        "manifest": {
            "group": str(manifest_info["group"]),
            "path": str(manifest_path),
            "seed": int(cfg.seed),
        },
    }
    missing = [row["name"] for row in rows if not row.get("manifest_sha256")]
    if missing:
        raise ValueError(f"Benchmark runs missing immutable manifest hashes: {missing}")
    manifest_hashes = {str(row.get("manifest_sha256", "")) for row in rows}
    if len(manifest_hashes) > 1:
        raise ValueError(f"Benchmark runs must share one manifest, got hashes: {sorted(manifest_hashes)}")
    write_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", rows)
    write_metrics_jsonl(output_dir / "metrics.jsonl", rows)
    summary["per_env_table"] = write_per_env_table(output_dir / "per_env_summary.csv", rows)
    plots = write_default_plots(output_dir / "plots", rows)
    summary["plots"] = plots
    write_json(output_dir / "summary.json", summary)
    write_review_html(output_dir / "review.html", str(cfg.title), rows, outputs, plots=plots, expected_cells=len(resolved))
    print(f"[benchmark] wrote {output_dir / 'summary.json'}")
    print(f"[benchmark] wrote {output_dir / 'review.html'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an MWM benchmark matrix.")
    parser.add_argument("config", help="Benchmark YAML config")
    parser.add_argument("--roles", nargs="+", help="Optional role filter, e.g. upstream_lewm_converted")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. seed=1")
    args = parser.parse_args()
    main(args.config, roles=args.roles, overrides=args.set)
