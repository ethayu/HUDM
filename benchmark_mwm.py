from __future__ import annotations

import contextlib
import io
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.artifacts import (
    eval_summary_row,
    file_sha256,
    load_json,
    write_default_plots,
    write_json,
    write_metrics_jsonl,
    write_per_env_table,
    write_run_sidecars,
    write_review_html,
    write_summary_csv,
)
from mwm.eval.reference import REFERENCE_ROLE
from eval_mwm import main as run_eval_mwm


DEFAULTS = {
    "output_dir": "rollouts/mwm_benchmark",
    "title": "MWM Benchmark",
    "require_shared_manifests": True,
    "gate": {
        "enabled": True,
        "env_ids": ["swm/PushT-v1", "swm/TwoRoom-v1"],
        "seeds": [0, 1, 2],
        "roles": ["upstream_lewm_converted", "retrained_lewm_single", "mwm_scheduled"],
    },
    "runs": [],
}


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return text.strip("_") or "run"


def _merged_run_config(run: Any) -> tuple[str, Any]:
    if "config" not in run:
        raise ValueError("Each benchmark run must define a eval_mwm config path under `config`.")
    name = _safe_name(str(run.get("name", Path(str(run.config)).stem)))
    cfg = OmegaConf.load(str(run.config))
    overrides = run.get("overrides", {})
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    return name, cfg


def _write_temp_config(cfg: Any) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="mwm_benchmark_", delete=False)
    tmp.write(OmegaConf.to_yaml(cfg))
    tmp.close()
    return tmp.name


def _run_key(run: Any, cfg: Any) -> tuple[str, str, int]:
    env_id = str(cfg.get("env_id", ""))
    key = str(run.get("manifest_group", env_id))
    seed = int(cfg.eval.seed)
    return key, env_id, seed


def _manifest_path(output_dir: Path, group: str, seed: int) -> Path:
    return output_dir / "manifests" / f"{_safe_name(group)}_seed{int(seed)}.json"


def _role(run: Any, cfg: Any) -> str:
    del cfg
    if run.get("role", None):
        return str(run.role)
    return "mwm_scheduled"


def _normalize_role_filter(roles: Any = None) -> set[str]:
    if roles is None:
        return set()
    if isinstance(roles, str):
        raw_items = [roles]
    else:
        raw_items = list(roles)
    selected: set[str] = set()
    for item in raw_items:
        for part in str(item).split(","):
            role = part.strip()
            if role:
                selected.add(role)
    return selected


def _filter_resolved_by_roles(cfg: Any, resolved: list[tuple[Any, Any]], roles: Any = None) -> list[tuple[Any, Any]]:
    selected = _normalize_role_filter(roles)
    if not selected:
        return resolved
    filtered = [(run, run_cfg) for run, run_cfg in resolved if _role(run, run_cfg) in selected]
    if not filtered:
        raise ValueError(f"No benchmark runs matched requested roles: {sorted(selected)}")
    if bool(cfg.get("gate", {}).get("enabled", False)):
        cfg.gate.roles = sorted({_role(run, run_cfg) for run, run_cfg in filtered})
    return filtered


def _validate_gate_matrix(cfg: Any, resolved: list[tuple[Any, Any]]) -> None:
    gate = cfg.get("gate", {})
    if not bool(gate.get("enabled", False)):
        return
    expected = {
        (str(env), int(seed), str(role))
        for env in gate.get("env_ids", [])
        for seed in gate.get("seeds", [])
        for role in gate.get("roles", [])
    }
    identities = [(str(run_cfg.get("env_id", "")), int(run_cfg.eval.seed), _role(run, run_cfg)) for run, run_cfg in resolved]
    present = set(identities)
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ValueError(f"Benchmark gate has duplicate cells: {duplicates}")
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"Benchmark gate is missing required cells: {missing}")


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


def main(cfg_path: str, *, roles: Any = None) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    manifest_groups: dict[tuple[str, int], Path] = {}

    runs = list(cfg.runs)
    if not runs:
        raise ValueError("Benchmark config must include at least one run.")
    resolved = []
    for run in runs:
        _, run_cfg = _merged_run_config(run)
        resolved.append((run, run_cfg))
    resolved = _filter_resolved_by_roles(cfg, resolved, roles)
    _validate_gate_matrix(cfg, resolved)

    for idx, (run, run_cfg) in enumerate(resolved):
        name = _safe_name(str(run.get("name", Path(str(run.config)).stem)))
        run_dir = output_dir / f"{idx:03d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_cfg.eval.output_path = str(run_dir / "eval.json")
        if _role(run, run_cfg) == REFERENCE_ROLE:
            run_cfg.eval.reference_policy = True
        group, env_id, seed = _run_key(run, run_cfg)
        manifest_path = _manifest_path(output_dir, group or env_id, seed)
        manifest_key = (group or env_id, seed)
        if manifest_key in manifest_groups:
            run_cfg.eval.manifest_path = str(manifest_groups[manifest_key])
            run_cfg.eval.write_manifest_path = None
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            run_cfg.eval.manifest_path = None
            run_cfg.eval.write_manifest_path = str(manifest_path)
            manifest_groups[manifest_key] = manifest_path
        run_cfg.eval.video_path = str(run_dir / "videos")
        if "save_video" not in run_cfg.eval:
            run_cfg.eval.save_video = True
        resolved_config_path = run_dir / "resolved_config.yaml"
        resolved_config_path.write_text(OmegaConf.to_yaml(run_cfg), encoding="utf-8")
        temp_cfg = _write_temp_config(run_cfg)

        print(f"[benchmark] {idx + 1}/{len(runs)} {name}")
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
        payload["role"] = _role(run, run_cfg)
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
        "manifest_groups": {f"{k[0]}:seed{k[1]}": str(v) for k, v in manifest_groups.items()},
    }
    if bool(cfg.require_shared_manifests):
        missing = [row["name"] for row in rows if not row.get("manifest_sha256")]
        if missing:
            raise ValueError(f"Benchmark runs missing immutable manifest hashes: {missing}")
    write_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", rows)
    write_metrics_jsonl(output_dir / "metrics.jsonl", rows)
    summary["per_env_table"] = write_per_env_table(output_dir / "per_env_summary.csv", rows)
    plots = write_default_plots(output_dir / "plots", rows)
    summary["plots"] = plots
    write_json(output_dir / "summary.json", summary)
    expected_cells = None
    if bool(cfg.get("gate", {}).get("enabled", False)):
        expected_cells = len(cfg.gate.env_ids) * len(cfg.gate.seeds) * len(cfg.gate.roles)
    write_review_html(output_dir / "review.html", str(cfg.title), rows, outputs, plots=plots, expected_cells=expected_cells)
    print(f"[benchmark] wrote {output_dir / 'summary.json'}")
    print(f"[benchmark] wrote {output_dir / 'review.html'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an MWM benchmark matrix.")
    parser.add_argument("config", help="Benchmark YAML config")
    parser.add_argument("--roles", nargs="+", help="Optional role filter, e.g. upstream_lewm_converted")
    args = parser.parse_args()
    main(args.config, roles=args.roles)
