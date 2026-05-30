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
from eval_mwm import main as run_eval_mwm


DEFAULTS = {
    "output_dir": "rollouts/mwm_benchmark",
    "title": "MWM Benchmark",
    "env_id": None,
    "seed": 0,
    "eval_config": None,
    "manifest": {},
    "runs": [],
}

def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return text.strip("_") or "run"


def _require_no_legacy_fields(cfg: Any) -> None:
    if "gate" in cfg:
        raise ValueError("legacy benchmark field `gate` is no longer supported; derive roles from runs.")


def _require_no_legacy_run_fields(run: Any) -> None:
    for field in ("manifest_group", "overrides", "config"):
        if field in run:
            raise ValueError(f"legacy benchmark field runs[].{field} is no longer supported.")


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping, got {type(value).__name__}.")
    return dict(value)


def _load_manifest_config(cfg: Any) -> dict[str, Any]:
    raw = _as_plain_dict(cfg.get("manifest", {}))
    if "config" in raw:
        manifest_cfg = OmegaConf.load(str(raw.pop("config")))
        merged = {**_as_plain_dict(manifest_cfg), **raw}
    else:
        merged = raw
    group = str(merged.get("group", "")).strip()
    if not group:
        raise ValueError("Benchmark config must define manifest.group or manifest.config with group.")
    if "path" not in merged:
        manifest_dir = Path(str(merged.get("dir", "rollouts/manifests")))
        merged["path"] = str(manifest_dir / f"{_safe_name(group)}_seed{int(cfg.seed)}.json")
    merged["group"] = group
    merged["path"] = str(merged["path"])
    return merged


def _checkpoint_mapping(run: Any) -> dict[str, Any]:
    if "checkpoint" not in run:
        raise ValueError("Each benchmark run must define runs[].checkpoint.")
    raw = run.checkpoint
    if isinstance(raw, str):
        return {"run_dir": raw, "epoch": None}
    checkpoint = _as_plain_dict(raw)
    if "run_dir" not in checkpoint:
        raise ValueError("runs[].checkpoint must be a path string or mapping with run_dir.")
    checkpoint.setdefault("epoch", None)
    return checkpoint


def _benchmark_eval_template(cfg: Any) -> Any:
    if not cfg.get("eval_config", None):
        raise ValueError("Benchmark config must define top-level eval_config.")
    if not cfg.get("env_id", None):
        raise ValueError("Benchmark config must define top-level env_id.")
    template = OmegaConf.load(str(cfg.eval_config))
    template_env = str(template.get("env_id", cfg.env_id))
    if template_env != str(cfg.env_id):
        raise ValueError(f"Benchmark env_id={cfg.env_id!r} does not match eval_config env_id={template_env!r}.")
    template.env_id = str(cfg.env_id)
    template.eval.seed = int(cfg.seed)
    return template


def _merged_run_config(cfg: Any, run: Any) -> tuple[str, Any]:
    _require_no_legacy_run_fields(run)
    name = _safe_name(str(run.get("name", run.get("role", "run"))))
    run_cfg = OmegaConf.create(OmegaConf.to_container(_benchmark_eval_template(cfg), resolve=True))
    run_cfg.checkpoint = _checkpoint_mapping(run)
    planner = run.get("planner", None)
    if planner is not None:
        run_cfg.planner = OmegaConf.merge(run_cfg.get("planner", {}), planner)
    run_eval = run.get("eval", None)
    if run_eval is not None:
        run_cfg.eval = OmegaConf.merge(run_cfg.get("eval", {}), run_eval)
        run_cfg.eval.seed = int(cfg.seed)
    return name, run_cfg


def _write_temp_config(cfg: Any) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="mwm_benchmark_", delete=False)
    tmp.write(OmegaConf.to_yaml(cfg))
    tmp.close()
    return tmp.name


def _manifest_path(cfg: Any) -> Path:
    return Path(str(_load_manifest_config(cfg)["path"]))


def _role(run: Any, cfg: Any) -> str:
    del cfg
    if run.get("role", None):
        return str(run.role)
    return str(run.get("name", "run"))


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
    del cfg
    selected = _normalize_role_filter(roles)
    if not selected:
        return resolved
    filtered = [(run, run_cfg) for run, run_cfg in resolved if _role(run, run_cfg) in selected]
    if not filtered:
        raise ValueError(f"No benchmark runs matched requested roles: {sorted(selected)}")
    return filtered


def _validate_benchmark_matrix(cfg: Any, resolved: list[tuple[Any, Any]]) -> None:
    _require_no_legacy_fields(cfg)
    _load_manifest_config(cfg)
    identities = [(str(run_cfg.get("env_id", "")), int(run_cfg.eval.seed), _role(run, run_cfg)) for run, run_cfg in resolved]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ValueError(f"Benchmark has duplicate cells: {duplicates}")
    env_ids = {identity[0] for identity in identities}
    seeds = {identity[1] for identity in identities}
    if env_ids != {str(cfg.env_id)}:
        raise ValueError(f"Benchmark runs must all use env_id={cfg.env_id!r}, got {sorted(env_ids)}.")
    if seeds != {int(cfg.seed)}:
        raise ValueError(f"Benchmark runs must all use seed={int(cfg.seed)}, got {sorted(seeds)}.")


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
    _require_no_legacy_fields(cfg)
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    manifest_path = _manifest_path(cfg)
    manifest_info = _load_manifest_config(cfg)

    runs = list(cfg.runs)
    if not runs:
        raise ValueError("Benchmark config must include at least one run.")
    resolved = []
    for run in runs:
        _, run_cfg = _merged_run_config(cfg, run)
        resolved.append((run, run_cfg))
    resolved = _filter_resolved_by_roles(cfg, resolved, roles)
    _validate_benchmark_matrix(cfg, resolved)

    for idx, (run, run_cfg) in enumerate(resolved):
        name = _safe_name(str(run.get("name", run.get("role", "run"))))
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
        temp_cfg = _write_temp_config(run_cfg)

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
    args = parser.parse_args()
    main(args.config, roles=args.roles)
