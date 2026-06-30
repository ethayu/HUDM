from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mwm.io import jsonable


def eval_summary_row(name: str, output_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    swm_results = dict(payload.get("swm_results", {}))
    diagnostics = dict(payload.get("planning_diagnostics", {}))
    manifest = dict(payload.get("manifest", {}))
    config = dict(payload.get("config", {}))
    level_counts = diagnostics.get("schedule_level_counts", {})
    return {
        "name": str(name),
        "env_id": str(payload.get("env_id", "")),
        "checkpoint_epoch": payload.get("checkpoint_epoch", ""),
        "checkpoint_run_dir": str(payload.get("checkpoint_run_dir", "")),
        "config_sha256": str(config.get("sha256", "")),
        "manifest_sha256": str(manifest.get("manifest_sha256", "")),
        "manifest_file_sha256": str(manifest.get("sha256", "")),
        "episodes": int(payload.get("episodes", 0)),
        "goal_offset": int(payload.get("goal_offset", 0)),
        "success_rate": float(swm_results.get("success_rate", float("nan"))),
        "plans": int(diagnostics.get("plans", 0)),
        "steps": int(diagnostics.get("steps", 0)),
        "bits_used_total": int(diagnostics.get("bits_used_total", 0)),
        "dynamics_flops_total": int(diagnostics.get("dynamics_flops_total", 0)),
        "plan_time_total_sec": float(diagnostics.get("plan_time_total_sec", 0.0)),
        "wall_time_sec": float(payload.get("wall_time_sec", 0.0)),
        "schedule_level_counts": json.dumps(jsonable(level_counts), sort_keys=True),
        "schedule": str(payload.get("schedule", "")),
        "role": str(payload.get("role", "")),
        "seed": int(payload.get("seed", payload.get("eval_seed", 0))),
        "output_json": str(output_path),
    }


def write_summary_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "env_id",
        "checkpoint_epoch",
        "checkpoint_run_dir",
        "config_sha256",
        "manifest_sha256",
        "manifest_file_sha256",
        "episodes",
        "goal_offset",
        "success_rate",
        "plans",
        "steps",
        "bits_used_total",
        "dynamics_flops_total",
        "plan_time_total_sec",
        "wall_time_sec",
        "schedule_level_counts",
        "schedule",
        "role",
        "seed",
        "output_json",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_per_env_table(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("env_id", "")), str(row.get("role", ""))), []).append(row)
    out_rows = []
    for (env_id, role), group in sorted(grouped.items()):
        rates = [float(r.get("success_rate", float("nan"))) for r in group]
        out_rows.append(
            {
                "env_id": env_id,
                "role": role,
                "runs": len(group),
                "mean_success_rate": float(np.nanmean(rates)) if rates else float("nan"),
                "seeds": ",".join(str(r.get("seed", "")) for r in sorted(group, key=lambda x: int(x.get("seed", 0)))),
            }
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["env_id", "role", "runs", "mean_success_rate", "seeds"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    return str(out)


__all__ = ["eval_summary_row", "write_per_env_table", "write_summary_csv"]
