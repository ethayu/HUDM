from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Sequence

from hudm.config import resolve_benchmark_spec
from hudm.experiment import run_experiment
from hudm.specs import BenchmarkSpec


def load_benchmark_spec(cfg_path: str) -> BenchmarkSpec:
    return resolve_benchmark_spec(cfg_path)


def _write_rows_csv(path: str, rows: Sequence[dict]) -> None:
    if len(rows) <= 0:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_benchmark(spec_or_path: str | BenchmarkSpec) -> str:
    spec = load_benchmark_spec(spec_or_path) if isinstance(spec_or_path, str) else spec_or_path
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(str(spec.output_root), f"benchmark_{spec.name}_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    rows: list[dict] = []
    for entry in spec.entries:
        experiment_dir = run_experiment(entry.experiment_config, output_root=run_dir)
        summary_path = os.path.join(experiment_dir, "summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        rows.append(
            {
                "benchmark_name": spec.name,
                "experiment_name": entry.name,
                "experiment_dir": experiment_dir,
                "baseline_variant": summary.get("baseline_variant"),
                "num_rollouts": summary.get("num_rollouts"),
                "num_variants": len(summary.get("summary", [])),
            }
        )

    with open(os.path.join(run_dir, "benchmark_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": spec.name,
                "experiments": [
                    {"name": entry.name, "config": entry.experiment_config}
                    for entry in spec.entries
                ],
            },
            f,
            indent=2,
        )
    _write_rows_csv(os.path.join(run_dir, "experiments.csv"), rows)
    print(f"[benchmark] wrote results to {run_dir}")
    return run_dir
