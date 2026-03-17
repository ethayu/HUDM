from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Sequence


EXPERIMENT_SCHEMA_VERSION = 1
REVIEWER_SCHEMA_VERSION = 1

EXPERIMENT_JSON = "experiment.json"
RUNS_CSV = "runs.csv"
VARIANTS_CSV = "variants.csv"
PAIRED_VS_BASELINE_CSV = "paired_vs_baseline.csv"
SELECTED_ROLLOUTS_JSON = "selected_rollouts.json"
TRACES_DIRNAME = "traces"
REVIEW_CACHE_DIRNAME = "review_cache"
REVIEW_DERIVED_DIRNAME = "derived"
REVIEW_MEDIA_DIRNAME = "media"


@dataclass(frozen=True)
class ExperimentBundlePaths:
    root: str
    experiment_json: str
    runs_csv: str
    variants_csv: str
    paired_vs_baseline_csv: str
    selected_rollouts_json: str
    traces_dir: str
    review_cache_dir: str


def bundle_paths(run_dir: str) -> ExperimentBundlePaths:
    root = os.path.abspath(run_dir)
    return ExperimentBundlePaths(
        root=root,
        experiment_json=os.path.join(root, EXPERIMENT_JSON),
        runs_csv=os.path.join(root, RUNS_CSV),
        variants_csv=os.path.join(root, VARIANTS_CSV),
        paired_vs_baseline_csv=os.path.join(root, PAIRED_VS_BASELINE_CSV),
        selected_rollouts_json=os.path.join(root, SELECTED_ROLLOUTS_JSON),
        traces_dir=os.path.join(root, TRACES_DIRNAME),
        review_cache_dir=os.path.join(root, REVIEW_CACHE_DIRNAME),
    )


def trace_dir(run_dir: str, variant_name: str, rollout_id: str) -> str:
    return os.path.join(os.path.abspath(run_dir), TRACES_DIRNAME, str(variant_name), str(rollout_id))


def review_cache_dir(run_dir: str) -> str:
    return os.path.join(os.path.abspath(run_dir), REVIEW_CACHE_DIRNAME)


def review_derived_dir(run_dir: str) -> str:
    return os.path.join(review_cache_dir(run_dir), REVIEW_DERIVED_DIRNAME)


def review_media_dir(run_dir: str, variant_name: str, rollout_id: str) -> str:
    return os.path.join(
        review_cache_dir(run_dir),
        REVIEW_MEDIA_DIRNAME,
        str(variant_name),
        str(rollout_id),
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_rows_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
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


def write_experiment_bundle(
    run_dir: str,
    *,
    experiment_payload: dict[str, Any],
    selected_rollouts: Sequence[dict[str, Any]],
    run_rows: Sequence[dict[str, Any]],
    variant_rows: Sequence[dict[str, Any]],
    paired_rows: Sequence[dict[str, Any]],
) -> ExperimentBundlePaths:
    paths = bundle_paths(run_dir)
    ensure_dir(paths.root)
    ensure_dir(paths.traces_dir)
    with open(paths.experiment_json, "w", encoding="utf-8") as f:
        json.dump(experiment_payload, f, indent=2)
    with open(paths.selected_rollouts_json, "w", encoding="utf-8") as f:
        json.dump(list(selected_rollouts), f, indent=2)
    write_rows_csv(paths.runs_csv, run_rows)
    write_rows_csv(paths.variants_csv, variant_rows)
    write_rows_csv(paths.paired_vs_baseline_csv, paired_rows)
    return paths


def _read_csv_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def migrate_legacy_experiment_dir(run_dir: str, *, overwrite: bool = False) -> ExperimentBundlePaths:
    paths = bundle_paths(run_dir)
    if os.path.isfile(paths.experiment_json) and not overwrite:
        return paths

    legacy_resolved = os.path.join(paths.root, "experiment_resolved.json")
    legacy_summary = os.path.join(paths.root, "summary.json")
    legacy_runs = os.path.join(paths.root, "per_rollout.csv")
    legacy_variants = os.path.join(paths.root, "summary.csv")
    legacy_selected = os.path.join(paths.root, SELECTED_ROLLOUTS_JSON)
    legacy_paired = os.path.join(paths.root, "paired_deltas_vs_baseline.csv")

    if not os.path.isfile(legacy_resolved):
        raise FileNotFoundError(f"Legacy experiment_resolved.json not found under {paths.root}")
    if not os.path.isfile(legacy_runs):
        raise FileNotFoundError(f"Legacy per_rollout.csv not found under {paths.root}")
    if not os.path.isfile(legacy_variants):
        raise FileNotFoundError(f"Legacy summary.csv not found under {paths.root}")
    if not os.path.isfile(legacy_selected):
        raise FileNotFoundError(f"Legacy selected_rollouts.json not found under {paths.root}")

    with open(legacy_resolved, "r", encoding="utf-8") as f:
        resolved = json.load(f)
    summary_payload: dict[str, Any] = {}
    if os.path.isfile(legacy_summary):
        with open(legacy_summary, "r", encoding="utf-8") as f:
            summary_payload = json.load(f)
    with open(legacy_selected, "r", encoding="utf-8") as f:
        selected_rollouts = json.load(f)

    variant_entries = list(resolved.get("variants", []))
    variant_order = [str(item.get("name", "")) for item in variant_entries]
    experiment_payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "reviewer_version": REVIEWER_SCHEMA_VERSION,
        "experiment_name": resolved.get("name", os.path.basename(paths.root)),
        "created_at": summary_payload.get("created_at", ""),
        "baseline_variant": summary_payload.get("baseline_variant", resolved.get("baseline", "")),
        "variant_order": variant_order,
        "num_rollouts": int(summary_payload.get("num_rollouts", len(selected_rollouts))),
        "rollouts": resolved.get("rollouts", {}),
        "execution": resolved.get("execution", {}),
        "terminal": resolved.get("terminal", {}),
        "reporting": resolved.get("reporting", {}),
        "shared_plan": resolved.get("shared_plan", {}),
        "variants": variant_entries,
    }
    run_rows = _read_csv_rows(legacy_runs)
    variant_rows = _read_csv_rows(legacy_variants)
    paired_rows = _read_csv_rows(legacy_paired)

    write_experiment_bundle(
        paths.root,
        experiment_payload=experiment_payload,
        selected_rollouts=selected_rollouts,
        run_rows=run_rows,
        variant_rows=variant_rows,
        paired_rows=paired_rows,
    )
    if len(paired_rows) <= 0 and not os.path.exists(paths.paired_vs_baseline_csv):
        with open(paths.paired_vs_baseline_csv, "w", encoding="utf-8") as f:
            f.write("")
    ensure_dir(paths.review_cache_dir)
    return paths
