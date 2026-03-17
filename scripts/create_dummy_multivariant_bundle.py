#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from copy import deepcopy
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hudm.experiment import aggregate_summary
from hudm.experiment_bundle import (
    RUNS_CSV,
    SELECTED_ROLLOUTS_JSON,
    TRACES_DIRNAME,
    VARIANTS_CSV,
    bundle_paths,
    trace_dir,
    write_experiment_bundle,
)


FAKE_VARIANTS = (
    "gt_env_finest_a",
    "gt_env_finest_b",
    "gt_env_finest_c",
    "gt_env_finest_d",
)


def _read_csv_rows(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _copy_trace_tree(source_root: str, output_root: str, source_variant: str, row: dict[str, Any]) -> None:
    rollout_id = str(row["rollout_id"])
    reassigned_variant = str(row["variant_name"])
    src_trace_dir = trace_dir(source_root, source_variant, rollout_id)
    dst_trace_dir = trace_dir(output_root, reassigned_variant, rollout_id)
    if not os.path.isdir(src_trace_dir):
        raise FileNotFoundError(f"Missing trace dir for rollout '{rollout_id}': {src_trace_dir}")
    os.makedirs(os.path.dirname(dst_trace_dir), exist_ok=True)
    shutil.copytree(src_trace_dir, dst_trace_dir, dirs_exist_ok=False)


def _prepare_rows(
    output_root: str,
    source_runs: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    if len(source_runs) <= 0:
        raise ValueError("Source runs.csv has no rows.")
    source_variant = str(source_runs[0]["variant_name"])
    reassigned_rows: list[dict[str, Any]] = []
    for src_row in source_runs:
        rollout_id = str(src_row["rollout_id"])
        for variant_name in FAKE_VARIANTS:
            row = dict(src_row)
            row["variant_name"] = variant_name
            rel_base = os.path.join(output_root, TRACES_DIRNAME, variant_name, rollout_id)
            row["run_dir"] = rel_base
            row["trace_json"] = os.path.join(rel_base, "trace.json")
            row["trace_npz"] = os.path.join(rel_base, "trace.npz")
            row["run_log"] = os.path.join(rel_base, "run.log")
            reassigned_rows.append(row)
    return source_variant, reassigned_rows


def build_dummy_bundle(source_root: str, output_root: str, *, overwrite: bool) -> None:
    source_paths = bundle_paths(source_root)
    output_paths = bundle_paths(output_root)
    if not os.path.isfile(source_paths.experiment_json):
        raise FileNotFoundError(f"Missing source experiment.json: {source_paths.experiment_json}")
    if not os.path.isfile(source_paths.runs_csv):
        raise FileNotFoundError(f"Missing source runs.csv: {source_paths.runs_csv}")
    if not os.path.isfile(source_paths.selected_rollouts_json):
        raise FileNotFoundError(f"Missing source selected_rollouts.json: {source_paths.selected_rollouts_json}")

    if os.path.exists(output_paths.root):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_paths.root}. Use --overwrite to replace it.")
        shutil.rmtree(output_paths.root)
    os.makedirs(output_paths.root, exist_ok=True)
    os.makedirs(output_paths.traces_dir, exist_ok=True)

    with open(source_paths.experiment_json, "r", encoding="utf-8") as f:
        source_payload = json.load(f)
    with open(source_paths.selected_rollouts_json, "r", encoding="utf-8") as f:
        selected_rollouts = json.load(f)
    source_runs = _read_csv_rows(source_paths.runs_csv)
    source_variant, reassigned_rows = _prepare_rows(output_root, source_runs)

    for row in reassigned_rows:
        _copy_trace_tree(source_root, output_root, source_variant, row)

    variant_rows, paired_rows = aggregate_summary(reassigned_rows, variant_order=list(FAKE_VARIANTS))

    payload = deepcopy(source_payload)
    payload["experiment_name"] = f"{source_payload.get('experiment_name', 'experiment')}_dummy_multivariant"
    payload.pop("baseline_variant", None)
    payload["variant_order"] = list(FAKE_VARIANTS)
    payload["num_rollouts"] = len(selected_rollouts)

    source_variant_specs = payload.get("variants", [])
    template_variant = deepcopy(source_variant_specs[0]) if source_variant_specs else {"name": source_variant}
    payload["variants"] = []
    for name in FAKE_VARIANTS:
        spec = deepcopy(template_variant)
        spec["name"] = name
        payload["variants"].append(spec)

    write_experiment_bundle(
        output_paths.root,
        experiment_payload=payload,
        selected_rollouts=selected_rollouts,
        run_rows=reassigned_rows,
        variant_rows=variant_rows,
        paired_rows=paired_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a dummy multi-variant experiment bundle from a single-variant source bundle.")
    parser.add_argument("--source", required=True, help="Path to source experiment bundle.")
    parser.add_argument("--output", required=True, help="Path to output experiment bundle.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dummy_bundle(
        source_root=os.path.abspath(args.source),
        output_root=os.path.abspath(args.output),
        overwrite=bool(args.overwrite),
    )
    print(f"Created dummy multi-variant bundle at: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
