from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.eval_compaction import EvalCompactionError, compact_completed_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_TARGETS = (
    (
        "configs/research/release20260728_dense_ogb_cube_all_fidelity_schedules.yaml",
        "reports/research/release20260728_dense_ogb_cube_all_fidelity_schedules",
    ),
    (
        "configs/research/release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules.yaml",
        "reports/research/release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules",
    ),
    (
        "configs/research/release20260728_dense_pusht_all_fidelity_schedules.yaml",
        "reports/research/release20260728_dense_pusht_all_fidelity_schedules",
    ),
    (
        "configs/research/release20260728_dense_pusht_goal50_plan50_execute20_all_fidelity_schedules.yaml",
        "reports/research/release20260728_dense_pusht_goal50_plan50_execute20_all_fidelity_schedules",
    ),
    (
        "configs/research/release20260728_dense_reacher_all_fidelity_schedules.yaml",
        "reports/research/release20260728_dense_reacher_all_fidelity_schedules",
    ),
    (
        "configs/research/release20260728_dense_reacher_goal50_plan50_execute20_all_fidelity_schedules.yaml",
        "reports/research/release20260728_dense_reacher_goal50_plan50_execute20_all_fidelity_schedules",
    ),
    (
        "configs/research/release20260728_dense_tworoom_all_fidelity_schedules.yaml",
        "reports/research/release20260728_dense_tworoom_all_fidelity_schedules",
    ),
    (
        "configs/research/release20260728_dense_tworoom_goal50_plan50_execute20_all_fidelity_schedules.yaml",
        "reports/research/release20260728_dense_tworoom_goal50_plan50_execute20_all_fidelity_schedules",
    ),
)


def release_output_roots(repo_root: str | Path = REPO_ROOT) -> tuple[Path, ...]:
    root = Path(repo_root).resolve()
    outputs: list[Path] = []
    for config_relative, output_relative in BENCHMARK_TARGETS:
        cfg_path = root / config_relative
        cfg = OmegaConf.load(cfg_path)
        configured_output = Path(str(cfg.output_dir))
        configured_output = configured_output if configured_output.is_absolute() else root / configured_output
        expected_output = root / output_relative
        if configured_output.resolve() != expected_output.resolve():
            raise RuntimeError(
                f"Refusing unexpected output root in {cfg_path}: {configured_output}; expected {expected_output}"
            )
        outputs.append(expected_output)
    if len(outputs) != 8 or len(set(outputs)) != 8:
        raise RuntimeError("release20260728 compactor must resolve exactly eight distinct output roots")
    return tuple(outputs)


def compact_release_outputs(
    *,
    repo_root: str | Path = REPO_ROOT,
    dry_run: bool = False,
    target_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    counts = {"compacted": 0, "already_compacted": 0, "partial": 0, "would_compact": 0, "errors": 0}
    reclaimed = 0
    errors: list[str] = []
    per_root: list[dict[str, Any]] = []
    output_roots = release_output_roots(repo_root)
    selected = tuple(range(len(output_roots))) if target_indices is None else target_indices
    if len(set(selected)) != len(selected) or any(index < 0 or index >= len(output_roots) for index in selected):
        raise ValueError(f"target indices must be distinct values in [0, {len(output_roots) - 1}]")
    for target_index in selected:
        output_root = output_roots[target_index]
        root_counts = dict.fromkeys(counts, 0)
        root_reclaimed = 0
        if output_root.is_dir():
            for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
                try:
                    result = compact_completed_eval(run_dir, dry_run=dry_run)
                except (EvalCompactionError, OSError, ValueError, json.JSONDecodeError) as exc:
                    message = f"{run_dir}: {exc}"
                    result = {"status": "errors", "run_dir": str(run_dir), "error": message}
                    errors.append(message)
                status = str(result["status"])
                root_counts[status] += 1
                counts[status] += 1
                root_reclaimed += int(result.get("reclaimed_bytes", 0))
        reclaimed += root_reclaimed
        per_root.append(
            {
                "target_index": target_index,
                "output_root": str(output_root),
                "counts": root_counts,
                "reclaimed_bytes": root_reclaimed,
            }
        )
    return {"counts": counts, "reclaimed_bytes": reclaimed, "errors": errors, "roots": per_root}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove losslessly duplicated policy diagnostics from the eight release20260728 benchmarks."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate eligible cells without rewriting eval.json.")
    parser.add_argument(
        "--target-index",
        action="append",
        type=int,
        choices=range(len(BENCHMARK_TARGETS)),
        help="Process one exact allowlisted output root by index; repeat to select more than one.",
    )
    args = parser.parse_args()
    targets = tuple(args.target_index) if args.target_index is not None else None
    report = compact_release_outputs(dry_run=args.dry_run, target_indices=targets)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["counts"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
