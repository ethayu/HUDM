from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mwm.benchmark.eval_artifacts import EvalArtifactError, compress_completed_eval
from scripts.research.compact_release20260728_policy_diagnostics import (
    BENCHMARK_TARGETS,
    REPO_ROOT,
    release_output_roots,
)


STATUSES = (
    "compressed",
    "already_compressed",
    "repaired",
    "partial",
    "would_compress",
    "would_repair",
    "errors",
)


def compress_release_outputs(
    *,
    repo_root: str | Path = REPO_ROOT,
    dry_run: bool = False,
    target_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    roots = release_output_roots(repo_root)
    selected = tuple(range(len(roots))) if target_indices is None else target_indices
    if len(set(selected)) != len(selected) or any(index < 0 or index >= len(roots) for index in selected):
        raise ValueError(f"target indices must be distinct values in [0, {len(roots) - 1}]")
    counts = dict.fromkeys(STATUSES, 0)
    reclaimed = 0
    errors: list[str] = []
    per_root: list[dict[str, Any]] = []
    for target_index in selected:
        output_root = roots[target_index]
        root_counts = dict.fromkeys(STATUSES, 0)
        root_reclaimed = 0
        if output_root.is_dir():
            for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
                try:
                    result = compress_completed_eval(run_dir, dry_run=dry_run)
                except (EvalArtifactError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    result = {"status": "errors", "error": str(exc)}
                    errors.append(f"{run_dir}: {exc}")
                status = str(result["status"])
                if status not in root_counts:
                    raise RuntimeError(f"unexpected compressor status {status!r}")
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
        description="Losslessly compress completed cells in the eight release20260728 schedule benchmarks."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate eligible cells without writing archives.")
    parser.add_argument(
        "--target-index",
        action="append",
        type=int,
        choices=range(len(BENCHMARK_TARGETS)),
        help="Process one exact allowlisted output root; repeat to select several roots.",
    )
    args = parser.parse_args()
    selected = tuple(args.target_index) if args.target_index is not None else None
    report = compress_release_outputs(dry_run=args.dry_run, target_indices=selected)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["counts"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compress_release_outputs", "main"]
