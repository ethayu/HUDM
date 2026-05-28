from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mwm.benchmark.artifacts import (
    load_json,
    write_default_plots,
    write_json,
    write_metrics_jsonl,
    write_per_env_table,
    write_review_html,
    write_summary_csv,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_benchmark_review(output_dir: str | Path, *, title: str | None = None) -> dict[str, Any]:
    root = Path(output_dir)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing benchmark summary: {summary_path}")

    summary = load_json(summary_path)
    rows = list(summary.get("runs") or _read_jsonl(root / "metrics.jsonl"))
    if not rows:
        raise ValueError(f"no benchmark rows found in {summary_path} or {root / 'metrics.jsonl'}")

    outputs: list[dict[str, Any]] = []
    for row in rows:
        output_json = Path(str(row.get("output_json", "")))
        if output_json.is_file():
            outputs.append(load_json(output_json))

    write_summary_csv(root / "summary.csv", rows)
    write_metrics_jsonl(root / "metrics.jsonl", rows)
    summary["runs"] = rows
    summary["output_dir"] = str(root)
    summary["per_env_table"] = write_per_env_table(root / "per_env_summary.csv", rows)
    plots = write_default_plots(root / "plots", rows)
    summary["plots"] = plots
    write_json(summary_path, summary)

    review_title = str(title or summary.get("title") or "SWM MWM Benchmark Review")
    write_review_html(root / "review.html", review_title, rows, outputs, plots=plots)
    return {
        "output_dir": str(root),
        "review": str(root / "review.html"),
        "plots": [str(Path(plot).name) for plot in plots],
        "runs": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render plots and a static HTML review for an existing SWM benchmark output directory.")
    parser.add_argument("output_dir", nargs="?", default="rollouts/mwm_benchmark")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    report = render_benchmark_review(args.output_dir, title=args.title)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
