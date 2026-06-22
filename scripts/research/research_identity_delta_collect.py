from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPORT_DIR = Path("reports/research/identity_delta")
SWEEP_DIR = REPORT_DIR / "seed_sweep"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _failure_indices(eval_path: Path) -> list[int]:
    payload = _load_json(eval_path)
    successes = payload.get("swm_results", {}).get("episode_successes", [])
    return [idx for idx, success in enumerate(successes) if not bool(success)]


def _run_dir_for_role(seed_dir: Path, row: dict[str, str]) -> Path:
    name = row.get("name") or row.get("role") or "run"
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_")
    index = 0 if row.get("role") == "upstream_lewm_converted" else 1
    return seed_dir / f"{index:03d}_{safe_name}"


def collect() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for summary_csv in sorted(SWEEP_DIR.glob("*_seed*/summary.csv")):
        seed_dir = summary_csv.parent
        env, seed_text = seed_dir.name.rsplit("_seed", 1)
        seed = int(seed_text)
        csv_rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
        by_role: dict[str, dict[str, Any]] = {}
        for row in csv_rows:
            role = row["role"]
            run_dir = _run_dir_for_role(seed_dir, row)
            failures = _failure_indices(run_dir / "eval.json")
            record = {
                "env": env,
                "seed": seed,
                "role": role,
                "success_rate": float(row["success_rate"]),
                "episodes": int(float(row["episodes"])),
                "manifest_sha256": row.get("manifest_sha256", ""),
                "manifest_file_sha256": row.get("manifest_file_sha256", ""),
                "config_sha256": row.get("config_sha256", ""),
                "failure_indices": failures,
                "output_json": str(run_dir / "eval.json"),
            }
            rows.append(record)
            by_role[role] = record
        upstream = by_role.get("upstream_lewm_converted")
        identity = by_role.get("retrained_lewm_identity")
        if upstream and identity:
            upstream_failures = set(upstream["failure_indices"])
            identity_failures = set(identity["failure_indices"])
            pairs.append(
                {
                    "env": env,
                    "seed": seed,
                    "upstream_success_rate": upstream["success_rate"],
                    "identity_success_rate": identity["success_rate"],
                    "identity_minus_upstream": identity["success_rate"] - upstream["success_rate"],
                    "manifest_sha256": upstream["manifest_sha256"],
                    "shared_failure_indices": sorted(upstream_failures & identity_failures),
                    "upstream_only_failure_indices": sorted(upstream_failures - identity_failures),
                    "identity_only_failure_indices": sorted(identity_failures - upstream_failures),
                }
            )

    rows.sort(key=lambda item: (item["env"], item["seed"], item["role"]))
    pairs.sort(key=lambda item: (item["env"], item["seed"]))

    env_summary: dict[str, Any] = {}
    for env in sorted({pair["env"] for pair in pairs}):
        env_pairs = [pair for pair in pairs if pair["env"] == env]
        deltas = [float(pair["identity_minus_upstream"]) for pair in env_pairs]
        upstream_rates = [float(pair["upstream_success_rate"]) for pair in env_pairs]
        identity_rates = [float(pair["identity_success_rate"]) for pair in env_pairs]
        env_summary[env] = {
            "seeds": [pair["seed"] for pair in env_pairs],
            "mean_identity_minus_upstream": mean(deltas),
            "population_std_identity_minus_upstream": pstdev(deltas) if len(deltas) > 1 else 0.0,
            "min_identity_minus_upstream": min(deltas),
            "max_identity_minus_upstream": max(deltas),
            "mean_upstream_success_rate": mean(upstream_rates),
            "mean_identity_success_rate": mean(identity_rates),
        }

    return {
        "rows": rows,
        "paired_results": pairs,
        "env_summary": env_summary,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = collect()
    (REPORT_DIR / "seed_sweep_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (REPORT_DIR / "seed_sweep_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "env",
            "seed",
            "upstream_success_rate",
            "identity_success_rate",
            "identity_minus_upstream",
            "manifest_sha256",
            "shared_failure_indices",
            "upstream_only_failure_indices",
            "identity_only_failure_indices",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in payload["paired_results"]:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key] for key in fieldnames})
    print(json.dumps(payload["env_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
