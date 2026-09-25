from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


SEEDS = (0, 1, 2, 42, 100)
ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/reacher_k192_spt_sigreg009_n500_20260804"
ROLLOUT_TEMPLATE = "mwm_reacher_k192_spt_sigreg009_seed{seed}_n500_20260804"
EXPECTED_ROLES = ("upstream_lewm_converted", "spt_isolateddec_sigreg009")


def main() -> None:
    paired = []
    manifest_hashes: dict[int, str] = {}
    config_hashes: dict[int, dict[str, str]] = {}
    for seed in SEEDS:
        summary = ROOT / "rollouts" / ROLLOUT_TEMPLATE.format(seed=seed) / "summary.csv"
        if not summary.is_file():
            raise FileNotFoundError(f"Missing N=500 result for seed {seed}: {summary}")
        with summary.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_role = {row["role"]: row for row in rows}
        if set(by_role) != set(EXPECTED_ROLES):
            raise RuntimeError(f"Seed {seed} roles differ: {sorted(by_role)}")
        if any(int(row["episodes"]) != 500 or int(row["seed"]) != seed for row in rows):
            raise RuntimeError(f"Seed/episode provenance mismatch in {summary}")
        hashes = {row["manifest_sha256"] for row in rows}
        file_hashes = {row["manifest_file_sha256"] for row in rows}
        if len(hashes) != 1 or len(file_hashes) != 1:
            raise RuntimeError(f"Seed {seed} did not use one paired manifest")
        upstream = float(by_role[EXPECTED_ROLES[0]]["success_rate"])
        candidate = float(by_role[EXPECTED_ROLES[1]]["success_rate"])
        paired.append(
            {
                "seed": seed,
                "episodes": 500,
                "upstream_success_rate": upstream,
                "candidate_success_rate": candidate,
                "candidate_minus_upstream": candidate - upstream,
                "manifest_sha256": next(iter(hashes)),
                "manifest_file_sha256": next(iter(file_hashes)),
                "summary_csv": str(summary.relative_to(ROOT)),
            }
        )
        manifest_hashes[seed] = next(iter(hashes))
        config_hashes[seed] = {role: by_role[role]["config_sha256"] for role in EXPECTED_ROLES}

    upstream_rates = [row["upstream_success_rate"] for row in paired]
    candidate_rates = [row["candidate_success_rate"] for row in paired]
    deltas = [row["candidate_minus_upstream"] for row in paired]
    payload = {
        "status": "pass",
        "environment": "swm/ReacherDMControl-v0",
        "episodes_per_seed": 500,
        "seeds": list(SEEDS),
        "paired_results": paired,
        "aggregate": {
            "upstream_mean_success_rate": mean(upstream_rates),
            "upstream_sample_stdev": stdev(upstream_rates),
            "candidate_mean_success_rate": mean(candidate_rates),
            "candidate_sample_stdev": stdev(candidate_rates),
            "mean_paired_delta": mean(deltas),
            "paired_delta_sample_stdev": stdev(deltas),
        },
        "config_sha256_by_seed_and_role": config_hashes,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = REPORT_ROOT / "n500_summary.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
