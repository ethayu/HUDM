#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/k192_sepopt_published_parity_20260924"
ENVIRONMENTS = ("ogb_cube", "reacher", "pusht", "tworoom")
STAGES = {
    "screen": ("screen_n50", (0, 1, 2, 42, 100), 50),
    "final": ("final_n100", (2002, 71623, 82715, 86604, 91943), 100),
}
T_975_DF4 = 2.7764451051977987
NONINFERIORITY_SCREEN_MARGIN_PP = -5.0


def mean_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_975_DF4 * sd / math.sqrt(len(values))
    return {"mean": mean, "sample_sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def load_environment(stage: str, environment: str) -> dict:
    stage_dir, seeds, expected_episodes = STAGES[stage]
    paired_results = []
    for seed in seeds:
        seed_dir = REPORT_ROOT / stage_dir / environment / f"seed_{seed}"
        summary_path = seed_dir / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
        by_role = {row["role"]: row for row in rows}
        published = by_role["upstream_lewm_converted"]
        retrained = by_role["sepopt_k192_20260922"]
        if int(published["episodes"]) != expected_episodes or int(retrained["episodes"]) != expected_episodes:
            raise RuntimeError(f"episode mismatch in {summary_path}")
        if published["manifest_file_sha256"] != retrained["manifest_file_sha256"]:
            raise RuntimeError(f"manifest mismatch in {summary_path}")
        protocol_fields = ("goal_offset", "pop_size", "elite_frac", "topk", "n_iter")
        if any(published[field] != retrained[field] for field in protocol_fields):
            raise RuntimeError(f"protocol mismatch in {summary_path}")
        published_rate = float(published["success_rate"])
        retrained_rate = float(retrained["success_rate"])
        paired_results.append(
            {
                "environment": environment,
                "seed": seed,
                "episodes_per_model": expected_episodes,
                "manifest_file_sha256": published["manifest_file_sha256"],
                "published_success_rate": published_rate,
                "retrained_success_rate": retrained_rate,
                "retrained_minus_published_pp": retrained_rate - published_rate,
                "goal_offset": int(published["goal_offset"]),
                "population": int(published["pop_size"]),
                "elite_fraction": float(published["elite_frac"]),
                "top_k": int(published["topk"]),
                "cem_iterations": int(published["n_iter"]),
                "published_dynamics_flops_total": int(published["dynamics_flops_total"]),
                "retrained_dynamics_flops_total": int(retrained["dynamics_flops_total"]),
            }
        )

    published_stats = mean_ci([row["published_success_rate"] for row in paired_results])
    retrained_stats = mean_ci([row["retrained_success_rate"] for row in paired_results])
    delta_stats = mean_ci([row["retrained_minus_published_pp"] for row in paired_results])
    payload = {
        "stage": stage,
        "environment": environment,
        "seeds": list(seeds),
        "episodes_per_seed_per_model": expected_episodes,
        "paired_results": paired_results,
        "published": published_stats,
        "retrained": retrained_stats,
        "paired_delta_pp": delta_stats,
    }
    if stage == "screen":
        payload["qualification_margin_pp"] = NONINFERIORITY_SCREEN_MARGIN_PP
        payload["qualified_for_n100"] = delta_stats["mean"] >= NONINFERIORITY_SCREEN_MARGIN_PP
    return payload


def write_environment(stage: str, environment: str) -> Path:
    payload = load_environment(stage, environment)
    path = REPORT_ROOT / f"{stage}_{environment}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "environment": environment,
        "stage": stage,
        "published_mean": payload["published"]["mean"],
        "retrained_mean": payload["retrained"]["mean"],
        "paired_delta_mean": payload["paired_delta_pp"]["mean"],
        "qualified_for_n100": payload.get("qualified_for_n100"),
    }, sort_keys=True))
    return path


def write_all_final() -> None:
    summaries = []
    skipped = []
    for environment in ENVIRONMENTS:
        path = REPORT_ROOT / f"final_{environment}_summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        screen_path = REPORT_ROOT / f"screen_{environment}_summary.json"
        if not screen_path.is_file():
            raise FileNotFoundError(
                f"neither a final result nor a screen decision exists for {environment}"
            )
        screen = json.loads(screen_path.read_text(encoding="utf-8"))
        if screen.get("qualified_for_n100") is not False:
            raise RuntimeError(
                f"{environment} qualified for n=100 but has no final result"
            )
        skipped.append(
            {
                "environment": environment,
                "reason": "screen mean paired delta was below the predeclared -5 pp margin",
                "screen_summary": str(screen_path.relative_to(ROOT)),
                "screen_paired_delta_pp": screen["paired_delta_pp"],
            }
        )
    payload = {
        "status": "complete" if not skipped else "complete_with_screen_skips",
        "comparison": "published converted LeWorldModel vs September 22 K=192 sepopt baseline",
        "environments": summaries,
        "skipped_after_screen": skipped,
    }
    (REPORT_ROOT / "final_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (REPORT_ROOT / "final_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "environment", "episodes_per_seed_per_model", "seeds",
            "published_mean", "published_ci95_low", "published_ci95_high",
            "retrained_mean", "retrained_ci95_low", "retrained_ci95_high",
            "paired_delta_mean", "paired_delta_ci95_low", "paired_delta_ci95_high",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow({
                "environment": item["environment"],
                "episodes_per_seed_per_model": item["episodes_per_seed_per_model"],
                "seeds": json.dumps(item["seeds"]),
                "published_mean": item["published"]["mean"],
                "published_ci95_low": item["published"]["ci95_low"],
                "published_ci95_high": item["published"]["ci95_high"],
                "retrained_mean": item["retrained"]["mean"],
                "retrained_ci95_low": item["retrained"]["ci95_low"],
                "retrained_ci95_high": item["retrained"]["ci95_high"],
                "paired_delta_mean": item["paired_delta_pp"]["mean"],
                "paired_delta_ci95_low": item["paired_delta_pp"]["ci95_low"],
                "paired_delta_ci95_high": item["paired_delta_pp"]["ci95_high"],
            })
    print(REPORT_ROOT / "final_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--environment", choices=ENVIRONMENTS)
    parser.add_argument("--all-final", action="store_true")
    args = parser.parse_args()
    if args.all_final:
        write_all_final()
        return
    if not args.stage or not args.environment:
        parser.error("--stage and --environment are required unless --all-final is used")
    write_environment(args.stage, args.environment)


if __name__ == "__main__":
    main()
