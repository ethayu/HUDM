from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/single_level_reacher_parity_20260806"
UPSTREAM = {
    "threshold_0_05": {
        "report": REPORT_ROOT / "upstream_eval_native_h5_seed42_iter30.txt",
        "log": ROOT / "logs/upstream_reacher_h5_7433956.out",
        "stable_worldmodel_commit": "44c45bd1f774995aac0cc2b41444ebea24c1d5ad",
    },
    "threshold_0_10": {
        "report": REPORT_ROOT / "upstream_eval_native_h5_seed42_iter30_threshold010.txt",
        "log": ROOT / "logs/upstream_reacher_h5_7433961.out",
        "stable_worldmodel_commit": "2096f6a17498f6283881e141d1c0579536815485",
    },
}
MWM = {
    "threshold_0_05": ROOT
    / "rollouts/reacher_historical_goal24_native_h5_seed42_n50_iter30_threshold005_20260806/eval.json",
    "threshold_0_10": ROOT
    / "rollouts/reacher_historical_goal24_native_h5_seed42_n50_iter30_threshold010_20260806/eval.json",
}
UPSTREAM_PAPER_CEM = {
    "report": REPORT_ROOT / "upstream_eval_native_h5_seed42_iter10_threshold010.txt",
    "log": ROOT / "logs/upstream_reacher_h5_7433977.out",
}
MWM_HISTORICAL_EXECUTABLE = {
    10: ROOT
    / "rollouts/reacher_historical_goal24_native_h5_seed42_n50_iter10_threshold010_exact_executable_20260806/eval.json",
    30: ROOT
    / "rollouts/reacher_historical_goal24_native_h5_seed42_n50_iter30_threshold010_exact_executable_20260806/eval.json",
}
DESTINATION = REPORT_ROOT / "upstream_evaluator_parity.json"


def parse_upstream_report(path: Path) -> tuple[float, list[bool]]:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(
        r"metrics: \{'success_rate': ([0-9.]+), 'episode_successes': array\(\[(.*?)\]\)",
        text,
        flags=re.DOTALL,
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one metrics block in {path}, found {len(matches)}")
    rate_text, labels_text = matches[0]
    labels = [token == "True" for token in re.findall(r"True|False", labels_text)]
    if len(labels) != 50:
        raise RuntimeError(f"Expected 50 labels in {path}, found {len(labels)}")
    rate = float(rate_text)
    if rate != 100.0 * sum(labels) / len(labels):
        raise RuntimeError(f"Rate/label mismatch in {path}")
    return rate, labels


def parse_start_rows(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"valid starting points found for evaluation\.\s*\n\[(.*?)\]\s*\nCEM solve time",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"Could not parse selected start rows from {path}")
    rows = [int(token) for token in re.findall(r"\d+", match.group(1))]
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 start rows in {path}, found {len(rows)}")
    return rows


def compare_labels(left: list[bool], right: list[bool]) -> dict[str, int]:
    return {
        "changed": sum(a != b for a, b in zip(left, right, strict=True)),
        "left_only_successes": sum(a and not b for a, b in zip(left, right, strict=True)),
        "right_only_successes": sum(b and not a for a, b in zip(left, right, strict=True)),
    }


def main() -> None:
    upstream_results = {}
    mwm_results = {}
    rows_by_key = {}
    for key, spec in UPSTREAM.items():
        rate, labels = parse_upstream_report(spec["report"])
        rows = parse_start_rows(spec["log"])
        upstream_results[key] = {
            "success_rate": rate,
            "episode_successes": labels,
            "stable_worldmodel_commit": spec["stable_worldmodel_commit"],
            "report": str(spec["report"].relative_to(ROOT)),
            "log": str(spec["log"].relative_to(ROOT)),
        }
        rows_by_key[key] = rows

        mwm = json.loads(MWM[key].read_text(encoding="utf-8"))
        mwm_labels = [bool(value) for value in mwm["swm_results"]["episode_successes"]]
        mwm_rows = [int(pair["start_row"]) for batch in mwm["batches"] for pair in batch["pairs"]]
        if rows != mwm_rows:
            raise RuntimeError(f"Start rows differ between untouched upstream and MWM for {key}")
        expected_threshold = 0.05 if key.endswith("0_05") else 0.1
        if float(mwm["env_runtime"]["reacher_qpos_threshold"]) != expected_threshold:
            raise RuntimeError(f"MWM threshold provenance mismatch for {key}")
        if mwm.get("goal_indexing", {}).get("effective_offset") != 24:
            raise RuntimeError(f"MWM historical goal provenance mismatch for {key}")
        mwm_results[key] = {
            "success_rate": float(mwm["swm_results"]["success_rate"]),
            "episode_successes": mwm_labels,
            "eval": str(MWM[key].relative_to(ROOT)),
        }

    if rows_by_key["threshold_0_05"] != rows_by_key["threshold_0_10"]:
        raise RuntimeError("Untouched controls did not use identical start rows")

    paired = {}
    for key in UPSTREAM:
        upstream_row = upstream_results[key]
        mwm_row = mwm_results[key]
        paired[key] = {
            "upstream_success_rate": upstream_row["success_rate"],
            "mwm_success_rate": mwm_row["success_rate"],
            "mwm_minus_upstream_percentage_points": mwm_row["success_rate"]
            - upstream_row["success_rate"],
            "label_comparison": compare_labels(
                upstream_row["episode_successes"], mwm_row["episode_successes"]
            ),
        }

    historical_executable = {}
    upstream_by_iterations = {
        10: parse_upstream_report(UPSTREAM_PAPER_CEM["report"]),
        30: (
            upstream_results["threshold_0_10"]["success_rate"],
            upstream_results["threshold_0_10"]["episode_successes"],
        ),
    }
    upstream_rows_by_iterations = {
        10: parse_start_rows(UPSTREAM_PAPER_CEM["log"]),
        30: rows_by_key["threshold_0_10"],
    }
    for iterations, path in MWM_HISTORICAL_EXECUTABLE.items():
        mwm = json.loads(path.read_text(encoding="utf-8"))
        labels = [bool(value) for value in mwm["swm_results"]["episode_successes"]]
        rows = [int(pair["start_row"]) for batch in mwm["batches"] for pair in batch["pairs"]]
        if rows != upstream_rows_by_iterations[iterations]:
            raise RuntimeError(f"Historical executable start rows differ for CEM={iterations}")
        diagnostics = mwm["planning_diagnostics"]
        if mwm.get("env_runtime", {}).get("post_reset_validation") != "passed":
            raise RuntimeError(f"Historical CEM={iterations} lacks post-reset threshold validation")
        expected_calls = 100 * iterations
        if int(diagnostics["cem_cost_calls"]) != expected_calls:
            raise RuntimeError(
                f"Historical CEM={iterations} made {diagnostics['cem_cost_calls']} calls, expected {expected_calls}"
            )
        for field in ("rollout_semantics_counts", "policy_semantics_counts"):
            if diagnostics.get(field) != {"upstream_lewm_historical": expected_calls}:
                raise RuntimeError(f"Historical semantics provenance mismatch for CEM={iterations}: {field}")
        upstream_rate, upstream_labels = upstream_by_iterations[iterations]
        comparison = compare_labels(upstream_labels, labels)
        mwm_rate = float(mwm["swm_results"]["success_rate"])
        if comparison["changed"] or mwm_rate != upstream_rate:
            raise RuntimeError(
                f"MWM did not exactly reproduce untouched historical evaluator at CEM={iterations}: "
                f"upstream={upstream_rate}, mwm={mwm_rate}, labels={comparison}"
            )
        historical_executable[str(iterations)] = {
            "cem_iterations": iterations,
            "upstream_success_rate": upstream_rate,
            "mwm_success_rate": mwm_rate,
            "label_comparison": comparison,
            "mwm_eval": str(path.relative_to(ROOT)),
            "upstream_report": str(
                (UPSTREAM_PAPER_CEM["report"] if iterations == 10 else UPSTREAM["threshold_0_10"]["report"])
                .relative_to(ROOT)
            ),
        }

    payload = {
        "status": "pass",
        "episodes": 50,
        "seed": 42,
        "requested_goal_offset": 25,
        "effective_goal_offset": 24,
        "cem_iterations": 30,
        "native_hdf5_sha256": "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06",
        "lewm_commit": "83f97d72ad067855bc89a1b74b4aff11d4dfdf0c",
        "start_rows_equal_across_all_controls": True,
        "superseded_pre_fix_mwm_comparison": {
            "reason": (
                "Both legacy MWM files effectively used threshold 0.05 because reset-time task "
                "recompilation discarded the requested 0.1 override."
            ),
            "requested_threshold_0_10_effective_threshold": 0.05,
            "paired": paired,
        },
        "historical_executable_exact_replay": historical_executable,
        "superseded_pre_fix_threshold_effect": {
            "upstream": compare_labels(
                upstream_results["threshold_0_05"]["episode_successes"],
                upstream_results["threshold_0_10"]["episode_successes"],
            ),
            "mwm": compare_labels(
                mwm_results["threshold_0_05"]["episode_successes"],
                mwm_results["threshold_0_10"]["episode_successes"],
            ),
        },
        "sources": {
            "upstream": upstream_results,
            "mwm": mwm_results,
            "historical_inference_parity": "reports/research/single_level_reacher_parity_20260806/historical_lewm_inference_parity.json",
            "historical_cem_parity": "reports/research/single_level_reacher_parity_20260806/historical_cem_parity.json",
        },
    }
    DESTINATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
