from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/dense_tworoom_lr_horizon_epochs_20260913"
HORIZONS = (10, 100)
SEEDS = (0, 1, 2, 42, 100)
ROLES = ("fixed_k96", "fixed_k144", "fixed_k192", "k192_cost_k96", "k192_cost_k144")
ROLE_DIAGNOSTIC_CONTRACT = {
    "fixed_k96": (96, None),
    "fixed_k144": (144, None),
    "fixed_k192": (192, None),
    "k192_cost_k96": (192, 96),
    "k192_cost_k144": (192, 144),
}
T95_DF4 = 2.7764451051977987


def _success_count(row: dict[str, str]) -> int:
    output_path = Path(row["output_json"])
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    outcomes = payload.get("swm_results", {}).get("episode_successes", [])
    episodes = int(row["episodes"])
    if not isinstance(outcomes, list) or len(outcomes) != episodes:
        raise RuntimeError(f"Incomplete episode outcomes in {output_path}: {len(outcomes)} != {episodes}")
    successes = sum(bool(value) for value in outcomes)
    expected_rate = successes / episodes * 100.0
    if abs(float(row["success_rate"]) - expected_rate) > 1e-9:
        raise RuntimeError(
            f"Success-rate mismatch in {output_path}: {row['success_rate']} != {expected_rate}"
        )
    return successes


def _diagnostic_contract(row: dict[str, str], role: str) -> dict[str, object]:
    output_path = Path(row["output_json"])
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    diagnostics = payload.get("planning_diagnostics", {})
    trace = diagnostics.get("trace", [])
    if not isinstance(trace, list) or not trace:
        raise RuntimeError(f"Missing planning trace in {output_path}")
    expected_terminal_k, expected_cost_prefix_k = ROLE_DIAGNOSTIC_CONTRACT[role]
    for index, item in enumerate(trace):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid planning trace entry {index} in {output_path}")
        for key in ("base_k", "terminal_k", "model_base_k", "model_terminal_k"):
            if int(item.get(key, -1)) != expected_terminal_k:
                raise RuntimeError(
                    f"{role}: {key} mismatch at trace {index} in {output_path}: "
                    f"{item.get(key)!r} != {expected_terminal_k}"
                )
        if item.get("model_prefix_criterion") is not True:
            raise RuntimeError(f"{role}: prefix criterion missing at trace {index} in {output_path}")
        observed_prefix = item.get("model_research_cost_prefix_k")
        if expected_cost_prefix_k is None:
            if observed_prefix is not None:
                raise RuntimeError(
                    f"{role}: unexpected research cost prefix at trace {index} in {output_path}: "
                    f"{observed_prefix!r}"
                )
        elif int(observed_prefix or -1) != expected_cost_prefix_k:
            raise RuntimeError(
                f"{role}: research cost prefix mismatch at trace {index} in {output_path}: "
                f"{observed_prefix!r} != {expected_cost_prefix_k}"
            )
    expected_calls = int(diagnostics.get("summary", {}).get("cem_cost_calls", -1))
    if len(trace) != expected_calls:
        raise RuntimeError(
            f"Planning trace/cost-call mismatch in {output_path}: {len(trace)} != {expected_calls}"
        )
    return {
        "diagnostic_trace_count": len(trace),
        "diagnostic_terminal_k": expected_terminal_k,
        "diagnostic_cost_prefix_k": expected_cost_prefix_k,
    }


def _summary(values: list[float]) -> dict[str, object]:
    center = mean(values)
    spread = stdev(values)
    half_width = T95_DF4 * spread / math.sqrt(len(values))
    return {
        "mean": center,
        "sample_stdev": spread,
        "ci95": [center - half_width, center + half_width],
        "by_seed": dict(zip((str(seed) for seed in SEEDS), values, strict=True)),
    }


def main() -> None:
    values: dict[int, dict[str, list[float]]] = {
        horizon: {role: [] for role in ROLES} for horizon in HORIZONS
    }
    rows_by_cell: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
    manifest_contract: dict[int, tuple[str, str]] = {}
    flat_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            path = REPORT_ROOT / "terminal_n500" / f"h{horizon}" / f"seed{seed}" / "summary.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_role = {row["role"]: row for row in rows}
            if set(by_role) != set(ROLES):
                raise RuntimeError(f"Unexpected roles in {path}: {sorted(by_role)}")
            hashes = {row["manifest_sha256"] for row in rows}
            file_hashes = {row["manifest_file_sha256"] for row in rows}
            if len(hashes) != 1 or len(file_hashes) != 1:
                raise RuntimeError(f"Manifest mismatch within {path}")
            contract = (next(iter(hashes)), next(iter(file_hashes)))
            if seed in manifest_contract and manifest_contract[seed] != contract:
                raise RuntimeError(f"Manifest mismatch across horizons for seed {seed}")
            manifest_contract[seed] = contract
            rows_by_cell[(horizon, seed)] = by_role
            for role in ROLES:
                row = by_role[role]
                if int(row["episodes"]) != 500 or int(row["seed"]) != seed:
                    raise RuntimeError(f"Evaluation contract mismatch in {path}: {role}")
                rate = float(row["success_rate"])
                diagnostic_contract = _diagnostic_contract(row, role)
                values[horizon][role].append(rate)
                flat_rows.append(
                    {
                        "lr_horizon": horizon,
                        "seed": seed,
                        "role": role,
                        "success_rate": rate,
                        "successes": _success_count(row),
                        "episodes": int(row["episodes"]),
                        "manifest_sha256": contract[0],
                        "manifest_file_sha256": contract[1],
                        "checkpoint": row["checkpoint_run_dir"],
                        **diagnostic_contract,
                    }
                )

    horizon_effects = {
        role: _summary(
            [
                values[100][role][index] - values[10][role][index]
                for index in range(len(SEEDS))
            ]
        )
        for role in ROLES
    }
    within_horizon: dict[str, dict[str, object]] = {}
    for horizon in HORIZONS:
        contrasts = {
            "fixed_k192_minus_fixed_k96": [
                values[horizon]["fixed_k192"][index] - values[horizon]["fixed_k96"][index]
                for index in range(len(SEEDS))
            ],
            "fixed_k192_minus_fixed_k144": [
                values[horizon]["fixed_k192"][index] - values[horizon]["fixed_k144"][index]
                for index in range(len(SEEDS))
            ],
            "k192_cost_k96_minus_full_k192": [
                values[horizon]["k192_cost_k96"][index] - values[horizon]["fixed_k192"][index]
                for index in range(len(SEEDS))
            ],
            "k192_cost_k144_minus_full_k192": [
                values[horizon]["k192_cost_k144"][index] - values[horizon]["fixed_k192"][index]
                for index in range(len(SEEDS))
            ],
        }
        within_horizon[str(horizon)] = {
            "success_rates": {role: _summary(role_values) for role, role_values in values[horizon].items()},
            "contrasts": {name: _summary(samples) for name, samples in contrasts.items()},
        }

    payload = {
        "status": "pass",
        "objective": (
            "Compare dense TwoRoom 10-epoch training under cosine horizons 10 and 100, "
            "and separate transition fidelity from terminal-cost prefix effects."
        ),
        "episodes_per_seed": 500,
        "seeds": list(SEEDS),
        "horizons": list(HORIZONS),
        "roles": list(ROLES),
        "horizon_effects": horizon_effects,
        "within_horizon": within_horizon,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_ROOT / "terminal_n500_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = REPORT_ROOT / "terminal_n500_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
