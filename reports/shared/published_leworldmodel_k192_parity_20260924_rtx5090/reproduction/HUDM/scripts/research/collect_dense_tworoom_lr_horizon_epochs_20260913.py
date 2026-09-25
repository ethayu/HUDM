from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/dense_tworoom_lr_horizon_epochs_20260913"
HORIZONS = (10, 100)
EPOCHS = tuple(range(10))
ROLES = ("fixed_k96", "fixed_k144", "fixed_k192", "k192_cost_k96", "k192_cost_k144")
ROLE_DIAGNOSTIC_CONTRACT = {
    "fixed_k96": (96, None),
    "fixed_k144": (144, None),
    "fixed_k192": (192, None),
    "k192_cost_k96": (192, 96),
    "k192_cost_k144": (192, 144),
}


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


def _trajectory_analysis(rows: list[dict[str, object]]) -> dict[str, object]:
    rates = {
        (int(row["lr_horizon"]), int(row["epoch"]), str(row["role"])): float(row["success_rate"])
        for row in rows
    }
    trajectories: dict[str, object] = {}
    for role in ROLES:
        role_payload: dict[str, object] = {}
        for horizon in HORIZONS:
            values = [rates[(horizon, epoch, role)] for epoch in EPOCHS]
            best_rate = max(values)
            role_payload[f"h{horizon}"] = {
                "by_epoch": dict(zip((str(epoch) for epoch in EPOCHS), values, strict=True)),
                "best_rate": best_rate,
                "best_epochs": [epoch for epoch, value in zip(EPOCHS, values, strict=True) if value == best_rate],
                "terminal_rate": values[-1],
                "terminal_minus_epoch8": values[-1] - values[-2],
            }
        role_payload["h100_minus_h10_by_epoch"] = {
            str(epoch): rates[(100, epoch, role)] - rates[(10, epoch, role)] for epoch in EPOCHS
        }
        role_payload["terminal_h100_minus_h10"] = rates[(100, EPOCHS[-1], role)] - rates[(10, EPOCHS[-1], role)]
        trajectories[role] = role_payload

    contrasts: dict[str, object] = {}
    contrast_specs = {
        "fixed_k192_minus_fixed_k96": ("fixed_k192", "fixed_k96"),
        "fixed_k192_minus_fixed_k144": ("fixed_k192", "fixed_k144"),
        "k192_cost_k96_minus_full_k192": ("k192_cost_k96", "fixed_k192"),
        "k192_cost_k144_minus_full_k192": ("k192_cost_k144", "fixed_k192"),
    }
    for horizon in HORIZONS:
        contrasts[f"h{horizon}"] = {
            name: {
                str(epoch): rates[(horizon, epoch, lhs)] - rates[(horizon, epoch, rhs)]
                for epoch in EPOCHS
            }
            for name, (lhs, rhs) in contrast_specs.items()
        }
    return {
        "status": "pass",
        "seed": 42,
        "episodes_per_cell": 50,
        "horizons": list(HORIZONS),
        "epochs": list(EPOCHS),
        "roles": list(ROLES),
        "trajectories": trajectories,
        "within_horizon_contrasts_by_epoch": contrasts,
    }


def main() -> None:
    collected: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for epoch in EPOCHS:
            path = REPORT_ROOT / "screen" / f"h{horizon}" / f"epoch_{epoch:03d}" / "summary.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_role = {row["role"]: row for row in rows}
            if set(by_role) != set(ROLES):
                raise RuntimeError(f"Unexpected roles in {path}: {sorted(by_role)}")
            hashes = {row["manifest_sha256"] for row in rows}
            file_hashes = {row["manifest_file_sha256"] for row in rows}
            if len(hashes) != 1 or len(file_hashes) != 1:
                raise RuntimeError(f"Manifest mismatch in {path}")
            for role in ROLES:
                row = by_role[role]
                if int(row["episodes"]) != 50 or int(row["seed"]) != 42:
                    raise RuntimeError(f"Evaluation contract mismatch in {path}: {role}")
                diagnostic_contract = _diagnostic_contract(row, role)
                collected.append(
                    {
                        "lr_horizon": horizon,
                        "epoch": epoch,
                        "role": role,
                        "success_rate": float(row["success_rate"]),
                        "successes": _success_count(row),
                        "episodes": int(row["episodes"]),
                        "manifest_sha256": next(iter(hashes)),
                        "manifest_file_sha256": next(iter(file_hashes)),
                        "checkpoint": row["checkpoint_run_dir"],
                        **diagnostic_contract,
                    }
                )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_ROOT / "epoch_screen_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(collected[0]))
        writer.writeheader()
        writer.writerows(collected)
    json_path = REPORT_ROOT / "epoch_screen_summary.json"
    json_path.write_text(json.dumps(collected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    analysis_path = REPORT_ROOT / "epoch_screen_analysis.json"
    analysis_path.write_text(
        json.dumps(_trajectory_analysis(collected), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated and collected {len(collected)} matched evaluation rows")


if __name__ == "__main__":
    main()
