from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/dense_tworoom_h100_continue20_20260914"
OLD_ROOT = ROOT / "reports/research/dense_tworoom_lr_horizon_epochs_20260913"
SEEDS = (0, 1, 2, 42, 100)
ROLES = ("fixed_k96", "fixed_k144", "fixed_k192", "k192_cost_k96", "k192_cost_k144")
ROLE_CONTRACT = {
    "fixed_k96": (96, None),
    "fixed_k144": (144, None),
    "fixed_k192": (192, None),
    "k192_cost_k96": (192, 96),
    "k192_cost_k144": (192, 144),
}
T95_DF4 = 2.7764451051977987
METRICS = (
    "validate/loss_epoch",
    "validate/pred_loss_epoch",
    "validate/recon_loss_epoch",
    "validate/sigreg_loss_epoch",
    *(f"validate/pred_loss_l{level}_epoch" for level in range(5)),
    *(f"validate/recon_loss_l{level}_epoch" for level in range(5)),
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _validate_role(row: dict[str, str], role: str, episodes: int, seed: int) -> int:
    if int(row["episodes"]) != episodes or int(row["seed"]) != seed:
        raise RuntimeError(f"Evaluation contract mismatch for {role}: {row}")
    payload = json.loads(_resolve(row["output_json"]).read_text(encoding="utf-8"))
    outcomes = payload.get("swm_results", {}).get("episode_successes", [])
    if not isinstance(outcomes, list) or len(outcomes) != episodes:
        raise RuntimeError(f"Incomplete episode outcomes in {row['output_json']}")
    successes = sum(bool(value) for value in outcomes)
    if abs(float(row["success_rate"]) - successes / episodes * 100.0) > 1e-9:
        raise RuntimeError(f"Success-rate mismatch in {row['output_json']}")
    terminal_k, prefix_k = ROLE_CONTRACT[role]
    diagnostics = payload.get("planning_diagnostics", {})
    trace = diagnostics.get("trace", [])
    if not isinstance(trace, list) or not trace:
        raise RuntimeError(f"Missing planning trace in {row['output_json']}")
    if len(trace) != int(diagnostics.get("summary", {}).get("cem_cost_calls", -1)):
        raise RuntimeError(f"Planning trace count mismatch in {row['output_json']}")
    for item in trace:
        for key in ("base_k", "terminal_k", "model_base_k", "model_terminal_k"):
            if int(item.get(key, -1)) != terminal_k:
                raise RuntimeError(f"{role} {key} mismatch in {row['output_json']}")
        observed = item.get("model_research_cost_prefix_k")
        if prefix_k is None and observed is not None:
            raise RuntimeError(f"Unexpected cost prefix for {role} in {row['output_json']}")
        if prefix_k is not None and int(observed or -1) != prefix_k:
            raise RuntimeError(f"Cost prefix mismatch for {role} in {row['output_json']}")
    return successes


def _read_cell(path: Path, episodes: int, seed: int) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_role = {row["role"]: row for row in rows}
    if set(by_role) != set(ROLES):
        raise RuntimeError(f"Unexpected roles in {path}: {sorted(by_role)}")
    if len({row["manifest_sha256"] for row in rows}) != 1:
        raise RuntimeError(f"Manifest hash mismatch within {path}")
    if len({row["manifest_file_sha256"] for row in rows}) != 1:
        raise RuntimeError(f"Manifest file hash mismatch within {path}")
    for role, row in by_role.items():
        _validate_role(row, role, episodes, seed)
    return by_role


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


def _collect_terminal() -> tuple[dict[str, object], list[dict[str, object]]]:
    branches = {
        "h10_epoch9": lambda seed: OLD_ROOT / "terminal_n500/h10" / f"seed{seed}/summary.csv",
        "h100_epoch9": lambda seed: OLD_ROOT / "terminal_n500/h100" / f"seed{seed}/summary.csv",
        "h10_tail_epoch14": lambda seed: REPORT_ROOT / "h10_tail_terminal_n500/epoch_014" / f"seed{seed}/summary.csv",
        "h100_epoch14": lambda seed: REPORT_ROOT / "terminal_n500/epoch_014" / f"seed{seed}/summary.csv",
        "h100_epoch19": lambda seed: REPORT_ROOT / "terminal_n500/epoch_019" / f"seed{seed}/summary.csv",
    }
    values = {branch: {role: [] for role in ROLES} for branch in branches}
    flat: list[dict[str, object]] = []
    seed_manifests: dict[int, tuple[str, str]] = {}
    for branch, path_for_seed in branches.items():
        for seed in SEEDS:
            cell = _read_cell(path_for_seed(seed), 500, seed)
            contract = (
                next(iter({row["manifest_sha256"] for row in cell.values()})),
                next(iter({row["manifest_file_sha256"] for row in cell.values()})),
            )
            if seed in seed_manifests and seed_manifests[seed] != contract:
                raise RuntimeError(f"Manifest mismatch across branches for seed {seed}")
            seed_manifests[seed] = contract
            for role, row in cell.items():
                rate = float(row["success_rate"])
                values[branch][role].append(rate)
                flat.append(
                    {
                        "branch": branch,
                        "seed": seed,
                        "role": role,
                        "success_rate": rate,
                        "successes": round(rate * 5),
                        "episodes": 500,
                        "manifest_sha256": contract[0],
                        "manifest_file_sha256": contract[1],
                        "checkpoint": row["checkpoint_run_dir"],
                    }
                )

    comparisons = {
        "h100_epoch14_minus_epoch9": ("h100_epoch14", "h100_epoch9"),
        "h100_epoch19_minus_epoch9": ("h100_epoch19", "h100_epoch9"),
        "h100_epoch19_minus_epoch14": ("h100_epoch19", "h100_epoch14"),
        "h10_tail_epoch14_minus_h10_epoch9": ("h10_tail_epoch14", "h10_epoch9"),
        "epoch14_h100_minus_h10_tail": ("h100_epoch14", "h10_tail_epoch14"),
    }
    effects = {
        name: {
            role: _summary(
                [values[lhs][role][i] - values[rhs][role][i] for i in range(len(SEEDS))]
            )
            for role in ROLES
        }
        for name, (lhs, rhs) in comparisons.items()
    }
    within: dict[str, object] = {}
    for branch, role_values in values.items():
        contrasts = {
            "fixed_k192_minus_fixed_k96": [
                role_values["fixed_k192"][i] - role_values["fixed_k96"][i]
                for i in range(len(SEEDS))
            ],
            "fixed_k192_minus_fixed_k144": [
                role_values["fixed_k192"][i] - role_values["fixed_k144"][i]
                for i in range(len(SEEDS))
            ],
            "k192_cost_k96_minus_full_k192": [
                role_values["k192_cost_k96"][i] - role_values["fixed_k192"][i]
                for i in range(len(SEEDS))
            ],
            "k192_cost_k144_minus_full_k192": [
                role_values["k192_cost_k144"][i] - role_values["fixed_k192"][i]
                for i in range(len(SEEDS))
            ],
        }
        within[branch] = {
            "success_rates": {role: _summary(samples) for role, samples in role_values.items()},
            "contrasts": {name: _summary(samples) for name, samples in contrasts.items()},
        }
    return {
        "status": "pass",
        "episodes_per_seed": 500,
        "seeds": list(SEEDS),
        "roles": list(ROLES),
        "within_branch": within,
        "paired_effects": effects,
    }, flat


def _collect_screen() -> tuple[dict[str, object], list[dict[str, object]]]:
    trajectories: dict[str, dict[int, dict[str, float]]] = {"h100": {}, "h10_tail": {}}
    flat: list[dict[str, object]] = []
    specs = [
        ("h100", 9, OLD_ROOT / "screen/h100/epoch_009/summary.csv"),
        ("h10_tail", 9, OLD_ROOT / "screen/h10/epoch_009/summary.csv"),
        *[("h100", epoch, REPORT_ROOT / f"screen/epoch_{epoch:03d}/summary.csv") for epoch in range(10, 20)],
        *[("h10_tail", epoch, REPORT_ROOT / f"h10_tail_screen/epoch_{epoch:03d}/summary.csv") for epoch in range(10, 15)],
    ]
    for branch, epoch, path in specs:
        cell = _read_cell(path, 50, 42)
        trajectories[branch][epoch] = {
            role: float(row["success_rate"]) for role, row in cell.items()
        }
        for role, row in cell.items():
            flat.append(
                {
                    "branch": branch,
                    "epoch": epoch,
                    "completed_epochs": epoch + 1,
                    "role": role,
                    "success_rate": float(row["success_rate"]),
                    "checkpoint": row["checkpoint_run_dir"],
                }
            )
    payload: dict[str, object] = {"status": "pass", "episodes": 50, "seed": 42, "branches": {}}
    for branch, by_epoch in trajectories.items():
        branch_payload: dict[str, object] = {}
        for role in ROLES:
            role_values = {str(epoch): by_epoch[epoch][role] for epoch in sorted(by_epoch)}
            best = max(role_values.values())
            branch_payload[role] = {
                "by_epoch": role_values,
                "best_rate": best,
                "best_epochs": [int(epoch) for epoch, value in role_values.items() if value == best],
                "terminal_minus_epoch9": role_values[str(max(by_epoch))] - role_values["9"],
            }
        payload["branches"][branch] = branch_payload
    return payload, flat


def _validation_rows(path: Path, expected_epochs: range) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = [row for row in csv.DictReader(handle) if row.get("validate/loss_epoch")]
    rows = []
    for raw in raw_rows:
        if any(not raw.get(metric) for metric in METRICS):
            raise RuntimeError(f"Missing validation metrics in {path} epoch {raw.get('epoch')}")
        rows.append(
            {
                "epoch": int(raw["epoch"]),
                "step": int(raw["step"]),
                "metrics": {metric: float(raw[metric]) for metric in METRICS},
            }
        )
    rows.sort(key=lambda row: int(row["epoch"]))
    if [row["epoch"] for row in rows] != list(expected_epochs):
        raise RuntimeError(f"Unexpected validation epochs in {path}: {[row['epoch'] for row in rows]}")
    return rows


def _collect_training() -> dict[str, object]:
    old = ROOT / "logs/mwm_training"
    specs = {
        "h100": [
            (old / "mwm_dense_tworoom_lr_horizon_h100_epochs_20260913/csv_logs/version_0/metrics.csv", range(0, 10)),
            (old / "mwm_dense_tworoom_h100_continue20_20260914/csv_logs/version_0/metrics.csv", range(10, 20)),
        ],
        "h10_tail": [
            (old / "mwm_dense_tworoom_lr_horizon_h10_epochs_20260913/csv_logs/version_0/metrics.csv", range(0, 10)),
            (old / "mwm_dense_tworoom_h10_tail15_20260914/csv_logs/version_0/metrics.csv", range(10, 15)),
        ],
    }
    payload: dict[str, object] = {"status": "pass", "metrics": list(METRICS), "branches": {}}
    for branch, segments in specs.items():
        rows = [row for path, epochs in segments for row in _validation_rows(path, epochs)]
        payload["branches"][branch] = {
            "rows": rows,
            "best_epochs_by_metric": {
                metric: [
                    row["epoch"]
                    for row in rows
                    if row["metrics"][metric] == min(candidate["metrics"][metric] for candidate in rows)
                ]
                for metric in METRICS
            },
            "terminal_minus_epoch9": {
                metric: rows[-1]["metrics"][metric] - rows[9]["metrics"][metric]
                for metric in METRICS
            },
        }
    return payload


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    terminal, terminal_rows = _collect_terminal()
    screen, screen_rows = _collect_screen()
    training = _collect_training()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "terminal_n500_summary.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (REPORT_ROOT / "epoch_screen_analysis.json").write_text(
        json.dumps(screen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (REPORT_ROOT / "training_trajectory.json").write_text(
        json.dumps(training, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(REPORT_ROOT / "terminal_n500_rows.csv", terminal_rows)
    _write_csv(REPORT_ROOT / "epoch_screen_rows.csv", screen_rows)
    print(json.dumps({"status": "pass", "terminal_rows": len(terminal_rows), "screen_rows": len(screen_rows)}))


if __name__ == "__main__":
    main()
