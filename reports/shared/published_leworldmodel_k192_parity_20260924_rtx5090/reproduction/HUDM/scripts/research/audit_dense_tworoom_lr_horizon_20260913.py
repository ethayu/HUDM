from __future__ import annotations

import copy
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import torch
from omegaconf import OmegaConf

from mwm.checkpoint_io import validate_checkpoint_directory
from mwm.io import file_sha256


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/dense_tworoom_lr_horizon_epochs_20260913"
HORIZONS = (10, 100)
EPOCHS = tuple(range(10))
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
RUN_NAMES = {
    10: "mwm_dense_tworoom_lr_horizon_h10_epochs_20260913",
    100: "mwm_dense_tworoom_lr_horizon_h100_epochs_20260913",
}
TRAINING_METRICS = (
    "validate/loss_epoch",
    "validate/pred_loss_epoch",
    "validate/recon_loss_epoch",
    "validate/sigreg_loss_epoch",
    *(f"validate/pred_loss_l{level}_epoch" for level in range(5)),
    *(f"validate/recon_loss_l{level}_epoch" for level in range(5)),
)
EPOCH_PATTERN = re.compile(r"^epoch=(?P<epoch>\d+)-step=(?P<step>\d+)\.ckpt$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolved_config(horizon: int) -> dict[str, Any]:
    path = ROOT / "checkpoints_mwm" / f"{RUN_NAMES[horizon]}_epoch_exports" / "resolved_training_config.yaml"
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    _require(isinstance(value, dict), f"Resolved config is not a mapping: {path}")
    return value


def _check_controlled_configs() -> dict[str, Any]:
    configs = {horizon: _resolved_config(horizon) for horizon in HORIZONS}
    for horizon, cfg in configs.items():
        _require(cfg["schedule"]["max_epochs"] == 10, f"h{horizon}: max_epochs is not 10")
        _require(cfg["schedule"]["lr_max_epochs"] == horizon, f"h{horizon}: LR horizon mismatch")
        _require(cfg["train"]["run_name"] == RUN_NAMES[horizon], f"h{horizon}: run name mismatch")
        _require(cfg["model"]["K"] == [96, 120, 144, 168, 192], f"h{horizon}: K mismatch")
        _require(cfg["seed"] == 3072, f"h{horizon}: seed mismatch")
    normalized: dict[int, dict[str, Any]] = {}
    execution_provenance: dict[str, dict[str, str]] = {}
    for horizon, cfg in configs.items():
        value = copy.deepcopy(cfg)
        value["schedule"]["lr_max_epochs"] = "<controlled>"
        value["train"]["run_name"] = "<controlled>"
        provenance = {
            "slurm.job_id": str(value.pop("slurm.job_id", "")),
            "slurm.task_id": str(value.pop("slurm.task_id", "")),
        }
        _require(all(provenance.values()), f"h{horizon}: missing Slurm execution provenance")
        execution_provenance[f"h{horizon}"] = provenance
        normalized[horizon] = value
    _require(
        normalized[10] == normalized[100],
        "Training configs differ beyond LR horizon, run name, and Slurm execution provenance",
    )
    return {
        "controlled_differences": ["schedule.lr_max_epochs", "train.run_name"],
        "excluded_execution_provenance": ["slurm.job_id", "slurm.task_id"],
        "execution_provenance": execution_provenance,
        "lr_max_epochs": {str(horizon): configs[horizon]["schedule"]["lr_max_epochs"] for horizon in HORIZONS},
    }


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _check_checkpoints() -> dict[str, Any]:
    totals = {"numbered_lightning": 0, "last_lightning": 0, "canonical": 0}
    branch_details: dict[str, Any] = {}
    for horizon in HORIZONS:
        run_name = RUN_NAMES[horizon]
        trainer_dir = ROOT / "logs/mwm_training" / run_name / "csv_logs/version_0/checkpoints"
        exports = ROOT / "checkpoints_mwm" / f"{run_name}_epoch_exports"
        numbered: list[tuple[int, int, Path]] = []
        for path in trainer_dir.glob("epoch=*-step=*.ckpt"):
            match = EPOCH_PATTERN.match(path.name)
            if match:
                numbered.append((int(match.group("epoch")), int(match.group("step")), path))
        numbered.sort()
        _require([epoch for epoch, _, _ in numbered] == list(EPOCHS), f"h{horizon}: numbered checkpoints incomplete")
        _require(all(step > 0 for _, step, _ in numbered), f"h{horizon}: non-positive checkpoint step")
        _require([step for _, step, _ in numbered] == sorted(step for _, step, _ in numbered), f"h{horizon}: steps not monotonic")
        _require((trainer_dir / "last.ckpt").is_file(), f"h{horizon}: last.ckpt missing")

        manifest_path = exports / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(isinstance(manifest, list) and len(manifest) == 10, f"h{horizon}: export manifest length mismatch")
        _require([int(row["epoch"]) for row in manifest] == list(EPOCHS), f"h{horizon}: manifest epochs mismatch")
        resolved_hash = file_sha256(exports / "resolved_training_config.yaml")
        for expected_epoch, row in zip(EPOCHS, manifest, strict=True):
            lightning = _resolve(str(row["lightning_checkpoint"]))
            canonical = _resolve(str(row["canonical_checkpoint"]))
            _require(lightning.is_file(), f"Missing Lightning checkpoint: {lightning}")
            _require(file_sha256(lightning) == row["lightning_sha256"], f"Lightning hash mismatch: {lightning}")
            _require(canonical == exports / f"epoch_{expected_epoch:03d}", f"Canonical path mismatch: {canonical}")
            _, metadata = validate_checkpoint_directory(canonical)
            _require(int(metadata.get("epoch", -1)) == expected_epoch, f"Canonical epoch mismatch: {canonical}")
            for filename, key in (
                ("weights.pt", "weights_sha256"),
                ("config.json", "config_sha256"),
                ("world_metadata.json", "metadata_sha256"),
            ):
                _require(file_sha256(canonical / filename) == row[key], f"Canonical hash mismatch: {canonical / filename}")
            _require(row["resolved_training_config_sha256"] == resolved_hash, f"Resolved-config hash mismatch: h{horizon}")

        actual_dirs = sorted(path.name for path in exports.glob("epoch_*" ) if path.is_dir())
        expected_dirs = [f"epoch_{epoch:03d}" for epoch in EPOCHS]
        _require(actual_dirs == expected_dirs, f"h{horizon}: unexpected canonical directory set")
        totals["numbered_lightning"] += len(numbered)
        totals["last_lightning"] += 1
        totals["canonical"] += len(actual_dirs)
        branch_details[str(horizon)] = {
            "numbered_epochs": [epoch for epoch, _, _ in numbered],
            "global_steps": [step for _, step, _ in numbered],
            "canonical_directories": actual_dirs,
            "manifest_sha256": file_sha256(manifest_path),
        }
    _require(totals == {"numbered_lightning": 20, "last_lightning": 2, "canonical": 20}, f"Checkpoint totals mismatch: {totals}")
    return {"totals": totals, "branches": branch_details}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_training_trajectory() -> dict[str, Any]:
    path = REPORT_ROOT / "training_trajectory.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("status") == "pass", "Training trajectory status is not pass")
    _require(payload.get("horizons") == list(HORIZONS), "Training trajectory horizons mismatch")
    _require(payload.get("epochs") == list(EPOCHS), "Training trajectory epochs mismatch")
    _require(payload.get("metrics") == list(TRAINING_METRICS), "Training trajectory metric set mismatch")
    branch_rows: dict[int, list[dict[str, Any]]] = {}
    for horizon in HORIZONS:
        branch = payload["branches"][f"h{horizon}"]
        _require(branch["run_name"] == RUN_NAMES[horizon], f"h{horizon}: training trajectory run mismatch")
        expected_csv = ROOT / "logs/mwm_training" / RUN_NAMES[horizon] / "csv_logs/version_0/metrics.csv"
        observed_csv = _resolve(str(branch["metrics_csv"]))
        _require(observed_csv == expected_csv, f"h{horizon}: metrics CSV path mismatch")
        _require(file_sha256(observed_csv) == branch["metrics_csv_sha256"], f"h{horizon}: metrics CSV hash mismatch")
        raw_rows = [row for row in _read_csv(observed_csv) if row.get("validate/loss_epoch")]
        raw_rows.sort(key=lambda row: int(row["epoch"]))
        rows = branch["rows"]
        _require(len(raw_rows) == len(rows) == len(EPOCHS), f"h{horizon}: validation row count mismatch")
        _require([int(row["epoch"]) for row in rows] == list(EPOCHS), f"h{horizon}: trajectory epochs mismatch")
        for raw, observed in zip(raw_rows, rows, strict=True):
            epoch = int(raw["epoch"])
            _require(int(observed["epoch"]) == epoch, f"h{horizon}: epoch alignment mismatch")
            _require(int(observed["step"]) == int(raw["step"]), f"h{horizon}/epoch{epoch}: step mismatch")
            _require(set(observed["metrics"]) == set(TRAINING_METRICS), f"h{horizon}/epoch{epoch}: metric keys mismatch")
            for metric in TRAINING_METRICS:
                _require(
                    float(observed["metrics"][metric]) == float(raw[metric]),
                    f"h{horizon}/epoch{epoch}/{metric}: metric mismatch",
                )
        branch_rows[horizon] = rows
        for metric in TRAINING_METRICS:
            values = [float(row["metrics"][metric]) for row in rows]
            expected_best = [epoch for epoch, value in zip(EPOCHS, values, strict=True) if value == min(values)]
            _require(
                branch["best_epochs_by_metric"][metric] == expected_best,
                f"h{horizon}/{metric}: best-epoch mismatch",
            )

    for metric in TRAINING_METRICS:
        for epoch in EPOCHS:
            expected_delta = (
                float(branch_rows[100][epoch]["metrics"][metric])
                - float(branch_rows[10][epoch]["metrics"][metric])
            )
            _require(
                abs(float(payload["h100_minus_h10_by_epoch"][metric][str(epoch)]) - expected_delta) <= 1e-12,
                f"epoch{epoch}/{metric}: training delta mismatch",
            )
        denominator = float(branch_rows[10][-1]["metrics"][metric])
        _require(denominator != 0.0, f"{metric}: zero h10 terminal denominator")
        expected_ratio = float(branch_rows[100][-1]["metrics"][metric]) / denominator
        _require(
            abs(float(payload["terminal_h100_over_h10"][metric]) - expected_ratio) <= 1e-12,
            f"{metric}: terminal training ratio mismatch",
        )
    return {
        "branches": len(HORIZONS),
        "epoch_rows": len(HORIZONS) * len(EPOCHS),
        "metrics_per_epoch": len(TRAINING_METRICS),
        "hash_bound_metrics_csvs": len(HORIZONS),
    }


def _check_terminal_scheduler_states() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for horizon in HORIZONS:
        trainer_dir = ROOT / "logs/mwm_training" / RUN_NAMES[horizon] / "csv_logs/version_0/checkpoints"
        candidates = list(trainer_dir.glob("epoch=9-step=*.ckpt"))
        _require(len(candidates) == 1, f"h{horizon}: expected one terminal numbered checkpoint")
        checkpoint = torch.load(candidates[0], map_location="cpu", weights_only=False, mmap=True)
        _require(int(checkpoint["epoch"]) == 9, f"h{horizon}: terminal checkpoint epoch mismatch")
        _require(int(checkpoint["global_step"]) == 116820, f"h{horizon}: terminal global step mismatch")
        schedulers = checkpoint.get("lr_schedulers", [])
        optimizers = checkpoint.get("optimizer_states", [])
        _require(len(schedulers) == len(optimizers) == 2, f"h{horizon}: optimizer/scheduler count mismatch")
        expected_max_steps = horizon * 5841
        expected_warmup_steps = max(1, int(0.01 * expected_max_steps))
        expected_last_epoch = 58410
        scheduler_rows: list[dict[str, Any]] = []
        for index, (scheduler, optimizer) in enumerate(zip(schedulers, optimizers, strict=True)):
            _require(int(scheduler["max_steps"]) == expected_max_steps, f"h{horizon}/scheduler{index}: max steps mismatch")
            _require(
                int(scheduler["warmup_steps"]) == expected_warmup_steps,
                f"h{horizon}/scheduler{index}: warmup steps mismatch",
            )
            _require(
                int(scheduler["last_epoch"]) == expected_last_epoch,
                f"h{horizon}/scheduler{index}: scheduler update count mismatch",
            )
            _require(
                int(scheduler["_step_count"]) == expected_last_epoch + 1,
                f"h{horizon}/scheduler{index}: scheduler step count mismatch",
            )
            base_lrs = [float(value) for value in scheduler["base_lrs"]]
            last_lrs = [float(value) for value in scheduler["_last_lr"]]
            optimizer_lrs = [float(group["lr"]) for group in optimizer["param_groups"]]
            _require(base_lrs == [5.0e-5], f"h{horizon}/scheduler{index}: base LR mismatch")
            if expected_last_epoch < expected_warmup_steps:
                expected_lr = base_lrs[0] * expected_last_epoch / expected_warmup_steps
            else:
                expected_lr = base_lrs[0] * (
                    1.0
                    + math.cos(
                        math.pi
                        * (expected_last_epoch - expected_warmup_steps)
                        / (expected_max_steps - expected_warmup_steps)
                    )
                ) / 2.0
            _require(len(last_lrs) == len(optimizer_lrs) == 1, f"h{horizon}/scheduler{index}: LR group mismatch")
            _require(abs(last_lrs[0] - expected_lr) <= 1e-15, f"h{horizon}/scheduler{index}: final LR mismatch")
            _require(abs(optimizer_lrs[0] - expected_lr) <= 1e-15, f"h{horizon}/optimizer{index}: final LR mismatch")
            scheduler_rows.append(
                {
                    "index": index,
                    "warmup_steps": expected_warmup_steps,
                    "max_steps": expected_max_steps,
                    "last_epoch": expected_last_epoch,
                    "base_lr": base_lrs[0],
                    "final_lr": expected_lr,
                    "final_lr_fraction": expected_lr / base_lrs[0],
                }
            )
        evidence[f"h{horizon}"] = {
            "checkpoint": str(candidates[0].relative_to(ROOT)),
            "checkpoint_epoch": 9,
            "global_step": int(checkpoint["global_step"]),
            "schedulers": scheduler_rows,
        }
        del checkpoint
    return evidence


def _check_rate(row: dict[str, Any]) -> None:
    episodes = int(row["episodes"])
    successes = int(row["successes"])
    _require(0 <= successes <= episodes, f"Invalid success count: {row}")
    _require(abs(float(row["success_rate"]) - successes / episodes * 100.0) <= 1e-9, f"Rate/count mismatch: {row}")


def _check_diagnostic_row(row: dict[str, Any]) -> None:
    role = str(row["role"])
    expected_terminal_k, expected_cost_prefix_k = ROLE_DIAGNOSTIC_CONTRACT[role]
    _require(int(row["diagnostic_trace_count"]) > 0, f"Empty diagnostic trace: {row}")
    _require(
        int(row["diagnostic_terminal_k"]) == expected_terminal_k,
        f"Terminal-K diagnostic mismatch: {row}",
    )
    observed_prefix = row.get("diagnostic_cost_prefix_k")
    if expected_cost_prefix_k is None:
        _require(observed_prefix in (None, ""), f"Unexpected cost-prefix diagnostic: {row}")
    else:
        _require(int(observed_prefix) == expected_cost_prefix_k, f"Cost-prefix diagnostic mismatch: {row}")


def _check_stat_summary(observed: dict[str, Any], values: list[float], label: str) -> None:
    _require(len(values) == len(SEEDS), f"{label}: expected {len(SEEDS)} paired values")
    center = mean(values)
    spread = stdev(values)
    half_width = T95_DF4 * spread / math.sqrt(len(values))
    expected_ci = [center - half_width, center + half_width]
    _require(abs(float(observed["mean"]) - center) <= 1e-9, f"{label}: mean mismatch")
    _require(abs(float(observed["sample_stdev"]) - spread) <= 1e-9, f"{label}: stdev mismatch")
    _require(
        all(abs(float(actual) - expected) <= 1e-9 for actual, expected in zip(observed["ci95"], expected_ci, strict=True)),
        f"{label}: CI mismatch",
    )
    _require(set(observed["by_seed"]) == {str(seed) for seed in SEEDS}, f"{label}: by-seed keys mismatch")
    for seed, expected in zip(SEEDS, values, strict=True):
        _require(
            abs(float(observed["by_seed"][str(seed)]) - expected) <= 1e-9,
            f"{label}: seed {seed} mismatch",
        )


def _check_epoch_screen() -> dict[str, Any]:
    json_rows = json.loads((REPORT_ROOT / "epoch_screen_summary.json").read_text(encoding="utf-8"))
    csv_rows = _read_csv(REPORT_ROOT / "epoch_screen_summary.csv")
    _require(len(json_rows) == len(csv_rows) == 100, "Epoch screen does not contain exactly 100 rows")
    cells = {(int(row["lr_horizon"]), int(row["epoch"]), row["role"]) for row in json_rows}
    expected = {(horizon, epoch, role) for horizon in HORIZONS for epoch in EPOCHS for role in ROLES}
    _require(cells == expected, "Epoch screen cell set mismatch")
    for row in json_rows:
        _require(int(row["episodes"]) == 50, f"Epoch-screen episode mismatch: {row}")
        _check_rate(row)
        _check_diagnostic_row(row)
        horizon = int(row["lr_horizon"])
        epoch = int(row["epoch"])
        expected_checkpoint = ROOT / "checkpoints_mwm" / f"{RUN_NAMES[horizon]}_epoch_exports" / f"epoch_{epoch:03d}"
        _require(_resolve(str(row["checkpoint"])) == expected_checkpoint, f"Epoch-screen checkpoint mismatch: {row}")
    _require(len({(row["manifest_sha256"], row["manifest_file_sha256"]) for row in json_rows}) == 1, "Epoch-screen manifests differ")

    analysis = json.loads((REPORT_ROOT / "epoch_screen_analysis.json").read_text(encoding="utf-8"))
    _require(analysis.get("status") == "pass", "Epoch-screen analysis status is not pass")
    _require(analysis.get("seed") == 42, "Epoch-screen analysis seed mismatch")
    _require(analysis.get("episodes_per_cell") == 50, "Epoch-screen analysis episode mismatch")
    _require(analysis.get("horizons") == list(HORIZONS), "Epoch-screen analysis horizons mismatch")
    _require(analysis.get("epochs") == list(EPOCHS), "Epoch-screen analysis epochs mismatch")
    _require(analysis.get("roles") == list(ROLES), "Epoch-screen analysis roles mismatch")
    rates = {
        (int(row["lr_horizon"]), int(row["epoch"]), str(row["role"])): float(row["success_rate"])
        for row in json_rows
    }
    for role in ROLES:
        role_analysis = analysis["trajectories"][role]
        for horizon in HORIZONS:
            observed = role_analysis[f"h{horizon}"]
            values = [rates[(horizon, epoch, role)] for epoch in EPOCHS]
            _require(
                [float(observed["by_epoch"][str(epoch)]) for epoch in EPOCHS] == values,
                f"h{horizon}/{role}: trajectory mismatch",
            )
            _require(float(observed["best_rate"]) == max(values), f"h{horizon}/{role}: best rate mismatch")
            _require(
                observed["best_epochs"] == [epoch for epoch, value in zip(EPOCHS, values, strict=True) if value == max(values)],
                f"h{horizon}/{role}: best epochs mismatch",
            )
            _require(float(observed["terminal_rate"]) == values[-1], f"h{horizon}/{role}: terminal rate mismatch")
            _require(
                abs(float(observed["terminal_minus_epoch8"]) - (values[-1] - values[-2])) <= 1e-9,
                f"h{horizon}/{role}: terminal change mismatch",
            )
        deltas = role_analysis["h100_minus_h10_by_epoch"]
        for epoch in EPOCHS:
            expected_delta = rates[(100, epoch, role)] - rates[(10, epoch, role)]
            _require(abs(float(deltas[str(epoch)]) - expected_delta) <= 1e-9, f"{role}: horizon trajectory mismatch")
        expected_terminal_delta = rates[(100, EPOCHS[-1], role)] - rates[(10, EPOCHS[-1], role)]
        _require(
            abs(float(role_analysis["terminal_h100_minus_h10"]) - expected_terminal_delta) <= 1e-9,
            f"{role}: terminal horizon contrast mismatch",
        )

    contrast_specs = {
        "fixed_k192_minus_fixed_k96": ("fixed_k192", "fixed_k96"),
        "fixed_k192_minus_fixed_k144": ("fixed_k192", "fixed_k144"),
        "k192_cost_k96_minus_full_k192": ("k192_cost_k96", "fixed_k192"),
        "k192_cost_k144_minus_full_k192": ("k192_cost_k144", "fixed_k192"),
    }
    for horizon in HORIZONS:
        observed_contrasts = analysis["within_horizon_contrasts_by_epoch"][f"h{horizon}"]
        _require(set(observed_contrasts) == set(contrast_specs), f"h{horizon}: screen contrast set mismatch")
        for name, (lhs, rhs) in contrast_specs.items():
            for epoch in EPOCHS:
                expected_delta = rates[(horizon, epoch, lhs)] - rates[(horizon, epoch, rhs)]
                _require(
                    abs(float(observed_contrasts[name][str(epoch)]) - expected_delta) <= 1e-9,
                    f"h{horizon}/{name}/epoch{epoch}: screen contrast mismatch",
                )
    return {
        "rows": len(json_rows),
        "cells": len(cells),
        "manifest_pairs": 1,
        "diagnostic_trace_entries": sum(int(row["diagnostic_trace_count"]) for row in json_rows),
        "trajectory_analysis": "validated",
    }


def _check_terminal() -> dict[str, Any]:
    rows = _read_csv(REPORT_ROOT / "terminal_n500_rows.csv")
    _require(len(rows) == 50, "Terminal confirmation does not contain exactly 50 rows")
    cells = {(int(row["lr_horizon"]), int(row["seed"]), row["role"]) for row in rows}
    expected = {(horizon, seed, role) for horizon in HORIZONS for seed in SEEDS for role in ROLES}
    _require(cells == expected, "Terminal confirmation cell set mismatch")
    manifest_by_seed: dict[int, set[tuple[str, str]]] = {seed: set() for seed in SEEDS}
    for row in rows:
        _require(int(row["episodes"]) == 500, f"Terminal episode mismatch: {row}")
        _check_rate(row)
        _check_diagnostic_row(row)
        horizon = int(row["lr_horizon"])
        seed = int(row["seed"])
        expected_checkpoint = ROOT / "checkpoints_mwm" / f"{RUN_NAMES[horizon]}_epoch_exports" / "epoch_009"
        _require(_resolve(row["checkpoint"]) == expected_checkpoint, f"Terminal checkpoint mismatch: {row}")
        manifest_by_seed[seed].add((row["manifest_sha256"], row["manifest_file_sha256"]))
    _require(all(len(values) == 1 for values in manifest_by_seed.values()), "Terminal manifests differ across horizons or roles")

    summary = json.loads((REPORT_ROOT / "terminal_n500_summary.json").read_text(encoding="utf-8"))
    _require(summary.get("status") == "pass", "Terminal summary status is not pass")
    _require(summary.get("horizons") == list(HORIZONS), "Terminal summary horizons mismatch")
    _require(summary.get("seeds") == list(SEEDS), "Terminal summary seeds mismatch")
    _require(summary.get("roles") == list(ROLES), "Terminal summary roles mismatch")
    _require(set(summary.get("horizon_effects", {})) == set(ROLES), "Missing paired horizon effects")
    rates = {
        (int(row["lr_horizon"]), int(row["seed"]), str(row["role"])): float(row["success_rate"])
        for row in rows
    }
    contrast_specs = {
        "fixed_k192_minus_fixed_k96": ("fixed_k192", "fixed_k96"),
        "fixed_k192_minus_fixed_k144": ("fixed_k192", "fixed_k144"),
        "k192_cost_k96_minus_full_k192": ("k192_cost_k96", "fixed_k192"),
        "k192_cost_k144_minus_full_k192": ("k192_cost_k144", "fixed_k192"),
    }
    expected_contrasts = set(contrast_specs)
    for role in ROLES:
        deltas = [rates[(100, seed, role)] - rates[(10, seed, role)] for seed in SEEDS]
        _check_stat_summary(summary["horizon_effects"][role], deltas, f"horizon effect/{role}")
    for horizon in HORIZONS:
        horizon_summary = summary["within_horizon"][str(horizon)]
        _require(
            set(horizon_summary["success_rates"]) == set(ROLES),
            f"h{horizon}: terminal role summaries mismatch",
        )
        _require(
            set(horizon_summary["contrasts"]) == expected_contrasts,
            f"h{horizon}: terminal contrasts mismatch",
        )
        for role in ROLES:
            values = [rates[(horizon, seed, role)] for seed in SEEDS]
            _check_stat_summary(horizon_summary["success_rates"][role], values, f"h{horizon}/{role}")
        for name, (lhs, rhs) in contrast_specs.items():
            values = [rates[(horizon, seed, lhs)] - rates[(horizon, seed, rhs)] for seed in SEEDS]
            _check_stat_summary(horizon_summary["contrasts"][name], values, f"h{horizon}/{name}")
    return {
        "rows": len(rows),
        "cells": len(cells),
        "paired_horizon_effects": len(ROLES),
        "diagnostic_trace_entries": sum(int(row["diagnostic_trace_count"]) for row in rows),
        "validated_statistical_summaries": len(ROLES) + len(HORIZONS) * (len(ROLES) + len(contrast_specs)),
    }


def main() -> None:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def run(name: str, operation: Callable[[], dict[str, Any]]) -> None:
        try:
            evidence = operation()
        except Exception as exc:  # Preserve every audit result in one artifact.
            checks.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
            failures.append(name)
        else:
            checks.append({"name": name, "status": "pass", "evidence": evidence})

    run("controlled_training_configs", _check_controlled_configs)
    run("training_validation_trajectories", _check_training_trajectory)
    run("terminal_scheduler_states", _check_terminal_scheduler_states)
    run("per_epoch_checkpoint_artifacts", _check_checkpoints)
    run("matched_epoch_screening", _check_epoch_screen)
    run("matched_terminal_confirmation", _check_terminal)

    payload = {
        "status": "pass" if not failures else "fail",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "ledger": "reports/research/dense_tworoom_lr_horizon_epochs_20260913/experiment_ledger.json",
        "checks": checks,
        "failures": failures,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = REPORT_ROOT / "completion_audit.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
