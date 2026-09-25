from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch

from mwm.io import file_sha256


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/dense_tworoom_h100_continue20_20260914"
TRAINING_ROOT = ROOT / "logs/mwm_training"
CHECKPOINT_ROOT = ROOT / "checkpoints_mwm"
ROLES = {"fixed_k96", "fixed_k144", "fixed_k192", "k192_cost_k96", "k192_cost_k144"}
ORDERED_ROLES = ("fixed_k96", "fixed_k144", "fixed_k192", "k192_cost_k96", "k192_cost_k144")
ROLE_CONTRACT = {
    "fixed_k96": (96, None),
    "fixed_k144": (144, None),
    "fixed_k192": (192, None),
    "k192_cost_k96": (192, 96),
    "k192_cost_k144": (192, 144),
}
SEEDS = (0, 1, 2, 42, 100)
T95_DF4 = 2.7764451051977987


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty file: {path}")


def _audit_resume_inputs() -> dict[str, object]:
    h100_source = (
        TRAINING_ROOT
        / "mwm_dense_tworoom_lr_horizon_h100_epochs_20260913"
        / "csv_logs/version_0/checkpoints/epoch=9-step=116820.ckpt"
    )
    h100_copy = REPORT_ROOT / "resume_inputs/h100/epoch=9-step=116820.ckpt"
    h100_provenance = json.loads(
        (REPORT_ROOT / "resume_inputs/h100/provenance.json").read_text(encoding="utf-8")
    )
    h100_source_hash = file_sha256(h100_source)
    h100_copy_hash = file_sha256(h100_copy)
    if h100_source_hash != h100_provenance["source_checkpoint_sha256"]:
        raise RuntimeError("The h100 resume hash does not match provenance")
    if h100_source_hash != h100_provenance["initial_isolated_copy_sha256"]:
        raise RuntimeError("The recorded initial h100 isolated-copy hash does not match its source")
    h100_log = ROOT / "logs/train_tr_h100_e20_8347223.out"
    launch_line = next(
        line for line in h100_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("{'resume_contract':")
    )
    h100_launch_contract = ast.literal_eval(launch_line)["resume_contract"]
    if h100_launch_contract != {
        "epoch": 9,
        "global_step": 116820,
        "optimizer_count": 2,
        "scheduler_count": 2,
        "scheduler_last_epochs": [58410, 58410],
        "scheduler_max_steps": [584100, 584100],
        "sha256": h100_source_hash,
    }:
        raise RuntimeError(f"Unexpected h100 launch contract: {h100_launch_contract}")

    h10_source = (
        TRAINING_ROOT
        / "mwm_dense_tworoom_lr_horizon_h10_epochs_20260913"
        / "csv_logs/version_0/checkpoints/epoch=9-step=116820.ckpt"
    )
    h10_derived = REPORT_ROOT / "resume_inputs/h10_epoch009_tail15.ckpt"
    h10_provenance = json.loads(
        (REPORT_ROOT / "resume_inputs/h10_epoch009_tail15.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if file_sha256(h10_source) != h10_provenance["source_checkpoint_sha256"]:
        raise RuntimeError("The h10 source checkpoint changed after tail derivation")
    h10_log = ROOT / "logs/train_tr_h10_tail15_8347131.out"
    h10_launch_line = next(
        line for line in h10_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("{'resume_contract':")
    )
    h10_launch_contract = ast.literal_eval(h10_launch_line)["resume_contract"]
    if h10_launch_contract != {
        "epoch": 9,
        "global_step": 116820,
        "optimizer_count": 2,
        "scheduler_count": 2,
        "scheduler_last_epochs": [58410, 58410],
        "scheduler_max_steps": [87615, 87615],
    }:
        raise RuntimeError(f"Unexpected h10-tail launch contract: {h10_launch_contract}")

    overwritten = {
        "h100": (
            h100_copy,
            TRAINING_ROOT / "mwm_dense_tworoom_h100_continue20_20260914/csv_logs/version_0/checkpoints/last.ckpt",
            19,
            233640,
        ),
        "h10_tail": (
            h10_derived,
            TRAINING_ROOT / "mwm_dense_tworoom_h10_tail15_20260914/csv_logs/version_0/checkpoints/last.ckpt",
            14,
            175230,
        ),
    }
    overwritten_hashes: dict[str, str] = {}
    for branch, (resume_path, terminal_path, epoch, step) in overwritten.items():
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        terminal_state = torch.load(terminal_path, map_location="cpu", weights_only=False)
        if int(resume_state.get("epoch", -1)) != epoch or int(resume_state.get("global_step", -1)) != step:
            raise RuntimeError(f"The automatic saver did not leave terminal state at {resume_path}")
        if resume_state["state_dict"].keys() != terminal_state["state_dict"].keys() or not all(
            torch.equal(resume_state["state_dict"][key], terminal_state["state_dict"][key])
            for key in resume_state["state_dict"]
        ):
            raise RuntimeError(f"Automatic-saver terminal weights differ for {branch}")
        overwritten_hashes[branch] = file_sha256(resume_path)
    if overwritten_hashes["h100"] != h100_provenance["post_training_isolated_path_sha256"]:
        raise RuntimeError("The post-training h100 isolated-path hash does not match provenance")
    return {
        "h100_source_sha256": h100_source_hash,
        "h100_initial_isolated_copy_sha256": h100_provenance["initial_isolated_copy_sha256"],
        "h100_launch_contract": h100_launch_contract,
        "h100_post_training_isolated_path_sha256": h100_copy_hash,
        "h10_source_sha256": h10_provenance["source_checkpoint_sha256"],
        "h10_initial_derived_sha256": h10_provenance["derived_checkpoint_sha256"],
        "h10_launch_contract": h10_launch_contract,
        "h10_post_training_resume_path_sha256": overwritten_hashes["h10_tail"],
        "h10_tail_resume_lr": h10_provenance["resume_lr"],
        "automatic_saver_overwrite": "isolated resume paths now contain terminal state; original sources remain hash-identical",
    }


def _audit_training_run(
    run_name: str,
    expected_epochs: range,
    terminal_epoch: int,
    terminal_global_step: int,
    scheduler_last_epoch: int,
    scheduler_max_steps: int,
) -> dict[str, object]:
    trainer_dir = TRAINING_ROOT / run_name / "csv_logs/version_0/checkpoints"
    export_root = CHECKPOINT_ROOT / f"{run_name}_epoch_exports"
    _require_file(trainer_dir / "last.ckpt")
    numbered = sorted(trainer_dir.glob("epoch=*-step=*.ckpt"))
    expected_names = [f"epoch={epoch}-step={(epoch + 1) * 11682}.ckpt" for epoch in expected_epochs]
    if [path.name for path in numbered] != expected_names:
        raise RuntimeError(
            f"Unexpected numbered checkpoints for {run_name}: {[path.name for path in numbered]}"
        )
    manifest_path = export_root / "manifest.json"
    _require_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if [row["epoch"] for row in manifest] != list(expected_epochs):
        raise RuntimeError(f"Incomplete canonical export manifest for {run_name}")
    for row in manifest:
        epoch = int(row["epoch"])
        canonical = ROOT / row["canonical_checkpoint"]
        for name in ("config.json", "weights.pt", "world_metadata.json"):
            _require_file(canonical / name)
        metadata = json.loads((canonical / "world_metadata.json").read_text(encoding="utf-8"))
        if int(metadata.get("epoch", -1)) != epoch:
            raise RuntimeError(f"Canonical epoch metadata mismatch in {canonical}")
        if metadata.get("levels") != [96, 120, 144, 168, 192]:
            raise RuntimeError(f"Canonical K levels mismatch in {canonical}")
        if file_sha256(canonical / "weights.pt") != row["weights_sha256"]:
            raise RuntimeError(f"Canonical weight hash mismatch in {canonical}")
    terminal = torch.load(trainer_dir / "last.ckpt", map_location="cpu", weights_only=False)
    schedulers = terminal.get("lr_schedulers", [])
    actual = {
        "epoch": int(terminal.get("epoch", -1)),
        "global_step": int(terminal.get("global_step", -1)),
        "optimizer_count": len(terminal.get("optimizer_states", [])),
        "scheduler_count": len(schedulers),
        "scheduler_last_epochs": [int(item.get("last_epoch", -1)) for item in schedulers],
        "scheduler_max_steps": [int(item.get("max_steps", -1)) for item in schedulers],
    }
    expected = {
        "epoch": terminal_epoch,
        "global_step": terminal_global_step,
        "optimizer_count": 2,
        "scheduler_count": 2,
        "scheduler_last_epochs": [scheduler_last_epoch, scheduler_last_epoch],
        "scheduler_max_steps": [scheduler_max_steps, scheduler_max_steps],
    }
    if actual != expected:
        raise RuntimeError(f"Terminal checkpoint mismatch for {run_name}: {actual} != {expected}")
    return {
        "numbered_lightning_checkpoints": len(numbered),
        "canonical_epoch_exports": len(manifest),
        "terminal_contract": actual,
    }


def _audit_eval_dir(path: Path, episodes: int, seed: int) -> None:
    _require_file(path / "summary.csv")
    with (path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row["role"] for row in rows} != ROLES:
        raise RuntimeError(f"Role mismatch in {path}")
    if any(int(row["episodes"]) != episodes or int(row["seed"]) != seed for row in rows):
        raise RuntimeError(f"Evaluation size/seed mismatch in {path}")
    for row in rows:
        role = row["role"]
        output = Path(row["output_json"])
        if not output.is_absolute():
            output = ROOT / output
        payload = json.loads(output.read_text(encoding="utf-8"))
        outcomes = payload.get("swm_results", {}).get("episode_successes", [])
        if not isinstance(outcomes, list) or len(outcomes) != episodes:
            raise RuntimeError(f"Episode-outcome coverage mismatch in {output}")
        observed_rate = sum(bool(value) for value in outcomes) / episodes * 100.0
        if not math.isclose(observed_rate, float(row["success_rate"]), abs_tol=1e-12):
            raise RuntimeError(f"Success-rate mismatch in {output}")
        terminal_k, prefix_k = ROLE_CONTRACT[role]
        diagnostics = payload.get("planning_diagnostics", {})
        trace = diagnostics.get("trace", [])
        if not isinstance(trace, list) or not trace:
            raise RuntimeError(f"Missing planning trace in {output}")
        if len(trace) != int(diagnostics.get("summary", {}).get("cem_cost_calls", -1)):
            raise RuntimeError(f"Planning trace/cost-call mismatch in {output}")
        for item in trace:
            for key in ("base_k", "terminal_k", "model_base_k", "model_terminal_k"):
                if int(item.get(key, -1)) != terminal_k:
                    raise RuntimeError(f"{role} {key} mismatch in {output}")
            if item.get("model_prefix_criterion") is not True:
                raise RuntimeError(f"Prefix criterion missing in {output}")
            observed_prefix = item.get("model_research_cost_prefix_k")
            if prefix_k is None and observed_prefix is not None:
                raise RuntimeError(f"Unexpected research cost prefix in {output}")
            if prefix_k is not None and int(observed_prefix or -1) != prefix_k:
                raise RuntimeError(f"Research cost prefix mismatch in {output}")


def _audit_evaluations() -> dict[str, int]:
    counts = {"h100_screen_cells": 0, "h10_tail_screen_cells": 0, "terminal_cells": 0}
    for epoch in range(10, 20):
        _audit_eval_dir(REPORT_ROOT / f"screen/epoch_{epoch:03d}", 50, 42)
        counts["h100_screen_cells"] += 1
    for epoch in range(10, 15):
        _audit_eval_dir(REPORT_ROOT / f"h10_tail_screen/epoch_{epoch:03d}", 50, 42)
        counts["h10_tail_screen_cells"] += 1
    for epoch in (14, 19):
        for seed in (0, 1, 2, 42, 100):
            _audit_eval_dir(REPORT_ROOT / f"terminal_n500/epoch_{epoch:03d}/seed{seed}", 500, seed)
            counts["terminal_cells"] += 1
    for seed in (0, 1, 2, 42, 100):
        _audit_eval_dir(REPORT_ROOT / f"h10_tail_terminal_n500/epoch_014/seed{seed}", 500, seed)
        counts["terminal_cells"] += 1
    return counts


def _expected_summary(samples: list[float]) -> dict[str, object]:
    center = mean(samples)
    spread = stdev(samples)
    half_width = T95_DF4 * spread / math.sqrt(len(samples))
    return {
        "mean": center,
        "sample_stdev": spread,
        "ci95": [center - half_width, center + half_width],
        "by_seed": dict(zip((str(seed) for seed in SEEDS), samples, strict=True)),
    }


def _assert_summary(actual: dict[str, object], samples: list[float], label: str) -> None:
    expected = _expected_summary(samples)
    for key in ("mean", "sample_stdev"):
        if not math.isclose(float(actual[key]), float(expected[key]), abs_tol=1e-12):
            raise RuntimeError(f"{label} {key} mismatch: {actual[key]} != {expected[key]}")
    if any(
        not math.isclose(float(observed), float(reference), abs_tol=1e-12)
        for observed, reference in zip(actual["ci95"], expected["ci95"], strict=True)
    ):
        raise RuntimeError(f"{label} ci95 mismatch")
    if actual["by_seed"] != expected["by_seed"]:
        raise RuntimeError(f"{label} per-seed values mismatch")


def _audit_statistics() -> dict[str, int]:
    rows_path = REPORT_ROOT / "terminal_n500_rows.csv"
    with rows_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    branches = ("h10_epoch9", "h100_epoch9", "h10_tail_epoch14", "h100_epoch14", "h100_epoch19")
    if len(rows) != len(branches) * len(SEEDS) * len(ORDERED_ROLES):
        raise RuntimeError(f"Unexpected terminal row count: {len(rows)}")
    values: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (row["branch"], row["role"], int(row["seed"]))
        if key in values:
            raise RuntimeError(f"Duplicate terminal row: {key}")
        if key[0] not in branches or key[1] not in ROLES or key[2] not in SEEDS:
            raise RuntimeError(f"Unexpected terminal key: {key}")
        rate = float(row["success_rate"])
        if int(row["episodes"]) != 500 or int(row["successes"]) != round(rate * 5):
            raise RuntimeError(f"Terminal count mismatch: {key}")
        values[key] = rate
    summary = json.loads((REPORT_ROOT / "terminal_n500_summary.json").read_text(encoding="utf-8"))
    checked = 0
    contrast_specs = {
        "fixed_k192_minus_fixed_k96": ("fixed_k192", "fixed_k96"),
        "fixed_k192_minus_fixed_k144": ("fixed_k192", "fixed_k144"),
        "k192_cost_k96_minus_full_k192": ("k192_cost_k96", "fixed_k192"),
        "k192_cost_k144_minus_full_k192": ("k192_cost_k144", "fixed_k192"),
    }
    for branch in branches:
        branch_summary = summary["within_branch"][branch]
        for role in ORDERED_ROLES:
            samples = [values[(branch, role, seed)] for seed in SEEDS]
            _assert_summary(branch_summary["success_rates"][role], samples, f"{branch}/{role}")
            checked += 1
        for name, (lhs, rhs) in contrast_specs.items():
            samples = [values[(branch, lhs, seed)] - values[(branch, rhs, seed)] for seed in SEEDS]
            _assert_summary(branch_summary["contrasts"][name], samples, f"{branch}/{name}")
            checked += 1
    effect_specs = {
        "h100_epoch14_minus_epoch9": ("h100_epoch14", "h100_epoch9"),
        "h100_epoch19_minus_epoch9": ("h100_epoch19", "h100_epoch9"),
        "h100_epoch19_minus_epoch14": ("h100_epoch19", "h100_epoch14"),
        "h10_tail_epoch14_minus_h10_epoch9": ("h10_tail_epoch14", "h10_epoch9"),
        "epoch14_h100_minus_h10_tail": ("h100_epoch14", "h10_tail_epoch14"),
    }
    for name, (lhs, rhs) in effect_specs.items():
        for role in ORDERED_ROLES:
            samples = [values[(lhs, role, seed)] - values[(rhs, role, seed)] for seed in SEEDS]
            _assert_summary(summary["paired_effects"][name][role], samples, f"{name}/{role}")
            checked += 1
    with (REPORT_ROOT / "epoch_screen_rows.csv").open(encoding="utf-8", newline="") as handle:
        screen_rows = list(csv.DictReader(handle))
    if len(screen_rows) != 85:
        raise RuntimeError(f"Unexpected screen row count: {len(screen_rows)}")
    screen_keys = {(row["branch"], int(row["epoch"]), row["role"]) for row in screen_rows}
    if len(screen_keys) != len(screen_rows):
        raise RuntimeError("Duplicate screen summary rows")
    return {"terminal_rows": len(rows), "screen_rows": len(screen_rows), "statistical_summaries": checked}


def _audit_training_collection() -> dict[str, int]:
    payload = json.loads((REPORT_ROOT / "training_trajectory.json").read_text(encoding="utf-8"))
    specs = {"h100": range(20), "h10_tail": range(15)}
    checked = 0
    for branch, epochs in specs.items():
        rows = payload["branches"][branch]["rows"]
        if [int(row["epoch"]) for row in rows] != list(epochs):
            raise RuntimeError(f"Training trajectory epochs mismatch for {branch}")
        if any(int(row["step"]) != (int(row["epoch"]) + 1) * 5841 - 1 for row in rows):
            raise RuntimeError(f"Training trajectory step mismatch for {branch}")
        checked += len(rows)
    return {"validation_epoch_rows": checked}


def main() -> None:
    for filename in (
        "terminal_n500_summary.json",
        "epoch_screen_analysis.json",
        "training_trajectory.json",
        "terminal_n500_rows.csv",
        "epoch_screen_rows.csv",
    ):
        _require_file(REPORT_ROOT / filename)
    for filename in ("terminal_n500_summary.json", "epoch_screen_analysis.json", "training_trajectory.json"):
        payload = json.loads((REPORT_ROOT / filename).read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise RuntimeError(f"Collector did not pass: {filename}")
    payload = {
        "status": "pass",
        "resume_inputs": _audit_resume_inputs(),
        "training": {
            "h100": _audit_training_run(
                "mwm_dense_tworoom_h100_continue20_20260914",
                range(10, 20),
                terminal_epoch=19,
                terminal_global_step=233640,
                scheduler_last_epoch=116820,
                scheduler_max_steps=584100,
            ),
            "h10_tail": _audit_training_run(
                "mwm_dense_tworoom_h10_tail15_20260914",
                range(10, 15),
                terminal_epoch=14,
                terminal_global_step=175230,
                scheduler_last_epoch=87615,
                scheduler_max_steps=87615,
            ),
        },
        "evaluations": _audit_evaluations(),
        "statistics": _audit_statistics(),
        "training_collection": _audit_training_collection(),
    }
    destination = REPORT_ROOT / "completion_audit.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
