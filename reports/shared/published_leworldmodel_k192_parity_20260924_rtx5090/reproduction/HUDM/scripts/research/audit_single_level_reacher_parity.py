from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = Path("reports/research/single_level_reacher_parity_20260806")
ARCH_REPORT = Path("reports/research/single_level_upstream_parity_20260806/report.json")
OPTIMIZER_ORDER_REPORT = REPORT_ROOT / "optimizer_parameter_order_forensics.json"
WORLD_PARAMETER_COUNT = 18_034_478
HDF5_SHA256 = "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06"
EVAL_SEEDS = (0, 1, 2, 42, 100)
MODEL_INIT_SEEDS = (0, 1, 2, 42, 100, 3072)
SWEEP_EXECUTION_FILES = (
    "scripts/research/slurm_launch_reacher_model_init_sweep_20260806.sbatch",
    "scripts/research/launch_reacher_model_init_sweep.py",
    "scripts/research/slurm_research_train_reacher_model_init_sweep_20260806.sbatch",
    "scripts/research/slurm_research_eval_reacher_model_init_sweep_20260806.sbatch",
    "scripts/research/research_reacher_checkpoint_n500.sh",
    "scripts/research/slurm_collect_reacher_model_init_n500.sbatch",
    "scripts/research/collect_reacher_model_init_n500.py",
    "scripts/research/slurm_finalize_single_level_reacher_parity_20260806.sbatch",
    "scripts/research/audit_single_level_reacher_parity.py",
)
BRANCHES = {
    "paper10": {
        "job_id": 7434499,
        "slurm_log": "logs/train_reacher_paper10_7434499.out",
        "run_name": "mwm_reacher_k192_paper10_nativeh5_nodecoder_init3072_fit0_20260806",
        "lr_max_epochs": 10,
        "report_prefix": "single_level_reacher_model_init_paper10_parity_20260806",
        "candidate_role": "paper10_nativeh5_nodecoder_init3072_fit0",
        "paper_report_tag": "single_level_reacher_paper10_nativeh5_parity_20260806",
        "public_report_tag": (
            "single_level_reacher_paper10_nativeh5_publiccem30_parity_20260806"
        ),
    },
    "epoch10_horizon100": {
        "job_id": 7434500,
        "slurm_log": "logs/train_reacher_h100_7434500.out",
        "run_name": (
            "mwm_reacher_k192_epoch10_horizon100_nativeh5_nodecoder_init3072_fit0_20260806"
        ),
        "lr_max_epochs": 100,
        "report_prefix": (
            "single_level_reacher_model_init_epoch10_horizon100_parity_20260806"
        ),
        "candidate_role": "epoch10_horizon100_nativeh5_nodecoder_init3072_fit0",
        "paper_report_tag": (
            "single_level_reacher_epoch10_horizon100_nativeh5_parity_20260806"
        ),
        "public_report_tag": (
            "single_level_reacher_epoch10_horizon100_nativeh5_publiccem30_parity_20260806"
        ),
    },
}
TEST_FILES = (
    "tests/test_mwm_core.py",
    "tests/test_mwm_artifacts.py",
    "tests/test_mwm_repo_hygiene.py",
    "tests/test_mwm_swm_env_runtime.py",
    "tests/test_mwm_spt_optimizer_parity.py",
    "tests/test_research_reacher_scheduler_collector.py",
    "tests/test_research_reacher_init_launcher.py",
    "tests/test_research_reacher_model_init_collector.py",
    "tests/test_research_single_level_reacher_completion_audit.py",
)


class EvidenceError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceError(f"Expected a JSON object in {path}")
    return payload


def _load_lightning_training_config(path: Path, *, label: str) -> dict[str, Any]:
    """Read the resolved training recipe saved by Lightning/SPT.

    Canonical ``config.json`` is intentionally a model-construction contract,
    not the trainer configuration.  The latter is stored under
    ``hyper_parameters`` in ``last.ckpt`` and is the authoritative record for
    data, optimizer, scheduler, loader, and seed settings.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(config, dict) or not config:
        raise EvidenceError(f"{label} Lightning checkpoint has no resolved training config")
    return config


def _validate_canonical_export_directory(
    path: Path, *, label: str, lightning_checkpoint: Path
) -> None:
    from mwm.checkpoint_io import validate_checkpoint_directory

    try:
        _, metadata = validate_checkpoint_directory(
            path, strict_artifacts=True, strict_metadata=True
        )
    except Exception as exc:
        raise EvidenceError(f"{label} canonical export contract failed: {exc}") from exc
    expected_metadata = {
        "epoch": 9,
        "best_model_score": None,
        "checkpoint_monitor": None,
        "export_checkpoint": "last",
    }
    mismatches = {
        field: {"actual": metadata.get(field), "expected": expected}
        for field, expected in expected_metadata.items()
        if metadata.get(field) != expected
    }
    if mismatches:
        raise EvidenceError(f"{label} checkpoint-selection metadata mismatch: {mismatches}")
    root = path.resolve().parents[1]
    expected_checkpoint = lightning_checkpoint.resolve()
    _validate_unmonitored_best_checkpoint(
        metadata,
        label=label,
        root=root,
        terminal_numbered_checkpoint=(
            lightning_checkpoint.parent / "epoch=9-step=127960.ckpt"
        ),
    )
    for field in ("last_checkpoint", "selected_lightning_checkpoint"):
        recorded = metadata.get(field)
        if not recorded or (root / str(recorded)).resolve() != expected_checkpoint:
            raise EvidenceError(
                f"{label} {field} does not identify the exported last.ckpt: {recorded!r}"
            )


def _validate_unmonitored_best_checkpoint(
    metadata: dict[str, Any],
    *,
    label: str,
    root: Path,
    terminal_numbered_checkpoint: Path,
) -> None:
    """Validate Lightning's unmonitored ``best_model_path`` bookkeeping.

    Depending on the Lightning version, an unmonitored ``ModelCheckpoint``
    reports either no best path or the terminal numbered checkpoint as its
    ``best_model_path``.  Neither denotes metric-based model selection when
    both ``monitor`` and ``best_model_score`` are null.  A non-null path is
    accepted only when it resolves to the exact terminal checkpoint.
    """
    recorded = metadata.get("best_checkpoint")
    if recorded is None:
        return
    actual = (root / str(recorded)).resolve()
    expected = terminal_numbered_checkpoint.resolve()
    if actual != expected:
        raise EvidenceError(
            f"{label} unmonitored best_checkpoint is not the terminal numbered "
            f"checkpoint: {recorded!r}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is not finite: {value!r}")
    return result


def _requirement(
    root: Path,
    name: str,
    relative_paths: tuple[Path, ...],
    validator: Callable[[list[dict[str, Any]], list[Path]], dict[str, Any] | None],
) -> dict[str, Any]:
    paths = [root / path for path in relative_paths]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        return {"name": name, "status": "pending", "missing": missing}
    try:
        payloads = [_load_json(path) for path in paths]
        detail = validator(payloads, paths) or {}
    except Exception as exc:
        return {
            "name": name,
            "status": "fail",
            "evidence": [str(path.relative_to(root)) for path in paths],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "name": name,
        "status": "pass",
        "evidence": [str(path.relative_to(root)) for path in paths],
        **detail,
    }


def _architecture(payloads: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    payload = payloads[0]
    if payload.get("status") != "pass":
        raise EvidenceError("architecture report did not pass")
    for field in ("world_parameter_count", "fresh_world_parameter_count"):
        if int(payload.get(field, -1)) != WORLD_PARAMETER_COUNT:
            raise EvidenceError(f"{field} is not {WORLD_PARAMETER_COUNT}")
    exact_fields = (
        "weights",
        "fresh_initialization",
        "encode",
        "action_encode",
        "predict",
        "pred_loss",
        "sigreg_loss",
        "world_loss",
        "preclip_gradient_norm",
    )
    maxima = payload.get("max_abs", {})
    for field in exact_fields:
        if float(maxima.get(field, math.inf)) != 0.0:
            raise EvidenceError(f"architecture parity field {field} is not bitwise exact")
    fp32_tolerances = {
        # Raw and wrapped graphs have different allocation layouts. Even with
        # deterministic math SDPA, B200 MIG profiles vary at the last FP32
        # reduction bits; the BF16 training-path gate below remains exact.
        "rollout": 2e-6,
        "planning_cost": 5e-5,
        "world_gradients": 1e-7,
        "post_adamw_update": 3e-8,
    }
    for field, tolerance in fp32_tolerances.items():
        if float(maxima.get(field, math.inf)) > tolerance:
            raise EvidenceError(f"{field} exceeds the {tolerance:g} deterministic tolerance")
    bf16 = payload.get("current_runtime_bf16", {})
    bf16_maxima = bf16.get("max_abs", {})
    bf16_exact_fields = (
        "pred_loss",
        "sigreg_loss",
        "world_loss",
        "world_gradients",
        "preclip_gradient_norm",
        "upstream_order_preclip_gradient_norm",
        "post_adamw_update",
    )
    for field in bf16_exact_fields:
        if float(bf16_maxima.get(field, math.inf)) != 0.0:
            raise EvidenceError(f"current-runtime BF16 parity field {field} is not bitwise exact")
    module_order_delta = _finite(
        bf16_maxima.get("module_order_preclip_gradient_norm"),
        "current-runtime natural module-order norm diagnostic",
    )
    if module_order_delta < 0.0:
        raise EvidenceError("natural module-order norm diagnostic cannot be negative")
    if bf16.get("autocast_dtype") != "torch.bfloat16":
        raise EvidenceError("current-runtime training-step parity did not use BF16 autocast")

    expected_boundaries = [
        (0, "encoder.embeddings.cls_token", [1, 1, 192]),
        (198, "predictor.pos_embedding", [1, 3, 192]),
        (279, "action_encoder.patch_embed.weight", [10, 10, 1]),
        (285, "projector.net.0.weight", [2048, 192]),
        (291, "pred_proj.net.0.weight", [2048, 192]),
    ]
    layout = payload.get("upstream_optimizer_parameter_layout", {})
    actual_boundaries = [
        (row.get("index"), row.get("raw_name"), row.get("shape"))
        for row in layout.get("module_boundaries", [])
    ]
    if int(layout.get("tensor_count", -1)) != 297 or actual_boundaries != expected_boundaries:
        raise EvidenceError("raw LeWM optimizer parameter layout does not match the 297-tensor contract")
    last_tensor = layout.get("last_tensor", {})
    if (
        last_tensor.get("index"),
        last_tensor.get("raw_name"),
        last_tensor.get("shape"),
    ) != (296, "pred_proj.net.3.bias", [192]):
        raise EvidenceError("raw LeWM optimizer parameter layout has the wrong final tensor")

    causal = payloads[1]
    if causal.get("status") != "pass":
        raise EvidenceError("optimizer-order causal forensics did not pass")
    natural = causal.get("natural_mwm_order_observation", {})
    ordered = causal.get("upstream_order_observation", {})
    if _finite(natural.get("bf16_loss_max_abs"), "natural-order BF16 loss") != 0.0:
        raise EvidenceError("natural-order causal run did not preserve exact BF16 loss")
    if _finite(natural.get("bf16_per_tensor_gradient_max_abs"), "natural-order gradients") != 0.0:
        raise EvidenceError("natural-order causal run did not preserve every BF16 gradient")
    if _finite(natural.get("preclip_gradient_norm_max_abs"), "natural-order clip norm") <= 0.0:
        raise EvidenceError("natural-order causal run did not expose a clipping discrepancy")
    if _finite(natural.get("post_adamw_update_max_abs"), "natural-order AdamW update") <= 0.0:
        raise EvidenceError("natural-order causal run did not expose an update discrepancy")
    for field in (
        "bf16_loss_max_abs",
        "bf16_per_tensor_gradient_max_abs",
        "preclip_gradient_norm_max_abs",
        "post_adamw_update_max_abs",
    ):
        if _finite(ordered.get(field), f"upstream-order {field}") != 0.0:
            raise EvidenceError(f"upstream-order causal field {field} is not exact")
    audit_root = paths[0].parents[3]
    for section in (natural, ordered, causal.get("profile_dependence_control", {})):
        source_log = audit_root / str(section.get("source_log", ""))
        if not source_log.is_file() or _sha256(source_log) != section.get("source_log_sha256"):
            raise EvidenceError(f"optimizer-order source log is absent or changed: {source_log}")
    trajectory = causal.get("same_mig_training_trajectory_observation", {})
    expected_device = trajectory.get("cuda_visible_device")
    for field in ("superseded_environment", "upstream_ordered_environment"):
        environment = _load_json(audit_root / str(trajectory.get(field, "")))
        actual_device = environment.get("environment_variables", {}).get("CUDA_VISIBLE_DEVICES")
        if actual_device != expected_device:
            raise EvidenceError(
                f"optimizer-order trajectory is not a same-MIG comparison: {actual_device!r}"
            )
    metric_rows: list[dict[str, str]] = []
    for field in ("superseded_metrics", "upstream_ordered_metrics"):
        with (audit_root / str(trajectory.get(field, ""))).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            first_row = next(csv.DictReader(handle), None)
        if first_row is None:
            raise EvidenceError(f"optimizer-order trajectory metrics are empty: {field}")
        metric_rows.append(first_row)
    old_row, new_row = metric_rows
    expected_first_step = int(trajectory.get("first_logged_step", -1))
    if int(old_row.get("step", -1)) != expected_first_step or int(new_row.get("step", -1)) != expected_first_step:
        raise EvidenceError("optimizer-order trajectories do not begin at the asserted common step")
    recorded_deltas = trajectory.get("new_minus_superseded", {})
    for metric in ("fit/loss", "fit/pred_loss", "fit/sigreg_loss"):
        actual_delta = float(new_row[metric]) - float(old_row[metric])
        if actual_delta != float(recorded_deltas.get(metric, math.inf)):
            raise EvidenceError(
                f"same-MIG optimizer-order trajectory delta changed for {metric}: {actual_delta}"
            )
    if not any(float(value) != 0.0 for value in recorded_deltas.values()):
        raise EvidenceError("same-MIG optimizer-order trajectories did not diverge")
    epoch_rows: list[dict[str, str]] = []
    for field in ("superseded_metrics", "upstream_ordered_metrics"):
        with (audit_root / str(trajectory.get(field, ""))).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            epoch_row = next(
                (
                    row
                    for row in csv.DictReader(handle)
                    if row.get("validate/world_loss_epoch") not in (None, "")
                ),
                None,
            )
        if epoch_row is None:
            raise EvidenceError(f"same-MIG trajectory lacks first-epoch validation: {field}")
        epoch_rows.append(epoch_row)
    old_epoch, new_epoch = epoch_rows
    expected_epoch_step = int(trajectory.get("first_epoch_csv_summary_step", -1))
    if (
        int(float(old_epoch.get("step", -1))) != expected_epoch_step
        or int(float(new_epoch.get("step", -1))) != expected_epoch_step
    ):
        raise EvidenceError("same-MIG first-epoch summaries do not share the asserted step")
    recorded_epoch_deltas = trajectory.get(
        "first_epoch_validation_new_minus_superseded", {}
    )
    epoch_fields = {
        "world_loss": "validate/world_loss_epoch",
        "prediction_loss": "validate/pred_loss_epoch",
        "sigreg_loss": "validate/sigreg_loss_epoch",
    }
    for field, csv_field in epoch_fields.items():
        actual_delta = float(new_epoch[csv_field]) - float(old_epoch[csv_field])
        if actual_delta != float(recorded_epoch_deltas.get(field, math.inf)):
            raise EvidenceError(
                f"same-MIG first-epoch trajectory delta changed for {field}: {actual_delta}"
            )
    return {
        "world_parameter_count": WORLD_PARAMETER_COUNT,
        "current_runtime_bf16_step": "bitwise_exact_with_upstream_parameter_order",
        "natural_order_causal_delta_observed": True,
        "same_mig_training_trajectory_diverged": True,
        "current_profile_natural_order_norm_delta": module_order_delta,
    }


def _native_data(payloads: list[dict[str, Any]], _: list[Path]) -> dict[str, Any]:
    payload = payloads[0]
    if payload.get("status") != "pass":
        raise EvidenceError("native-HDF5 batch parity did not pass")
    dataset = payload.get("dataset", {})
    if dataset.get("archive_member_sha256") != HDF5_SHA256:
        raise EvidenceError("native HDF5 member hash mismatch")
    if int(dataset.get("drop_last_batches_at_128", -1)) != 12_796:
        raise EvidenceError("training batch geometry mismatch")
    for section in ("single_process", "production_worker_path"):
        values = payload.get(section, {})
        false_fields = [key for key, value in values.items() if key.endswith("equal") and value is not True]
        if false_fields:
            raise EvidenceError(f"{section} parity failed: {false_fields}")
    return {"hdf5_sha256": HDF5_SHA256}


def _checkpoint(payloads: list[dict[str, Any]], _: list[Path]) -> dict[str, Any]:
    payload = payloads[0]
    if payload.get("status") != "pass":
        raise EvidenceError("checkpoint forensics did not pass")
    geometry = payload.get("training_geometry", {})
    if int(geometry.get("inferred_completed_epochs", -1)) != 10:
        raise EvidenceError("released checkpoint is not proven to be a 10-epoch artifact")
    if int(geometry.get("train_batches_per_epoch", -1)) != 12_796:
        raise EvidenceError("released checkpoint batch geometry mismatch")
    counters = payload.get("source", {}).get("batch_norm_num_batches_tracked", {})
    if set(map(int, counters.values())) != {127_960}:
        raise EvidenceError("released checkpoint BatchNorm counters mismatch")
    return {"released_training_epochs": 10, "released_optimizer_updates": 127_960}


def _historical_inference(payloads: list[dict[str, Any]], _: list[Path]) -> dict[str, Any]:
    inference, cem = payloads
    if inference.get("status") != "pass" or cem.get("status") != "pass":
        raise EvidenceError("historical inference/CEM parity did not pass")
    metrics = inference.get("metrics", {})
    for field in (
        "historical_mwm_encode_max_abs",
        "historical_mwm_action_encode_max_abs",
        "historical_mwm_rollout_max_abs",
        "historical_mwm_cost_max_abs",
    ):
        if float(metrics.get(field, math.inf)) != 0.0:
            raise EvidenceError(f"historical inference field {field} is not bitwise exact")
    comparisons = cem.get("comparisons", {})
    for field in ("actions_bitwise_equal", "costs_bitwise_equal", "mean_bitwise_equal", "var_bitwise_equal"):
        if comparisons.get(field) is not True:
            raise EvidenceError(f"historical CEM field {field} failed")
    return {"historical_rollout_and_cem": "bitwise_exact"}


def _historical_evaluator(payloads: list[dict[str, Any]], _: list[Path]) -> dict[str, Any]:
    payload = payloads[0]
    if payload.get("status") != "pass" or payload.get("start_rows_equal_across_all_controls") is not True:
        raise EvidenceError("historical evaluator parity did not pass")
    if int(payload.get("requested_goal_offset", -1)) != 25 or int(payload.get("effective_goal_offset", -1)) != 24:
        raise EvidenceError("historical goal-indexing contract mismatch")
    exact = payload.get("historical_executable_exact_replay", {})
    expected_rates = {"10": 86.0, "30": 94.0}
    for iterations, expected_rate in expected_rates.items():
        row = exact.get(iterations, {})
        if float(row.get("upstream_success_rate", -1)) != expected_rate:
            raise EvidenceError(f"unexpected upstream rate for CEM={iterations}")
        if float(row.get("mwm_success_rate", -1)) != expected_rate:
            raise EvidenceError(f"unexpected MWM rate for CEM={iterations}")
        if int(row.get("label_comparison", {}).get("changed", -1)) != 0:
            raise EvidenceError(f"episode labels differ for CEM={iterations}")
    return {"paper_replay_success_rate": 86.0, "public_yaml_replay_success_rate": 94.0}


def _initialization_provenance(payloads: list[dict[str, Any]], _: list[Path]) -> dict[str, Any]:
    payload = payloads[0]
    if payload.get("status") != "pass":
        raise EvidenceError("model-initialization forensics did not pass")
    source = payload.get("source_order", {})
    if source.get("global_seed_calls_before_model") != []:
        raise EvidenceError("upstream unexpectedly seeds model construction")
    if int(source.get("model_encoder_construction_line", 10**9)) >= int(source.get("manager_construction_line", -1)):
        raise EvidenceError("upstream model/Manager construction order mismatch")
    checkpoint = payload.get("released_checkpoint", {})
    if checkpoint.get("contains_rng_state") is not False or checkpoint.get("contains_optimizer_state") is not False:
        raise EvidenceError("released checkpoint unexpectedly contains recoverable construction state")
    return {"exact_released_construction_rng": "proven_unrecoverable"}


def _forensic_inventory(payloads: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    allowed = {
        "data_parity.json": {"mismatch_proven"},
        "eval_action_scaler_forensics.json": {"pass"},
        "action_policy_forensics.json": {"pass"},
        "legacy_reacher_dataset_replay.json": {"pass_with_last_bit_dependency_drift"},
        "dependency_forensics.json": {"pass_with_one_unrecorded_runtime_axis"},
        "scheduler_horizon_forensics.json": {"pass"},
        "reacher_eval_threshold_forensics.json": {"pass"},
        "upstream_eval_sampling_parity.json": {"pass"},
    }
    for payload, path in zip(payloads, paths, strict=True):
        if payload.get("status") not in allowed[path.name]:
            raise EvidenceError(f"unexpected forensic status for {path.name}: {payload.get('status')!r}")
        if path.name == "scheduler_horizon_forensics.json":
            audit_root = path.parents[3]
            branch_rows = {
                "paper_10_epoch_horizon": BRANCHES["paper10"],
                "public_100_epoch_horizon": BRANCHES["epoch10_horizon100"],
            }
            observations = (
                ("observed_upstream_ordered_first_epoch", 1, 12_796),
                ("observed_upstream_ordered_second_epoch", 2, 25_592),
                ("observed_upstream_ordered_third_epoch", 3, 38_388),
            )
            comparisons = {
                "world_loss": "validate/world_loss_epoch",
                "prediction_loss": "validate/pred_loss_epoch",
                "sigreg_loss": "validate/sigreg_loss_epoch",
            }
            for observation_key, epoch_number, completed_updates in observations:
                observed = payload.get(observation_key, {})
                if int(observed.get("completed_optimizer_updates", -1)) != completed_updates:
                    raise EvidenceError(
                        f"{observation_key} optimizer-update count mismatch"
                    )
                expected_step = completed_updates - 1
                if int(observed.get("csv_summary_step", -1)) != expected_step:
                    raise EvidenceError(f"{observation_key} summary-step mismatch")
                for section, branch_spec in branch_rows.items():
                    metrics_path = (
                        audit_root
                        / "logs"
                        / "mwm_training"
                        / str(branch_spec["run_name"])
                        / "csv_logs"
                        / "version_0"
                        / "metrics.csv"
                    )
                    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                        epoch_rows = [
                            row
                            for row in csv.DictReader(handle)
                            if row.get("validate/world_loss_epoch") not in (None, "")
                        ]
                    if len(epoch_rows) < epoch_number:
                        raise EvidenceError(
                            f"missing epoch-{epoch_number} validation row for {section}"
                        )
                    epoch_row = epoch_rows[epoch_number - 1]
                    if int(float(epoch_row.get("step", -1))) != expected_step:
                        raise EvidenceError(
                            f"epoch-{epoch_number} CSV step mismatch for {section}"
                        )
                    recorded = observed.get(section, {})
                    for field, csv_field in comparisons.items():
                        if float(recorded.get(field, math.inf)) != float(epoch_row[csv_field]):
                            raise EvidenceError(
                                f"recorded epoch-{epoch_number} {section}/{field} "
                                "does not match live CSV"
                            )
    data = payloads[0]
    if data.get("official_hdf5", {}).get("sha256") != HDF5_SHA256:
        raise EvidenceError("data forensic official HDF5 hash mismatch")
    replay = payloads[3]
    if replay.get("released_archive_action_replay", {}).get("all_values_bitwise_equal") is not True:
        raise EvidenceError("released random-action stream replay is not bitwise exact")
    return {"identified_recipe_axes": [path.stem for path in paths]}


def _scheduler(payloads: list[dict[str, Any]], _: list[Path]) -> dict[str, Any]:
    payload, paper10_paper, paper10_public, horizon100_paper, horizon100_public = payloads
    if payload.get("status") != "pass":
        raise EvidenceError("scheduler comparison did not pass")
    expected_interpretation = (
        "Operational branch choice for the follow-on construction-seed sensitivity sweep; "
        "it is not proof of the released checkpoint's historical LR horizon because the "
        "released construction RNG and training runtime are unrecoverable."
    )
    if payload.get("selection_interpretation") != expected_interpretation:
        raise EvidenceError("scheduler selection provenance limitation is missing")
    if payload.get("historical_scheduler_identification") != (
        "unrecoverable_from_released_artifact"
    ):
        raise EvidenceError("historical scheduler is incorrectly presented as identified")
    branch = payload.get("selected_scheduler_branch")
    if branch not in BRANCHES:
        raise EvidenceError(f"unsupported selected scheduler branch: {branch!r}")
    comparisons = payload.get("comparisons", {})
    expected_recipes = {
        "paper_cem": {
            "threshold": 0.1,
            "planner": {"elite_frac": 0.1, "n_iter": 10, "pop_size": 300, "topk": 30},
        },
        "public_cem": {
            "threshold": 0.05,
            "planner": {"elite_frac": 0.1, "n_iter": 30, "pop_size": 300, "topk": 30},
        },
    }
    expected_data = {
        "dataset_path": "data/upstream/reacher.h5",
        "dataset_format": "hdf5",
        "dataset_sha256": HDF5_SHA256,
        "goal_offset": 25,
        "goal_indexing": "upstream_lewm_end_exclusive",
        "effective_goal_offset": 24,
        "manifest_pairs": 500,
        "eval_budget_per_batch": 50,
        "action_preprocessing": "standard_scaler",
        "sampling": "upstream_lewm",
        "rollout_semantics": "upstream_lewm_historical",
        "policy_semantics": "upstream_lewm_historical",
    }
    if payload.get("evaluated_recipe_axes") != expected_recipes:
        raise EvidenceError("scheduler comparison evaluator recipe axes are missing or incorrect")
    if payload.get("evaluation_data") != expected_data:
        raise EvidenceError("scheduler comparison native-HDF5/goal provenance is missing")
    for planner in ("paper_cem", "public_cem"):
        row = comparisons.get(planner, {})
        for field in (
            "upstream_mean_success_rate",
            "paper10_candidate_mean_success_rate",
            "horizon100_candidate_mean_success_rate",
            "paper10_mean_paired_delta",
            "horizon100_mean_paired_delta",
        ):
            _finite(row.get(field), f"{planner}.{field}")
        for field in ("paper10_mean_paired_delta_ci95", "horizon100_mean_paired_delta_ci95"):
            if len(row.get(field, [])) != 2:
                raise EvidenceError(f"{planner}.{field} is not a two-sided interval")
    summaries = {
        ("paper10", "paper_cem"): paper10_paper,
        ("paper10", "public_cem"): paper10_public,
        ("epoch10_horizon100", "paper_cem"): horizon100_paper,
        ("epoch10_horizon100", "public_cem"): horizon100_public,
    }
    for (summary_branch, planner), summary in summaries.items():
        if summary.get("status") != "pass":
            raise EvidenceError(f"{summary_branch}/{planner} N=500 summary did not pass")
        if summary.get("candidate_role") != BRANCHES[summary_branch]["candidate_role"]:
            raise EvidenceError(f"{summary_branch}/{planner} candidate role mismatch")
        if summary.get("seeds") != list(EVAL_SEEDS) or int(summary.get("episodes_per_seed", -1)) != 500:
            raise EvidenceError(f"{summary_branch}/{planner} seed/episode contract mismatch")
        runtime = summary.get("environment_runtime", {})
        expected_threshold = float(expected_recipes[planner]["threshold"])
        if float(runtime.get("reacher_qpos_threshold", -1)) != expected_threshold:
            raise EvidenceError(f"{summary_branch}/{planner} threshold mismatch")
        if runtime.get("post_reset_validation") != "passed":
            raise EvidenceError(f"{summary_branch}/{planner} threshold was not validated after reset")
        if summary.get("planner_params") != expected_recipes[planner]["planner"]:
            raise EvidenceError(f"{summary_branch}/{planner} planner parameters mismatch")
        if summary.get("evaluation_data") != expected_data:
            raise EvidenceError(f"{summary_branch}/{planner} native-HDF5/goal provenance mismatch")
        rows = summary.get("paired_results", [])
        if [row.get("seed") for row in rows] != list(EVAL_SEEDS):
            raise EvidenceError(f"{summary_branch}/{planner} paired rows mismatch")
        for row in rows:
            if (
                int(row.get("episodes", -1)) != 500
                or row.get("env_runtime") != runtime
                or row.get("planner_params") != summary.get("planner_params")
                or row.get("evaluation_data") != summary.get("evaluation_data")
            ):
                raise EvidenceError(f"{summary_branch}/{planner} paired provenance mismatch")
            upstream = _finite(row.get("upstream_success_rate"), "upstream_success_rate")
            candidate = _finite(row.get("candidate_success_rate"), "candidate_success_rate")
            delta = _finite(row.get("candidate_minus_upstream"), "candidate_minus_upstream")
            if abs((candidate - upstream) - delta) > 1e-9:
                raise EvidenceError(f"{summary_branch}/{planner} paired delta mismatch")
            if not row.get("manifest_sha256") or not row.get("manifest_file_sha256"):
                raise EvidenceError(f"{summary_branch}/{planner} manifest provenance missing")
    for planner in ("paper_cem", "public_cem"):
        left = summaries[("paper10", planner)]["paired_results"]
        right = summaries[("epoch10_horizon100", planner)]["paired_results"]
        for left_row, right_row in zip(left, right, strict=True):
            for field in ("upstream_success_rate", "manifest_sha256", "manifest_file_sha256"):
                if left_row[field] != right_row[field]:
                    raise EvidenceError(
                        f"{planner} cross-branch upstream {field} mismatch at seed {left_row['seed']}"
                    )
    return {
        "selected_scheduler_branch": branch,
        "historical_scheduler_identification": "unrecoverable_from_released_artifact",
        "paired_seed_rows_validated": 20,
    }


def _canonical_checkpoints(
    root: Path, scheduler: dict[str, Any] | None
) -> dict[str, Any]:
    paths: list[Path] = []
    missing: list[str] = []
    try:
        for branch, spec in BRANCHES.items():
            checkpoint = root / "checkpoints_mwm" / spec["run_name"]
            lightning_checkpoint = (
                root
                / "logs"
                / "mwm_training"
                / spec["run_name"]
                / "csv_logs"
                / "version_0"
                / "checkpoints"
                / "last.ckpt"
            )
            required = [
                checkpoint / "config.json",
                checkpoint / "weights.pt",
                checkpoint / "world_metadata.json",
                lightning_checkpoint,
                root
                / "logs"
                / "mwm_training"
                / str(spec["run_name"])
                / "csv_logs"
                / "version_0"
                / "metrics.csv",
                root / str(spec["slurm_log"]),
                root
                / "logs"
                / "mwm_training"
                / str(spec["run_name"])
                / "environment.json",
            ]
            paths.extend(required)
            missing.extend(str(path.relative_to(root)) for path in required if not path.is_file())
        if missing:
            return {"name": "canonical_scheduler_checkpoints", "status": "pending", "missing": missing}
        for branch, spec in BRANCHES.items():
            checkpoint = root / "checkpoints_mwm" / spec["run_name"]
            lightning_checkpoint = (
                root
                / "logs"
                / "mwm_training"
                / spec["run_name"]
                / "csv_logs"
                / "version_0"
                / "checkpoints"
                / "last.ckpt"
            )
            config = _load_lightning_training_config(
                lightning_checkpoint, label=branch
            )
            _validate_training_config(
                config,
                label=branch,
                run_name=str(spec["run_name"]),
                model_init_seed=3072,
                lr_max_epochs=int(spec["lr_max_epochs"]),
                node_local_source_prefix="mwm-reacher-",
                expected_slurm_job_id=int(spec["job_id"]),
            )
            _validate_canonical_export_directory(
                checkpoint,
                label=branch,
                lightning_checkpoint=lightning_checkpoint,
            )
            if (checkpoint / "weights.pt").stat().st_size <= 0:
                raise EvidenceError(f"{branch} weights are empty")
            _validate_export_matches_lightning(
                checkpoint / "weights.pt",
                lightning_checkpoint,
                label=branch,
            )
            _validate_live_optimizer_scheduler_checkpoint(
                lightning_checkpoint,
                label=branch,
                lr_max_epochs=int(spec["lr_max_epochs"]),
            )
            _validate_canonical_runtime_evidence(
                root,
                label=branch,
                run_name=str(spec["run_name"]),
                job_id=int(spec["job_id"]),
                slurm_log=Path(str(spec["slurm_log"])),
            )
    except Exception as exc:
        return {
            "name": "canonical_scheduler_checkpoints",
            "status": "fail",
            "evidence": [str(path.relative_to(root)) for path in paths if path.is_file()],
            "error": f"{type(exc).__name__}: {exc}",
        }
    detail: dict[str, Any] = {}
    if scheduler is not None:
        detail["selected_scheduler_branch"] = scheduler.get("selected_scheduler_branch")
    return {
        "name": "canonical_scheduler_checkpoints",
        "status": "pass",
        "evidence": [str(path.relative_to(root)) for path in paths],
        **detail,
    }


def _validate_canonical_runtime_evidence(
    root: Path,
    *,
    label: str,
    run_name: str,
    job_id: int,
    slurm_log: Path,
) -> None:
    log_path = root / slurm_log
    text = log_path.read_text(encoding="utf-8", errors="replace")
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    required_fragments = (
        f"Job ID: {job_id}",
        "Configured optimizer 'world_opt' (modules=367, param_tensors=297, "
        "total_params=18034478) with LinearWarmupCosineAnnealingLR scheduler.",
        "[LogUnusedParametersOnce] Registered hooks on 341 leaf parameters.",
    )
    missing_fragments = [fragment for fragment in required_fragments if fragment not in plain]
    if missing_fragments:
        raise EvidenceError(f"{label} runtime log is missing parity receipts: {missing_fragments}")
    expected_prefit = {
        "world_loss": "4.970762252807617",
        "pred_loss": "0.06451215595006943",
        "sigreg_loss": "54.5",
    }
    missing_prefit = [
        metric
        for metric, value in expected_prefit.items()
        if re.search(
            rf"validate/{re.escape(metric)}\s*\|\s*{re.escape(value)}(?:\s|\|)",
            plain,
        )
        is None
    ]
    if missing_prefit:
        raise EvidenceError(f"{label} runtime log has the wrong pre-fit metrics: {missing_prefit}")
    unused_start = plain.find("The following parameters did NOT receive gradients")
    unused_end = plain.find("Hooks removed, callback disabled.", unused_start)
    if unused_start < 0 or unused_end < 0:
        raise EvidenceError(f"{label} runtime log does not contain the first-backward audit")
    unused_block = plain[unused_start:unused_end]
    unused_parameters = re.findall(
        r"-\s+(model\.[A-Za-z0-9_.]+)\s*$", unused_block, re.MULTILINE
    )
    if len(unused_parameters) != 44:
        raise EvidenceError(
            f"{label} first backward reported {len(unused_parameters)} unused tensors, expected 44"
        )
    non_decoder = [name for name in unused_parameters if not name.startswith("model.decoders.0.")]
    if non_decoder:
        raise EvidenceError(f"{label} world tensors were unused on first backward: {non_decoder[:8]}")

    environment_path = (
        root / "logs" / "mwm_training" / run_name / "environment.json"
    )
    environment = _load_json(environment_path)
    slurm = environment.get("slurm", {})
    expected_slurm = {
        "SLURM_JOB_ID": str(job_id),
        "SLURM_JOB_PARTITION": "b200-mig45",
        "SLURM_CPUS_PER_TASK": "6",
        "SLURM_SUBMIT_DIR": str(root),
    }
    slurm_mismatches = {
        key: {"actual": slurm.get(key), "expected": value}
        for key, value in expected_slurm.items()
        if slurm.get(key) != value
    }
    if slurm_mismatches:
        raise EvidenceError(f"{label} runtime Slurm receipt mismatch: {slurm_mismatches}")
    packages = environment.get("packages", {}).get("key_packages", {})
    expected_packages = {
        "torch": "2.11.0",
        "lightning": "2.6.1",
        "stable-pretraining": "0.1.6",
        "stable-worldmodel": "0.1.0",
        "transformers": "5.7.0",
    }
    package_mismatches = {
        key: {"actual": packages.get(key), "expected": value}
        for key, value in expected_packages.items()
        if packages.get(key) != value
    }
    if package_mismatches:
        raise EvidenceError(f"{label} runtime package receipt mismatch: {package_mismatches}")

    metrics_path = (
        root
        / "logs"
        / "mwm_training"
        / run_name
        / "csv_logs"
        / "version_0"
        / "metrics.csv"
    )
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fit_rows = [row for row in rows if row.get("fit/world_loss")]
    expected_fit_steps = list(range(49, 127_960, 50))
    actual_fit_steps = [int(row["step"]) for row in fit_rows]
    if actual_fit_steps != expected_fit_steps:
        raise EvidenceError(
            f"{label} fit metric steps are incomplete: "
            f"count={len(actual_fit_steps)}, first={actual_fit_steps[:2]}, last={actual_fit_steps[-2:]}"
        )
    validation_rows = [row for row in rows if row.get("validate/world_loss_epoch")]
    expected_validation_steps = [(epoch + 1) * 12_796 - 1 for epoch in range(10)]
    actual_validation_steps = [int(row["step"]) for row in validation_rows]
    if actual_validation_steps != expected_validation_steps:
        raise EvidenceError(
            f"{label} validation epoch summaries are incomplete: {actual_validation_steps}"
        )
    for stage_rows, fields in (
        (fit_rows, ("fit/world_loss", "fit/pred_loss", "fit/sigreg_loss")),
        (
            validation_rows,
            (
                "validate/world_loss_epoch",
                "validate/pred_loss_epoch",
                "validate/sigreg_loss_epoch",
            ),
        ),
    ):
        for row in stage_rows:
            for field in fields:
                _finite(row.get(field), f"{label}.{field}@{row.get('step')}")


def _validate_live_optimizer_scheduler_checkpoint(
    path: Path, *, label: str, lr_max_epochs: int
) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_updates = 127_960
    expected_max_steps = 12_796 * lr_max_epochs
    expected_warmup_steps = max(1, int(0.01 * expected_max_steps))
    if int(payload.get("epoch", -1)) != 9:
        raise EvidenceError(f"{label} Lightning checkpoint did not finish epoch 10")
    if int(payload.get("global_step", -1)) != expected_updates:
        raise EvidenceError(f"{label} live optimizer update count mismatch")
    optimizers = payload.get("optimizer_states", [])
    schedulers = payload.get("lr_schedulers", [])
    if len(optimizers) != 1 or len(schedulers) != 1:
        raise EvidenceError(f"{label} strict checkpoint must contain one optimizer and scheduler")
    optimizer = optimizers[0]
    optimizer_state = optimizer.get("state", {})
    if len(optimizer_state) != 297:
        raise EvidenceError(f"{label} optimizer does not contain the 297 world tensors")
    param_groups = optimizer.get("param_groups", [])
    if len(param_groups) != 1:
        raise EvidenceError(f"{label} optimizer param-group count mismatch")
    if param_groups[0].get("params") != list(range(297)):
        raise EvidenceError(f"{label} optimizer parameter IDs are not the canonical 0..296 sequence")
    expected_adamw_group = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-3,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    adamw_mismatches = {
        field: {"actual": param_groups[0].get(field), "expected": expected}
        for field, expected in expected_adamw_group.items()
        if param_groups[0].get(field) != expected
    }
    if adamw_mismatches:
        raise EvidenceError(f"{label} AdamW defaults mismatch: {adamw_mismatches}")

    # Adam assigns state IDs in optimizer-group order. These boundary tensors
    # distinguish raw LeWM's encoder -> predictor -> action encoder -> projector
    # -> prediction projector registration from MWM's natural module order.
    expected_order_sentinels = {
        0: (1, 1, 192),
        198: (1, 3, 192),
        279: (10, 10, 1),
        285: (2048, 192),
        291: (2048, 192),
        296: (192,),
    }
    actual_order_sentinels: dict[int, tuple[int, ...] | None] = {}
    for state_id, expected_shape in expected_order_sentinels.items():
        state = optimizer_state.get(state_id, optimizer_state.get(str(state_id), {}))
        exp_avg = state.get("exp_avg") if isinstance(state, dict) else None
        actual_shape = tuple(exp_avg.shape) if torch.is_tensor(exp_avg) else None
        actual_order_sentinels[state_id] = actual_shape
        if actual_shape != expected_shape:
            raise EvidenceError(
                f"{label} optimizer parameter order mismatch at state {state_id}: "
                f"{actual_shape!r} != {expected_shape!r}"
            )
    scheduler = schedulers[0]
    expected_scheduler = {
        "last_epoch": expected_updates,
        "max_steps": expected_max_steps,
        "warmup_steps": expected_warmup_steps,
        "base_lrs": [5e-5],
    }
    mismatches = {
        field: {"actual": scheduler.get(field), "expected": expected}
        for field, expected in expected_scheduler.items()
        if scheduler.get(field) != expected
    }
    if mismatches:
        raise EvidenceError(f"{label} live scheduler state mismatch: {mismatches}")
    expected_lr = 5e-5 * (
        1
        + math.cos(
            math.pi
            * (expected_updates - expected_warmup_steps)
            / (expected_max_steps - expected_warmup_steps)
        )
    ) / 2
    scheduler_lrs = scheduler.get("_last_lr", [])
    optimizer_lrs = [group.get("lr") for group in optimizer["param_groups"]]
    for source, values in (("scheduler", scheduler_lrs), ("optimizer", optimizer_lrs)):
        if len(values) != 1 or abs(float(values[0]) - expected_lr) > 1e-15:
            raise EvidenceError(
                f"{label} {source} LR mismatch: {values!r} != {[expected_lr]!r}"
            )


def _validate_export_matches_lightning(
    export_path: Path, lightning_path: Path, *, label: str
) -> None:
    import torch

    exported = torch.load(export_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(lightning_path, map_location="cpu", weights_only=False)
    if not isinstance(exported, dict) or not exported:
        raise EvidenceError(f"{label} exported weights are not a nonempty state dict")
    state = checkpoint.get("state_dict", {}) if isinstance(checkpoint, dict) else {}
    lightning_model = {
        str(key).removeprefix("model."): value
        for key, value in state.items()
        if str(key).startswith("model.")
    }
    if set(exported) != set(lightning_model):
        missing = sorted(set(lightning_model) - set(exported))
        extra = sorted(set(exported) - set(lightning_model))
        raise EvidenceError(
            f"{label} export/Lightning model keys differ: missing={missing[:8]}, extra={extra[:8]}"
        )
    for key, value in exported.items():
        source = lightning_model[key]
        if not torch.is_tensor(value) or not torch.is_tensor(source) or not torch.equal(value, source):
            raise EvidenceError(f"{label} exported tensor differs from last.ckpt at {key}")


def _validate_training_config(
    config: dict[str, Any],
    *,
    label: str,
    run_name: str,
    model_init_seed: int,
    lr_max_epochs: int,
    node_local_source_prefix: str | None = None,
    expected_slurm_job_id: int | None = None,
) -> None:
    expected_scalars = {
        "seed": (config.get("seed"), 3072),
        "env_id": (config.get("env_id"), "swm/ReacherDMControl-v0"),
        "base.family": (config.get("base", {}).get("family"), "lewm"),
        "base.checkpoint": (
            config.get("base", {}).get("checkpoint"),
            "models--quentinll--lewm-reacher",
        ),
        "model.K": (config.get("model", {}).get("K"), [192]),
        "model.D": (config.get("model", {}).get("D"), 192),
        "model.action_dim": (config.get("model", {}).get("action_dim"), "auto"),
        "model.image_shape": (config.get("model", {}).get("image_shape"), "auto"),
        "model.action_block": (config.get("model", {}).get("action_block"), 5),
        "model.history_size": (config.get("model", {}).get("history_size"), 3),
        "model.num_preds": (config.get("model", {}).get("num_preds"), 1),
        "train.batch_size": (config.get("train", {}).get("batch_size"), 128),
        "train.horizon": (config.get("train", {}).get("horizon"), 4),
        "train.num_workers": (config.get("train", {}).get("num_workers"), 6),
        "train.drop_last": (config.get("train", {}).get("drop_last"), True),
        "train.prefetch_factor": (config.get("train", {}).get("prefetch_factor"), 3),
        "train.pin_memory": (config.get("train", {}).get("pin_memory"), True),
        "train.precision": (config.get("train", {}).get("precision"), "bf16"),
        "train.no_cuda": (config.get("train", {}).get("no_cuda"), False),
        "train.devices": (config.get("train", {}).get("devices"), 1),
        "train.strategy": (config.get("train", {}).get("strategy"), "auto"),
        "train.num_nodes": (config.get("train", {}).get("num_nodes"), 1),
        "train.sync_batchnorm": (config.get("train", {}).get("sync_batchnorm"), False),
        "train.use_distributed_sampler": (
            config.get("train", {}).get("use_distributed_sampler"),
            True,
        ),
        "train.backend": (
            config.get("train", {}).get("backend"),
            "stable_worldmodel_lewm",
        ),
        "train.timestamp_run_dir": (
            config.get("train", {}).get("timestamp_run_dir"),
            False,
        ),
        "train.clean_trainer_root": (
            config.get("train", {}).get("clean_trainer_root"),
            True,
        ),
        "train.limit_train_batches": (
            config.get("train", {}).get("limit_train_batches"),
            1.0,
        ),
        "train.limit_val_batches": (
            config.get("train", {}).get("limit_val_batches"),
            1.0,
        ),
        "train.matmul_precision": (config.get("train", {}).get("matmul_precision"), "highest"),
        "train.checkpoint_every_n_train_steps": (
            config.get("train", {}).get("checkpoint_every_n_train_steps"),
            63_980,
        ),
        "train.checkpoint_monitor": (
            config.get("train", {}).get("checkpoint_monitor"),
            None,
        ),
        "train.save_top_k": (config.get("train", {}).get("save_top_k"), -1),
        "train.export_checkpoint": (
            config.get("train", {}).get("export_checkpoint"),
            "last",
        ),
        "train.resume_checkpoint": (
            config.get("train", {}).get("resume_checkpoint"),
            None,
        ),
        "train.slurm_auto_requeue": (
            config.get("train", {}).get("slurm_auto_requeue"),
            False,
        ),
        "train.gradient_clip_val": (config.get("train", {}).get("gradient_clip_val"), 1.0),
        "train.model_init_seed": (
            config.get("train", {}).get("model_init_seed"),
            model_init_seed,
        ),
        "train.fit_seed": (config.get("train", {}).get("fit_seed"), 0),
        "train.run_name": (config.get("train", {}).get("run_name"), run_name),
        "optim.lr": (config.get("optim", {}).get("lr"), 5e-5),
        "optim.weight_decay": (config.get("optim", {}).get("weight_decay"), 1e-3),
        "decoder_training.enabled": (
            config.get("decoder_training", {}).get("enabled"),
            False,
        ),
        "decoder_training.mode": (
            config.get("decoder_training", {}).get("mode"),
            "separate_optimizer",
        ),
        "loss.rollout_weight": (config.get("loss", {}).get("rollout_weight"), 1.0),
        "loss.sigreg_weight": (config.get("loss", {}).get("sigreg_weight"), 0.09),
        "loss.sigreg_knots": (config.get("loss", {}).get("sigreg_knots"), 17),
        "loss.sigreg_num_proj": (config.get("loss", {}).get("sigreg_num_proj"), 1024),
        "loss.history_size": (config.get("loss", {}).get("history_size"), 3),
        "loss.num_preds": (config.get("loss", {}).get("num_preds"), 1),
        "schedule.max_epochs": (config.get("schedule", {}).get("max_epochs"), 10),
        "schedule.lr_max_epochs": (
            config.get("schedule", {}).get("lr_max_epochs"),
            lr_max_epochs,
        ),
    }
    mismatches = {
        field: {"actual": actual, "expected": expected}
        for field, (actual, expected) in expected_scalars.items()
        if actual != expected
    }
    if mismatches:
        raise EvidenceError(f"{label} exported training config mismatch: {mismatches}")
    data = config.get("data", {})
    source = data.get("training_source", {})
    expected_data = {
        "path": "data/upstream/reacher.lance",
        "format": "lance",
        "split_ratio": 0.9,
        "pixels_key": "pixels",
        "action_key": "action",
        "frameskip": 5,
        "keys_to_load": ["pixels", "action", "observation"],
        "keys_to_cache": ["action", "observation"],
    }
    data_mismatches = {
        field: {"actual": data.get(field), "expected": expected}
        for field, expected in expected_data.items()
        if data.get(field) != expected
    }
    expected_source = {
        "format": "hdf5",
        "archive": "data/upstream/reacher.tar.zst",
        "archive_sha256": "4ff2385e49712caa89f21b8e0a246e2614b621d3f22cf2d1224d845e879a1cc2",
        "member": "reacher.h5",
        "extracted_sha256": HDF5_SHA256,
    }
    source_mismatches = {
        field: {"actual": source.get(field), "expected": expected}
        for field, expected in expected_source.items()
        if source.get(field) != expected
    }
    if data_mismatches or source_mismatches:
        raise EvidenceError(
            f"{label} training-data contract mismatch: "
            f"data={data_mismatches}, source={source_mismatches}"
        )
    source_path = str(source.get("path", ""))
    if node_local_source_prefix is None:
        if source_path != "data/upstream/reacher.h5":
            raise EvidenceError(f"{label} logical training-source path mismatch: {source_path!r}")
    else:
        recorded_job_id = str(config.get("slurm.job_id", ""))
        if not recorded_job_id.isdigit():
            raise EvidenceError(f"{label} resolved recipe has no Slurm job ID")
        if expected_slurm_job_id is not None and recorded_job_id != str(expected_slurm_job_id):
            raise EvidenceError(
                f"{label} resolved recipe Slurm job mismatch: "
                f"{recorded_job_id!r} != {expected_slurm_job_id!r}"
            )
        expected_pattern = (
            rf"^/.*/{re.escape(node_local_source_prefix)}{recorded_job_id}\."
            r"[^/]+/reacher\.h5$"
        )
        if re.fullmatch(expected_pattern, source_path) is None:
            raise EvidenceError(
                f"{label} node-local training-source path mismatch: {source_path!r}"
            )
    if config.get("restore", {}).get("import_path") != (
        "mwm.swm.restore.reacher_qpos_match_restore_spec"
    ):
        raise EvidenceError(f"{label} Reacher restore contract mismatch")
    expected_policy = {
        "shared": ["latent_producer"],
        "per_level": ["transition"],
        "reconstructor": ["decoder"],
    }
    if config.get("mwm", {}).get("component_policy") != expected_policy:
        raise EvidenceError(f"{label} component policy mismatch")
    expected_loss_scope = {
        "regularizers": "shared_latent",
        "reconstructor_detach_encoder": True,
        "reconstructor_contributes_to_encoder_loss": False,
    }
    if config.get("mwm", {}).get("loss_terms") != expected_loss_scope:
        raise EvidenceError(f"{label} loss-scope contract mismatch")
    loss = config.get("loss", {})
    if "recon_weight" in loss or float(loss.get("recon_latent_weight", 0.0)) != 0.0:
        raise EvidenceError(f"{label} world objective contains reconstruction feedback")


def _initialization_checkpoints(root: Path, selected: str) -> dict[str, Any]:
    spec = BRANCHES[selected]
    paths: list[Path] = []
    missing: list[str] = []
    for init_seed in MODEL_INIT_SEEDS[:-1]:
        run_name = (
            f"mwm_reacher_k192_{selected}_nativeh5_nodecoder_init{init_seed}_fit0_20260806"
        )
        checkpoint = root / "checkpoints_mwm" / run_name
        lightning_checkpoint = (
            root
            / "logs"
            / "mwm_training"
            / run_name
            / "csv_logs"
            / "version_0"
            / "checkpoints"
            / "last.ckpt"
        )
        required = [
            checkpoint / "config.json",
            checkpoint / "weights.pt",
            checkpoint / "world_metadata.json",
            lightning_checkpoint,
        ]
        paths.extend(required)
        missing.extend(str(path.relative_to(root)) for path in required if not path.is_file())
    if missing:
        return {"name": "selected_construction_seed_checkpoints", "status": "pending", "missing": missing}
    try:
        for init_seed in MODEL_INIT_SEEDS[:-1]:
            run_name = (
                f"mwm_reacher_k192_{selected}_nativeh5_nodecoder_init{init_seed}_fit0_20260806"
            )
            checkpoint = root / "checkpoints_mwm" / run_name
            lightning_checkpoint = (
                root
                / "logs"
                / "mwm_training"
                / run_name
                / "csv_logs"
                / "version_0"
                / "checkpoints"
                / "last.ckpt"
            )
            _validate_training_config(
                _load_lightning_training_config(
                    lightning_checkpoint, label=f"{selected}/init{init_seed}"
                ),
                label=f"{selected}/init{init_seed}",
                run_name=run_name,
                model_init_seed=init_seed,
                lr_max_epochs=int(spec["lr_max_epochs"]),
                node_local_source_prefix="mwm-reacher-init-",
            )
            _validate_canonical_export_directory(
                checkpoint,
                label=f"{selected}/init{init_seed}",
                lightning_checkpoint=lightning_checkpoint,
            )
            if (checkpoint / "weights.pt").stat().st_size <= 0:
                raise EvidenceError(f"{selected}/init{init_seed} weights are empty")
            _validate_export_matches_lightning(
                checkpoint / "weights.pt",
                lightning_checkpoint,
                label=f"{selected}/init{init_seed}",
            )
            _validate_live_optimizer_scheduler_checkpoint(
                lightning_checkpoint,
                label=f"{selected}/init{init_seed}",
                lr_max_epochs=int(spec["lr_max_epochs"]),
            )
    except Exception as exc:
        return {
            "name": "selected_construction_seed_checkpoints",
            "status": "fail",
            "evidence": [str(path.relative_to(root)) for path in paths],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "name": "selected_construction_seed_checkpoints",
        "status": "pass",
        "evidence": [str(path.relative_to(root)) for path in paths],
        "model_init_seeds": list(MODEL_INIT_SEEDS[:-1]),
    }


def _sweep_submission(
    payloads: list[dict[str, Any]], paths: list[Path], scheduler: dict[str, Any]
) -> dict[str, Any]:
    payload = payloads[0]
    selected = scheduler.get("selected_scheduler_branch")
    if payload.get("status") != "submitted" or payload.get("selected_scheduler_branch") != selected:
        raise EvidenceError("construction-seed sweep submission does not match scheduler selection")
    comparison = paths[0].parents[0] / "scheduler_branch_comparison.json"
    if payload.get("scheduler_comparison_sha256") != _sha256(comparison):
        raise EvidenceError("scheduler comparison changed after sweep submission")
    expected_sweep_seeds = list(MODEL_INIT_SEEDS[:-1])
    if payload.get("model_init_seeds") != expected_sweep_seeds:
        raise EvidenceError("construction-seed sweep submission has the wrong model seeds")
    if payload.get("evaluation_seeds") != list(EVAL_SEEDS):
        raise EvidenceError("construction-seed sweep submission has the wrong evaluation seeds")
    if int(payload.get("episodes_per_evaluation_seed", -1)) != 500:
        raise EvidenceError("construction-seed sweep submission has the wrong episode count")
    spec = BRANCHES[str(selected)]
    expected_recipe = {
        "train_config": (
            "configs/research/"
            f"train_mwm_lewm_reacher_k192_{selected}_nativeh5_nodecoder_"
            "init3072_fit0_20260806.yaml"
        ),
        "lr_max_epochs": str(spec["lr_max_epochs"]),
        "sweep_tag": str(selected),
        "report_prefix": str(spec["report_prefix"]),
        "eval_config": "configs/eval/paper_reacher.yaml",
        "baseline_report_tag": str(spec["paper_report_tag"]),
        "baseline_candidate_role": str(spec["candidate_role"]),
    }
    if payload.get("recipe") != expected_recipe:
        raise EvidenceError("construction-seed sweep submission recipe mismatch")
    repository_root = paths[0].resolve().parents[3]
    expected_recipe_hashes = {
        "train_config": _sha256(repository_root / expected_recipe["train_config"]),
        "eval_config": _sha256(repository_root / expected_recipe["eval_config"]),
    }
    if payload.get("recipe_file_sha256s") != expected_recipe_hashes:
        raise EvidenceError("construction-seed sweep recipe files changed after submission")
    expected_execution_hashes = {
        path: _sha256(repository_root / path) for path in SWEEP_EXECUTION_FILES
    }
    if payload.get("execution_file_sha256s") != expected_execution_hashes:
        raise EvidenceError("construction-seed sweep execution files changed after submission")
    jobs = payload.get("jobs", {})
    if set(jobs) != {"training_array", "evaluation_array", "collector", "final_audit"}:
        raise EvidenceError("construction-seed sweep job receipt is incomplete")
    if not all(str(job).isdigit() for job in jobs.values()):
        raise EvidenceError("construction-seed sweep job IDs are invalid")
    return {
        "selected_scheduler_branch": selected,
        "model_init_seeds": expected_sweep_seeds,
        "evaluation_seeds": list(EVAL_SEEDS),
        "episodes_per_evaluation_seed": 500,
        "jobs": jobs,
    }


def _seed_sweep(
    payloads: list[dict[str, Any]],
    _: list[Path],
    *,
    expected_baseline_report_tag: str,
) -> dict[str, Any]:
    payload = payloads[0]
    if payload.get("status") != "pass":
        raise EvidenceError("construction-seed summary did not pass")
    if payload.get("evaluation_seeds") != list(EVAL_SEEDS):
        raise EvidenceError("construction-seed evaluation-seed contract mismatch")
    if payload.get("model_init_seeds") != list(MODEL_INIT_SEEDS):
        raise EvidenceError("construction-seed model-seed contract mismatch")
    if int(payload.get("episodes_per_eval_seed", -1)) != 500:
        raise EvidenceError("construction-seed episode count mismatch")
    if payload.get("baseline_report_tag") != expected_baseline_report_tag:
        raise EvidenceError("construction-seed baseline report does not match selected branch")
    runtime = payload.get("environment_runtime", {})
    if float(runtime.get("reacher_qpos_threshold", -1)) != 0.1:
        raise EvidenceError("construction-seed sweep did not use paper threshold 0.1")
    if runtime.get("post_reset_validation") != "passed":
        raise EvidenceError("construction-seed sweep threshold was not validated after reset")
    expected_planner = {"elite_frac": 0.1, "n_iter": 10, "pop_size": 300, "topk": 30}
    expected_data = {
        "dataset_path": "data/upstream/reacher.h5",
        "dataset_format": "hdf5",
        "dataset_sha256": HDF5_SHA256,
        "goal_offset": 25,
        "goal_indexing": "upstream_lewm_end_exclusive",
        "effective_goal_offset": 24,
        "manifest_pairs": 500,
        "eval_budget_per_batch": 50,
        "action_preprocessing": "standard_scaler",
        "sampling": "upstream_lewm",
        "rollout_semantics": "upstream_lewm_historical",
        "policy_semantics": "upstream_lewm_historical",
    }
    if payload.get("planner_params") != expected_planner:
        raise EvidenceError("construction-seed sweep paper planner provenance mismatch")
    if payload.get("evaluation_data") != expected_data:
        raise EvidenceError("construction-seed sweep native-HDF5/goal provenance mismatch")
    branches = payload.get("by_model_init_seed", [])
    if [row.get("model_init_seed") for row in branches] != list(MODEL_INIT_SEEDS):
        raise EvidenceError("construction-seed result rows mismatch")
    for branch in branches:
        rows = branch.get("paired_results", [])
        if [row.get("eval_seed") for row in rows] != list(EVAL_SEEDS):
            raise EvidenceError("construction-seed paired evaluation rows mismatch")
        for row in rows:
            if (
                int(row.get("episodes", -1)) != 500
                or row.get("env_runtime") != runtime
                or row.get("planner_params") != expected_planner
                or row.get("evaluation_data") != expected_data
            ):
                raise EvidenceError("construction-seed paired provenance mismatch")
            upstream = _finite(row.get("upstream_success_rate"), "upstream_success_rate")
            candidate = _finite(row.get("candidate_success_rate"), "candidate_success_rate")
            delta = _finite(row.get("candidate_minus_upstream"), "candidate_minus_upstream")
            if not (0 <= upstream <= 100 and 0 <= candidate <= 100):
                raise EvidenceError("construction-seed success rate outside [0, 100]")
            if abs((candidate - upstream) - delta) > 1e-9:
                raise EvidenceError("construction-seed paired delta is inconsistent")
            if not row.get("manifest_sha256") or not row.get("manifest_file_sha256"):
                raise EvidenceError("construction-seed manifest provenance is missing")
            if not row.get("upstream_summary_csv") or not row.get("candidate_summary_csv"):
                raise EvidenceError("construction-seed source-summary provenance is missing")
        aggregate = branch.get("aggregate", {})
        if len(aggregate.get("mean_paired_delta_ci95", [])) != 2:
            raise EvidenceError("construction-seed paired interval missing")
        if not isinstance(aggregate.get("zero_in_mean_paired_delta_ci95"), bool):
            raise EvidenceError("construction-seed zero-in-interval flag missing")
        candidate_rates = [float(row["candidate_success_rate"]) for row in rows]
        deltas = [float(row["candidate_minus_upstream"]) for row in rows]
        if abs(float(aggregate.get("candidate_mean_success_rate", math.inf)) - sum(candidate_rates) / len(candidate_rates)) > 1e-12:
            raise EvidenceError("construction-seed candidate aggregate is inconsistent")
        if abs(float(aggregate.get("mean_paired_delta", math.inf)) - sum(deltas) / len(deltas)) > 1e-12:
            raise EvidenceError("construction-seed delta aggregate is inconsistent")
    sensitivity = payload.get("construction_seed_sensitivity", {})
    closest = int(sensitivity.get("descriptive_closest_model_init_seed", -1))
    if closest not in MODEL_INIT_SEEDS:
        raise EvidenceError("descriptive closest construction seed is invalid")
    candidate_means = [
        float(branch["aggregate"]["candidate_mean_success_rate"]) for branch in branches
    ]
    expected_range = max(candidate_means) - min(candidate_means)
    if abs(float(sensitivity.get("candidate_mean_success_rate_range", math.inf)) - expected_range) > 1e-12:
        raise EvidenceError("construction-seed sensitivity range is inconsistent")
    expected_closest = min(
        branches,
        key=lambda branch: abs(float(branch["aggregate"]["mean_paired_delta"])),
    )
    if int(expected_closest["model_init_seed"]) != closest:
        raise EvidenceError("descriptive closest construction seed is inconsistent")
    expected_closest_delta = float(expected_closest["aggregate"]["mean_paired_delta"])
    if abs(float(sensitivity.get("descriptive_closest_mean_paired_delta", math.inf)) - expected_closest_delta) > 1e-12:
        raise EvidenceError("descriptive closest construction delta is inconsistent")
    return {
        "upstream_mean_success_rate": _finite(
            payload.get("upstream", {}).get("mean_success_rate"), "upstream mean"
        ),
        "descriptive_closest_model_init_seed": closest,
        "descriptive_closest_mean_paired_delta": _finite(
            sensitivity.get("descriptive_closest_mean_paired_delta"), "closest delta"
        ),
        "candidate_mean_success_rate_range_across_initializations": _finite(
            sensitivity.get("candidate_mean_success_rate_range"), "construction range"
        ),
    }


def run_test_gate(root: Path, python: str) -> dict[str, Any]:
    completed = subprocess.run(
        [python, "-m", "pytest", "-q", *TEST_FILES],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": completed.returncode,
        "command": [python, "-m", "pytest", "-q", *TEST_FILES],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def audit(root: Path, test_result: dict[str, Any] | None = None) -> dict[str, Any]:
    requirements: list[dict[str, Any]] = []
    requirements.append(
        _requirement(
            root,
            "architecture_loss_gradient_update_parity",
            (ARCH_REPORT, OPTIMIZER_ORDER_REPORT),
            _architecture,
        )
    )
    requirements.append(
        _requirement(root, "native_hdf5_training_batch_parity", (REPORT_ROOT / "native_hdf5_batch_parity.json",), _native_data)
    )
    requirements.append(
        _requirement(root, "released_checkpoint_training_geometry", (REPORT_ROOT / "checkpoint_training_forensics.json",), _checkpoint)
    )
    requirements.append(
        _requirement(
            root,
            "historical_inference_and_cem_parity",
            (REPORT_ROOT / "historical_lewm_inference_parity.json", REPORT_ROOT / "historical_cem_parity.json"),
            _historical_inference,
        )
    )
    requirements.append(
        _requirement(root, "historical_reacher_executable_replay", (REPORT_ROOT / "upstream_evaluator_parity.json",), _historical_evaluator)
    )
    requirements.append(
        _requirement(root, "released_model_initialization_provenance", (REPORT_ROOT / "model_initialization_forensics.json",), _initialization_provenance)
    )
    forensic_paths = tuple(
        REPORT_ROOT / name
        for name in (
            "data_parity.json",
            "eval_action_scaler_forensics.json",
            "action_policy_forensics.json",
            "legacy_reacher_dataset_replay.json",
            "dependency_forensics.json",
            "scheduler_horizon_forensics.json",
            "reacher_eval_threshold_forensics.json",
            "upstream_eval_sampling_parity.json",
        )
    )
    requirements.append(_requirement(root, "recipe_and_dependency_forensic_inventory", forensic_paths, _forensic_inventory))
    scheduler_requirement = _requirement(
        root,
        "paired_scheduler_branch_evaluation",
        (
            REPORT_ROOT / "scheduler_branch_comparison.json",
            Path("reports/research")
            / BRANCHES["paper10"]["paper_report_tag"]
            / "n500_summary.json",
            Path("reports/research")
            / BRANCHES["paper10"]["public_report_tag"]
            / "n500_summary.json",
            Path("reports/research")
            / BRANCHES["epoch10_horizon100"]["paper_report_tag"]
            / "n500_summary.json",
            Path("reports/research")
            / BRANCHES["epoch10_horizon100"]["public_report_tag"]
            / "n500_summary.json",
        ),
        _scheduler,
    )
    requirements.append(scheduler_requirement)
    scheduler_detail = scheduler_requirement if scheduler_requirement["status"] == "pass" else None
    requirements.append(_canonical_checkpoints(root, scheduler_detail))
    receipt_path = REPORT_ROOT / "model_init_sweep_submission.json"
    if scheduler_detail is None:
        requirements.append(
            {
                "name": "construction_seed_sweep_submission",
                "status": "pending",
                "missing": ["successful scheduler selection"],
            }
        )
        requirements.append(
            {
                "name": "paired_construction_seed_evaluation",
                "status": "pending",
                "missing": ["successful scheduler selection"],
            }
        )
        requirements.append(
            {
                "name": "selected_construction_seed_checkpoints",
                "status": "pending",
                "missing": ["successful scheduler selection"],
            }
        )
    else:
        requirements.append(
            _requirement(
                root,
                "construction_seed_sweep_submission",
                (receipt_path,),
                lambda payloads, paths: _sweep_submission(payloads, paths, scheduler_detail),
            )
        )
        selected = str(scheduler_detail["selected_scheduler_branch"])
        seed_summary = (
            Path("reports/research")
            / BRANCHES[selected]["report_prefix"]
            / "model_init_n500_summary.json"
        )
        requirements.append(
            _requirement(
                root,
                "paired_construction_seed_evaluation",
                (seed_summary,),
                lambda payloads, paths: _seed_sweep(
                    payloads,
                    paths,
                    expected_baseline_report_tag=str(
                        BRANCHES[selected]["paper_report_tag"]
                    ),
                ),
            )
        )
        requirements.append(
            _initialization_checkpoints(
                root, str(scheduler_detail["selected_scheduler_branch"])
            )
        )
    if test_result is None:
        requirements.append(
            {
                "name": "current_worktree_parity_tests",
                "status": "pending",
                "missing": ["audit invocation with --run-tests"],
            }
        )
    elif int(test_result.get("returncode", -1)) != 0:
        requirements.append(
            {
                "name": "current_worktree_parity_tests",
                "status": "fail",
                "command": test_result.get("command"),
                "stdout": test_result.get("stdout"),
                "stderr": test_result.get("stderr"),
            }
        )
    else:
        requirements.append(
            {
                "name": "current_worktree_parity_tests",
                "status": "pass",
                "command": test_result.get("command"),
                "summary": str(test_result.get("stdout", "")).strip().splitlines()[-1],
            }
        )
    statuses = [str(requirement["status"]) for requirement in requirements]
    status = "fail" if "fail" in statuses else "pending" if "pending" in statuses else "pass"
    return {
        "status": status,
        "objective": "single-level MWM versus upstream LeWM end-to-end Reacher parity",
        "requirements": requirements,
        "counts": {value: statuses.count(value) for value in ("pass", "pending", "fail")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all evidence required for single-level Reacher parity.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument(
        "--python",
        default="/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / REPORT_ROOT / "completion_audit.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    test_result = run_test_gate(root, args.python) if args.run_tests else None
    payload = audit(root, test_result)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "pass" else 2 if payload["status"] == "pending" else 1)


if __name__ == "__main__":
    main()
