from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from mwm.checkpoint_io import load_world_model_from_checkpoint
from mwm.io import file_sha256


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def canonical_signature(trace: dict[str, Any]) -> str:
    payload = {
        "batch_size": int(trace["batch_end"]) - int(trace["batch_start"]),
        "num_samples": int(trace["num_samples"]),
        "horizon": int(trace["horizon"]),
        "action_dim": int(trace["action_dim"]),
        "history": int(trace["model_history_size"]),
        "rollout_ks": [int(value) for value in trace["model_rollout_ks"]],
        "rollout_levels": [int(value) for value in trace["model_rollout_level_indices"]],
        "rollout_semantics": str(trace["model_rollout_semantics"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def audit_signature(model: Any, signature_text: str, device: torch.device) -> int:
    signature = json.loads(signature_text)
    batch_size = int(signature["batch_size"])
    num_samples = int(signature["num_samples"])
    horizon = int(signature["horizon"])
    action_dim = int(signature["action_dim"])
    history = int(signature["history"])
    rollout_ks = [int(value) for value in signature["rollout_ks"]]
    rollout_levels = [int(value) for value in signature["rollout_levels"]]

    if len(rollout_ks) != horizon or len(rollout_levels) != horizon:
        raise ValueError(f"Rollout schedule does not match horizon: {signature}")
    if int(model.action_dim) != action_dim:
        raise ValueError(f"Checkpoint action_dim={model.action_dim}, trace action_dim={action_dim}")
    if int(model.history_size) != history:
        raise ValueError(f"Checkpoint history={model.history_size}, trace history={history}")
    if str(model.planning_rollout_semantics) != str(signature["rollout_semantics"]):
        model.planning_rollout_semantics = str(signature["rollout_semantics"])

    dtype = next(model.parameters()).dtype
    infos = {
        "pixels": torch.zeros(
            batch_size,
            num_samples,
            history,
            1,
            device=device,
            dtype=dtype,
        ),
        "emb": torch.zeros(
            batch_size,
            num_samples,
            history,
            int(model.D),
            device=device,
            dtype=dtype,
        ),
    }
    candidates = torch.zeros(
        batch_size,
        num_samples,
        horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )
    with torch.inference_mode():
        if bool(model.supports_arbitrary_k):
            result = model.rollout_with_k_schedule(
                infos,
                candidates,
                rollout_ks,
                flop_accounting="dynamics_audit",
            )
        else:
            result = model.rollout_with_schedule(
                infos,
                candidates,
                rollout_levels,
                flop_accounting="dynamics_audit",
            )
    error = result.get("_mwm_flop_audit_error")
    if error:
        raise RuntimeError(f"FLOP audit failed: {error}")
    value = int(result.get("_mwm_dynamics_flops", 0))
    if value <= 0:
        raise RuntimeError(f"FLOP audit returned non-positive count: {value}")
    return value


def audit_cell(
    cell_dir: Path,
    device: torch.device,
    model_cache: dict[Path, tuple[Any, dict[str, Any]]],
) -> dict[str, Any]:
    eval_path = cell_dir / "eval.json"
    diagnostics_path = cell_dir / "planning_diagnostics.json"
    eval_payload = load_json(eval_path)
    diagnostics = load_json(diagnostics_path)
    trace = diagnostics.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"Missing planning trace in {diagnostics_path}")
    if any(str(row.get("model_flop_accounting", "none")) != "none" for row in trace):
        raise ValueError(f"Expected an unaudited trace in {diagnostics_path}")
    if any(int(row.get("model_dynamics_flops", 0)) != 0 for row in trace):
        raise ValueError(f"Expected zero original dynamics FLOPs in {diagnostics_path}")

    checkpoint = Path(str(eval_payload["checkpoint_run_dir"]))
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    if checkpoint not in model_cache:
        model, metadata, _ = load_world_model_from_checkpoint(checkpoint, None, device)
        model_cache[checkpoint] = (model, metadata)
    model, metadata = model_cache[checkpoint]
    counts = Counter(canonical_signature(row) for row in trace)
    signature_rows: list[dict[str, Any]] = []
    dynamics_flops_total = 0
    for signature_text, count in sorted(counts.items()):
        per_call = audit_signature(model, signature_text, device)
        subtotal = int(per_call) * int(count)
        dynamics_flops_total += subtotal
        signature_rows.append(
            {
                "signature": json.loads(signature_text),
                "calls": int(count),
                "dynamics_flops_per_call": int(per_call),
                "dynamics_flops_subtotal": int(subtotal),
            }
        )

    original_summary = diagnostics.get("summary", {})
    trace_calls = sum(int(row.get("cem_cost_calls", 0)) for row in trace)
    if trace_calls != len(trace) or trace_calls != int(original_summary.get("cem_cost_calls", -1)):
        raise RuntimeError(f"Trace call-count mismatch in {diagnostics_path}")
    episodes = int(eval_payload.get("episodes", len(eval_payload.get("outcomes", []))))
    return {
        "cell": cell_dir.name,
        "role": str(eval_payload.get("role", eval_payload.get("name", cell_dir.name))),
        "checkpoint": str(checkpoint),
        "checkpoint_weights_sha256": file_sha256(checkpoint / "weights.pt"),
        "planning_diagnostics": str(diagnostics_path.resolve()),
        "planning_diagnostics_sha256": file_sha256(diagnostics_path),
        "episodes": episodes,
        "success_rate": float(eval_payload["swm_results"]["success_rate"]),
        "cem_cost_calls": trace_calls,
        "original_flop_accounting": "none",
        "original_dynamics_flops_total": int(original_summary.get("dynamics_flops_total", 0)),
        "backfill_method": "exact_unique_trace_signature_dynamics_audit",
        "dynamics_flops_total": int(dynamics_flops_total),
        "dynamics_flops_per_episode": float(dynamics_flops_total) / episodes,
        "signatures": signature_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill exact dynamics FLOPs for completed unaudited quick benchmarks."
    )
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    benchmark_root = args.benchmark_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    cell_dirs = sorted(path.parent for path in benchmark_root.glob("*/planning_diagnostics.json"))
    if not cell_dirs:
        raise FileNotFoundError(f"No benchmark cells found under {benchmark_root}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_cache: dict[Path, tuple[Any, dict[str, Any]]] = {}
    rows = [audit_cell(cell_dir, device, model_cache) for cell_dir in cell_dirs]
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "hudm.quick_benchmark_dynamics_flops_backfill/v1",
        "benchmark_root": str(benchmark_root),
        "source_summary_sha256": file_sha256(benchmark_root / "summary.json"),
        "method": (
            "Profile each unique recorded dynamics-call signature once with torch "
            "FlopCounterMode via flop_accounting=dynamics_audit, then multiply by its "
            "exact trace frequency. Original benchmark artifacts are not modified."
        ),
        "rows": rows,
    }
    json_path = output_dir / "dynamics_flops_backfill.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output_dir / "dynamics_flops_backfill.csv"
    fieldnames = [
        "cell",
        "role",
        "episodes",
        "success_rate",
        "cem_cost_calls",
        "original_flop_accounting",
        "original_dynamics_flops_total",
        "backfill_method",
        "dynamics_flops_total",
        "dynamics_flops_per_episode",
        "checkpoint",
        "checkpoint_weights_sha256",
        "planning_diagnostics",
        "planning_diagnostics_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    checksums = output_dir / "SHA256SUMS"
    checksums.write_text(
        f"{file_sha256(csv_path)}  {csv_path.name}\n{file_sha256(json_path)}  {json_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
