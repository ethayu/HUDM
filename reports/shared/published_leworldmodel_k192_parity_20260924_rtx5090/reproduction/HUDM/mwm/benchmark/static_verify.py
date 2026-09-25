from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.checkpoint_verify import (
    load_checkpoint_metadata_for_benchmark,
    validate_benchmark_role_checkpoint_contract,
)
from mwm.benchmark.config import DEFAULTS as BENCHMARK_DEFAULTS
from mwm.benchmark.config import load_manifest_config, role
from mwm.benchmark.matrix_identity import expected_cells_from_resolved, load_expected_resolved
from mwm.benchmark.paper_targets import normalize_paper_target_config


def verify_benchmark_static(
    cfg_path: str | Path = "configs/benchmark/scheduled_pusht.yaml",
    *,
    roles: Any = None,
    check_checkpoints: bool = True,
) -> dict[str, Any]:
    cfg = OmegaConf.merge(BENCHMARK_DEFAULTS, OmegaConf.load(str(cfg_path)))
    resolved = load_expected_resolved(cfg, roles=roles)
    errors: list[str] = []
    paper_targets = normalize_paper_target_config(cfg)

    checked_checkpoints: set[str] = set()
    if check_checkpoints:
        for run, run_cfg in resolved:
            checkpoint_ref = OmegaConf.select(run_cfg, "checkpoint.run_dir")
            run_role = role(run, run_cfg)
            if checkpoint_ref is None:
                errors.append(f"static benchmark run {run.get('name', '<unnamed>')} missing checkpoint.run_dir")
                continue
            checkpoint_dir = Path(str(checkpoint_ref))
            checkpoint_metadata = load_checkpoint_metadata_for_benchmark(checkpoint_dir, errors)
            if checkpoint_metadata:
                validate_benchmark_role_checkpoint_contract(
                    {"role": run_role, "checkpoint_run_dir": str(checkpoint_dir)},
                    checkpoint_metadata,
                    errors,
                )
                checked_checkpoints.add(str(checkpoint_dir))
    errors = list(dict.fromkeys(errors))
    if errors:
        raise ValueError("Benchmark static verification failed:\n- " + "\n- ".join(errors))

    return {
        "config": str(cfg_path),
        "output_dir": str(cfg.output_dir),
        "runs": len(resolved),
        "env_id": str(cfg.env_id),
        "seed": int(cfg.seed),
        "manifest": load_manifest_config(cfg),
        "expected_cells": sorted(expected_cells_from_resolved(resolved)),
        "paper_targets": paper_targets,
        "checkpoint_contracts": sorted(checked_checkpoints),
        "check_checkpoints": bool(check_checkpoints),
        "static_only": True,
    }


__all__ = ["verify_benchmark_static"]
