from __future__ import annotations

from pathlib import Path
from typing import Any

from mwm.adapters.builder import STABLE_CONFIG_TARGET
from mwm.adapters.constants import LEWM_BASE_ADAPTER_ARCH
from mwm.checkpoint_contract import checkpoint_full_latent_dim
from mwm.checkpoint_io import METADATA_FILENAME, validate_checkpoint_directory
from mwm.io import load_json


def load_checkpoint_metadata_for_benchmark(checkpoint_dir: Path, errors: list[str]) -> dict[str, Any]:
    try:
        _, metadata = validate_checkpoint_directory(checkpoint_dir, strict_artifacts=True, strict_metadata=True)
    except Exception as exc:  # noqa: BLE001 - verifier should aggregate failures
        errors.append(f"MWM checkpoint contract failed: {checkpoint_dir}: {exc}")
        try:
            return load_json(checkpoint_dir / METADATA_FILENAME)
        except Exception:
            return {}
    return metadata


def validate_benchmark_role_checkpoint_contract(row: dict[str, Any], metadata: dict[str, Any], errors: list[str]) -> None:
    role = str(row.get("role", ""))
    checkpoint_dir = Path(str(row.get("checkpoint_run_dir", "")))
    levels = [int(k) for k in metadata.get("levels", [])] if isinstance(metadata.get("levels", []), list) else []
    model_meta = metadata.get("model", {})
    target = str(model_meta.get("target", "")) if isinstance(model_meta, dict) else ""
    backend = str(metadata.get("training_backend", ""))
    try:
        d: int | None = checkpoint_full_latent_dim(metadata)
    except ValueError as exc:
        errors.append(f"MWM checkpoint contract failed: {checkpoint_dir}: {exc}")
        d = None
    if role == "upstream_lewm_converted":
        if metadata.get("role") != "upstream_lewm_converted":
            errors.append(f"upstream role checkpoint missing upstream_lewm_converted metadata role: {checkpoint_dir}")
        if d is not None and levels != [d]:
            errors.append(f"upstream role checkpoint must be identity-parity K=[D={d}], got {levels}: {checkpoint_dir}")
        if target != STABLE_CONFIG_TARGET:
            errors.append(f"upstream role checkpoint must export the generic base-adaptive target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"upstream role checkpoint missing corrected architecture version: {checkpoint_dir}")
    elif role == "retrained_lewm_identity":
        if d is not None and levels != [d]:
            errors.append(f"retrained identity checkpoint must be K=[D={d}], got {levels}: {checkpoint_dir}")
        if backend != "stable_worldmodel_lewm":
            errors.append(
                f"retrained identity checkpoint must use the Le-WM base-adapter backend, got {backend!r}: {checkpoint_dir}"
            )
        if target != STABLE_CONFIG_TARGET:
            errors.append(f"retrained identity checkpoint must export the generic base-adaptive target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"retrained identity checkpoint missing corrected architecture version: {checkpoint_dir}")
    elif role == "mwm_scheduled":
        if levels != [48, 96, 144]:
            errors.append(f"scheduled MWM checkpoint must be K=[48,96,144], got {levels}: {checkpoint_dir}")
        if target != STABLE_CONFIG_TARGET:
            errors.append(f"scheduled MWM checkpoint must export the generic base-adaptive target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"scheduled MWM checkpoint missing corrected architecture version: {checkpoint_dir}")
    elif role == "mwm_dense":
        if levels != [6, 12, 48, 96, 144, 192]:
            errors.append(f"dense MWM checkpoint must be K=[6,12,48,96,144,192], got {levels}: {checkpoint_dir}")
        if target != STABLE_CONFIG_TARGET:
            errors.append(f"dense MWM checkpoint must export the generic base-adaptive target: {checkpoint_dir}")
        if metadata.get("architecture_version") != LEWM_BASE_ADAPTER_ARCH:
            errors.append(f"dense MWM checkpoint missing corrected architecture version: {checkpoint_dir}")


__all__ = ["load_checkpoint_metadata_for_benchmark", "validate_benchmark_role_checkpoint_contract"]
