from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mwm.data.metadata import dataset_metadata_path, load_dataset_metadata


DEFAULT_CONFIGS = [
    "configs/train/mwm_lewm_pusht.yaml",
    "configs/train/mwm_lewm_tworoom.yaml",
    "configs/train/mwm_lewm_scheduled_pusht.yaml",
    "configs/train/mwm_lewm_scheduled_tworoom.yaml",
    "configs/train/mwm_lewm_dense_pusht.yaml",
    "configs/train/mwm_lewm_dense_reacher.yaml",
    "configs/train/mwm_lewm_dense_ogb_cube.yaml",
    "configs/train/mwm_lewm_dense_tworoom.yaml",
    "configs/eval/mwm_lewm_pusht.yaml",
    "configs/eval/mwm_lewm_tworoom.yaml",
]

PAPER_PARITY_CONFIGS = [
    "configs/train/mwm_lewm_pusht_upstream.yaml",
    "configs/train/mwm_lewm_reacher_upstream.yaml",
    "configs/train/mwm_lewm_ogb_cube_upstream.yaml",
    "configs/train/mwm_lewm_tworoom_upstream.yaml",
    "configs/eval/paper_pusht.yaml",
    "configs/eval/paper_reacher.yaml",
    "configs/eval/paper_ogb_cube.yaml",
    "configs/eval/paper_tworoom.yaml",
]


def _data_cfg(path: str | Path) -> dict[str, Any] | None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(path))
    data = cfg.get("data", None)
    if not isinstance(data, dict) and not OmegaConf.is_config(data):
        return None
    return dict(OmegaConf.to_container(data, resolve=True))


def verify_data_configs(config_paths: list[str | Path] | None = None) -> dict[str, Any]:
    paths = [Path(p) for p in (config_paths or DEFAULT_CONFIGS)]
    errors: list[str] = []
    seen: dict[str, dict[str, Any]] = {}
    for cfg_path in paths:
        data = _data_cfg(cfg_path)
        if not data:
            continue
        fmt = str(data.get("format", "lance")).lower()
        if fmt != "lance":
            errors.append(f"{cfg_path}: MWM runtime requires format lance, got {fmt!r}")
            continue
        dataset_path = Path(str(data.get("path", "")))
        if not dataset_path.exists():
            errors.append(f"{cfg_path}: missing Lance dataset {dataset_path}")
            continue
        metadata_path = dataset_metadata_path(dataset_path)
        metadata = load_dataset_metadata(dataset_path, required=False)
        if not metadata:
            errors.append(f"{cfg_path}: missing dataset metadata {metadata_path}")
            continue
        if str(metadata.get("format", "")) != "swm_lance":
            errors.append(f"{cfg_path}: metadata format must be swm_lance, got {metadata.get('format')!r}")
        for key in ("env_id", "restore_spec", "image_shape", "action_dim", "action_low", "action_high", "dataset"):
            if key not in metadata:
                errors.append(f"{cfg_path}: metadata {metadata_path} missing {key!r}")
        seen[str(dataset_path)] = {
            "config": str(cfg_path),
            "metadata": str(metadata_path),
            "env_id": metadata.get("env_id"),
            "restore_spec": metadata.get("restore_spec"),
            "format": "lance",
        }
    if errors:
        raise ValueError("MWM dataset verification failed:\n- " + "\n- ".join(errors))
    return {"datasets": seen, "count": len(seen)}


def _resolve_cli_paths(argv: list[str]) -> list[str | Path] | None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify MWM Lance dataset configs and sidecar metadata.")
    parser.add_argument("configs", nargs="*", help="Optional config paths to verify instead of the default smoke set.")
    parser.add_argument("--paper-parity", action="store_true", help="Verify paper-parity train/eval configs.")
    parser.add_argument("--all", dest="all_configs", action="store_true", help="Verify default and paper-parity configs.")
    args = parser.parse_args(argv)

    selected_modes = int(bool(args.paper_parity)) + int(bool(args.all_configs))
    if selected_modes > 1:
        parser.error("--paper-parity and --all are mutually exclusive")
    if args.configs and selected_modes:
        parser.error("explicit config paths cannot be combined with --paper-parity or --all")

    if not args.configs and not selected_modes:
        return None
    if args.paper_parity:
        return PAPER_PARITY_CONFIGS
    if args.all_configs:
        return [*DEFAULT_CONFIGS, *PAPER_PARITY_CONFIGS]
    return args.configs


def main(argv: list[str] | None = None) -> None:
    import sys

    args = [*argv] if argv is not None else sys.argv[1:]
    report = verify_data_configs(_resolve_cli_paths(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CONFIGS",
    "PAPER_PARITY_CONFIGS",
    "main",
    "verify_data_configs",
]
