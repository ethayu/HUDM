from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from omegaconf import OmegaConf

from mwm.checkpoint_io import validate_checkpoint_directory
from mwm.io import file_sha256
from mwm.training.stable_wm_export import export_stable_wm_adapter_lightning_checkpoint


EPOCH_PATTERN = re.compile(r"^epoch=(?P<epoch>\d+)-step=(?P<step>\d+)\.ckpt$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export every epoch Lightning checkpoint as a canonical MWM weight bundle."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--trainer-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-start-epoch", type=int, default=0)
    parser.add_argument("--expected-epochs", type=int, required=True)
    args = parser.parse_args()

    if args.expected_start_epoch < 0:
        raise ValueError("--expected-start-epoch must be non-negative")
    if args.expected_epochs <= 0:
        raise ValueError("--expected-epochs must be positive")

    checkpoints: list[tuple[int, int, Path]] = []
    for path in args.trainer_checkpoint_dir.glob("epoch=*-step=*.ckpt"):
        match = EPOCH_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group("epoch")), int(match.group("step")), path))
    checkpoints.sort()
    expected = list(
        range(args.expected_start_epoch, args.expected_start_epoch + args.expected_epochs)
    )
    actual = [epoch for epoch, _, _ in checkpoints]
    if actual != expected:
        raise RuntimeError(f"Expected epoch checkpoints {expected}, found {actual}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    first_state = torch.load(checkpoints[0][2], map_location="cpu", weights_only=False)
    resolved_config = first_state.get("hyper_parameters")
    if not isinstance(resolved_config, dict):
        raise RuntimeError(f"Missing resolved training configuration in {checkpoints[0][2]}")
    resolved_config_path = args.output_root / "resolved_training_config.yaml"
    resolved_config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(resolved_config)), encoding="utf-8")
    manifest: list[dict[str, object]] = []
    for epoch, step, checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_config = state.get("hyper_parameters")
        if checkpoint_config != resolved_config:
            raise RuntimeError(f"Resolved training configuration changed at {checkpoint}")
        output_dir = args.output_root / f"epoch_{epoch:03d}"
        export_stable_wm_adapter_lightning_checkpoint(
            str(resolved_config_path),
            str(checkpoint),
            output_dir=str(output_dir),
        )
        _, metadata = validate_checkpoint_directory(output_dir)
        exported_epoch = int(metadata.get("epoch", -1))
        if exported_epoch != epoch:
            raise RuntimeError(
                f"Exported epoch mismatch for {checkpoint}: metadata={exported_epoch}, expected={epoch}"
            )
        manifest.append(
            {
                "epoch": epoch,
                "global_step": step,
                "lightning_checkpoint": str(checkpoint),
                "lightning_sha256": file_sha256(checkpoint),
                "canonical_checkpoint": str(output_dir),
                "weights_sha256": file_sha256(output_dir / "weights.pt"),
                "config_sha256": file_sha256(output_dir / "config.json"),
                "metadata_sha256": file_sha256(output_dir / "world_metadata.json"),
                "resolved_training_config_sha256": file_sha256(resolved_config_path),
            }
        )

    destination = args.output_root / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported and validated {len(manifest)} epoch checkpoints to {args.output_root}")


if __name__ == "__main__":
    main()
