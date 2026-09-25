from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from mwm.config_cli import load_config
from mwm.io import file_sha256
from mwm.training.stable_wm_config import DEFAULTS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an audited h10 checkpoint with a five-epoch cosine tail extension."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cfg = load_config(DEFAULTS, args.config, [])
    if int(cfg.schedule.max_epochs) != 15 or int(cfg.schedule.lr_max_epochs) != 15:
        raise RuntimeError("Tail config must use max_epochs=15 and lr_max_epochs=15")

    state = torch.load(args.source, map_location="cpu", weights_only=False)
    schedulers = state.get("lr_schedulers", [])
    optimizers = state.get("optimizer_states", [])
    source_contract = {
        "epoch": int(state.get("epoch", -1)),
        "global_step": int(state.get("global_step", -1)),
        "optimizer_count": len(optimizers),
        "scheduler_count": len(schedulers),
        "scheduler_last_epochs": [int(item.get("last_epoch", -1)) for item in schedulers],
        "scheduler_max_steps": [int(item.get("max_steps", -1)) for item in schedulers],
    }
    expected_source = {
        "epoch": 9,
        "global_step": 116820,
        "optimizer_count": 2,
        "scheduler_count": 2,
        "scheduler_last_epochs": [58410, 58410],
        "scheduler_max_steps": [58410, 58410],
    }
    if source_contract != expected_source:
        raise RuntimeError(f"Unexpected h10 source contract: {source_contract}")

    steps_per_epoch_per_scheduler = 5841
    max_steps = 15 * steps_per_epoch_per_scheduler
    warmup_steps = max(1, int(0.01 * max_steps))
    last_epoch = 10 * steps_per_epoch_per_scheduler
    base_lr = 5e-5
    tail_lr = base_lr * (
        1.0
        + math.cos(
            math.pi
            * (last_epoch - warmup_steps)
            / (max_steps - warmup_steps)
        )
    ) / 2.0

    for scheduler in schedulers:
        scheduler["warmup_steps"] = warmup_steps
        scheduler["max_steps"] = max_steps
        scheduler["base_lrs"] = [base_lr]
        scheduler["last_epoch"] = last_epoch
        scheduler["_step_count"] = last_epoch + 1
        scheduler["_last_lr"] = [tail_lr]
    for optimizer in optimizers:
        for group in optimizer["param_groups"]:
            group["lr"] = tail_lr
            group["initial_lr"] = base_lr

    state["hyper_parameters"] = OmegaConf.to_container(cfg, resolve=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    provenance = {
        "derived_checkpoint_sha256": file_sha256(args.output),
        "method": "preserve weights and optimizer moments; extend cosine endpoint to 15 epochs",
        "new_scheduler_last_epoch": last_epoch,
        "new_scheduler_max_steps": max_steps,
        "new_scheduler_warmup_steps": warmup_steps,
        "resume_lr": tail_lr,
        "source_checkpoint": str(args.source),
        "source_checkpoint_sha256": file_sha256(args.source),
        "source_contract": source_contract,
    }
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
