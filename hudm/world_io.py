from __future__ import annotations

import os
import re
import json
from typing import Any, Iterable

import torch


_ENCODER_RE = re.compile(r"^encoder_epoch(\d+)\.pt$")


def checkpoint_epochs(run_dir: str) -> list[int]:
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Checkpoint run directory not found: {run_dir}")
    epochs: list[int] = []
    for name in os.listdir(run_dir):
        match = _ENCODER_RE.match(name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def latest_checkpoint_epoch(run_dir: str) -> int:
    epochs = checkpoint_epochs(run_dir)
    if not epochs:
        raise FileNotFoundError(
            f"No epoch checkpoints found under {run_dir}. "
            "Expected files like encoder_epoch<N>.pt."
        )
    return int(epochs[-1])


def _per_level_paths(prefix: str, epoch: int, num_levels: int) -> Iterable[tuple[int, str]]:
    for li in range(int(num_levels)):
        yield li, f"{prefix}_l{li}_epoch{int(epoch)}.pt"


METADATA_FILENAME = "world_metadata.json"


def save_world_metadata(run_dir: str, metadata: dict[str, Any]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, METADATA_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_world_metadata(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, METADATA_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing world metadata: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def save_world_checkpoint(model, run_dir: str, epoch: int, metadata: dict[str, Any] | None = None) -> None:
    epoch = int(epoch)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.encoder.state_dict(), os.path.join(run_dir, f"encoder_epoch{epoch}.pt"))
    if model.decoder_mode == "per_level":
        for li, filename in _per_level_paths("decoder", epoch, len(model.K)):
            torch.save(model.decoders[li].state_dict(), os.path.join(run_dir, filename))
    else:
        torch.save(model.decoder.state_dict(), os.path.join(run_dir, f"decoder_epoch{epoch}.pt"))
    if model.dynamics_mode == "per_level":
        for li, filename in _per_level_paths("dyn", epoch, len(model.K)):
            torch.save(model.dynamics[li].state_dict(), os.path.join(run_dir, filename))
    else:
        torch.save(model.dynamics.state_dict(), os.path.join(run_dir, f"dyn_epoch{epoch}.pt"))
    if metadata is not None:
        meta = dict(metadata)
        meta["epoch"] = epoch
        save_world_metadata(run_dir, meta)


def load_world_checkpoint(model, run_dir: str, epoch: int, device: torch.device) -> None:
    epoch = int(epoch)
    enc_path = os.path.join(run_dir, f"encoder_epoch{epoch}.pt")
    if not os.path.isfile(enc_path):
        raise FileNotFoundError(f"Missing encoder checkpoint: {enc_path}")
    model.encoder.load_state_dict(torch.load(enc_path, map_location=device))

    if model.dynamics_mode == "per_level":
        for li, filename in _per_level_paths("dyn", epoch, len(model.K)):
            path = os.path.join(run_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing dynamics checkpoint: {path}")
            model.dynamics[li].load_state_dict(torch.load(path, map_location=device))
    else:
        path = os.path.join(run_dir, f"dyn_epoch{epoch}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing dynamics checkpoint: {path}")
        model.dynamics.load_state_dict(torch.load(path, map_location=device))

    if model.decoder_mode == "per_level":
        for li, filename in _per_level_paths("decoder", epoch, len(model.K)):
            path = os.path.join(run_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing decoder checkpoint: {path}")
            model.decoders[li].load_state_dict(torch.load(path, map_location=device))
    else:
        path = os.path.join(run_dir, f"decoder_epoch{epoch}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing decoder checkpoint: {path}")
        model.decoder.load_state_dict(torch.load(path, map_location=device))


def build_world_model_from_metadata(metadata: dict[str, Any]):
    from models.world.model import HierWorldModel

    model_cfg = dict(metadata.get("model", {}))
    return HierWorldModel(
        K=[int(k) for k in model_cfg["K"]],
        D=int(model_cfg["D"]),
        action_dim=int(metadata["action_dim"]),
        input=str(model_cfg.get("input", "images")),
        decoder_mode=str(model_cfg.get("decoder_mode", "per_level")),
        dynamics_mode=str(model_cfg.get("dynamics_mode", "per_level")),
        image_shape=tuple(int(x) for x in metadata["image_shape"]),
    )


def load_world_model_from_checkpoint(
    run_dir: str,
    epoch: int | None,
    device: torch.device,
):
    metadata = load_world_metadata(run_dir)
    model = build_world_model_from_metadata(metadata).to(device)
    ckpt_epoch = latest_checkpoint_epoch(run_dir) if epoch is None else int(epoch)
    load_world_checkpoint(model, run_dir, ckpt_epoch, device=device)
    model.eval()
    return model, metadata, ckpt_epoch
