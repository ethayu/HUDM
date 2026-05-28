from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from mwm.adapters.lewm import LeWMObjectImporter
from mwm.checkpoints import save_world_checkpoint
from mwm.dependency_refs import dependency_refs


DEFAULTS = {
    "output_root": "checkpoints_mwm",
    "sources": [
        {
            "name": "upstream_lewm_pusht",
            "repo": "quentinll/lewm-pusht",
            "env_id": "swm/PushT-v1",
            "restore_spec": "pusht_state_goal_state",
            "image_shape": [224, 224],
            "action_dim": 10,
            "base_action_dim": 2,
            "action_block": 5,
        },
        {
            "name": "upstream_lewm_tworoom",
            "repo": "quentinll/lewm-tworooms",
            "env_id": "swm/TwoRoom-v1",
            "restore_spec": "point_state_goal_state",
            "image_shape": [224, 224],
            "action_dim": 10,
            "base_action_dim": 2,
            "action_block": 5,
        },
    ],
}


def _action_spec(spec: Any) -> dict[str, int]:
    block = int(spec.action_block)
    dim = int(spec.action_dim)
    base_dim = int(spec.get("base_action_dim", dim // max(1, block)))
    if block <= 0:
        raise ValueError(f"{spec.name} action_block must be positive, got {block}.")
    if base_dim * block != dim:
        raise ValueError(
            f"{spec.name} action spec is inconsistent: base_action_dim={base_dim}, "
            f"action_block={block}, action_dim={dim}."
        )
    return {"dim": dim, "base_dim": base_dim, "block": block}


def _load_upstream(repo_or_path: str) -> torch.nn.Module:
    path = Path(repo_or_path)
    if path.is_file():
        obj = torch.load(path, map_location="cpu", weights_only=False)
    else:
        from stable_worldmodel.wm.utils import load_pretrained

        obj = load_pretrained(repo_or_path)
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"Expected upstream Le-WM object module, got {type(obj).__name__}")
    return obj


def prepare_one(spec: Any, output_root: str | Path) -> Path:
    root = Path(output_root)
    out_dir = root / str(spec.name)
    action_spec = _action_spec(spec)
    source_dir = root / "upstream_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / f"{spec.name}_object.pt"
    if not source_path.is_file():
        obj = _load_upstream(str(spec.get("object_checkpoint", spec.get("repo"))))
        torch.save(obj, source_path)
    model = LeWMObjectImporter(
        str(source_path),
        D=int(spec.get("D", 192)),
        K=tuple(int(k) for k in spec.get("K", [192])),
        action_dim=int(action_spec["dim"]),
        action_block=int(action_spec["block"]),
        image_shape=tuple(int(x) for x in spec.image_shape),
        normalize_imagenet=bool(spec.get("normalize_imagenet", True)),
        expected_class_name=spec.get("expected_class_name", None),
    ).import_model()
    metadata = {
        "env_id": str(spec.env_id),
        "restore_spec": str(spec.restore_spec),
        "image_shape": [int(x) for x in spec.image_shape],
        "action_dim": int(action_spec["base_dim"]),
        "action_block": int(action_spec["block"]),
        "action_preprocessing": "standard_scaler",
        "source_history_size": int(getattr(model, "source_history_size", spec.get("history_size", 3))),
        "action_spec": action_spec,
        "levels": [int(k) for k in spec.get("K", [192])],
        "role": "upstream_lewm_converted",
        "upstream": {
            "repo": str(spec.get("repo", "")),
            "object_checkpoint": str(source_path),
        },
        "dataset": {
            "pixels_key": str(spec.get("pixels_key", "pixels")),
            "action_key": str(spec.get("action_key", "action")),
        },
        "model": {
            "target": "mwm.adapters.lewm.build_mwm_lewm_from_object",
            "D": int(spec.get("D", 192)),
            "K": [int(k) for k in spec.get("K", [192])],
            "action_dim": int(action_spec["dim"]),
            "action_block": int(action_spec["block"]),
        },
        "dependencies": dependency_refs(Path(__file__).resolve().parent),
    }
    for key in ("action_low", "action_high"):
        if key in spec:
            metadata[key] = [float(x) for x in spec[key]]
    save_world_checkpoint(model, out_dir, metadata=metadata)
    return out_dir


def main(cfg_path: str | None = None) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path) if cfg_path else {})
    for spec in cfg.sources:
        out = prepare_one(spec, cfg.output_root)
        print(out)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
