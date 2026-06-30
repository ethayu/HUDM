from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
        {
            "name": "upstream_lewm_reacher",
            "repo": "quentinll/lewm-reacher",
            "env_id": "swm/ReacherDMControl-v0",
            "restore_spec": "reacher_qpos_match_qpos_qvel",
            "image_shape": [224, 224],
            "action_dim": 10,
            "base_action_dim": 2,
            "action_block": 5,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
        },
        {
            "name": "upstream_lewm_ogb_cube",
            "repo": "quentinll/lewm-cube",
            "env_id": "swm/OGBCube-v0",
            "restore_spec": "ogbench_cube_single_qpos_qvel_target_pose",
            "image_shape": [224, 224],
            "action_dim": 25,
            "base_action_dim": 5,
            "action_block": 5,
            "action_low": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "action_high": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
    ],
}


EXPECTED_UPSTREAM_CLASS = "stable_worldmodel.wm.lewm.lewm.LeWM"


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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _resolve_upstream(repo_or_path: str) -> tuple[Any, dict[str, Any], Path]:
    import torch

    path = Path(repo_or_path)
    if path.is_file():
        obj = torch.load(path, map_location="cpu", weights_only=False)
        config_path = path.parent / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Upstream Le-WM object file requires sibling config.json: {config_path}")
        source_config = _read_json(config_path)
    elif path.is_dir():
        from stable_worldmodel.wm.utils import load_pretrained

        obj = load_pretrained(str(path))
        config_path = path / "config.json"
        source_config = _read_json(config_path)
    else:
        from stable_worldmodel.data import ensure_dir_exists, get_cache_dir
        from stable_worldmodel.wm.utils import _resolve, load_pretrained

        cache_dir = get_cache_dir(None, sub_folder="checkpoints")
        ensure_dir_exists(cache_dir)
        checkpoint_path, source_config = _resolve(repo_or_path, cache_dir)
        obj = load_pretrained(repo_or_path)
        config_path = checkpoint_path.parent / "config.json"
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"Expected upstream Le-WM object module, got {type(obj).__name__}")
    return obj, source_config, config_path


def _validate_upstream_object(obj: Any, *, expected_class_name: str | None) -> None:
    import torch

    missing = [name for name in ("encoder", "predictor", "action_encoder") if not hasattr(obj, name)]
    if missing:
        raise ValueError(f"Trusted Le-WM object is missing components: {missing}")
    if not expected_class_name:
        return
    class_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
    allowed = {str(expected_class_name), str(expected_class_name).rsplit(".", 1)[-1]}
    if class_name not in allowed and type(obj).__name__ not in allowed:
        raise ValueError(f"Trusted Le-WM object has class {class_name!r}; expected {expected_class_name!r}.")


def _copy_upstream_lewm_weights(model: Any, upstream: Any) -> None:
    target_state = model.state_dict()
    copied: set[str] = set()
    prefixes = (
        ("encoder.", "encoder."),
        ("projector.", "projector."),
        ("action_encoder.", "transitions.0.action_encoder."),
        ("predictor.", "transitions.0.predictor."),
        ("pred_proj.", "transitions.0.pred_proj."),
    )
    for source_name, source_tensor in upstream.state_dict().items():
        for source_prefix, target_prefix in prefixes:
            if not source_name.startswith(source_prefix):
                continue
            target_name = target_prefix + source_name[len(source_prefix) :]
            if target_name not in target_state:
                raise ValueError(f"Upstream Le-WM parameter {source_name!r} has no MWM target {target_name!r}.")
            if tuple(target_state[target_name].shape) != tuple(source_tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {source_name!r} -> {target_name!r}: "
                    f"{tuple(source_tensor.shape)} != {tuple(target_state[target_name].shape)}."
                )
            target_state[target_name] = source_tensor.detach().clone()
            copied.add(target_name)
            break

    required_prefixes = ("encoder.", "projector.", "transitions.0.")
    missing = sorted(name for name in target_state if name.startswith(required_prefixes) and name not in copied)
    if missing:
        raise ValueError(f"Upstream Le-WM conversion did not populate MWM parameter(s): {missing[:8]}")
    model.load_state_dict(target_state, strict=True)


def prepare_one(spec: Any, output_root: str | Path) -> Path:
    from mwm.adapters.builder import STABLE_CONFIG_TARGET, build_mwm_from_stable_config
    from mwm.adapters.stable_config import stable_config_sha256
    from mwm.checkpoint_io import save_world_checkpoint
    from mwm.dependency_refs import dependency_refs

    root = Path(output_root)
    out_dir = root / str(spec.name)
    action_spec = _action_spec(spec)
    source_ref = str(spec.get("source", spec.get("repo")))
    upstream, source_config, config_path = _resolve_upstream(source_ref)
    _validate_upstream_object(
        upstream,
        expected_class_name=spec.get("expected_class_name", EXPECTED_UPSTREAM_CLASS),
    )
    predictor_config = source_config.get("predictor", {}) if isinstance(source_config.get("predictor", {}), dict) else {}
    history_size = int(spec.get("history_size", predictor_config.get("num_frames", source_config.get("history_size", 3))))
    num_preds = int(spec.get("num_preds", 1))
    levels = [int(k) for k in spec.get("K", [spec.get("D", 192)])]
    model = build_mwm_from_stable_config(
        family="lewm",
        source_config=source_config,
        source_config_sha256=stable_config_sha256(config_path),
        training_recipe={
            "history_size": history_size,
            "num_preds": num_preds,
            "loss_scope": {"regularizers": "shared_latent"},
        },
        K=tuple(levels),
        action_dim=int(action_spec["dim"]),
        expected_D=int(spec.get("D", 192)),
        action_block=int(action_spec["block"]),
        image_shape=tuple(int(x) for x in spec.image_shape),
        normalize_imagenet=bool(spec.get("normalize_imagenet", True)),
    )
    _copy_upstream_lewm_weights(model, upstream)
    metadata = {
        "env_id": str(spec.env_id),
        "restore_spec": str(spec.restore_spec),
        "image_shape": [int(x) for x in spec.image_shape],
        "action_dim": int(action_spec["base_dim"]),
        "action_block": int(action_spec["block"]),
        "action_preprocessing": "standard_scaler",
        "source_history_size": history_size,
        "action_spec": action_spec,
        "levels": levels,
        "role": "upstream_lewm_converted",
        "fresh_init": False,
        "upstream": {
            "repo": str(spec.get("repo", "")),
            "source": source_ref,
            "source_config": str(config_path),
        },
        "dataset": {
            "pixels_key": str(spec.get("pixels_key", "pixels")),
            "action_key": str(spec.get("action_key", "action")),
        },
        "model": {
            "target": STABLE_CONFIG_TARGET,
            "D": int(spec.get("D", 192)),
            "K": levels,
            "action_dim": int(action_spec["dim"]),
            "action_block": int(action_spec["block"]),
        },
            "dependencies": dependency_refs(Path(__file__).resolve().parents[2]),
    }
    for key in ("action_low", "action_high"):
        if key in spec:
            metadata[key] = [float(x) for x in spec[key]]
    save_world_checkpoint(model, out_dir, metadata=metadata)
    return out_dir


def main(cfg_path: str | None = None) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path) if cfg_path else {})
    for spec in cfg.sources:
        out = prepare_one(spec, cfg.output_root)
        print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare upstream Le-WM checkpoints as MWM checkpoints.")
    parser.add_argument("config", nargs="?", default=None, help="Optional YAML config overriding the built-in sources")
    args = parser.parse_args()
    main(args.config)
