from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings


DEFAULTS = {
    "K": None,
    "checkpoint": {"run_dir": "checkpoints_mwm/run", "epoch": None},
    "data": {
        "path": "data/swm_dataset.lance",
        "format": "lance",
        "split_ratio": 0.8,
        "pixels_key": "pixels",
        "action_key": "action",
        "keys_to_cache": ["action", "proprio", "state"],
        "action_preprocessing": "auto",
    },
    "eval": {
        "episodes": 4,
        "goal_offset": 30,
        "seed": 0,
        "budget": 50,
        "num_envs": 4,
        "output_path": "rollouts/mwm_eval.json",
        "manifest_path": None,
        "write_manifest_path": None,
        "sampling": "mwm",
        "save_video": False,
        "video_path": "rollouts/mwm_eval_videos",
    },
    "env": {
        "max_episode_steps": 100,
        "goal_conditioned": True,
        "kwargs": {},
    },
    "restore": {"import_path": None},
    "planner": {
        "horizon": 16,
        "receding_horizon": 1,
        "action_block": 1,
        "batch_size": "auto",
        "pop_size": 256,
        "topk": None,
        "elite_frac": 0.1,
        "n_iter": 5,
        "init_std": 1.0,
        "seed": None,
        "warm_start": True,
        "clamp_actions": False,
        "std_unbiased": True,
        "flop_accounting": "none",
        "scheduler": {
            "enabled": True,
            "mpc": {"mode": "fixed", "level": "finest"},
            "cem": {"mode": "fixed", "level": "base"},
            "rollout": {"mode": "fixed", "level": "base"},
        },
    },
    "device": "auto",
}


@dataclass
class EvalRuntime:
    cfg: Any
    device: Any
    model: Any
    metadata: dict[str, Any]
    epoch: int
    dataset: Any
    process: dict[str, Any]
    env_id: str
    image_shape: tuple[int, int]
    restore_spec_id: str
    eval_callables: list[dict[str, Any]]

    def close(self) -> None:
        from mwm.eval.validation import close_dataset

        close_dataset(self.dataset)


def resolve_device(raw: str):
    import torch

    if str(raw) == "auto":
        if torch.cuda.is_available():
            try:
                probe = torch.empty(1, device="cuda")
                del probe
                return torch.device("cuda")
            except Exception as exc:
                warnings.warn(
                    "CUDA was reported available but a test allocation failed; "
                    f"falling back to CPU for device=auto: {exc}",
                    RuntimeWarning,
                )
        return torch.device("cpu")
    return torch.device(str(raw))


def _load_eval_config(cfg_path: str, overrides: list[str] | None = None):
    """Load eval config while keeping the scheduler schema closed.

    OmegaConf recursively merges mappings, which would otherwise add the
    default level selectors back into an explicit literal-K scheduler.
    """
    from omegaconf import OmegaConf

    from mwm.config_cli import load_config

    override_values = list(overrides or [])
    cfg = load_config(DEFAULTS, cfg_path, override_values)
    explicit = OmegaConf.load(str(cfg_path))
    if override_values:
        explicit = OmegaConf.merge(explicit, OmegaConf.from_dotlist(override_values))
    explicit_scheduler = OmegaConf.select(explicit, "planner.scheduler")
    if explicit_scheduler is not None:
        cfg.planner.scheduler = OmegaConf.create(
            OmegaConf.to_container(explicit_scheduler, resolve=True)
        )
    return cfg


def load_eval_runtime(cfg_path: str, *, overrides: list[str] | None = None) -> EvalRuntime:
    from mwm.checkpoint_io import load_world_model_from_checkpoint
    from mwm.data.paths import local_path
    from mwm.eval.action_preprocessing import (
        available_stat_keys_for_action_process,
        build_eval_process,
        uses_standardized_action_space,
    )
    from mwm.eval.validation import (
        close_dataset,
        eval_keys_to_load,
        validate_dataset_metadata,
    )
    from mwm.swm.envs import parse_image_shape
    from mwm.swm.restore import eval_callables_for_env

    cfg = _load_eval_config(cfg_path, overrides)
    device = resolve_device(str(cfg.device))
    data_format = str(cfg.data.get("format", "lance")).lower()
    if data_format != "lance":
        raise ValueError(f"MWM evaluation requires format lance, got format={data_format!r}.")
    from stable_worldmodel.data import load_dataset

    model, metadata, epoch = load_world_model_from_checkpoint(
        str(cfg.checkpoint.run_dir),
        None if cfg.checkpoint.epoch is None else int(cfg.checkpoint.epoch),
        device=device,
    )
    dataset = load_dataset(
        local_path(cfg.data.path),
        format=data_format,
        frameskip=1,
        num_steps=2,
        keys_to_load=eval_keys_to_load(cfg, model, metadata),
    )
    process = {}
    if uses_standardized_action_space(model, metadata, cfg):
        stats_dataset = load_dataset(
            local_path(cfg.data.path),
            format=data_format,
            frameskip=1,
            num_steps=2,
            keys_to_load=available_stat_keys_for_action_process(cfg, dataset.column_names),
        )
        try:
            process = build_eval_process(stats_dataset, model, metadata, cfg)
        finally:
            close_dataset(stats_dataset)
    env_id = str(metadata["env_id"])
    image_shape = parse_image_shape(metadata["image_shape"])
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    restore_spec_id, eval_callables = eval_callables_for_env(
        env_id,
        dataset.column_names,
        import_path=restore_import_path,
    )
    validate_dataset_metadata(dataset, metadata, cfg)
    if str(metadata["restore_spec"]) != restore_spec_id:
        raise ValueError(
            f"Runtime restore spec {restore_spec_id!r} does not match checkpoint restore_spec={metadata['restore_spec']!r}."
        )
    return EvalRuntime(
        cfg=cfg,
        device=device,
        model=model,
        metadata=metadata,
        epoch=int(epoch),
        dataset=dataset,
        process=process,
        env_id=env_id,
        image_shape=image_shape,
        restore_spec_id=restore_spec_id,
        eval_callables=eval_callables,
    )


def config_dependency_root() -> Path:
    return Path(__file__).resolve().parents[2]


__all__ = [
    "DEFAULTS",
    "EvalRuntime",
    "_load_eval_config",
    "config_dependency_root",
    "load_eval_runtime",
    "resolve_device",
]
