from __future__ import annotations

import warnings
from pathlib import Path


DEFAULTS = {
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


def main(cfg_path: str, *, overrides: list[str] | None = None) -> None:
    from omegaconf import OmegaConf

    from mwm.checkpoint_io import load_world_model_from_checkpoint
    from mwm.config_cli import load_config
    from mwm.data.paths import local_path
    from mwm.dependency_refs import dependency_refs
    from mwm.eval.action_preprocessing import (
        available_stat_keys_for_action_process,
        build_eval_process,
        uses_standardized_action_space,
    )
    from mwm.eval.execution import (
        combine_mwm_diagnostics,
        combine_policy_diagnostics,
        combine_swm_results,
        run_batch,
    )
    from mwm.eval.manifest import pairs_for_eval
    from mwm.eval.policy import model_accounting
    from mwm.eval.validation import (
        close_dataset,
        dataset_path,
        dataset_runtime_metadata,
        eval_keys_to_load,
        validate_dataset_metadata,
    )
    from mwm.io import jsonable, write_json
    from mwm.swm.envs import parse_image_shape
    from mwm.swm.restore import eval_callables_for_env

    cfg = load_config(DEFAULTS, cfg_path, overrides or [])
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

    pairs, manifest_info = pairs_for_eval(dataset=dataset, cfg=cfg, env_id=env_id, restore_spec_id=restore_spec_id)
    all_results = []
    batch_size = max(1, int(cfg.eval.num_envs))
    for batch_index, offset in enumerate(range(0, len(pairs), batch_size)):
        all_results.append(
            run_batch(
                env_id=env_id,
                image_shape=image_shape,
                model=model,
                metadata=metadata,
                dataset=dataset,
                pairs=pairs[offset : offset + batch_size],
                cfg=cfg,
                device=device,
                eval_callables=eval_callables,
                batch_index=batch_index,
                process=process,
            )
        )
    videos = [video for batch in all_results for video in batch.get("videos", [])]
    checkpoint_ref = str(cfg.checkpoint.run_dir)

    output = {
        "env_id": env_id,
        "checkpoint_run_dir": checkpoint_ref,
        "checkpoint_epoch": int(epoch),
        "dataset": dataset_path(dataset, cfg),
        "episodes": int(cfg.eval.episodes),
        "goal_offset": int(cfg.eval.goal_offset),
        "eval_budget": int(cfg.eval.budget),
        "restore_spec": restore_spec_id,
        "swm_results": combine_swm_results(all_results),
        "planning_diagnostics": combine_mwm_diagnostics(all_results),
        "policy_diagnostics": combine_policy_diagnostics(all_results),
        "model_accounting": model_accounting(model),
        "dataset_metadata": dataset_runtime_metadata(dataset, cfg),
        "manifest": manifest_info,
        "batches": all_results,
        "videos": videos,
        "dependencies": dependency_refs(Path(__file__).resolve().parents[2]),
    }
    output["schedule"] = jsonable(OmegaConf.to_container(cfg.planner.scheduler, resolve=True))
    output["seed"] = int(cfg.eval.seed)
    if cfg.get("config", None):
        output["config"] = jsonable(OmegaConf.to_container(cfg.config, resolve=True))
    output_path = Path(str(cfg.eval.output_path))
    write_json(output_path, output)
    print(f"Wrote MWM planning results to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an MWM checkpoint.")
    parser.add_argument("config", help="Evaluation YAML config")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. eval.seed=1")
    args = parser.parse_args()
    main(args.config, overrides=args.set)


__all__ = ["DEFAULTS", "main", "resolve_device"]
