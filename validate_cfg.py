"""Centralized config validation helpers for top-level entry scripts."""

from __future__ import annotations

import os
from typing import List


def _reject_unknown_keys(cfg_section, allowed: set[str], prefix: str) -> None:
    if cfg_section is None or not hasattr(cfg_section, "keys"):
        raise ValueError(f"{prefix} must be a mapping/object.")
    for key in cfg_section.keys():
        if key not in allowed:
            raise ValueError(f"Unknown config key: {prefix}.{key}")


def validate_plan_cfg(cfg) -> None:
    _reject_unknown_keys(
        cfg,
        {
            "backend",
            "env_id",
            "env",
            "world_model",
            "mpc",
            "cem",
            "objective",
            "fidelity",
            "gt_env",
            "init_goal",
            "render",
            "save",
        },
        "plan",
    )
    _reject_unknown_keys(
        cfg.env,
        {"with_velocity", "with_target", "add_noise", "noise_std"},
        "plan.env",
    )
    _reject_unknown_keys(
        cfg.world_model,
        {"config_path", "run_dir", "checkpoint_root", "device", "ensemble"},
        "plan.world_model",
    )
    _reject_unknown_keys(cfg.world_model.ensemble, {"enabled", "run_dirs"}, "plan.world_model.ensemble")
    _reject_unknown_keys(cfg.mpc, {"steps", "horizon", "replan_every"}, "plan.mpc")
    _reject_unknown_keys(
        cfg.cem,
        {"pop_size", "elite_frac", "n_iter", "init_std", "warm_start", "action_low", "action_high"},
        "plan.cem",
    )
    _reject_unknown_keys(
        cfg.objective,
        {
            "latent_metric",
            "terminal_weight",
            "running_weight",
            "action_l2_weight",
            "eef_weight",
            "block_pos_weight",
            "block_angle_weight",
            "state_l2_weight",
        },
        "plan.objective",
    )
    _reject_unknown_keys(cfg.fidelity, {"enabled", "num_levels", "mpc", "cem", "rollout"}, "plan.fidelity")
    _reject_unknown_keys(cfg.fidelity.mpc, {"mode", "level", "start_level", "end_level"}, "plan.fidelity.mpc")
    _reject_unknown_keys(cfg.fidelity.cem, {"mode", "level", "start_level", "end_level"}, "plan.fidelity.cem")
    _reject_unknown_keys(
        cfg.fidelity.rollout,
        {"mode", "level", "start_level", "end_level", "uncertainty"},
        "plan.fidelity.rollout",
    )
    _reject_unknown_keys(
        cfg.fidelity.rollout.uncertainty,
        {"criterion", "threshold", "percentile", "min_level", "max_downshifts_per_step"},
        "plan.fidelity.rollout.uncertainty",
    )
    _reject_unknown_keys(
        cfg.gt_env,
        {"rollout_samples", "objective_space", "progress", "progress_leave", "fidelity_env"},
        "plan.gt_env",
    )
    _reject_unknown_keys(
        cfg.gt_env.fidelity_env,
        {
            "mode",
            "blur_sigma_max",
            "pool_scale_max",
            "quantize_levels_min",
            "quantize_levels_max",
            "action_noise_std_max",
            "downsample_output",
            "min_downsample_size",
        },
        "plan.gt_env.fidelity_env",
    )
    _reject_unknown_keys(cfg.init_goal, {"source", "dataset"}, "plan.init_goal")
    _reject_unknown_keys(
        cfg.init_goal.dataset,
        {"zarr_path", "split", "split_ratio", "trajectory_len", "seed"},
        "plan.init_goal.dataset",
    )

    backend = str(cfg.backend).lower()
    if backend not in {"wm", "gt_env"}:
        raise ValueError(f"plan.backend must be 'wm' or 'gt_env', got {cfg.backend}")
    if int(cfg.mpc.steps) <= 0:
        raise ValueError(f"plan.mpc.steps must be > 0, got {cfg.mpc.steps}")
    if int(cfg.mpc.horizon) <= 0:
        raise ValueError(f"plan.mpc.horizon must be > 0, got {cfg.mpc.horizon}")
    if int(cfg.mpc.replan_every) <= 0:
        raise ValueError(f"plan.mpc.replan_every must be > 0, got {cfg.mpc.replan_every}")
    if int(cfg.mpc.replan_every) > int(cfg.mpc.horizon):
        raise ValueError(
            f"plan.mpc.replan_every must be <= plan.mpc.horizon, got "
            f"replan_every={cfg.mpc.replan_every}, horizon={cfg.mpc.horizon}"
        )
    if int(cfg.cem.pop_size) <= 0:
        raise ValueError(f"plan.cem.pop_size must be > 0, got {cfg.cem.pop_size}")
    if float(cfg.cem.elite_frac) <= 0.0 or float(cfg.cem.elite_frac) >= 1.0:
        raise ValueError(f"plan.cem.elite_frac must be in (0,1), got {cfg.cem.elite_frac}")
    if int(cfg.cem.n_iter) <= 0:
        raise ValueError(f"plan.cem.n_iter must be > 0, got {cfg.cem.n_iter}")
    if float(cfg.cem.init_std) <= 0.0:
        raise ValueError(f"plan.cem.init_std must be > 0, got {cfg.cem.init_std}")
    if not isinstance(cfg.cem.warm_start, bool):
        raise ValueError(f"plan.cem.warm_start must be a bool, got {type(cfg.cem.warm_start).__name__}")
    if int(cfg.gt_env.rollout_samples) <= 0:
        raise ValueError(f"plan.gt_env.rollout_samples must be > 0, got {cfg.gt_env.rollout_samples}")
    objective_space = str(getattr(cfg.gt_env, "objective_space", "image")).lower()
    if objective_space not in {"image", "state"}:
        raise ValueError(
            f"plan.gt_env.objective_space must be 'image' or 'state', got {cfg.gt_env.objective_space}"
        )
    if int(cfg.fidelity.num_levels) <= 0:
        raise ValueError(f"plan.fidelity.num_levels must be > 0, got {cfg.fidelity.num_levels}")
    if backend == "gt_env" and str(cfg.fidelity.rollout.mode).lower() == "uncertainty_downshift":
        raise ValueError(
            "plan.fidelity.rollout.mode=uncertainty_downshift is only supported for backend='wm'."
        )

    init_src = str(cfg.init_goal.source).lower()
    if init_src not in {"random", "dataset"}:
        raise ValueError(f"plan.init_goal.source must be 'random' or 'dataset', got {cfg.init_goal.source}")
    if init_src == "dataset" and int(cfg.init_goal.dataset.trajectory_len) <= 0:
        raise ValueError(
            "plan.init_goal.dataset.trajectory_len must be > 0 when source='dataset'."
        )

    ens_enabled = bool(getattr(cfg.world_model.ensemble, "enabled", False))
    run_dir = cfg.world_model.run_dir
    run_dirs = list(getattr(cfg.world_model.ensemble, "run_dirs", []))
    if backend == "wm" and ens_enabled:
        if len(run_dirs) < 2:
            raise ValueError(
                "plan.world_model.ensemble.enabled=true requires at least 2 run_dirs."
            )
        if run_dir is not None:
            raise ValueError(
                "plan.world_model.run_dir must be null when ensemble is enabled; "
                "use plan.world_model.ensemble.run_dirs."
            )
    if backend == "wm" and (not ens_enabled):
        if len(run_dirs) > 0:
            raise ValueError(
                "plan.world_model.ensemble.run_dirs is set but ensemble is disabled."
            )


def validate_world_cfg(cfg, wandb_available: bool = True) -> None:
    _reject_unknown_keys(
        cfg,
        {"seed", "data", "model", "train", "optim", "loss", "schedule", "wandb"},
        "world",
    )
    _reject_unknown_keys(
        cfg.data,
        {"zarr_path", "split_ratio", "action_dim", "action_mode", "synthetic"},
        "world.data",
    )
    _reject_unknown_keys(
        cfg.model,
        {"D", "K", "decoder_mode", "dynamics_mode"},
        "world.model",
    )
    _reject_unknown_keys(
        cfg.train,
        {"batch_size", "num_workers", "no_cuda", "checkpoint_dir", "run_name"},
        "world.train",
    )
    _reject_unknown_keys(cfg.optim, {"lr"}, "world.optim")
    _reject_unknown_keys(
        cfg.loss, {"recon_weight", "teacher_weight", "rollout_weight"}, "world.loss"
    )
    _reject_unknown_keys(
        cfg.schedule, {"max_epochs", "patience", "min_delta"}, "world.schedule"
    )
    _reject_unknown_keys(cfg.wandb, {"enable", "project", "run_name"}, "world.wandb")

    if not os.path.exists(str(cfg.data.zarr_path)):
        raise FileNotFoundError(f"data.zarr_path not found: {cfg.data.zarr_path}")
    split_ratio = float(cfg.data.split_ratio)
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"data.split_ratio must be in (0,1), got {cfg.data.split_ratio}")
    if int(cfg.data.action_dim) <= 0:
        raise ValueError(f"data.action_dim must be > 0, got {cfg.data.action_dim}")
    action_mode = str(getattr(cfg.data, "action_mode", "relative"))
    if action_mode != "relative":
        raise ValueError(
            "train_world.py requires data.action_mode='relative' so the injected "
            "null action remains identity."
        )

    K: List[int] = [int(k) for k in list(cfg.model.K)]
    D: int = int(cfg.model.D)
    if D <= 0:
        raise ValueError(f"model.D must be > 0, got {cfg.model.D}")
    if len(K) == 0:
        raise ValueError("model.K must contain at least one level.")
    if any(k <= 0 for k in K):
        raise ValueError(f"model.K must contain positive ints, got {K}")
    if any(K[i] >= K[i + 1] for i in range(len(K) - 1)):
        raise ValueError(f"model.K must be strictly increasing, got {K}")
    if K[-1] != D:
        raise ValueError(f"Largest model.K must equal model.D, got K[-1]={K[-1]}, D={D}")
    if str(getattr(cfg.model, "decoder_mode", "per_level")) not in {"per_level", "shared"}:
        raise ValueError("model.decoder_mode must be 'per_level' or 'shared'")
    if str(getattr(cfg.model, "dynamics_mode", "per_level")) not in {"per_level", "shared"}:
        raise ValueError("model.dynamics_mode must be 'per_level' or 'shared'")

    if int(cfg.train.batch_size) <= 0:
        raise ValueError(f"train.batch_size must be > 0, got {cfg.train.batch_size}")
    if int(cfg.train.num_workers) < 0:
        raise ValueError(f"train.num_workers must be >= 0, got {cfg.train.num_workers}")

    if float(cfg.optim.lr) <= 0.0:
        raise ValueError(f"optim.lr must be > 0, got {cfg.optim.lr}")

    recon_w = float(getattr(cfg.loss, "recon_weight", 1.0))
    teacher_w = float(getattr(cfg.loss, "teacher_weight", 1.0))
    rollout_w = float(getattr(cfg.loss, "rollout_weight", 1.0))
    if recon_w < 0.0 or teacher_w < 0.0 or rollout_w < 0.0:
        raise ValueError("loss weights must be non-negative.")
    if recon_w + teacher_w + rollout_w <= 0.0:
        raise ValueError("At least one loss weight must be > 0.")

    if int(getattr(cfg.schedule, "max_epochs", 30)) <= 0:
        raise ValueError(f"schedule.max_epochs must be > 0, got {cfg.schedule.max_epochs}")
    if int(cfg.schedule.patience) <= 0:
        raise ValueError(f"schedule.patience must be > 0, got {cfg.schedule.patience}")
    if float(cfg.schedule.min_delta) < 0.0:
        raise ValueError(f"schedule.min_delta must be >= 0, got {cfg.schedule.min_delta}")

    s_cfg = getattr(cfg.data, "synthetic", None)
    if s_cfg and bool(getattr(s_cfg, "enable", False)):
        _reject_unknown_keys(
            s_cfg,
            {"enable", "zarr_path", "frac", "total_train", "seed", "val_source"},
            "world.data.synthetic",
        )
        if not os.path.exists(str(s_cfg.zarr_path)):
            raise FileNotFoundError(f"data.synthetic.zarr_path not found: {s_cfg.zarr_path}")
        frac = float(getattr(s_cfg, "frac", 0.5))
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"data.synthetic.frac must be in [0,1], got {frac}")
        total_train = getattr(s_cfg, "total_train", None)
        if total_train is not None and int(total_train) <= 0:
            raise ValueError(
                f"data.synthetic.total_train must be > 0 or null, got {total_train}"
            )
        val_source = str(getattr(s_cfg, "val_source", "real")).lower()
        if val_source not in {"real", "synthetic", "mixed"}:
            raise ValueError(
                f"data.synthetic.val_source must be one of real|synthetic|mixed, got {val_source}"
            )

    if cfg.get("wandb", {}).get("enable", False) and not bool(wandb_available):
        raise ImportError("wandb is enabled in config but wandb is not installed.")
