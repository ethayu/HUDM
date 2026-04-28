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
            "particle_env",
            "init_goal",
            "render",
            "save",
        },
        "plan",
    )
    _reject_unknown_keys(
        cfg.env,
        {"with_velocity", "with_target", "add_noise", "noise_std", "render_size"},
        "plan.env",
    )
    _reject_unknown_keys(
        cfg.world_model,
        {"config_path", "run_dir", "epoch","checkpoint_root", "device", "ensemble"},
        "plan.world_model",
    )
    _reject_unknown_keys(cfg.world_model.ensemble, {"enabled", "run_dirs"}, "plan.world_model.ensemble")
    _reject_unknown_keys(cfg.mpc, {"steps", "horizon", "replan_every"}, "plan.mpc")
    _reject_unknown_keys(
        cfg.cem,
        {
            "pop_size",
            "elite_frac",
            "n_iter",
            "init_std",
            "inject_dataset_gt_actions",
            "gt_inject_count",
            "warm_start",
            "action_low",
            "action_high",
        },
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
            "num_levels",
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
    _reject_unknown_keys(
        cfg.particle_env,
        {"rollout_samples", "objective_space", "batch_mode", "progress", "progress_leave", "fidelity_env"},
        "plan.particle_env",
    )
    _reject_unknown_keys(
        cfg.particle_env.fidelity_env,
        {
            "particle_counts",
            "device",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "min_particles",
            "particle_radius",
            "radius_scale",
            "radius_clip_spacing",
            "stem_w",
            "stem_h",
            "bar_w",
            "bar_h",
            "pusher_radius",
            "sim_hz",
            "control_hz",
            "pusher_k_p",
            "pusher_k_v",
            "substeps",
            "iters",
            "mu",
            "contact_alpha",
            "ground_friction_accel",
            "rest_speed_eps",
            "lin_damp",
            "vel_damp",
            "alpha_rigid",
        },
        "plan.particle_env.fidelity_env",
    )
    _reject_unknown_keys(cfg.init_goal, {"source", "dataset"}, "plan.init_goal")
    _reject_unknown_keys(
        cfg.init_goal.dataset,
        {"zarr_path", "split", "split_ratio", "trajectory_len", "seed", "reconstruct_goal_state"},
        "plan.init_goal.dataset",
    )

    backend = str(cfg.backend).lower()
    if backend not in {"wm", "gt_env", "particle_sim"}:
        raise ValueError(f"plan.backend must be 'wm', 'gt_env', or 'particle_sim', got {cfg.backend}")
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
    if int(getattr(cfg.env, "render_size", 512)) <= 0:
        raise ValueError(f"plan.env.render_size must be > 0, got {cfg.env.render_size}")
    if not isinstance(cfg.cem.warm_start, bool):
        raise ValueError(f"plan.cem.warm_start must be a bool, got {type(cfg.cem.warm_start).__name__}")
    _inject_gt = getattr(cfg.cem, "inject_dataset_gt_actions", False)
    if not isinstance(_inject_gt, bool):
        raise ValueError(
            "plan.cem.inject_dataset_gt_actions must be a bool, "
            f"got {type(_inject_gt).__name__}"
        )
    if int(getattr(cfg.cem, "gt_inject_count", 1)) <= 0:
        raise ValueError(f"plan.cem.gt_inject_count must be > 0, got {cfg.cem.gt_inject_count}")
    if int(cfg.gt_env.rollout_samples) <= 0:
        raise ValueError(f"plan.gt_env.rollout_samples must be > 0, got {cfg.gt_env.rollout_samples}")
    objective_space = str(getattr(cfg.gt_env, "objective_space", "image")).lower()
    if objective_space not in {"image", "state"}:
        raise ValueError(
            f"plan.gt_env.objective_space must be 'image' or 'state', got {cfg.gt_env.objective_space}"
        )
    if int(cfg.particle_env.rollout_samples) <= 0:
        raise ValueError(
            f"plan.particle_env.rollout_samples must be > 0, got {cfg.particle_env.rollout_samples}"
        )
    particle_objective_space = str(getattr(cfg.particle_env, "objective_space", "state")).lower()
    if particle_objective_space not in {"image", "state"}:
        raise ValueError(
            "plan.particle_env.objective_space must be 'image' or 'state', "
            f"got {cfg.particle_env.objective_space}"
        )
    particle_batch_mode = str(getattr(cfg.particle_env, "batch_mode", "auto")).lower()
    if particle_batch_mode not in {"auto", "off", "force"}:
        raise ValueError(
            "plan.particle_env.batch_mode must be one of auto|off|force, "
            f"got {cfg.particle_env.batch_mode}"
        )
    particle_counts = [int(c) for c in list(getattr(cfg.particle_env.fidelity_env, "particle_counts", []))]
    if len(particle_counts) <= 0:
        raise ValueError("plan.particle_env.fidelity_env.particle_counts must contain at least one value.")
    prev_count = None
    for i, count in enumerate(particle_counts):
        if int(count) <= 0:
            raise ValueError(
                f"plan.particle_env.fidelity_env.particle_counts[{i}] must be >= 1, got {count}"
            )
        if prev_count is not None and int(count) <= int(prev_count):
            raise ValueError(
                "plan.particle_env.fidelity_env.particle_counts must be strictly increasing, "
                f"but index {i} has {count} after {prev_count}"
            )
        prev_count = int(count)
    min_particles = int(getattr(cfg.particle_env.fidelity_env, "min_particles", 1))
    if min_particles <= 0:
        raise ValueError(
            "plan.particle_env.fidelity_env.min_particles must be >= 1, "
            f"got {min_particles}"
        )
    radius_scale = float(getattr(cfg.particle_env.fidelity_env, "radius_scale", 1.0))
    if radius_scale <= 0.0:
        raise ValueError(
            "plan.particle_env.fidelity_env.radius_scale must be > 0, "
            f"got {radius_scale}"
        )
    particle_radius = getattr(cfg.particle_env.fidelity_env, "particle_radius", None)
    if particle_radius is not None and float(particle_radius) <= 0.0:
        raise ValueError(
            "plan.particle_env.fidelity_env.particle_radius must be > 0 or null, "
            f"got {particle_radius}"
        )
    sim_hz = int(getattr(cfg.particle_env.fidelity_env, "sim_hz", 100))
    if sim_hz <= 0:
        raise ValueError(
            "plan.particle_env.fidelity_env.sim_hz must be > 0, "
            f"got {sim_hz}"
        )
    control_hz = int(getattr(cfg.particle_env.fidelity_env, "control_hz", 10))
    if control_hz <= 0:
        raise ValueError(
            "plan.particle_env.fidelity_env.control_hz must be > 0, "
            f"got {control_hz}"
        )
    if sim_hz % control_hz != 0:
        raise ValueError(
            "plan.particle_env.fidelity_env.sim_hz must be divisible by control_hz, "
            f"got sim_hz={sim_hz}, control_hz={control_hz}"
        )
    pusher_k_p = float(getattr(cfg.particle_env.fidelity_env, "pusher_k_p", 100.0))
    if pusher_k_p < 0.0:
        raise ValueError(
            "plan.particle_env.fidelity_env.pusher_k_p must be >= 0, "
            f"got {pusher_k_p}"
        )
    pusher_k_v = float(getattr(cfg.particle_env.fidelity_env, "pusher_k_v", 20.0))
    if pusher_k_v < 0.0:
        raise ValueError(
            "plan.particle_env.fidelity_env.pusher_k_v must be >= 0, "
            f"got {pusher_k_v}"
        )
    contact_alpha = float(getattr(cfg.particle_env.fidelity_env, "contact_alpha", 0.35))
    if contact_alpha <= 0.0 or contact_alpha > 1.0:
        raise ValueError(
            "plan.particle_env.fidelity_env.contact_alpha must be in (0, 1], "
            f"got {contact_alpha}"
        )
    ground_friction_accel = float(getattr(cfg.particle_env.fidelity_env, "ground_friction_accel", 2.0))
    if ground_friction_accel < 0.0:
        raise ValueError(
            "plan.particle_env.fidelity_env.ground_friction_accel must be >= 0, "
            f"got {ground_friction_accel}"
        )
    rest_speed_eps = float(getattr(cfg.particle_env.fidelity_env, "rest_speed_eps", 0.01))
    if rest_speed_eps < 0.0:
        raise ValueError(
            "plan.particle_env.fidelity_env.rest_speed_eps must be >= 0, "
            f"got {rest_speed_eps}"
        )
    num_levels = getattr(cfg.fidelity, "num_levels", None)
    if num_levels is not None and int(num_levels) <= 0:
        raise ValueError(f"plan.fidelity.num_levels must be > 0, got {cfg.fidelity.num_levels}")
    gt_num_levels = int(getattr(cfg.gt_env.fidelity_env, "num_levels", 4))
    if gt_num_levels <= 0:
        raise ValueError(f"plan.gt_env.fidelity_env.num_levels must be > 0, got {gt_num_levels}")
    if backend != "wm" and str(cfg.fidelity.rollout.mode).lower() == "uncertainty_downshift":
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
    seed_cfg = getattr(cfg.init_goal.dataset, "seed", 0)
    if isinstance(seed_cfg, str):
        if seed_cfg.strip().lower() != "random":
            raise ValueError(
                "plan.init_goal.dataset.seed must be an int or the string 'random', "
                f"got {cfg.init_goal.dataset.seed!r}"
            )
    else:
        try:
            int(seed_cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "plan.init_goal.dataset.seed must be an int or the string 'random', "
                f"got {cfg.init_goal.dataset.seed!r}"
            ) from exc
    recon_mode = int(getattr(cfg.init_goal.dataset, "reconstruct_goal_state", 0))
    if recon_mode not in {0, 1, 2, 3}:
        raise ValueError(
            "plan.init_goal.dataset.reconstruct_goal_state must be one of {0,1,2,3}, "
            f"got {cfg.init_goal.dataset.reconstruct_goal_state!r}"
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
        {"zarr_path", "split_ratio", "action_dim", "action_mode", "synthetic", "horizon_T"},
        "world.data",
    )
    _reject_unknown_keys(
        cfg.model,
        {"D", "K", "decoder_mode", "dynamics_mode","input"},
        "world.model",
    )
    _reject_unknown_keys(
        cfg.train,
        {"batch_size", "num_workers", "no_cuda", "checkpoint_dir", "run_name","horizon"},
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
    model_input = str(getattr(cfg.model, "input", "images")).lower()
    if model_input not in {"images", "state", "mixed"}:
        raise ValueError("model.input must be one of 'images', 'state', or 'mixed'")

    if int(cfg.train.batch_size) <= 0:
        raise ValueError(f"train.batch_size must be > 0, got {cfg.train.batch_size}")
    if int(cfg.train.num_workers) < 0:
        raise ValueError(f"train.num_workers must be >= 0, got {cfg.train.num_workers}")
    if int(getattr(cfg.train, "horizon", getattr(cfg.data, "horizon_T", 0))) <= 0:
        raise ValueError(
            "train.horizon must be > 0 (or world.data.horizon_T must be > 0 when used as fallback)."
        )

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
