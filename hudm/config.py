from __future__ import annotations

import copy
import os
from typing import Any

from omegaconf import DictConfig, OmegaConf

from hudm.specs import BenchmarkEntry, BenchmarkSpec, ExperimentSpec, ExperimentVariant, make_plan_spec
from validate_cfg import validate_plan_cfg


def _reject_unknown_keys(cfg_section, allowed: set[str], prefix: str) -> None:
    if cfg_section is None or not hasattr(cfg_section, "keys"):
        raise ValueError(f"{prefix} must be a mapping/object.")
    for key in cfg_section.keys():
        if key not in allowed:
            raise ValueError(f"Unknown config key: {prefix}.{key}")


def _resolve_import_path(base_dir: str, import_path: str) -> str:
    if os.path.isabs(import_path):
        return import_path
    return os.path.normpath(os.path.join(base_dir, import_path))


def load_config_with_imports(cfg_path: str) -> DictConfig:
    cfg_path = os.path.abspath(cfg_path)
    root = OmegaConf.load(cfg_path)
    merged = OmegaConf.create()
    imports = list(root.get("imports", [])) if getattr(root, "get", None) is not None else []
    base_dir = os.path.dirname(cfg_path)
    for import_path in imports:
        merged = OmegaConf.merge(
            merged,
            load_config_with_imports(_resolve_import_path(base_dir, str(import_path))),
        )
    if "imports" in root:
        del root["imports"]
    return OmegaConf.merge(merged, root)


def _plan_defaults() -> dict[str, Any]:
    return {
        "task": {
            "env_id": "pusht",
            "env": {
                "with_velocity": True,
                "with_target": True,
                "add_noise": 0,
                "noise_std": 0.0,
                "render_size": 512,
            },
            "init_goal": {
                "source": "random",
                "dataset": {
                    "zarr_path": None,
                    "split": "valid",
                    "split_ratio": None,
                    "trajectory_len": 20,
                    "seed": 0,
                    "reconstruct_goal_state": 0,
                },
            },
        },
        "budget": {
            "max_env_steps": 50,
        },
        "planner": {
            "horizon": 20,
            "replan_every": 1,
            "cem": {
                "pop_size": 256,
                "elite_frac": 0.1,
                "n_iter": 5,
                "init_std": 1.0,
                "inject_dataset_gt_actions": False,
                "warm_start": True,
                "action_low": None,
                "action_high": None,
            },
            "fidelity": {
                "enabled": True,
                "num_levels": 4,
                "mpc": {
                    "mode": "linear",
                    "level": "finest",
                    "start_level": "coarsest",
                    "end_level": "finest",
                },
                "cem": {
                    "mode": "linear",
                    "level": "base",
                    "start_level": "base",
                    "end_level": "finest",
                },
                "rollout": {
                    "mode": "fixed",
                    "level": "base",
                    "start_level": "base",
                    "end_level": "coarsest",
                    "uncertainty": {
                        "criterion": "mean",
                        "threshold": 0.05,
                        "percentile": 0.8,
                        "min_level": "coarsest",
                        "max_downshifts_per_step": 1,
                    },
                },
            },
        },
        "backend": {
            "kind": "wm",
            "wm": {
                "world_model": {
                    "config_path": None,
                    "run_dir": None,
                    "epoch": None,
                    "checkpoint_root": "checkpoints_world",
                    "device": "auto",
                    "ensemble": {
                        "enabled": False,
                        "run_dirs": [],
                    },
                },
                "objective": {
                    "latent_metric": "l2",
                    "terminal_weight": 1.0,
                    "running_weight": 0.0,
                    "action_l2_weight": 0.0,
                },
            },
            "gt_env": {
                "rollout_samples": 1,
                "objective_space": "image",
                "progress": True,
                "progress_leave": False,
                "objective": {
                    "action_l2_weight": 0.0,
                    "eef_weight": 1.0,
                    "block_pos_weight": 1.0,
                    "block_angle_weight": 0.1,
                    "state_l2_weight": 0.0,
                },
                "fidelity_env": {
                    "mode": "blur_avgpool",
                    "blur_sigma_max": 2.0,
                    "pool_scale_max": 4,
                    "quantize_levels_min": 8,
                    "quantize_levels_max": 256,
                    "action_noise_std_max": 0.0,
                    "downsample_output": False,
                    "min_downsample_size": 12,
                },
            },
            "particle_sim": {
                "rollout_samples": 1,
                "objective_space": "state",
                "progress": True,
                "progress_leave": False,
                "objective": {
                    "action_l2_weight": 0.0,
                    "eef_weight": 1.0,
                    "block_pos_weight": 1.0,
                    "block_angle_weight": 0.1,
                    "state_l2_weight": 0.0,
                },
                "fidelity_env": {
                    "spacings": [0.020, 0.016, 0.013, 0.010],
                    "device": "auto",
                    "xmin": -0.25,
                    "xmax": 0.25,
                    "ymin": -0.25,
                    "ymax": 0.25,
                    "min_particles": 1,
                    "coarsest_single_particle": True,
                    "particle_radius": None,
                    "radius_scale": 1.0,
                    "radius_clip_spacing": False,
                    "stem_w": 0.05,
                    "stem_h": 0.10,
                    "bar_w": 0.12,
                    "bar_h": 0.04,
                    "pusher_radius": 0.015,
                    "pusher_speed": 0.6,
                    "pusher_interp_substeps": True,
                    "frame_dt": 1.0 / 60.0,
                    "substeps": 16,
                    "iters": 8,
                    "mu": 0.6,
                    "contact_alpha": 0.35,
                    "ground_friction_accel": 2.0,
                    "rest_speed_eps": 0.01,
                    "lin_damp": 0.995,
                    "vel_damp": 0.999,
                    "alpha_rigid": 1.0,
                },
            },
        },
        "artifacts": {
            "render": False,
            "save": False,
        },
    }


def _experiment_defaults() -> dict[str, Any]:
    return {
        "name": "experiment",
        "shared_plan": {
            "imports": [],
            "plan": {},
        },
        "rollouts": {
            "seed": 0,
            "num_rollouts": 4,
            "sample_without_replacement": True,
        },
        "execution": {
            "mode": "auto",
            "max_workers": 2,
        },
        "terminal": {
            "mode": "compact",
        },
        "reporting": {
            "output_root": "rollouts",
        },
        "baseline": None,
        "variants": [],
    }


def _benchmark_defaults() -> dict[str, Any]:
    return {
        "name": "benchmark",
        "output_root": "rollouts",
        "experiments": [],
    }


def _prune_inactive_backend(cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    kind = str(cfg.backend.kind).lower()
    for other in ("wm", "gt_env", "particle_sim"):
        if other != kind and other in cfg.backend:
            del cfg.backend[other]
    return cfg


def validate_plan_spec_cfg(cfg: DictConfig) -> None:
    _reject_unknown_keys(cfg, {"task", "budget", "planner", "backend", "artifacts"}, "plan")
    _reject_unknown_keys(cfg.task, {"env_id", "env", "init_goal"}, "plan.task")
    _reject_unknown_keys(
        cfg.task.env,
        {"with_velocity", "with_target", "add_noise", "noise_std", "render_size"},
        "plan.task.env",
    )
    _reject_unknown_keys(cfg.task.init_goal, {"source", "dataset"}, "plan.task.init_goal")
    _reject_unknown_keys(
        cfg.task.init_goal.dataset,
        {"zarr_path", "split", "split_ratio", "trajectory_len", "seed", "reconstruct_goal_state"},
        "plan.task.init_goal.dataset",
    )
    _reject_unknown_keys(cfg.budget, {"max_env_steps"}, "plan.budget")
    _reject_unknown_keys(cfg.planner, {"horizon", "replan_every", "cem", "fidelity"}, "plan.planner")
    _reject_unknown_keys(
        cfg.planner.cem,
        {
            "pop_size",
            "elite_frac",
            "n_iter",
            "init_std",
            "inject_dataset_gt_actions",
            "warm_start",
            "action_low",
            "action_high",
        },
        "plan.planner.cem",
    )
    _reject_unknown_keys(cfg.planner.fidelity, {"enabled", "num_levels", "mpc", "cem", "rollout"}, "plan.planner.fidelity")
    _reject_unknown_keys(cfg.planner.fidelity.mpc, {"mode", "level", "start_level", "end_level"}, "plan.planner.fidelity.mpc")
    _reject_unknown_keys(cfg.planner.fidelity.cem, {"mode", "level", "start_level", "end_level"}, "plan.planner.fidelity.cem")
    _reject_unknown_keys(
        cfg.planner.fidelity.rollout,
        {"mode", "level", "start_level", "end_level", "uncertainty"},
        "plan.planner.fidelity.rollout",
    )
    _reject_unknown_keys(
        cfg.planner.fidelity.rollout.uncertainty,
        {"criterion", "threshold", "percentile", "min_level", "max_downshifts_per_step"},
        "plan.planner.fidelity.rollout.uncertainty",
    )
    _reject_unknown_keys(cfg.backend, {"kind", "wm", "gt_env", "particle_sim"}, "plan.backend")
    _reject_unknown_keys(cfg.artifacts, {"render", "save"}, "plan.artifacts")

    kind = str(cfg.backend.kind).lower()
    if kind not in {"wm", "gt_env", "particle_sim"}:
        raise ValueError(f"plan.backend.kind must be wm|gt_env|particle_sim, got {cfg.backend.kind}")
    for other in {"wm", "gt_env", "particle_sim"}:
        if other != kind and getattr(cfg.backend, other, None) is not None:
            raise ValueError(
                f"Resolved plan must not carry inactive backend block plan.backend.{other} "
                f"when kind={kind}."
            )

    if kind == "wm":
        _reject_unknown_keys(cfg.backend.wm, {"world_model", "objective"}, "plan.backend.wm")
        _reject_unknown_keys(
            cfg.backend.wm.world_model,
            {"config_path", "run_dir", "epoch", "checkpoint_root", "device", "ensemble"},
            "plan.backend.wm.world_model",
        )
        _reject_unknown_keys(
            cfg.backend.wm.world_model.ensemble,
            {"enabled", "run_dirs"},
            "plan.backend.wm.world_model.ensemble",
        )
        _reject_unknown_keys(
            cfg.backend.wm.objective,
            {"latent_metric", "terminal_weight", "running_weight", "action_l2_weight"},
            "plan.backend.wm.objective",
        )
    elif kind == "gt_env":
        _reject_unknown_keys(
            cfg.backend.gt_env,
            {"rollout_samples", "objective_space", "progress", "progress_leave", "fidelity_env", "objective"},
            "plan.backend.gt_env",
        )
        _reject_unknown_keys(
            cfg.backend.gt_env.fidelity_env,
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
            "plan.backend.gt_env.fidelity_env",
        )
        _reject_unknown_keys(
            cfg.backend.gt_env.objective,
            {"action_l2_weight", "eef_weight", "block_pos_weight", "block_angle_weight", "state_l2_weight"},
            "plan.backend.gt_env.objective",
        )
    else:
        _reject_unknown_keys(
            cfg.backend.particle_sim,
            {"rollout_samples", "objective_space", "progress", "progress_leave", "fidelity_env", "objective"},
            "plan.backend.particle_sim",
        )
        _reject_unknown_keys(
            cfg.backend.particle_sim.fidelity_env,
            {
                "spacings",
                "device",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "min_particles",
                "coarsest_single_particle",
                "particle_radius",
                "radius_scale",
                "radius_clip_spacing",
                "stem_w",
                "stem_h",
                "bar_w",
                "bar_h",
                "pusher_radius",
                "pusher_speed",
                "pusher_interp_substeps",
                "frame_dt",
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
            "plan.backend.particle_sim.fidelity_env",
        )
        _reject_unknown_keys(
            cfg.backend.particle_sim.objective,
            {"action_l2_weight", "eef_weight", "block_pos_weight", "block_angle_weight", "state_l2_weight"},
            "plan.backend.particle_sim.objective",
        )

    if int(cfg.budget.max_env_steps) <= 0:
        raise ValueError("plan.budget.max_env_steps must be > 0")
    if int(cfg.planner.horizon) <= 0:
        raise ValueError("plan.planner.horizon must be > 0")
    if int(cfg.planner.replan_every) <= 0:
        raise ValueError("plan.planner.replan_every must be > 0")


def plan_spec_to_runtime_cfg(clean_cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(clean_cfg, resolve=True))
    validate_plan_spec_cfg(cfg)
    kind = str(cfg.backend.kind).lower()
    default_wm_world_model = copy.deepcopy(_plan_defaults()["backend"]["wm"]["world_model"])
    default_gt_env = copy.deepcopy(_plan_defaults()["backend"]["gt_env"])
    default_particle_env = copy.deepcopy(_plan_defaults()["backend"]["particle_sim"])
    default_gt_env.pop("objective", None)
    default_particle_env.pop("objective", None)

    runtime = {
        "backend": kind,
        "env_id": cfg.task.env_id,
        "env": OmegaConf.to_container(cfg.task.env, resolve=True),
        "world_model": (
            OmegaConf.to_container(cfg.backend.wm.world_model, resolve=True)
            if kind == "wm"
            else default_wm_world_model
        ),
        "mpc": {
            "steps": int(cfg.budget.max_env_steps),
            "horizon": int(cfg.planner.horizon),
            "replan_every": int(cfg.planner.replan_every),
        },
        "cem": OmegaConf.to_container(cfg.planner.cem, resolve=True),
        "objective": copy.deepcopy(
            OmegaConf.to_container(cfg.backend[kind].objective, resolve=True)
        ),
        "fidelity": OmegaConf.to_container(cfg.planner.fidelity, resolve=True),
        "gt_env": default_gt_env,
        "particle_env": default_particle_env,
        "init_goal": OmegaConf.to_container(cfg.task.init_goal, resolve=True),
        "render": bool(cfg.artifacts.render),
        "save": bool(cfg.artifacts.save),
    }
    if kind == "gt_env":
        gt_env_cfg = OmegaConf.to_container(cfg.backend.gt_env, resolve=True)
        gt_env_cfg.pop("objective", None)
        runtime["gt_env"] = gt_env_cfg
    elif kind == "particle_sim":
        particle_cfg = OmegaConf.to_container(cfg.backend.particle_sim, resolve=True)
        particle_cfg.pop("objective", None)
        runtime["particle_env"] = particle_cfg

    runtime_cfg = OmegaConf.create(runtime)
    validate_plan_cfg(runtime_cfg)
    return runtime_cfg


def resolve_plan_spec(cfg_path: str, *, name: str | None = None) -> Any:
    root = load_config_with_imports(cfg_path)
    clean_cfg = OmegaConf.merge(_plan_defaults(), root.get("plan", root))
    clean_cfg = _prune_inactive_backend(clean_cfg)
    validate_plan_spec_cfg(clean_cfg)
    runtime_cfg = plan_spec_to_runtime_cfg(clean_cfg)
    return make_plan_spec(
        name=name or os.path.splitext(os.path.basename(cfg_path))[0],
        config_path=os.path.abspath(cfg_path),
        clean_cfg=clean_cfg,
        runtime_cfg=runtime_cfg,
    )


def _resolve_plan_fragment_bundle(bundle_cfg, *, base_dir: str, name: str) -> Any:
    imports = list(getattr(bundle_cfg, "imports", []))
    merged = OmegaConf.create({"plan": copy.deepcopy(_plan_defaults())})
    for import_path in imports:
        imported = load_config_with_imports(_resolve_import_path(base_dir, str(import_path)))
        merged = OmegaConf.merge(merged, imported)
    merged = OmegaConf.merge(
        merged,
        OmegaConf.create({"plan": OmegaConf.to_container(getattr(bundle_cfg, "plan", {}), resolve=True)}),
    )
    clean_cfg = _prune_inactive_backend(OmegaConf.merge(_plan_defaults(), merged.get("plan", merged)))
    validate_plan_spec_cfg(clean_cfg)
    runtime_cfg = plan_spec_to_runtime_cfg(clean_cfg)
    return make_plan_spec(name=name, config_path=None, clean_cfg=clean_cfg, runtime_cfg=runtime_cfg)


def validate_experiment_cfg(cfg: DictConfig) -> None:
    _reject_unknown_keys(
        cfg,
        {"name", "shared_plan", "rollouts", "execution", "terminal", "reporting", "baseline", "variants"},
        "experiment",
    )
    _reject_unknown_keys(cfg.shared_plan, {"imports", "plan"}, "experiment.shared_plan")
    _reject_unknown_keys(
        cfg.rollouts,
        {"seed", "num_rollouts", "sample_without_replacement"},
        "experiment.rollouts",
    )
    _reject_unknown_keys(cfg.execution, {"mode", "max_workers"}, "experiment.execution")
    _reject_unknown_keys(cfg.terminal, {"mode"}, "experiment.terminal")
    _reject_unknown_keys(cfg.reporting, {"output_root"}, "experiment.reporting")
    if cfg.variants is None or len(cfg.variants) <= 0:
        raise ValueError("experiment.variants must contain at least one variant.")
    names: list[str] = []
    for idx, variant in enumerate(cfg.variants):
        _reject_unknown_keys(variant, {"name", "imports", "overrides"}, f"experiment.variants[{idx}]")
        if not str(variant.name).strip():
            raise ValueError(f"experiment.variants[{idx}].name must be non-empty.")
        names.append(str(variant.name))
        overrides = getattr(variant, "overrides", {})
        if hasattr(overrides, "keys"):
            if "task" in overrides or "budget" in overrides:
                raise ValueError(
                    f"experiment.variants[{idx}] may not override task or budget. "
                    "Create a separate experiment instead."
                )
    if len(set(names)) != len(names):
        raise ValueError("experiment.variants contains duplicate names.")
    if str(getattr(cfg.execution, "mode", "auto")).lower() not in {"auto", "serial", "process"}:
        raise ValueError("experiment.execution.mode must be one of auto|serial|process.")
    if str(getattr(cfg.terminal, "mode", "compact")).lower() not in {"quiet", "compact", "verbose"}:
        raise ValueError("experiment.terminal.mode must be one of quiet|compact|verbose.")
    if int(getattr(cfg.execution, "max_workers", 1)) <= 0:
        raise ValueError("experiment.execution.max_workers must be > 0.")
    if int(cfg.rollouts.num_rollouts) <= 0:
        raise ValueError("experiment.rollouts.num_rollouts must be > 0.")
    if int(cfg.rollouts.seed) < 0:
        raise ValueError("experiment.rollouts.seed must be >= 0.")
    if not isinstance(cfg.rollouts.sample_without_replacement, bool):
        raise ValueError("experiment.rollouts.sample_without_replacement must be a bool.")

def resolve_experiment_spec(cfg_path: str) -> ExperimentSpec:
    root = load_config_with_imports(cfg_path)
    cfg = OmegaConf.merge(_experiment_defaults(), root.get("experiment", root))
    validate_experiment_cfg(cfg)
    base_dir = os.path.dirname(os.path.abspath(cfg_path))
    shared_plan = _resolve_plan_fragment_bundle(
        cfg.shared_plan,
        base_dir=base_dir,
        name=f"{cfg.name}_shared",
    )

    variants: list[ExperimentVariant] = []
    for variant_cfg in cfg.variants:
        shared_plan_overrides = (
            OmegaConf.to_container(cfg.shared_plan.plan, resolve=True)
            if isinstance(getattr(cfg.shared_plan, "plan", {}), DictConfig)
            else dict(getattr(cfg.shared_plan, "plan", {}))
        )
        variant_overrides = getattr(variant_cfg, "overrides", {})
        variant_overrides = (
            OmegaConf.to_container(variant_overrides, resolve=True)
            if isinstance(variant_overrides, DictConfig)
            else dict(variant_overrides)
        )
        variant_bundle = OmegaConf.create(
            {
                "imports": list(getattr(cfg.shared_plan, "imports", []))
                + list(getattr(variant_cfg, "imports", [])),
                "plan": OmegaConf.to_container(
                    OmegaConf.merge(
                        OmegaConf.create(shared_plan_overrides),
                        OmegaConf.create(variant_overrides),
                    ),
                    resolve=True,
                ),
            }
        )
        plan = _resolve_plan_fragment_bundle(
            variant_bundle,
            base_dir=base_dir,
            name=str(variant_cfg.name),
        )
        if plan.task_signature() != shared_plan.task_signature():
            raise ValueError(
                f"Variant {variant_cfg.name!r} changes the shared task contract. "
                "Create a separate experiment instead."
            )
        if plan.rollout_signature() != shared_plan.rollout_signature():
            raise ValueError(
                f"Variant {variant_cfg.name!r} changes the rollout-selection contract. "
                "Create a separate experiment instead."
            )
        if plan.budget_signature() != shared_plan.budget_signature():
            raise ValueError(
                f"Variant {variant_cfg.name!r} changes the shared execution budget. "
                "Create a separate experiment instead."
            )
        variants.append(ExperimentVariant(name=str(variant_cfg.name), plan=plan))

    return ExperimentSpec(
        name=str(cfg.name),
        config_path=os.path.abspath(cfg_path),
        shared_plan=shared_plan,
        variants=variants,
        rollouts=dict(OmegaConf.to_container(cfg.rollouts, resolve=True)),
        execution=dict(OmegaConf.to_container(cfg.execution, resolve=True)),
        terminal=dict(OmegaConf.to_container(cfg.terminal, resolve=True)),
        reporting=dict(OmegaConf.to_container(cfg.reporting, resolve=True)),
    )


def validate_benchmark_cfg(cfg: DictConfig) -> None:
    _reject_unknown_keys(cfg, {"name", "output_root", "experiments"}, "benchmark")
    if cfg.experiments is None or len(cfg.experiments) <= 0:
        raise ValueError("benchmark.experiments must contain at least one entry.")
    names: list[str] = []
    for idx, entry in enumerate(cfg.experiments):
        _reject_unknown_keys(entry, {"name", "config"}, f"benchmark.experiments[{idx}]")
        if not str(entry.name).strip():
            raise ValueError(f"benchmark.experiments[{idx}].name must be non-empty.")
        if not str(entry.config).strip():
            raise ValueError(f"benchmark.experiments[{idx}].config must be non-empty.")
        names.append(str(entry.name))
    if len(set(names)) != len(names):
        raise ValueError("benchmark.experiments contains duplicate names.")


def resolve_benchmark_spec(cfg_path: str) -> BenchmarkSpec:
    root = load_config_with_imports(cfg_path)
    cfg = OmegaConf.merge(_benchmark_defaults(), root.get("benchmark", root))
    validate_benchmark_cfg(cfg)
    base_dir = os.path.dirname(os.path.abspath(cfg_path))
    entries = [
        BenchmarkEntry(
            name=str(entry.name),
            experiment_config=_resolve_import_path(base_dir, str(entry.config)),
        )
        for entry in cfg.experiments
    ]
    return BenchmarkSpec(
        name=str(cfg.name),
        config_path=os.path.abspath(cfg_path),
        entries=entries,
        output_root=str(cfg.output_root),
    )
