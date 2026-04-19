from __future__ import annotations

import inspect
import os
import random
from typing import Any

import gym
import numpy as np
import torch
from gym.envs.registration import register
from omegaconf import DictConfig, OmegaConf

from hudm.world_io import latest_checkpoint_epoch, load_world_checkpoint
from models.world.ensemble import WorldModelEnsemble
from models.world.model import HierWorldModel
from planning.gt_env_cem import GTEnvCEMPlanner
from planning.latent_cem import LatentCEMPlanner
from planning.particle_cem import ParticleCEMPlanner
from pusht.pusht_particle_backend import PushTParticleBackend
from pusht.pusht_wrapper import PushTWrapper


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def resolve_dataset_seed(seed_cfg: Any) -> int:
    if isinstance(seed_cfg, str):
        seed_s = seed_cfg.strip().lower()
        if seed_s == "random":
            return int(random.randrange(1_000_000))
        raise ValueError(
            f"plan.init_goal.dataset.seed must be an int or 'random', got {seed_cfg!r}."
        )
    try:
        return int(seed_cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"plan.init_goal.dataset.seed must be an int or 'random', got {seed_cfg!r}."
        ) from exc


def gym_make_versioned(env_id: str, env_cfg: DictConfig):
    make_kwargs = dict(OmegaConf.to_container(env_cfg, resolve=True))
    make_kwargs.setdefault("render_action", False)

    try:
        make_params = inspect.signature(gym.make).parameters
    except (TypeError, ValueError):
        make_params = {}
    if "disable_env_checker" in make_params:
        make_kwargs["disable_env_checker"] = True
    if "apply_api_compatibility" in make_params:
        make_kwargs["apply_api_compatibility"] = False
    return gym.make(str(env_id), **make_kwargs)


def image_to_model_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    #save the img to a file
    x = torch.as_tensor(img, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x * 2.0 - 1.0

    return x


def resolve_device(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg).lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("plan.world_model.device=cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if device_cfg != "auto":
        raise ValueError(
            f"plan.world_model.device must be one of auto|cpu|cuda, got '{device_cfg}'."
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_bits_human(bits: int) -> str:
    value = float(max(0, int(bits)))
    units = ["b", "Kb", "Mb", "Gb", "Tb"]
    unit_idx = 0
    while value >= 1000.0 and unit_idx < len(units) - 1:
        value /= 1000.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def bits_to_flops_estimate(bits: int) -> int:
    b = max(0, int(bits))
    return (b + 31) // 32


def format_flops_human(flops: int) -> str:
    value = float(max(0, int(flops)))
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
    unit_idx = 0
    while value >= 1000.0 and unit_idx < len(units) - 1:
        value /= 1000.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def latest_run_dir(root: str) -> str:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    run_dirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(run_dirs, key=os.path.getmtime)


def load_world_model_member(
    wm_cfg: DictConfig,
    run_dir: str,
    device: torch.device,
    epoch: int,
) -> HierWorldModel:
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"World-model run directory not found: {run_dir}")

    wm = HierWorldModel(
        K=list[int](wm_cfg.model.K),
        D=int(wm_cfg.model.D),
        action_dim=int(wm_cfg.data.action_dim),
        input=str(getattr(wm_cfg.model, "input", "images")),
        decoder_mode=str(getattr(wm_cfg.model, "decoder_mode", "per_level")),
        dynamics_mode=str(getattr(wm_cfg.model, "dynamics_mode", "per_level")),
    ).to(device)
    load_world_checkpoint(wm, run_dir, epoch=epoch, device=device)
    wm.eval()
    return wm


def wm_signature(wm_cfg: DictConfig) -> dict:
    return {
        "K": list(wm_cfg.model.K),
        "D": int(wm_cfg.model.D),
        "action_dim": int(wm_cfg.data.action_dim),
        "input": str(getattr(wm_cfg.model, "input", "images")),
        "decoder_mode": str(getattr(wm_cfg.model, "decoder_mode", "per_level")),
        "dynamics_mode": str(getattr(wm_cfg.model, "dynamics_mode", "per_level")),
    }


def load_run_world_cfg(run_dir: str) -> tuple[DictConfig, str]:
    cfg_path = os.path.join(run_dir, "world.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Missing world config in run directory: {cfg_path}. "
            "Set plan.world_model.config_path explicitly or provide run dirs with world.yaml."
        )
    return OmegaConf.load(cfg_path), cfg_path


def assert_world_cfg_compatible(base_sig: dict, other_cfg: DictConfig, label: str) -> None:
    other_sig = wm_signature(other_cfg)
    if other_sig != base_sig:
        raise ValueError(
            f"World-model config mismatch for {label}.\n"
            f"Expected: {base_sig}\n"
            f"Found:    {other_sig}"
        )


def resolve_world_epoch(run_dir: str, epoch_cfg: Any) -> int:
    if epoch_cfg is None:
        return latest_checkpoint_epoch(run_dir)
    return int(epoch_cfg)


def load_world_model(plan_cfg: DictConfig) -> tuple[object, DictConfig, str, torch.device]:
    device = resolve_device(str(plan_cfg.world_model.device))
    config_path = plan_cfg.world_model.config_path
    if config_path is not None:
        config_path = str(config_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"World-model config not found: {config_path}")
        cfg_override = OmegaConf.load(config_path)
    else:
        cfg_override = None

    ens_cfg = plan_cfg.world_model.get("ensemble", {})
    ens_enabled = bool(getattr(ens_cfg, "enabled", False))
    if ens_enabled:
        run_dirs = [str(rd) for rd in list(getattr(ens_cfg, "run_dirs", []))]
        if cfg_override is None:
            wm_cfg, _ = load_run_world_cfg(run_dirs[0])
        else:
            wm_cfg = cfg_override
        base_sig = wm_signature(wm_cfg)
        members = []
        for rd in run_dirs:
            if not os.path.isdir(rd):
                raise FileNotFoundError(f"World-model run directory not found: {rd}")
            run_cfg_path = os.path.join(rd, "world.yaml")
            if os.path.isfile(run_cfg_path):
                run_cfg = OmegaConf.load(run_cfg_path)
                assert_world_cfg_compatible(base_sig, run_cfg, label=rd)
            members.append(
                load_world_model_member(
                    wm_cfg,
                    rd,
                    device,
                    epoch=resolve_world_epoch(rd, getattr(plan_cfg.world_model, "epoch", None)),
                )
            )
        wm_backend = WorldModelEnsemble(members).to(device)
        return wm_backend, wm_cfg, ", ".join(run_dirs), device

    run_dir = plan_cfg.world_model.run_dir
    if run_dir is None:
        checkpoint_root = str(getattr(plan_cfg.world_model, "checkpoint_root", "checkpoints_world"))
        run_dir = latest_run_dir(checkpoint_root)
    run_dir = str(run_dir)
    if cfg_override is None:
        wm_cfg, _ = load_run_world_cfg(run_dir)
    else:
        wm_cfg = cfg_override
        run_cfg_path = os.path.join(run_dir, "world.yaml")
        if os.path.isfile(run_cfg_path):
            run_cfg = OmegaConf.load(run_cfg_path)
            assert_world_cfg_compatible(wm_signature(wm_cfg), run_cfg, label=run_dir)
    wm = load_world_model_member(
        wm_cfg,
        run_dir,
        device,
        epoch=resolve_world_epoch(run_dir, getattr(plan_cfg.world_model, "epoch", None)),
    )
    return wm, wm_cfg, run_dir, device


@torch.no_grad()
def encode_visual(wm: object, img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = image_to_model_tensor(img, device=device)
    return wm.encode(x)


def register_plan_env(cfg: DictConfig) -> None:
    try:
        register(
            id=str(cfg.env_id),
            entry_point="pusht.pusht_wrapper:PushTWrapper",
            max_episode_steps=300,
            reward_threshold=1.0,
        )
    except gym.error.Error:
        pass


def build_plan_runtime(cfg: DictConfig) -> dict:
    register_plan_env(cfg)
    env_wrapped = gym_make_versioned(str(cfg.env_id), cfg.env)
    env: PushTWrapper = unwrap_env(env_wrapped)
    backend = str(cfg.backend).lower()
    device = resolve_device(str(cfg.world_model.device))
    wm = None
    wm_cfg = None
    run_desc = backend

    if backend == "wm":
        wm, wm_cfg, run_desc, device = load_world_model(cfg)
        model_action_dim = int(wm_cfg.data.action_dim)
        if int(env.action_dim) != model_action_dim:
            raise ValueError(
                f"Action-dim mismatch: env.action_dim={env.action_dim}, world-model action_dim={model_action_dim}"
            )
        if not bool(cfg.env.with_velocity):
            print("[warn] plan.env.with_velocity=false may mismatch the world model's training distribution.")
        cfg.fidelity.num_levels = len(wm.K)
        planner = LatentCEMPlanner(
            world_model=wm,
            horizon=int(cfg.mpc.horizon),
            action_dim=int(env.action_dim),
            pop_size=int(cfg.cem.pop_size),
            elite_frac=float(cfg.cem.elite_frac),
            n_iter=int(cfg.cem.n_iter),
            init_std=float(cfg.cem.init_std),
            warm_start=bool(cfg.cem.warm_start),
            action_low=cfg.cem.action_low,
            action_high=cfg.cem.action_high,
            objective_cfg=OmegaConf.to_container(cfg.objective, resolve=True),
            fidelity_cfg=OmegaConf.to_container(cfg.fidelity, resolve=True),
            drop_tail_on_coarsen=True,
            device=device,
        )
    elif backend == "gt_env":
        planner = GTEnvCEMPlanner(
            env=env,
            horizon=int(cfg.mpc.horizon),
            action_dim=int(env.action_dim),
            pop_size=int(cfg.cem.pop_size),
            elite_frac=float(cfg.cem.elite_frac),
            n_iter=int(cfg.cem.n_iter),
            init_std=float(cfg.cem.init_std),
            warm_start=bool(cfg.cem.warm_start),
            action_low=cfg.cem.action_low,
            action_high=cfg.cem.action_high,
            objective_cfg=OmegaConf.to_container(cfg.objective, resolve=True),
            fidelity_cfg=OmegaConf.to_container(cfg.fidelity, resolve=True),
            gt_env_cfg=OmegaConf.to_container(cfg.gt_env, resolve=True),
            device=device,
        )
    else:
        spacings = list(cfg.particle_env.fidelity_env.spacings)
        cfg.fidelity.num_levels = len(spacings)
        particle_seed = resolve_dataset_seed(getattr(cfg.init_goal.dataset, "seed", 0))
        cfg.init_goal.dataset.seed = particle_seed
        particle_backend = PushTParticleBackend(
            with_velocity=bool(cfg.env.with_velocity),
            with_target=bool(cfg.env.with_target),
            render_size=int(getattr(env, "render_size", 512)),
            relative=bool(getattr(env, "relative", True)),
            action_scale=float(getattr(env, "action_scale", 100.0)),
            device=str(cfg.particle_env.fidelity_env.device),
            fidelity_spacings=[float(s) for s in spacings],
            warp_cfg=OmegaConf.to_container(cfg.particle_env.fidelity_env, resolve=True),
            seed=particle_seed,
        )
        planner = ParticleCEMPlanner(
            particle_backend=particle_backend,
            horizon=int(cfg.mpc.horizon),
            action_dim=int(env.action_dim),
            pop_size=int(cfg.cem.pop_size),
            elite_frac=float(cfg.cem.elite_frac),
            n_iter=int(cfg.cem.n_iter),
            init_std=float(cfg.cem.init_std),
            warm_start=bool(cfg.cem.warm_start),
            action_low=cfg.cem.action_low,
            action_high=cfg.cem.action_high,
            objective_cfg=OmegaConf.to_container(cfg.objective, resolve=True),
            fidelity_cfg=OmegaConf.to_container(cfg.fidelity, resolve=True),
            particle_env_cfg=OmegaConf.to_container(cfg.particle_env, resolve=True),
            device=device,
        )
    return {
        "env": env,
        "planner": planner,
        "wm": wm,
        "wm_cfg": wm_cfg,
        "backend": backend,
        "device": device,
        "run_desc": run_desc,
    }


def print_plan_runtime_summary(runtime: dict, cfg: DictConfig) -> None:
    backend = str(runtime["backend"])
    wm = runtime.get("wm", None)
    print(f"[backend] {backend}")
    if wm is not None:
        print(f"[world_model] loaded from {runtime['run_desc']}")
    else:
        print("[world_model] not used")
    if wm is not None and hasattr(wm, "num_members"):
        print(f"[ensemble] members={wm.num_members}")
    if wm is not None:
        print(f"[levels] K={wm.K}")
    elif backend == "gt_env":
        print(f"[levels] num_levels={int(cfg.fidelity.num_levels)} (gt_env)")
        print(f"[gt_env] objective_space={str(cfg.gt_env.objective_space).lower()}")
        print(f"[gt_env] progress={bool(cfg.gt_env.progress)}")
    else:
        print(f"[levels] num_levels={int(cfg.fidelity.num_levels)} (particle_sim)")
        print(f"[particle] objective_space={str(cfg.particle_env.objective_space).lower()}")
        print(f"[particle] spacings={list(cfg.particle_env.fidelity_env.spacings)}")
        print(f"[particle] progress={bool(cfg.particle_env.progress)}")
