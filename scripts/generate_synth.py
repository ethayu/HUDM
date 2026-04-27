#!/usr/bin/env python3
#python scripts/generate_synth.py synthetic/pusht_synth_planning.zarr --train_eps 100 --val_eps 1 --
#len_min 50 --len_max 160 --with_velocity --policy contact_aware --mode-profile planning --quality-profile strict --img-size 96 --workers 50 --contact-aware-ou-sigma 3.0
"""
Generate synthetic PushT rollouts by stepping the ground-truth environment and
save them in Zarr format for world-model training.

Usage:
  # Contact-aware default policy
  python scripts/generate_synth.py synthetic/pusht_synth.zarr \
      --train_eps 200 --val_eps 50 \
      --len_min 50 --len_max 160 \
      --seed 0 --with_velocity --img-size 96

  # Planning-biased synthetic data, with actions perturbed by OU noise with sigma=3.0, parallelized with 5 workers
  python scripts/generate_synth.py synthetic/pusht_synth.zarr \
      --train_eps 200 --val_eps 50 \
      --len_min 50 --len_max 160 \
      --policy contact_aware --mode-profile planning \
      --quality-profile strict --seed 0 --with_velocity --img-size 96 \
      --contact-aware-ou-sigma 3.0 --workers 5

Notes:
  - Outputs a Zarr store compatible with datasets/zarr_episodes.py
  - The Zarr store contains: data/img, data/action, data/state, data/action_abs, meta/episode_ends
  - Images are stored as float32 in [0, 255] range
  - data/action stores replay-safe env input actions (relative controls before env.action_scale)
  - data/action_abs stores absolute action targets for reference/debug
  - States are the pre-action environment state (float32)
  - Train/val split is handled via split_ratio when loading the dataset
"""

#Apr 13 python scripts/generate_synth.py synthetic/pusht_synth_training_translation.zarr --train_eps 10 --val_eps 1 --len_min 50 --len_max 150 --with_velocity --policy contact_aware --mode-profile train --quality-profile strict --img-size 96 --workers 50 --contact-aware-ou-sigma 3.0 --max-env-input-norm 0.3
#Apt 14 python scripts/generate_synth.py synthetic/pusht_synth_planning_translation_100.zarr --train_eps 100 --val_eps 100 --len_min 5 --len_max 150 --with_velocity --policy contact_aware --mode-profile planning --quality-profile strict --img-size 96 --workers 50 --contact-aware-ou-sigma 3.0 --max-env-input-norm 0.3
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
from multiprocessing.connection import wait
from typing import Any, Mapping, Optional, Sequence

# Add project root to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from hudm.synthetic_generation import (
    EpisodeAttempt,
    QUALITY_PROFILES,
    RolloutDiagnostics,
    analyze_rollout,
    bounded_env_action_from_abs_target,
    build_contact_candidates,
    choose_maneuver_family,
    reject_rollout,
    resolve_mode_weights,
    score_shadow_rollout,
    summarize_rejections,
    validate_spawn_state,
    wall_clearance,
    block_world_vertices,
)
from pusht.pusht_wrapper import PushTWrapper

try:
    import zarr
except Exception:
    zarr = None


def build_generator_env(*, with_velocity: bool, headless: bool) -> PushTWrapper:
    env = PushTWrapper(with_velocity=with_velocity, with_target=True, add_noise=0)
    env.headless = bool(headless)
    return env


def _parse_mu(s: Optional[str], dim: int) -> Optional[np.ndarray]:
    if s is None:
        return None
    vals = [float(x) for x in s.split(",") if x.strip() != ""]
    if len(vals) == 1:
        return np.full(dim, vals[0], dtype=np.float32)
    if len(vals) != dim:
        raise ValueError(f"--ou-mu must have length {dim} or be scalar")
    return np.asarray(vals, dtype=np.float32)


def sample_valid_init_state(
    rng: np.random.Generator,
    *,
    with_velocity: bool,
    quality_profile: str,
    max_attempts: int,
) -> np.ndarray:
    for _ in range(max(1, int(max_attempts))):
        state = np.asarray(
            [
                rng.uniform(40.0, 472.0),
                rng.uniform(40.0, 472.0),
                rng.uniform(110.0, 402.0),
                rng.uniform(110.0, 402.0),
                rng.uniform(-math.pi, math.pi),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        if not with_velocity:
            state = state[:5]
        valid, _, _ = validate_spawn_state(state, quality_profile)
        if valid:
            return state
    raise RuntimeError(
        f"Unable to sample a valid synthetic init state after {max_attempts} attempts "
        f"for quality_profile={quality_profile!r}."
    )


def _resize_frame(frame: np.ndarray, img_size: int) -> np.ndarray:
    im = Image.fromarray(frame.astype(np.uint8)) if frame.dtype != np.uint8 else Image.fromarray(frame)
    im = im.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(im).astype(np.float32)


def _apply_contact_aware_ou_offset(
    abs_target: np.ndarray,
    *,
    rng: np.random.Generator | None,
    ou_state: np.ndarray,
    theta: float,
    sigma: float,
    dt: float,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(abs_target, dtype=np.float32)
    state = np.asarray(ou_state, dtype=np.float32)
    if float(sigma) <= 0.0:
        return np.clip(target, low, high).astype(np.float32), state.copy()
    if rng is None:
        raise ValueError("rng is required when contact-aware OU sigma > 0.")
    noise = rng.normal(0.0, 1.0, size=target.shape).astype(np.float32)
    next_state = state + float(theta) * (-state) * float(dt) + float(sigma) * math.sqrt(float(dt)) * noise
    noisy_target = np.clip(target + next_state, low, high).astype(np.float32)
    return noisy_target, np.asarray(next_state, dtype=np.float32)


def render_episode_frames(
    env: PushTWrapper,
    *,
    init_state: np.ndarray,
    actions: np.ndarray,
    img_size: int,
    seed: int,
) -> np.ndarray:
    env.headless = False
    obs, _ = env.prepare(seed=seed, init_state=init_state)
    frames = []
    for act in actions:
        frames.append(_resize_frame(np.asarray(obs["visual"]), img_size))
        obs, _, _, _ = env.step(np.asarray(act, dtype=np.float32))
    return np.asarray(frames, dtype=np.float32)


def _simulate_candidate_rollout(
    shadow_env: PushTWrapper,
    *,
    current_state: np.ndarray,
    candidate,
    quality_profile: str,
    contact_aware_ou_state: np.ndarray | None,
    contact_aware_ou_seed: int | None,
    contact_aware_ou_theta: float,
    contact_aware_ou_sigma: float,
    contact_aware_ou_dt: float,
    max_env_input_norm: float,
    T: int,
) -> RolloutDiagnostics:
    shadow_env.headless = True
    _, state = shadow_env.prepare(seed=0, init_state=current_state)
    states_before = []
    states_after = []
    contacts = []
    quality = QUALITY_PROFILES[str(quality_profile)]
    ou_state = (
        np.zeros(2, dtype=np.float32)
        if contact_aware_ou_state is None
        else np.asarray(contact_aware_ou_state, dtype=np.float32).copy()
    )
    ou_rng = None if float(contact_aware_ou_sigma) <= 0.0 else np.random.default_rng(int(contact_aware_ou_seed))
    for abs_target in candidate.action_abs_seq:
        cur = np.asarray(state, dtype=np.float32)
        states_before.append(cur.copy())
        exec_abs_target, ou_state = _apply_contact_aware_ou_offset(
            np.asarray(abs_target, dtype=np.float32),
            rng=ou_rng,
            ou_state=ou_state,
            theta=float(contact_aware_ou_theta),
            sigma=float(contact_aware_ou_sigma),
            dt=float(contact_aware_ou_dt),
            low=quality.bounds_low,
            high=quality.bounds_high,
        )
        act, _ = bounded_env_action_from_abs_target(
            cur[:2],
            exec_abs_target,
            env_action_scale=float(shadow_env.action_scale),
            relative=bool(shadow_env.relative),
            max_env_input_norm=max_env_input_norm,
        )
        _, _, _, info = shadow_env.step(act)
        state = np.asarray(info["state"], dtype=np.float32)
        states_after.append(state.copy())
        contacts.append(int(info.get("n_contacts", 0)))
    return analyze_rollout(states_before, states_after, contacts, T=T, quality_profile=quality_profile)


def _select_contact_aware_candidate(
    shadow_env: PushTWrapper,
    *,
    state: np.ndarray,
    mode_weights: Mapping[str, float],
    rng: np.random.Generator,
    mode_profile: str,
    quality_profile: str,
    recent_motion_fraction: float | None,
    recent_contact_fraction: float | None,
    contact_aware_ou_state: np.ndarray,
    contact_aware_ou_theta: float,
    contact_aware_ou_sigma: float,
    contact_aware_ou_dt: float,
    max_env_input_norm: float,
    T: int,
) -> tuple[str, Any, int | None]:
    family = choose_maneuver_family(
        state,
        mode_weights,
        rng,
        mode_profile=mode_profile,
        quality_profile=quality_profile,
        recent_motion_fraction=recent_motion_fraction,
        recent_contact_fraction=recent_contact_fraction,
    )
    candidates = build_contact_candidates(state, family, rng, mode_profile=mode_profile)
    if not candidates and family != "recover":
        family = "recover"
        candidates = build_contact_candidates(state, family, rng, mode_profile=mode_profile)
    if not candidates:
        raise RuntimeError(f"No candidates available for family={family!r}")

    best_candidate = None
    best_score = None
    best_noise_seed: int | None = None
    for candidate in candidates:
        noise_seed = None if float(contact_aware_ou_sigma) <= 0.0 else int(rng.integers(0, 2**31 - 1))
        diag = _simulate_candidate_rollout(
            shadow_env,
            T=T,
            current_state=np.asarray(state, dtype=np.float32),
            candidate=candidate,
            quality_profile=quality_profile,
            contact_aware_ou_state=contact_aware_ou_state,
            contact_aware_ou_seed=noise_seed,
            contact_aware_ou_theta=contact_aware_ou_theta,
            contact_aware_ou_sigma=contact_aware_ou_sigma,
            contact_aware_ou_dt=contact_aware_ou_dt,
            max_env_input_norm=max_env_input_norm,
        )
        score = score_shadow_rollout(candidate.family, diag, mode_profile=mode_profile, quality_profile=quality_profile)
        if best_candidate is None or score > float(best_score):
            best_candidate = candidate
            best_score = score
            best_noise_seed = noise_seed
    assert best_candidate is not None
    return family, best_candidate, best_noise_seed


def rollout_contact_aware_episode(
    env: PushTWrapper,
    shadow_env: PushTWrapper,
    T: int,
    *,
    rng: np.random.Generator,
    with_velocity: bool,
    mode_profile: str,
    quality_profile: str,
    weights: Mapping[str, float],
    spawn_attempts: int,
    contact_aware_ou_theta: float,
    contact_aware_ou_sigma: float,
    contact_aware_ou_dt: float,
    max_env_input_norm: float,
) -> EpisodeAttempt:
    init_state = sample_valid_init_state(
        rng,
        with_velocity=with_velocity,
        quality_profile=quality_profile,
        max_attempts=spawn_attempts,
    )
    env.headless = True
    _, state = env.prepare(seed=int(rng.integers(0, 1_000_000)), init_state=init_state)

    states = []
    states_after = []
    actions = []
    actions_abs = []
    contacts = []
    selected_families: list[str] = []
    selected_candidates: list[str] = []
    recent_motion_flags: list[float] = []
    recent_contact_flags: list[float] = []
    current_candidate = None
    current_candidate_ou_rng = None
    segment_step = 0
    contact_aware_ou_state = np.zeros(2, dtype=np.float32)

    quality = QUALITY_PROFILES[str(quality_profile)]
    for _ in range(T):
        if current_candidate is None or segment_step >= len(current_candidate.action_abs_seq):
            recent_motion_fraction = None if len(recent_motion_flags) <= 0 else float(np.mean(recent_motion_flags[-3:]))
            recent_contact_fraction = None if len(recent_contact_flags) <= 0 else float(np.mean(recent_contact_flags[-3:]))
            family, current_candidate, current_candidate_ou_seed = _select_contact_aware_candidate(
                shadow_env,
                state=np.asarray(state, dtype=np.float32),
                T=T,
                mode_weights=weights,
                rng=rng,
                mode_profile=mode_profile,
                quality_profile=quality_profile,
                recent_motion_fraction=recent_motion_fraction,
                recent_contact_fraction=recent_contact_fraction,
                contact_aware_ou_state=contact_aware_ou_state,
                contact_aware_ou_theta=contact_aware_ou_theta,
                contact_aware_ou_sigma=contact_aware_ou_sigma,
                contact_aware_ou_dt=contact_aware_ou_dt,
                max_env_input_norm=max_env_input_norm,
            )
            current_candidate_ou_rng = (
                None
                if current_candidate_ou_seed is None
                else np.random.default_rng(int(current_candidate_ou_seed))
            )
            segment_step = 0
            selected_families.append(family)
            selected_candidates.append(current_candidate.name)

        current_state = np.asarray(state, dtype=np.float32)
        abs_target, contact_aware_ou_state = _apply_contact_aware_ou_offset(
            np.asarray(current_candidate.action_abs_seq[segment_step], dtype=np.float32),
            rng=current_candidate_ou_rng,
            ou_state=contact_aware_ou_state,
            theta=float(contact_aware_ou_theta),
            sigma=float(contact_aware_ou_sigma),
            dt=float(contact_aware_ou_dt),
            low=quality.bounds_low,
            high=quality.bounds_high,
        )
        act, executed_abs_target = bounded_env_action_from_abs_target(
            current_state[:2],
            abs_target,
            env_action_scale=float(env.action_scale),
            relative=bool(env.relative),
            max_env_input_norm=max_env_input_norm,
        )
        states.append(current_state.copy())
        actions.append(act.astype(np.float32))
        actions_abs.append(executed_abs_target.astype(np.float32))

        _, done, info = None, None, None
        _, _, done, info = env.step(act)
        next_state = np.asarray(info["state"], dtype=np.float32)
        states_after.append(next_state.copy())
        contact_count = int(info.get("n_contacts", 0))
        contacts.append(contact_count)

        step_motion = float(np.linalg.norm(next_state[2:4] - current_state[2:4]))
        recent_motion_flags.append(float(step_motion > quality.meaningful_motion_threshold))
        recent_contact_flags.append(float(contact_count > 0))
        if len(recent_motion_flags) > 6:
            recent_motion_flags = recent_motion_flags[-6:]
        if len(recent_contact_flags) > 6:
            recent_contact_flags = recent_contact_flags[-6:]

        state = next_state
        segment_step += 1
        block_clearance = wall_clearance(block_world_vertices(next_state), low=quality.bounds_low, high=quality.bounds_high)
        stalled = (
            segment_step >= 3
            and len(recent_motion_flags) >= 3
            and float(np.mean(recent_motion_flags[-3:])) <= 0.0
            and float(np.mean(recent_contact_flags[-3:])) < 0.34
        )
        if block_clearance < max(quality.spawn_clearance + 6.0, 18.0) or stalled:
            current_candidate = None
            current_candidate_ou_rng = None
        if bool(done):
            break

    diagnostics = analyze_rollout(states, states_after, contacts, T=T, quality_profile=quality_profile)
    rejected, reasons = reject_rollout(diagnostics, mode_profile=mode_profile, quality_profile=quality_profile)
    return EpisodeAttempt(
        init_state=np.asarray(init_state, dtype=np.float32),
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        actions_abs=np.asarray(actions_abs, dtype=np.float32),
        diagnostics=diagnostics,
        rejection_reasons=tuple(reasons if rejected else ()),
        selected_families=tuple(selected_families),
        selected_candidates=tuple(selected_candidates),
    )


def rollout_baseline_episode(
    env: PushTWrapper,
    *,
    T: int,
    policy: str,
    action_scale: float,
    rng: np.random.Generator,
    with_velocity: bool,
    quality_profile: str,
    spawn_attempts: int,
    ou_theta: float,
    ou_sigma: float,
    ou_dt: float,
    ou_mu: Optional[np.ndarray],
    max_env_input_norm: float,
) -> EpisodeAttempt:
    init_state = sample_valid_init_state(
        rng,
        with_velocity=with_velocity,
        quality_profile=quality_profile,
        max_attempts=spawn_attempts,
    )
    env.headless = True
    _, state = env.prepare(seed=int(rng.integers(0, 1_000_000)), init_state=init_state)
    x = np.zeros(env.action_dim, dtype=np.float32)
    mu = np.zeros(env.action_dim, dtype=np.float32) if ou_mu is None else np.asarray(ou_mu, dtype=np.float32)

    states = []
    states_after = []
    actions = []
    actions_abs = []
    contacts = []
    env_bounds = QUALITY_PROFILES[str(quality_profile)]
    for _ in range(T):
        current_state = np.asarray(state, dtype=np.float32)
        agent_pos = current_state[:2]

        if policy == "ou":
            noise = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32)
            x = x + ou_theta * (mu - x) * ou_dt + ou_sigma * math.sqrt(ou_dt) * noise
            act = x * float(action_scale)
        elif policy == "random":
            act = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32) * float(action_scale)
        else:
            raise ValueError(f"Unsupported baseline policy: {policy}")

        next_abs = agent_pos + act * float(env.action_scale) if env.relative else act * float(env.action_scale)
        next_abs = np.clip(next_abs, env_bounds.bounds_low, env_bounds.bounds_high).astype(np.float32)
        act, executed_abs_target = bounded_env_action_from_abs_target(
            agent_pos,
            next_abs,
            env_action_scale=float(env.action_scale),
            relative=bool(env.relative),
            max_env_input_norm=max_env_input_norm,
        )
        states.append(current_state.copy())
        actions.append(act.astype(np.float32))
        actions_abs.append(executed_abs_target.astype(np.float32))
        _, _, done, info = env.step(act)
        state = np.asarray(info["state"], dtype=np.float32)
        states_after.append(state.copy())
        contacts.append(int(info.get("n_contacts", 0)))
        if bool(done):
            break

    diagnostics = analyze_rollout(states, states_after, contacts, T=T, quality_profile=quality_profile)
    rejected, reasons = reject_rollout(diagnostics, mode_profile="balanced", quality_profile=quality_profile)
    return EpisodeAttempt(
        init_state=np.asarray(init_state, dtype=np.float32),
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        actions_abs=np.asarray(actions_abs, dtype=np.float32),
        diagnostics=diagnostics,
        rejection_reasons=tuple(reasons if rejected else ()),
        selected_families=(policy,),
        selected_candidates=(policy,),
    )


def save_zarr(
    zarr_out: str,
    frames_list: list[np.ndarray],
    actions_list: list[np.ndarray],
    actions_abs_list: list[np.ndarray],
    states_list: list[np.ndarray],
    *,
    env_action_scale: float,
    env_relative: bool,
    max_env_input_norm: float | None = None,
    extra_attrs: Mapping[str, Any] | None = None,
) -> None:
    if zarr is None:
        raise ImportError("zarr is not installed. Try: pip install zarr")
    lengths = [
        min(f.shape[0], a.shape[0], a_abs.shape[0], s.shape[0])
        for f, a, a_abs, s in zip(frames_list, actions_list, actions_abs_list, states_list)
    ]
    frames = np.concatenate([f[:L] for f, L in zip(frames_list, lengths)], axis=0)
    actions = np.concatenate([a[:L] for a, L in zip(actions_list, lengths)], axis=0)
    actions_abs = np.concatenate([a[:L] for a, L in zip(actions_abs_list, lengths)], axis=0)
    states = np.concatenate([s[:L] for s, L in zip(states_list, lengths)], axis=0)
    ends = np.cumsum(lengths) - 1

    root = zarr.group(zarr_out, overwrite=True)
    g_data = root.create_group('data')
    g_meta = root.create_group('meta')
    # NOTE: zarr 3.x requires an explicit shape when creating datasets from buffers.
    # Keep using create_dataset so arrays are materialized directly from numpy data.
    def _create_dataset(group, name: str, array: np.ndarray, chunks: tuple[int, ...]) -> None:
        group.create_dataset(
            name,
            shape=array.shape,
            data=array,
            chunks=chunks,
            dtype=array.dtype,
        )

    _create_dataset(
        g_data,
        'img',
        frames.astype(np.float32),
        chunks=(min(160, frames.shape[0]),) + frames.shape[1:],
    )
    _create_dataset(
        g_data,
        'action',
        actions.astype(np.float32),
        chunks=(min(160, actions.shape[0]), actions.shape[1]),
    )
    _create_dataset(
        g_data,
        'action_abs',
        actions_abs.astype(np.float32),
        chunks=(min(160, actions_abs.shape[0]), actions_abs.shape[1]),
    )
    _create_dataset(
        g_data,
        'state',
        states.astype(np.float32),
        chunks=(min(160, states.shape[0]), states.shape[1]),
    )
    _create_dataset(
        g_meta,
        'episode_ends',
        ends.astype(np.int64),
        chunks=(max(1, min(1024, len(ends))),),
    )
    root.attrs.update({
        "action_format": "env_input",
        "action_abs_format": "absolute_target",
        "env_action_scale": float(env_action_scale),
        "env_relative": bool(env_relative),
    })
    if max_env_input_norm is not None:
        root.attrs["max_env_input_norm"] = float(max_env_input_norm)
    if extra_attrs:
        root.attrs.update(dict(extra_attrs))


class ZarrWriter:
    """Streams episodes directly to a Zarr store, avoiding RAM accumulation of frames."""

    def __init__(
        self,
        zarr_out: str,
        *,
        env_action_scale: float,
        env_relative: bool,
        max_env_input_norm: float | None = None,
    ) -> None:
        if zarr is None:
            raise ImportError("zarr is not installed. Try: pip install zarr")
        self._root = zarr.group(zarr_out, overwrite=True)
        self._g_data = self._root.require_group("data")
        self._g_meta = self._root.require_group("meta")
        self._img: Any = None
        self._action: Any = None
        self._action_abs: Any = None
        self._state: Any = None
        self._episode_ends: list[int] = []
        self._total_steps: int = 0
        self._root.attrs.update({
            "action_format": "env_input",
            "action_abs_format": "absolute_target",
            "env_action_scale": float(env_action_scale),
            "env_relative": bool(env_relative),
        })
        if max_env_input_norm is not None:
            self._root.attrs["max_env_input_norm"] = float(max_env_input_norm)

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def n_episodes(self) -> int:
        return len(self._episode_ends)

    def append_episode(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        actions_abs: np.ndarray,
        states: np.ndarray,
    ) -> None:
        L = min(frames.shape[0], actions.shape[0], actions_abs.shape[0], states.shape[0])
        frames = frames[:L].astype(np.float32)
        actions = actions[:L].astype(np.float32)
        actions_abs = actions_abs[:L].astype(np.float32)
        states = states[:L].astype(np.float32)

        if self._img is None:
            H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]
            action_dim = actions.shape[1]
            state_dim = states.shape[1]
            self._img = self._g_data.zeros(
                "img", shape=(0, H, W, C), chunks=(160, H, W, C), dtype="f4"
            )
            self._action = self._g_data.zeros(
                "action", shape=(0, action_dim), chunks=(160, action_dim), dtype="f4"
            )
            self._action_abs = self._g_data.zeros(
                "action_abs", shape=(0, action_dim), chunks=(160, action_dim), dtype="f4"
            )
            self._state = self._g_data.zeros(
                "state", shape=(0, state_dim), chunks=(160, state_dim), dtype="f4"
            )

        self._img.append(frames)
        self._action.append(actions)
        self._action_abs.append(actions_abs)
        self._state.append(states)
        self._total_steps += L
        self._episode_ends.append(self._total_steps - 1)

    def finalize(self, extra_attrs: Mapping[str, Any] | None = None) -> None:
        ends = np.asarray(self._episode_ends, dtype=np.int64)
        self._g_meta.create_dataset(
            "episode_ends",
            shape=ends.shape,
            data=ends,
            chunks=(max(1, min(1024, len(ends))),),
            dtype=ends.dtype,
        )
        if extra_attrs:
            self._root.attrs.update(dict(extra_attrs))


def _mp_context() -> mp.context.BaseContext:
    if sys.platform != "win32":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context()


def _resolve_max_failed_episode_batches(n_eps: int, max_failed_episode_batches: int) -> int:
    if int(max_failed_episode_batches) < 0:
        return -1
    if int(max_failed_episode_batches) > 0:
        return int(max_failed_episode_batches)
    return max(32, 8 * int(n_eps))


def _generator_runtime_reasons(exc: RuntimeError) -> tuple[str, ...] | None:
    msg = str(exc)
    if msg.startswith("Unable to sample a valid synthetic init state"):
        return ("spawn_sampling_failed",)
    if msg.startswith("No candidates available for family="):
        return ("no_contact_candidates",)
    return None


def _generate_single_episode_task(task: Mapping[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(task["seed"]))
    with_velocity = bool(task["with_velocity"])
    policy = str(task["policy"])
    quality_profile = str(task["quality_profile"])
    mode_profile = str(task["mode_profile"])
    decision_env = build_generator_env(with_velocity=with_velocity, headless=True)
    shadow_env = build_generator_env(with_velocity=with_velocity, headless=True)
    ou_mu_raw = task.get("ou_mu", None)
    ou_mu = None if ou_mu_raw is None else np.asarray(ou_mu_raw, dtype=np.float32)
    rejection_reasons: list[Sequence[str]] = []

    for _ in range(int(task["max_attempts_per_episode"])):
        horizon = int(rng.integers(int(task["len_min"]), int(task["len_max"]) + 1))
        try:
            if policy in {"contact_aware", "advanced"}:
                attempt = rollout_contact_aware_episode(
                    decision_env,
                    shadow_env,
                    T=horizon,
                    rng=rng,
                    with_velocity=with_velocity,
                    mode_profile=mode_profile,
                    quality_profile=quality_profile,
                    weights=dict(task["weights"]),
                    spawn_attempts=int(task["max_attempts_per_episode"]),
                    contact_aware_ou_theta=float(task["contact_aware_ou_theta"]),
                    contact_aware_ou_sigma=float(task["contact_aware_ou_sigma"]),
                    contact_aware_ou_dt=float(task["contact_aware_ou_dt"]),
                    max_env_input_norm=float(task["max_env_input_norm"]),
                )
            else:
                attempt = rollout_baseline_episode(
                    decision_env,
                    T=horizon,
                    policy=policy,
                    action_scale=float(task["action_scale"]),
                    rng=rng,
                    with_velocity=with_velocity,
                    quality_profile=quality_profile,
                    spawn_attempts=int(task["max_attempts_per_episode"]),
                    ou_theta=float(task["ou_theta"]),
                    ou_sigma=float(task["ou_sigma"]),
                    ou_dt=float(task["ou_dt"]),
                    ou_mu=ou_mu,
                    max_env_input_norm=float(task["max_env_input_norm"]),
                )
        except RuntimeError as exc:
            reasons = _generator_runtime_reasons(exc)
            if reasons is None:
                raise
            rejection_reasons.append(reasons)
            continue
        if attempt.accepted:
            return {
                "accepted": True,
                "task_idx": int(task["task_idx"]),
                "render_seed": int(task["render_seed"]),
                "init_state": np.asarray(attempt.init_state, dtype=np.float32),
                "states": np.asarray(attempt.states, dtype=np.float32),
                "actions": np.asarray(attempt.actions, dtype=np.float32),
                "actions_abs": np.asarray(attempt.actions_abs, dtype=np.float32),
                "rejection_reasons": list(rejection_reasons),
            }
        rejection_reasons.append(attempt.rejection_reasons)

    return {
        "accepted": False,
        "task_idx": int(task["task_idx"]),
        "rejection_reasons": list(rejection_reasons),
    }


def _episode_worker(task: Mapping[str, Any], send_conn) -> None:
    try:
        send_conn.send({"ok": True, "result": _generate_single_episode_task(task)})
    except BaseException as exc:  # pragma: no cover - worker crash propagation
        send_conn.send(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "task_idx": int(task["task_idx"]),
            }
        )
    finally:
        send_conn.close()


def _run_parallel_episode_tasks(
    *,
    split_name: str,
    target_accepted: int,
    workers: int,
    task_factory,
    rejection_reasons: list[Sequence[str]],
    max_failed_episode_batches: int,
) -> tuple[list[dict[str, Any]], int]:
    ctx = _mp_context()
    active: dict[Any, tuple[mp.Process, Mapping[str, Any]]] = {}
    accepted: list[dict[str, Any]] = []
    next_task_idx = 0
    failed_episode_batches = 0
    displayed_accepted = 0
    progress = tqdm(
        total=int(target_accepted),
        desc=f"generate:{split_name}",
        unit="ep",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )

    def launch_next() -> None:
        nonlocal next_task_idx
        task = task_factory(int(next_task_idx))
        next_task_idx += 1
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_episode_worker, args=(task, send_conn))
        proc.start()
        send_conn.close()
        active[recv_conn] = (proc, task)

    try:
        for _ in range(max(1, int(workers))):
            launch_next()

        while active:
            ready = wait(list(active.keys()))
            for recv_conn in ready:
                proc, task = active.pop(recv_conn)
                payload = recv_conn.recv()
                recv_conn.close()
                proc.join()
                if proc.exitcode not in (0, None) and payload.get("ok", False):
                    raise RuntimeError(
                        f"Worker exited unexpectedly for split={split_name!r} "
                        f"task_idx={int(task['task_idx'])} with code {proc.exitcode}."
                    )
                if not payload.get("ok", False):
                    raise RuntimeError(
                        f"Parallel synthetic generation failed for split={split_name!r} "
                        f"task_idx={int(payload.get('task_idx', task['task_idx']))}: {payload['error']}"
                    )
                result = payload["result"]
                rejection_reasons.extend(result["rejection_reasons"])
                if bool(result.get("accepted", False)):
                    accepted.append(result)
                else:
                    failed_episode_batches += 1
                    if int(max_failed_episode_batches) >= 0 and failed_episode_batches > int(max_failed_episode_batches):
                        summary = summarize_rejections(rejection_reasons)
                        raise RuntimeError(
                            f"Unable to produce enough accepted episodes for split={split_name!r}. "
                            f"accepted={min(len(accepted), int(target_accepted))}/{int(target_accepted)} "
                            f"failed_episode_batches={failed_episode_batches} "
                            f"max_failed_episode_batches={max_failed_episode_batches}. "
                            f"Rejections: {summary}"
                        )
                new_displayed_accepted = min(len(accepted), int(target_accepted))
                if new_displayed_accepted > displayed_accepted:
                    progress.update(new_displayed_accepted - displayed_accepted)
                    displayed_accepted = new_displayed_accepted
                progress.set_postfix(
                    rejected=len(rejection_reasons),
                    failed_batches=failed_episode_batches,
                    refresh=False,
                )
                if len(accepted) < int(target_accepted):
                    launch_next()
    finally:
        progress.close()
        for recv_conn, (proc, _) in list(active.items()):
            recv_conn.close()
            if proc.is_alive():
                proc.terminate()
            proc.join()
    accepted.sort(key=lambda item: int(item["task_idx"]))
    return accepted[: int(target_accepted)], int(failed_episode_batches)


def generate_split(
    *,
    n_eps: int,
    split_name: str,
    len_min: int,
    len_max: int,
    policy: str,
    action_scale: float,
    rng: np.random.Generator,
    decision_env: PushTWrapper,
    shadow_env: PushTWrapper,
    render_env: PushTWrapper,
    with_velocity: bool,
    img_size: int,
    mode_profile: str,
    quality_profile: str,
    weights: Mapping[str, float],
    max_attempts_per_episode: int,
    max_failed_episode_batches: int,
    workers: int,
    contact_aware_ou_theta: float,
    contact_aware_ou_sigma: float,
    contact_aware_ou_dt: float,
    ou_theta: float,
    ou_sigma: float,
    ou_dt: float,
    ou_mu: Optional[np.ndarray],
    max_env_input_norm: float,
    zarr_writer: Optional["ZarrWriter"] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    actions_list: list[np.ndarray] = []
    actions_abs_list: list[np.ndarray] = []
    frames_list: list[np.ndarray] = []
    states_list: list[np.ndarray] = []
    rejection_reasons: list[Sequence[str]] = []
    failed_episode_batches = 0
    resolved_max_failed_episode_batches = _resolve_max_failed_episode_batches(
        int(n_eps), int(max_failed_episode_batches)
    )
    if int(workers) > 1:
        worker_count = min(int(workers), int(n_eps))
        def make_task(task_idx: int) -> dict[str, Any]:
            return {
                "task_idx": int(task_idx),
                "split_name": str(split_name),
                "seed": int(rng.integers(0, 2**31 - 1)),
                "render_seed": int(rng.integers(0, 2**31 - 1)),
                "len_min": int(len_min),
                "len_max": int(len_max),
                "policy": str(policy),
                "action_scale": float(action_scale),
                "with_velocity": bool(with_velocity),
                "mode_profile": str(mode_profile),
                "quality_profile": str(quality_profile),
                "weights": dict(weights),
                "max_attempts_per_episode": int(max_attempts_per_episode),
                "contact_aware_ou_theta": float(contact_aware_ou_theta),
                "contact_aware_ou_sigma": float(contact_aware_ou_sigma),
                "contact_aware_ou_dt": float(contact_aware_ou_dt),
                "ou_theta": float(ou_theta),
                "ou_sigma": float(ou_sigma),
                "ou_dt": float(ou_dt),
                "ou_mu": None if ou_mu is None else np.asarray(ou_mu, dtype=np.float32).tolist(),
                "max_env_input_norm": float(max_env_input_norm),
            }
        accepted, failed_episode_batches = _run_parallel_episode_tasks(
            split_name=split_name,
            target_accepted=int(n_eps),
            workers=worker_count,
            task_factory=make_task,
            rejection_reasons=rejection_reasons,
            max_failed_episode_batches=resolved_max_failed_episode_batches,
        )

        render_progress = tqdm(
            total=int(n_eps),
            desc=f"render:{split_name}",
            unit="ep",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
        try:
            for result in accepted:
                frames = render_episode_frames(
                    render_env,
                    init_state=np.asarray(result["init_state"], dtype=np.float32),
                    actions=np.asarray(result["actions"], dtype=np.float32),
                    img_size=img_size,
                    seed=int(result["render_seed"]),
                )
                ep_actions = np.asarray(result["actions"], dtype=np.float32)
                ep_actions_abs = np.asarray(result["actions_abs"], dtype=np.float32)
                ep_states = np.asarray(result["states"], dtype=np.float32)
                if zarr_writer is not None:
                    zarr_writer.append_episode(frames, ep_actions, ep_actions_abs, ep_states)
                else:
                    actions_list.append(ep_actions)
                    actions_abs_list.append(ep_actions_abs)
                    frames_list.append(frames)
                    states_list.append(ep_states)
                render_progress.update(1)
                render_progress.set_postfix(
                    rejected=len(rejection_reasons),
                    failed_batches=failed_episode_batches,
                    refresh=False,
                )
        finally:
            render_progress.close()

        stats = {
            "accepted_episodes": int(n_eps),
            "rejected_attempts": int(len(rejection_reasons)),
            "failed_episode_batches": int(failed_episode_batches),
            "rejection_counts": summarize_rejections(rejection_reasons),
        }
        print(
            f"[generate_synth] split={split_name} accepted={stats['accepted_episodes']} "
            f"rejected={stats['rejected_attempts']} failed_batches={stats['failed_episode_batches']} "
            f"reasons={stats['rejection_counts']}"
        )
        return actions_list, actions_abs_list, frames_list, states_list, stats

    progress = tqdm(
        total=int(n_eps),
        desc=f"generate:{split_name}",
        unit="ep",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    try:
        accepted_episodes = 0
        while accepted_episodes < int(n_eps):
            accepted_attempt = None
            for _ in range(int(max_attempts_per_episode)):
                horizon = int(rng.integers(int(len_min), int(len_max) + 1))
                try:
                    if policy in {"contact_aware", "advanced"}:
                        attempt = rollout_contact_aware_episode(
                            decision_env,
                            shadow_env,
                            T=horizon,
                            rng=rng,
                            with_velocity=with_velocity,
                            mode_profile=mode_profile,
                            quality_profile=quality_profile,
                            weights=weights,
                            spawn_attempts=max_attempts_per_episode,
                            contact_aware_ou_theta=contact_aware_ou_theta,
                            contact_aware_ou_sigma=contact_aware_ou_sigma,
                            contact_aware_ou_dt=contact_aware_ou_dt,
                            max_env_input_norm=max_env_input_norm,
                        )
                    else:
                        attempt = rollout_baseline_episode(
                            decision_env,
                            T=horizon,
                            policy=policy,
                            action_scale=action_scale,
                            rng=rng,
                            with_velocity=with_velocity,
                            quality_profile=quality_profile,
                            spawn_attempts=max_attempts_per_episode,
                            ou_theta=ou_theta,
                            ou_sigma=ou_sigma,
                            ou_dt=ou_dt,
                            ou_mu=ou_mu,
                            max_env_input_norm=max_env_input_norm,
                        )
                except RuntimeError as exc:
                    reasons = _generator_runtime_reasons(exc)
                    if reasons is None:
                        raise
                    rejection_reasons.append(reasons)
                    progress.set_postfix(
                        rejected=len(rejection_reasons),
                        failed_batches=failed_episode_batches,
                        refresh=False,
                    )
                    continue
                if attempt.accepted:
                    accepted_attempt = attempt
                    break
                rejection_reasons.append(attempt.rejection_reasons)
                progress.set_postfix(
                    rejected=len(rejection_reasons),
                    failed_batches=failed_episode_batches,
                    refresh=False,
                )
            if accepted_attempt is None:
                failed_episode_batches += 1
                if int(resolved_max_failed_episode_batches) >= 0 and failed_episode_batches > int(resolved_max_failed_episode_batches):
                    summary = summarize_rejections(rejection_reasons)
                    raise RuntimeError(
                        f"Unable to produce enough accepted episodes for split={split_name!r}. "
                        f"accepted={accepted_episodes}/{int(n_eps)} "
                        f"failed_episode_batches={failed_episode_batches} "
                        f"max_failed_episode_batches={resolved_max_failed_episode_batches}. "
                        f"Rejections: {summary}"
                    )
                progress.set_postfix(
                    rejected=len(rejection_reasons),
                    failed_batches=failed_episode_batches,
                    refresh=False,
                )
                continue

            frames = render_episode_frames(
                render_env,
                init_state=accepted_attempt.init_state,
                actions=accepted_attempt.actions,
                img_size=img_size,
                seed=int(rng.integers(0, 1_000_000)),
            )
            if zarr_writer is not None:
                zarr_writer.append_episode(frames, accepted_attempt.actions, accepted_attempt.actions_abs, accepted_attempt.states)
            else:
                actions_list.append(accepted_attempt.actions)
                actions_abs_list.append(accepted_attempt.actions_abs)
                frames_list.append(frames)
                states_list.append(accepted_attempt.states)
            accepted_episodes += 1
            progress.update(1)
            progress.set_postfix(
                rejected=len(rejection_reasons),
                failed_batches=failed_episode_batches,
                refresh=False,
            )
    finally:
        progress.close()

    stats = {
        "accepted_episodes": int(n_eps),
        "rejected_attempts": int(len(rejection_reasons)),
        "failed_episode_batches": int(failed_episode_batches),
        "rejection_counts": summarize_rejections(rejection_reasons),
    }
    print(
        f"[generate_synth] split={split_name} accepted={stats['accepted_episodes']} "
        f"rejected={stats['rejected_attempts']} failed_batches={stats['failed_episode_batches']} "
        f"reasons={stats['rejection_counts']}"
    )
    return actions_list, actions_abs_list, frames_list, states_list, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("zarr_out", help="Output Zarr store path (e.g., synthetic/pusht_synth.zarr)")
    p.add_argument("--train_eps", type=int, default=200, help="Number of training episodes to generate")
    p.add_argument("--val_eps", type=int, default=50, help="Number of validation episodes to generate")
    p.add_argument("--len_min", type=int, default=50, help="Minimum episode length")
    p.add_argument("--len_max", type=int, default=160, help="Maximum episode length")
    p.add_argument(
        "--policy",
        choices=["ou", "random", "contact_aware", "advanced"],
        default="contact_aware",
        help="Action policy: contact_aware (default), ou, random, or advanced (alias for contact_aware).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--with_velocity", action="store_true", help="Include velocity in state representation")
    p.add_argument("--action_scale", type=float, default=1.0, help="Relative action scaling before env.action_scale is applied")
    p.add_argument("--img-size", type=int, default=96, help="Image size (width and height)")
    p.add_argument("--mode-profile", choices=["train", "planning", "balanced"], default="train")
    p.add_argument("--quality-profile", choices=["strict", "balanced", "loose"], default="strict")
    p.add_argument("--max-attempts-per-episode", type=int, default=64)
    p.add_argument(
        "--max-failed-episode-batches",
        type=int,
        default=-1,
        help="Global cap on exhausted episode batches before the split aborts. -1 disables the cap; 0 uses an automatic limit.",
    )
    p.add_argument("--workers", type=int, default=1, help="Number of worker processes for episode generation.")
    p.add_argument("--weight-translate", type=float, default=None)
    p.add_argument("--weight-rotate", type=float, default=None)
    p.add_argument("--weight-sweep", type=float, default=None)
    p.add_argument("--weight-recover", type=float, default=None)
    p.add_argument("--contact-aware-ou-theta", type=float, default=0.35, help="OU mean reversion rate for contact-aware absolute-target offsets.")
    p.add_argument("--contact-aware-ou-sigma", type=float, default=0.0, help="OU sigma in absolute-target pixels for contact-aware generation.")
    p.add_argument("--contact-aware-ou-dt", type=float, default=1.0, help="OU timestep for contact-aware absolute-target offsets.")
    p.add_argument("--ou-theta", type=float, default=0.15, help="OU process mean reversion rate")
    p.add_argument("--ou-sigma", type=float, default=0.2, help="OU process volatility")
    p.add_argument("--ou-dt", type=float, default=1.0, help="OU process time step")
    p.add_argument("--ou-mu", type=str, default=None, help="OU process mean (comma-separated per-dim or scalar)")
    p.add_argument(
        "--max-env-input-norm",
        type=float,
        default=1.0,
        help="Maximum allowed norm for stored and executed env-input actions.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.train_eps) <= 0 or int(args.val_eps) <= 0:
        raise ValueError("train_eps and val_eps must both be > 0")
    if int(args.len_min) <= 0 or int(args.len_max) < int(args.len_min):
        raise ValueError("len_min must be > 0 and len_max must be >= len_min")
    if int(args.max_attempts_per_episode) <= 0:
        raise ValueError("max_attempts_per_episode must be > 0")
    if int(args.max_failed_episode_batches) < -1:
        raise ValueError("max_failed_episode_batches must be >= -1")
    if int(args.workers) <= 0:
        raise ValueError("workers must be > 0")
    if float(args.max_env_input_norm) <= 0.0:
        raise ValueError("max_env_input_norm must be > 0")
    if str(args.quality_profile) not in QUALITY_PROFILES:
        raise ValueError(f"Unknown quality profile: {args.quality_profile}")

    rng = np.random.default_rng(int(args.seed))
    decision_env = build_generator_env(with_velocity=bool(args.with_velocity), headless=True)
    shadow_env = build_generator_env(with_velocity=bool(args.with_velocity), headless=True)
    render_env = build_generator_env(with_velocity=bool(args.with_velocity), headless=False)
    mode_weights = resolve_mode_weights(
        str(args.mode_profile),
        overrides={
            "translate": args.weight_translate,
            "rotate": args.weight_rotate,
            "sweep": args.weight_sweep,
            "recover": args.weight_recover,
        },
    )
    policy = "contact_aware" if str(args.policy) == "advanced" else str(args.policy)
    ou_mu = _parse_mu(args.ou_mu, decision_env.action_dim)

    writer = ZarrWriter(
        args.zarr_out,
        env_action_scale=float(decision_env.action_scale),
        env_relative=bool(decision_env.relative),
        max_env_input_norm=float(args.max_env_input_norm),
    )
    shared_split_kwargs = dict(
        len_min=int(args.len_min),
        len_max=int(args.len_max),
        policy=policy,
        action_scale=float(args.action_scale),
        rng=rng,
        decision_env=decision_env,
        shadow_env=shadow_env,
        render_env=render_env,
        with_velocity=bool(args.with_velocity),
        img_size=int(args.img_size),
        mode_profile=str(args.mode_profile),
        quality_profile=str(args.quality_profile),
        weights=mode_weights,
        max_attempts_per_episode=int(args.max_attempts_per_episode),
        max_failed_episode_batches=int(args.max_failed_episode_batches),
        workers=int(args.workers),
        contact_aware_ou_theta=float(args.contact_aware_ou_theta),
        contact_aware_ou_sigma=float(args.contact_aware_ou_sigma),
        contact_aware_ou_dt=float(args.contact_aware_ou_dt),
        ou_theta=float(args.ou_theta),
        ou_sigma=float(args.ou_sigma),
        ou_dt=float(args.ou_dt),
        ou_mu=ou_mu,
        max_env_input_norm=float(args.max_env_input_norm),
        zarr_writer=writer,
    )
    _, _, _, _, train_stats = generate_split(n_eps=int(args.train_eps), split_name="train", **shared_split_kwargs)
    _, _, _, _, val_stats = generate_split(n_eps=int(args.val_eps), split_name="valid", **shared_split_kwargs)

    rejected_attempts = int(train_stats["rejected_attempts"]) + int(val_stats["rejected_attempts"])
    failed_episode_batches = int(train_stats["failed_episode_batches"]) + int(val_stats["failed_episode_batches"])
    writer.finalize(extra_attrs={
        "generator_policy": str(policy),
        "mode_profile": str(args.mode_profile),
        "quality_profile": str(args.quality_profile),
        "weight_translate": float(mode_weights["translate"]),
        "weight_rotate": float(mode_weights["rotate"]),
        "weight_sweep": float(mode_weights["sweep"]),
        "weight_recover": float(mode_weights["recover"]),
        "contact_aware_ou_theta": float(args.contact_aware_ou_theta),
        "contact_aware_ou_sigma": float(args.contact_aware_ou_sigma),
        "contact_aware_ou_dt": float(args.contact_aware_ou_dt),
        "accepted_episodes": int(args.train_eps) + int(args.val_eps),
        "rejected_attempts": rejected_attempts,
        "failed_episode_batches": failed_episode_batches,
    })
    print(f"Saved synthetic Zarr store to {args.zarr_out}")
    print(f"  Training episodes: {args.train_eps}")
    print(f"  Validation episodes: {args.val_eps}")
    print(f"  Total frames: {writer.total_steps}")
    print(f"  Total actions: {writer.total_steps}")
    print(f"  Rejected attempts: {rejected_attempts}")
    print(f"  Failed episode batches: {failed_episode_batches}")


if __name__ == "__main__":
    main()
