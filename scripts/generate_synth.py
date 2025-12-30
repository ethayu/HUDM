#!/usr/bin/env python3
"""
Generate synthetic PushT rollouts by stepping the ground-truth environment and
save them in Zarr format for world-model training.

Usage:
  # Basic OU policy
  python scripts/generate_synth.py synthetic/pusht_synth.zarr \
      --train_eps 200 --val_eps 50 \
      --len_min 50 --len_max 160 \
      --policy ou --ou-theta 0.15 --ou-sigma 0.2 \
      --seed 0 --with_velocity --img-size 96
  
  # Advanced policy (OU + goal-directed toward T block)
  python scripts/generate_synth.py synthetic/pusht_synth.zarr \
      --train_eps 200 --val_eps 50 \
      --len_min 50 --len_max 160 \
      --policy advanced --ou-theta 0.15 --ou-sigma 0.2 \
      --seed 0 --with_velocity --img-size 96

Notes:
  - Outputs a Zarr store compatible with datasets/zarr_episodes.py
  - The Zarr store contains: data/img, data/action, data/state, meta/episode_ends
  - Images are stored as float32 in [0, 255] range
  - Actions are absolute pixel targets (float32)
  - States are the pre-action environment state (float32)
  - Train/val split is handled via split_ratio when loading the dataset
"""

import os
import sys
import math
import argparse
from typing import List, Optional, Tuple

# Add project root to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gym.envs.registration import register
from pusht.pusht_wrapper import PushTWrapper
from PIL import Image

try:
    import zarr
except Exception:
    zarr = None


def rollout_episode(
    env: PushTWrapper,
    T: int,
    policy: str,
    action_scale: float,
    rng: np.random.Generator,
    ou_theta: float = 0.15,
    ou_sigma: float = 0.2,
    ou_dt: float = 1.0,
    ou_mu: Optional[np.ndarray] = None,
    collect_frames: bool = False,
    img_size: int = 96,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    init_state, goal = env.sample_random_init_goal_states(seed=int(rng.integers(0, 1_000_000)))
    obs, state = env.prepare(seed=int(rng.integers(0, 1_000_000)), init_state=init_state)

    states = []
    actions = []
    frames = []

    # OU state init
    x = np.zeros(env.action_dim, dtype=np.float32)
    mu = np.zeros(env.action_dim, dtype=np.float32) if ou_mu is None else np.asarray(ou_mu, dtype=np.float32)

    # Environment bounds (walls at 5 and 506, so safe range is roughly [15, 491])
    env_bounds = (15.0, 491.0)
    
    for t in range(T):
        current_state = state
        agent_pos = current_state[:2]
        block_pos = current_state[2:4]

        if policy == 'ou':
            # Ornstein–Uhlenbeck: x <- x + theta*(mu - x)*dt + sigma*sqrt(dt)*N(0,1)
            noise = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32)
            x = x + ou_theta * (mu - x) * ou_dt + ou_sigma * math.sqrt(ou_dt) * noise
            act = x * action_scale
        elif policy == 'random':
            act = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32) * action_scale
        elif policy == 'advanced':
            # Advanced policy: OU-style with goal-directed behavior toward T block
            # Get current state: [agent_x, agent_y, block_x, block_y, block_angle, ...]
            # Decide whether to be goal-directed (40% of the time)
            goal_directed_prob = 0.4
            is_goal_directed = rng.random() < goal_directed_prob
            
            if is_goal_directed and t > 5:  # Start goal-directed after a few steps
                # Compute direction toward block to create meaningful interactions
                # We want to approach the block from a good angle for pushing
                agent_to_block = block_pos - agent_pos
                agent_to_block_norm = np.linalg.norm(agent_to_block)
                
                if agent_to_block_norm > 1e-6:
                    # Base direction: toward the block
                    direction = agent_to_block / agent_to_block_norm
                    
                    # Add perpendicular component for better pushing angles
                    # This creates circular/arc movements around the block
                    perp = np.array([-direction[1], direction[0]])
                    perp_weight = rng.uniform(-0.3, 0.3)  # Vary the approach angle
                    
                    # Combine direction toward block with perpendicular component
                    desired_direction = direction + perp * perp_weight
                    desired_norm = np.linalg.norm(desired_direction)
                    
                    if desired_norm > 1e-6:
                        desired_direction = desired_direction / desired_norm
                        # Scale by distance factor: move faster when far, slower when close
                        distance_factor = min(1.0, agent_to_block_norm / 100.0)
                        mu_goal = desired_direction * 0.6 * distance_factor  # Bias toward block
                    else:
                        mu_goal = mu
                else:
                    # Already very close, use exploration
                    mu_goal = mu
            else:
                # Pure exploration mode - but still keep within bounds
                # Use OU with zero mean but add boundary repulsion
                agent_pos_normalized = (agent_pos - 256.0) / 256.0  # Center and normalize
                # Soft boundary repulsion: push away from walls
                boundary_repulsion = np.zeros(2, dtype=np.float32)
                if agent_pos[0] < 100:
                    boundary_repulsion[0] = 0.5  # Push right
                elif agent_pos[0] > 412:
                    boundary_repulsion[0] = -0.5  # Push left
                if agent_pos[1] < 100:
                    boundary_repulsion[1] = 0.5  # Push down
                elif agent_pos[1] > 412:
                    boundary_repulsion[1] = -0.5  # Push up
                
                mu_goal = boundary_repulsion
            
            # OU process with goal-directed mean
            noise = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32)
            x = x + ou_theta * (mu_goal - x) * ou_dt + ou_sigma * math.sqrt(ou_dt) * noise
            
            # Actions are relative (velocities), scaled by action_scale
            act = x * action_scale
            
            # Check if action would take agent out of bounds and dampen if needed
            next_pos = agent_pos + act
            if next_pos[0] < env_bounds[0] or next_pos[0] > env_bounds[1]:
                act[0] *= 0.03  # Dampen x component
            if next_pos[1] < env_bounds[0] or next_pos[1] > env_bounds[1]:
                act[1] *= 0.03  # Dampen y component
        else:
            # Fallback to random
            act = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32) * action_scale

        # Save pre-step state and corresponding absolute action target
        if env.relative:
            action_abs = agent_pos + act * env.action_scale
        else:
            action_abs = act * env.action_scale

        actions.append(action_abs.astype(np.float32))
        states.append(current_state.astype(np.float32))

        if collect_frames:
            fr = obs['visual']
            im = Image.fromarray(fr.astype(np.uint8)) if fr.dtype != np.uint8 else Image.fromarray(fr)
            im = im.resize((img_size, img_size), Image.BILINEAR)
            frames.append(np.asarray(im).astype(np.float32))

        obs, _, done, info = env.step(act)
        state = info['state'].astype(np.float32)
        if collect_frames:
            pass
        if done:
            break

    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    if collect_frames:
        frames = np.asarray(frames, dtype=np.float32)
    else:
        frames = None
    return states, actions, frames


def save_zarr(
    zarr_out: str,
    frames_list: List[np.ndarray],
    actions_list: List[np.ndarray],
    states_list: List[np.ndarray],
):
    if zarr is None:
        raise ImportError("zarr is not installed. Try: pip install zarr")
    # Flatten episodes; ensure per-episode frames match action count (pre-step frames)
    # frames_list[i].shape == (Li, H, W, 3); actions_list[i].shape == (Li, A)
    lengths = [
        min(f.shape[0], a.shape[0], s.shape[0])
        for f, a, s in zip(frames_list, actions_list, states_list)
    ]
    frames = np.concatenate([f[:L] for f, L in zip(frames_list, lengths)], axis=0)
    actions = np.concatenate([a[:L] for a, L in zip(actions_list, lengths)], axis=0)
    states = np.concatenate([s[:L] for s, L in zip(states_list, lengths)], axis=0)
    ends = np.cumsum(lengths) - 1

    root = zarr.group(zarr_out, overwrite=True)
    g_data = root.create_group('data')
    g_meta = root.create_group('meta')
    # Create arrays
    g_data.create('img', data=frames.astype(np.float32), chunks=(min(160, frames.shape[0]),) + frames.shape[1:])
    g_data.create('action', data=actions.astype(np.float32), chunks=(min(160, actions.shape[0]), actions.shape[1]))
    g_data.create('state', data=states.astype(np.float32), chunks=(min(160, states.shape[0]), states.shape[1]))
    g_meta.create('episode_ends', data=ends.astype(np.int64), chunks=(max(1, len(ends)),))
    # minimal attrs
    root.attrs.update({})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('zarr_out', help='Output Zarr store path (e.g., synthetic/pusht_synth.zarr)')
    p.add_argument('--train_eps', type=int, default=200, help='Number of training episodes to generate')
    p.add_argument('--val_eps', type=int, default=50, help='Number of validation episodes to generate')
    p.add_argument('--len_min', type=int, default=50, help='Minimum episode length')
    p.add_argument('--len_max', type=int, default=160, help='Maximum episode length')
    p.add_argument('--policy', choices=['ou','random','advanced'], default='ou', 
                   help='Action policy: ou (Ornstein-Uhlenbeck), random, or advanced (OU + goal-directed toward T block)')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    p.add_argument('--with_velocity', action='store_true', help='Include velocity in state representation')
    p.add_argument('--action_scale', type=float, default=1.0, help='Action scaling factor (default 100.0 for PushT)')
    # OU params
    p.add_argument('--ou-theta', type=float, default=0.15, help='OU process mean reversion rate')
    p.add_argument('--ou-sigma', type=float, default=0.2, help='OU process volatility')
    p.add_argument('--ou-dt', type=float, default=1.0, help='OU process time step')
    p.add_argument('--ou-mu', type=str, default=None, help='OU process mean (comma-separated per-dim or scalar)')
    p.add_argument('--img-size', type=int, default=96, help='Image size (width and height)')
    args = p.parse_args()

    # Register env id if needed
    try:
        register(
            id='pusht',
            entry_point='pusht.pusht_wrapper:PushTWrapper',
            max_episode_steps=300,
            reward_threshold=1.0,
        )
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    env = PushTWrapper(with_velocity=args.with_velocity, with_target=True, add_noise=0)

    def _parse_mu(s: Optional[str], dim: int) -> Optional[np.ndarray]:
        if s is None:
            return None
        vals = [float(x) for x in s.split(',') if x.strip() != '']
        if len(vals) == 1:
            return np.full(dim, vals[0], dtype=np.float32)
        assert len(vals) == dim, f"--ou-mu must have length {dim} or be scalar"
        return np.asarray(vals, dtype=np.float32)

    def gen_split(n_eps: int):
        acts_list, frames_list, states_list = [], [], []
        for _ in range(n_eps):
            T = int(rng.integers(args.len_min, args.len_max + 1))
            states, actions, frames = rollout_episode(
                env,
                T=T,
                policy=args.policy,
                action_scale=args.action_scale,
                rng=rng,
                ou_theta=args.ou_theta,
                ou_sigma=args.ou_sigma,
                ou_dt=args.ou_dt,
                ou_mu=_parse_mu(args.ou_mu, env.action_dim),
                collect_frames=True,  # Always collect frames for zarr output
                img_size=args.img_size,
            )
            acts_list.append(actions)
            frames_list.append(frames)
            states_list.append(states)
        return acts_list, frames_list, states_list

    train_actions, train_frames, train_states = gen_split(args.train_eps)
    val_actions, val_frames, val_states = gen_split(args.val_eps)

    # Save a single Zarr with both train and val concatenated
    # Train/val separation is handled via split_ratio when loading the dataset
    frames_all = train_frames + val_frames
    actions_all = train_actions + val_actions
    states_all = train_states + val_states
    save_zarr(args.zarr_out, frames_all, actions_all, states_all)
    print(f'Saved synthetic Zarr store to {args.zarr_out}')
    print(f'  Training episodes: {args.train_eps}')
    print(f'  Validation episodes: {args.val_eps}')
    print(f'  Total frames: {sum(len(f) for f in frames_all)}')
    print(f'  Total actions: {sum(len(a) for a in actions_all)}')


if __name__ == '__main__':
    main()
