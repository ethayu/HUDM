#!/usr/bin/env python3
"""
Generate synthetic PushT rollouts by stepping the ground-truth environment and
save them in the same .pth format expected by datasets/pusht_dset.py.

Usage:
  python scripts/generate_synth.py --out synthetic/pusht_dataset --train_eps 200 --val_eps 50 \
      --len_min 50 --len_max 160 --policy ou --ou-theta 0.15 --ou-sigma 0.2 --seed 0 --with_velocity

Notes:
  - The saved dataset layout mirrors the real dataset: <out>/{train,val}/{states,rel_actions,abs_actions,velocities}.pth, seq_lengths.pkl, shapes.pkl
  - Actions are saved as relative actions (rel_actions.pth). You can switch to absolute by --abs.
  - States are stored in angle representation (5D). Training will convert to sin/cos if configured.
"""

import os
import math
import argparse
import pickle
from typing import List, Optional

import numpy as np
import torch

from gym.envs.registration import register
from pusht.pusht_wrapper import PushTWrapper


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
):
    init_state, goal = env.sample_random_init_goal_states(seed=int(rng.integers(0, 1_000_000)))
    obs, state0 = env.prepare(seed=int(rng.integers(0, 1_000_000)), init_state=init_state)

    states = [state0]
    actions = []

    # OU state init
    x = np.zeros(env.action_dim, dtype=np.float32)
    mu = np.zeros(env.action_dim, dtype=np.float32) if ou_mu is None else np.asarray(ou_mu, dtype=np.float32)

    for t in range(T):
        if policy == 'ou':
            # Ornstein–Uhlenbeck: x <- x + theta*(mu - x)*dt + sigma*sqrt(dt)*N(0,1)
            noise = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32)
            x = x + ou_theta * (mu - x) * ou_dt + ou_sigma * math.sqrt(ou_dt) * noise
            act = x * action_scale
        elif policy == 'random':
            act = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32) * action_scale
        else:
            # Placeholder for more advanced policies (e.g., CEM planner)
            act = rng.normal(0.0, 1.0, size=env.action_dim).astype(np.float32) * action_scale

        actions.append(act.astype(np.float32))
        _, _, done, info = env.step(act)
        states.append(info['state'].astype(np.float32))
        if done:
            break

    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    return states, actions


def pad_stack(seqs: List[np.ndarray], pad_last: int) -> np.ndarray:
    """Pad variable-length sequences to the same length along axis 0 using zeros.
    pad_last is the target length.
    """
    dims = seqs[0].shape[1:]
    out = np.zeros((len(seqs), pad_last) + dims, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        L = min(s.shape[0], pad_last)
        out[i, :L] = s[:L]
    return out


def save_split(root: str, states_list: List[np.ndarray], actions_list: List[np.ndarray], with_velocity: bool):
    os.makedirs(root, exist_ok=True)
    lengths = [len(s) for s in states_list]
    T_max = max(lengths)

    # states include angle (+ velocities if enabled); we store angle-only in states.pth and velocities separately
    states_arr = pad_stack([s[:, :5] for s in states_list], pad_last=T_max)
    if with_velocity:
        vels_arr = pad_stack([s[:, 5:7] for s in states_list], pad_last=T_max)
    else:
        vels_arr = None

    # actions: make length match states by padding a zero at the end
    A = actions_list[0].shape[-1]
    acts_padded = []
    for a, L in zip(actions_list, lengths):
        ap = np.zeros((L + 1, A), dtype=np.float32)
        ap[:L] = a
        acts_padded.append(ap)
    actions_arr = pad_stack(acts_padded, pad_last=T_max)

    # Save tensors
    torch.save(torch.from_numpy(states_arr), os.path.join(root, 'states.pth'))
    torch.save(torch.from_numpy(actions_arr), os.path.join(root, 'rel_actions.pth'))
    if vels_arr is not None:
        torch.save(torch.from_numpy(vels_arr), os.path.join(root, 'velocities.pth'))

    with open(os.path.join(root, 'seq_lengths.pkl'), 'wb') as f:
        pickle.dump(lengths, f)
    with open(os.path.join(root, 'shapes.pkl'), 'wb') as f:
        pickle.dump(['T'] * len(lengths), f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='Output root (e.g., synthetic/pusht_dataset)')
    p.add_argument('--train_eps', type=int, default=200)
    p.add_argument('--val_eps', type=int, default=50)
    p.add_argument('--len_min', type=int, default=50)
    p.add_argument('--len_max', type=int, default=160)
    p.add_argument('--policy', choices=['ou','random'], default='ou')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--with_velocity', action='store_true')
    p.add_argument('--action_scale', type=float, default=100.0)
    # OU params
    p.add_argument('--ou-theta', type=float, default=0.15)
    p.add_argument('--ou-sigma', type=float, default=0.2)
    p.add_argument('--ou-dt', type=float, default=1.0)
    p.add_argument('--ou-mu', type=str, default=None, help='comma-separated list for per-dim mean; default zeros')
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
        states_list, acts_list = [], []
        for _ in range(n_eps):
            T = int(rng.integers(args.len_min, args.len_max + 1))
            states, actions = rollout_episode(
                env,
                T=T,
                policy=args.policy,
                action_scale=args.action_scale,
                rng=rng,
                ou_theta=args.ou_theta,
                ou_sigma=args.ou_sigma,
                ou_dt=args.ou_dt,
                ou_mu=_parse_mu(args.ou_mu, env.action_dim),
            )
            states_list.append(states)
            acts_list.append(actions)
        return states_list, acts_list

    train_states, train_actions = gen_split(args.train_eps)
    val_states, val_actions = gen_split(args.val_eps)

    save_split(os.path.join(args.out, 'train'), train_states, train_actions, args.with_velocity)
    save_split(os.path.join(args.out, 'val'),   val_states,   val_actions,   args.with_velocity)

    print('Saved synthetic dataset to', args.out)


if __name__ == '__main__':
    main()
