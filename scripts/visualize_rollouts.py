#!/usr/bin/env python3
"""
Visualize rollouts from the PushT datasets by rendering de-normalized states
through the ground-truth environment and saving GIFs.

Usage:
  python scripts/visualize_rollouts.py \
      --config configs/train.yaml \
      --split valid \
      --count 5 \
      --outdir rollouts/dataset_vis \
      --fps 15 \
      --source mixed

Notes:
  - Uses the same representation toggle as training (data.use_sincos).
  - De-normalizes states using fixed PushT constants.
  - Does not depend on stored videos; renders frames directly from states.
"""

import os
import argparse
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import imageio.v2 as imageio  # imageio>=2.9
except ModuleNotFoundError:  # pragma: no cover
    import imageio

from gym.envs.registration import register
from pusht.pusht_wrapper import PushTWrapper

from datasets.pusht_dset import (
    load_pusht_slice_train_val,
    STATE_MEAN_SINCOS,
    STATE_STD_SINCOS,
    STATE_MEAN_ANGLE,
    STATE_STD_ANGLE,
)
from datasets.state_repr import sincos_to_angle


def denorm_states(states_norm: torch.Tensor, use_sincos: bool) -> torch.Tensor:
    """De-normalize a (T, D) tensor of states using PushT constants.
    If use_sincos is True, also convert sin/cos → angle to match env expectation.
    """
    D = states_norm.size(-1)
    if use_sincos:
        mean = STATE_MEAN_SINCOS[:D]
        std = STATE_STD_SINCOS[:D]
    else:
        mean = STATE_MEAN_ANGLE[:D]
        std = STATE_STD_ANGLE[:D]
    s = states_norm * std + mean
    if use_sincos:
        s = sincos_to_angle(s)
    return s


def render_episode(states_denorm: np.ndarray, env: PushTWrapper) -> List[np.ndarray]:
    """Render a sequence of environment frames from de-normalized states.
    Returns a list of RGB frames (H, W, C, uint8 or float32).
    """
    frames: List[np.ndarray] = []
    for t in range(states_denorm.shape[0]):
        obs, _ = env.prepare(seed=t, init_state=states_denorm[t])
        frames.append(obs['visual'])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to training or sim config with data.* fields')
    ap.add_argument('--split', choices=['train','valid','val'], default='valid')
    ap.add_argument('--count', type=int, default=3, help='Number of episodes to visualize')
    ap.add_argument('--outdir', type=str, default='rollouts/dataset_vis')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--source', choices=['real','synthetic','mixed'], default='real', help='Which dataset to visualize')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    data = cfg.data
    use_sincos = bool(getattr(data, 'use_sincos', True))
    with_velocity = bool(getattr(data, 'with_velocity', True))

    # Helper to load trajectories from a given root
    def load_trajs_at(root_path: str):
        _, td = load_pusht_slice_train_val(
            n_rollout=data.n_rollout,
            data_path=root_path,
            normalize_action=data.normalize_action,
            split_ratio=data.split_ratio,
            num_hist=0,
            num_pred=0,
            frameskip=1,
            with_velocity=with_velocity,
            use_sincos=use_sincos,
        )
        return td

    split_name = 'valid' if args.split in ('valid','val') else 'train'

    # Choose source(s)
    real_traj = load_trajs_at(data.path)[split_name]
    synth_traj = None
    if getattr(data, 'synthetic', None) and getattr(data.synthetic, 'path', None):
        synth_traj = load_trajs_at(data.synthetic.path)[split_name]

    # Register env and instantiate one instance for rendering
    try:
        register(
            id='pusht',
            entry_point='pusht.pusht_wrapper:PushTWrapper',
            max_episode_steps=300,
            reward_threshold=1.0,
        )
    except Exception:
        pass
    env = PushTWrapper(with_velocity=with_velocity, with_target=True, add_noise=0)

    os.makedirs(args.outdir, exist_ok=True)

    def dump_from(traj, tag: str, max_count: int):
        n = min(max_count, len(traj))
        for i in range(n):
            _, _, state_seq, _ = traj[i]
            state_t = torch.as_tensor(state_seq, dtype=torch.float32)
            state_denorm = denorm_states(state_t, use_sincos=use_sincos).cpu().numpy()
            frames = render_episode(state_denorm, env)
            out_path = os.path.join(args.outdir, f"{split_name}_{tag}_ep{i:03d}.gif")
            imageio.mimwrite(out_path, frames, fps=args.fps)
            print(f"Wrote {out_path}  ({len(frames)} frames)")

    os.makedirs(args.outdir, exist_ok=True)
    if args.source == 'real':
        dump_from(real_traj, 'real', args.count)
    elif args.source == 'synthetic':
        if synth_traj is None:
            raise SystemExit('No synthetic.path configured in data.synthetic.path')
        dump_from(synth_traj, 'synth', args.count)
    else:
        if synth_traj is None:
            raise SystemExit('No synthetic.path configured in data.synthetic.path for mixed source')
        half = max(1, args.count // 2)
        dump_from(real_traj, 'real', half)
        dump_from(synth_traj, 'synth', args.count - half)


if __name__ == '__main__':
    main()
