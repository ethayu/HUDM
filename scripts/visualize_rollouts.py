#!/usr/bin/env python3
"""
Visualize rollouts from Zarr datasets by displaying them in a window.

Usage:
  # Visualize from world.yaml config with mixed dataset
  python scripts/visualize_rollouts.py \
      --config configs/world.yaml \
      --split valid \
      --count 5 \
      --fps 15 \
      --source mixed

Notes:
  - Works with zarr datasets (data.zarr_path and data.synthetic.zarr_path)
  - Images are stored directly in zarr, so no state rendering needed
  - Supports 'real', 'synthetic', or 'mixed' sources
  - Close the window to advance to the next episode
"""

import argparse
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import zarr
except Exception:
    zarr = None


def load_episodes_from_zarr(zarr_path: str, split: str = "train", split_ratio: float = 0.8) -> List[np.ndarray]:
    """Load full episodes from a zarr dataset.
    Returns a list of episode arrays, each of shape (T, H, W, C).
    """
    if zarr is None:
        raise ImportError("zarr not installed. pip install zarr")
    
    root = zarr.open_group(zarr_path, mode="r")
    data = root["data"]
    meta = root["meta"]
    
    img = data["img"]  # (N, H, W, C), float32
    ends = meta["episode_ends"][:]  # (E,), int64
    
    # Compute episode start indices
    starts = np.zeros_like(ends)
    starts[0] = 0
    for i in range(1, len(ends)):
        starts[i] = ends[i - 1] + 1
    
    # Split episodes
    n_ep = len(ends)
    n_train = int(split_ratio * n_ep)
    if split.lower() in ("train",):
        ep_idx = np.arange(0, n_train)
    else:
        ep_idx = np.arange(n_train, n_ep)
    
    episodes = []
    for ei in ep_idx:
        s = int(starts[ei])
        e = int(ends[ei])
        # Load full episode: images from s to e (inclusive)
        ep_imgs = img[s : e + 1]  # (T, H, W, C)
        episodes.append(ep_imgs)
    
    return episodes


def display_episode(episode: np.ndarray, title: str, fps: int = 15):
    """Display an episode in a window as an animation.
    episode: (T, H, W, C) array, values in [0, 1] (float32)
    """
    # Convert to uint8 [0, 255] for display
    if episode.dtype == np.float32 or episode.dtype == np.float64:
        episode = np.clip(episode, 0.0, 1.0)
        episode_display = (episode * 255).astype(np.uint8)
    else:
        episode_display = episode.astype(np.uint8)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    im = ax.imshow(episode_display[0])
    
    frame_text = ax.text(0.02, 0.98, f'Frame 0/{len(episode)-1}', 
                        transform=ax.transAxes, fontsize=12,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                        color='white')
    
    def update_frame(frame_idx):
        im.set_array(episode_display[frame_idx])
        frame_text.set_text(f'Frame {frame_idx}/{len(episode)-1}')
        return im, frame_text
    
    interval = 1000 / fps  # milliseconds per frame
    anim = animation.FuncAnimation(fig, update_frame, frames=len(episode),
                                  interval=interval, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    return anim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to world.yaml config')
    ap.add_argument('--split', choices=['train','valid','val'], default='valid')
    ap.add_argument('--count', type=int, default=3, help='Number of episodes to visualize')
    ap.add_argument('--fps', type=int, default=15, help='Frames per second for playback')
    ap.add_argument('--source', choices=['real','synthetic','mixed'], default='real',
                    help='Which dataset to visualize')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    data = cfg.data
    split_ratio = float(data.split_ratio)
    split_name = 'valid' if args.split in ('valid','val') else 'train'

    # Load real episodes
    real_episodes = load_episodes_from_zarr(data.zarr_path, split=split_name, split_ratio=split_ratio)
    
    # Load synthetic episodes if available
    synth_episodes = None
    if getattr(data, 'synthetic', None) and getattr(data.synthetic, 'zarr_path', None):
        synth_episodes = load_episodes_from_zarr(
            data.synthetic.zarr_path, 
            split=split_name, 
            split_ratio=split_ratio
        )

    # Collect episodes to display
    episodes_to_show = []
    titles = []
    
    if args.source == 'real':
        n = min(args.count, len(real_episodes))
        episodes_to_show = real_episodes[:n]
        titles = [f'Real Episode {i+1}/{n}' for i in range(n)]
    elif args.source == 'synthetic':
        if synth_episodes is None:
            raise SystemExit('No synthetic.zarr_path configured in data.synthetic.zarr_path')
        n = min(args.count, len(synth_episodes))
        episodes_to_show = synth_episodes[:n]
        titles = [f'Synthetic Episode {i+1}/{n}' for i in range(n)]
    else:  # mixed
        if synth_episodes is None:
            raise SystemExit('No synthetic.zarr_path configured in data.synthetic.zarr_path for mixed source')
        half = max(1, args.count // 2)
        n_real = min(half, len(real_episodes))
        n_synth = min(args.count - half, len(synth_episodes))
        episodes_to_show = real_episodes[:n_real] + synth_episodes[:n_synth]
        titles = ([f'Real Episode {i+1}/{n_real}' for i in range(n_real)] +
                 [f'Synthetic Episode {i+1}/{n_synth}' for i in range(n_synth)])
    
    # Display each episode
    print(f"Displaying {len(episodes_to_show)} episodes. Close window to advance to next episode.")
    for i, (ep, title) in enumerate(zip(episodes_to_show, titles)):
        print(f"\n[{i+1}/{len(episodes_to_show)}] {title} ({len(ep)} frames)")
        display_episode(ep, title, fps=args.fps)


if __name__ == '__main__':
    main()

