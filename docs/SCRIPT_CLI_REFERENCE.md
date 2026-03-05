# Script CLI Reference

This document covers the user-facing script entrypoints under `scripts/`.

## `scripts/generate_synth.py`

Generate synthetic PushT episodes and save a training-compatible Zarr dataset.

```bash
python scripts/generate_synth.py <zarr_out> [options]
```

Required:

- `zarr_out`: output Zarr path (example: `synthetic/pusht_synth.zarr`)

Options:

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--train_eps` | int | `200` | Number of training episodes to generate. |
| `--val_eps` | int | `50` | Number of validation episodes to generate. |
| `--len_min` | int | `50` | Minimum episode length. |
| `--len_max` | int | `160` | Maximum episode length. |
| `--policy` | `ou|random|advanced` | `ou` | Action policy used for rollout generation. |
| `--seed` | int | `0` | RNG seed. |
| `--with_velocity` | flag | off | Use velocity-augmented state in environment wrapper. |
| `--action_scale` | float | `1.0` | Relative action scale before wrapper action scaling. |
| `--ou-theta` | float | `0.15` | OU process mean reversion rate. |
| `--ou-sigma` | float | `0.2` | OU process volatility. |
| `--ou-dt` | float | `1.0` | OU process timestep. |
| `--ou-mu` | str | `None` | OU mean (scalar or comma-separated per action dim). |
| `--img-size` | int | `96` | Output frame size in pixels. |

Output schema:

- `data/img`
- `data/action`
- `data/state`
- `meta/episode_ends`

## `scripts/visualize_rollouts.py`

Display stored rollout episodes from Zarr data.

```bash
python scripts/visualize_rollouts.py --config <world.yaml> [options]
```

Required:

- `--config`: world config path used to resolve dataset paths

Options:

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--split` | `train|valid|val` | `valid` | Episode split to visualize. |
| `--count` | int | `3` | Number of episodes to display. |
| `--fps` | int | `15` | Playback speed. |
| `--source` | `real|synthetic|mixed` | `real` | Data source selection. |

Behavior:

- Opens a window per episode.
- Close each window to continue to the next episode.

## `scripts/visualize_world_decoder.py`

Create a reconstruction grid comparing ground-truth frames to decoder outputs at each latent level.

```bash
python scripts/visualize_world_decoder.py <world.yaml> [options]
```

Required:

- `config`: world config path

Options:

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--count` | int | `4` | Number of episodes/samples included in the grid. |
| `--out` | path | `rollouts/decoder_grid.png` | Output image path. |

Behavior:

- Loads the latest run from `train.checkpoint_dir` in the provided world config.
- Writes a PNG grid to `--out`.

## `scripts/visualize_fidelity.py`

Visualize image-space fidelity transforms at selected levels.

```bash
python scripts/visualize_fidelity.py [--config <world.yaml> | --zarr <path>] [options]
```

Options:

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--config` | path | `None` | World config path (used to read `data.zarr_path`). |
| `--zarr` | path | `None` | Zarr path override (takes precedence over `--config`). |
| `--indices` | str | `None` | Comma-separated frame indices. |
| `--num-frames` | int | `3` | Random frame count (used when `--indices` is unset). |
| `--seed` | int | `0` | RNG seed for random frame selection. |
| `--modes` | str | `blur_avgpool,blur_quantize` | Comma-separated fidelity modes. |
| `--levels` | str | `0.0,0.5,1.0` | Comma-separated fidelity levels. |
| `--blur-sigma-max` | float | `2.0` | Blur parameter upper bound. |
| `--pool-scale-max` | int | `4` | Pooling scale upper bound. |
| `--quantize-levels-min` | int | `8` | Minimum quantization bins. |
| `--quantize-levels-max` | int | `256` | Maximum quantization bins. |
| `--save` | path | `None` | If set, saves the figure to file; otherwise opens interactive view. |

## `scripts/test_ensemble.py`

Quick integration smoke script for `MaskedDynamicsEnsemble`.

```bash
python scripts/test_ensemble.py
```

No CLI flags.

## `scripts/debug_planning_backend.py`

Interactive/scripted backend debugger for planning backends.

```bash
python scripts/debug_planning_backend.py --config configs/plan.yaml --backend particle_sim --keyboard
```

Options:

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--config` | path | `configs/plan.yaml` | Planner config path. |
| `--backend` | `gt_env\|particle_sim` | from config | Backend to debug. |
| `--seed` | int | `0` | Seed for init/goal sampling. |
| `--fidelity-level` | int | `-1` | Initial fidelity level index (`-1` = finest). |
| `--actions` | str | `\"\"` | Scripted actions: `ax,ay;ax,ay;...`. |
| `--keyboard` | flag | off | Enable realtime key control (one env step per frame at `--fps`). |
| `--max-steps` | int | `0` | Max steps to run; `<=0` means no limit. |
| `--key-action-mag` | float | `0.25` | Action magnitude used by WASD keys. |
| `--fps` | float | `12.0` | Render/update rate. |
| `--render-size` | int | `224` | Backend render resolution before display upsampling. |
| `--display-size` | int | `560` | Displayed image side length in pixels. |
| `--panel-width` | int | `430` | Side HUD panel width in pixels. |
| `--font-scale` | float | `0.58` | HUD font scale. |
| `--save` | path | `\"\"` | Optional GIF output path. |
| `--no-window` | flag | off | Disable OpenCV window output. |
| `--stop-on-done` | flag | off | Stop automatically when the backend returns `done=true`. |

Keyboard controls:

- `W/A/S/D`: move pusher for the current frame.
- no key press: apply no-op action for the current frame.
- `[` / `]` (or `-` / `=`): decrease/increase fidelity level.
- `0-9`: jump directly to fidelity level index.
- `R`: reset episode.
- `Q` or `Esc`: quit.
