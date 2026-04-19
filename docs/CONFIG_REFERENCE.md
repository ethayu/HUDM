# YAML Config Reference

This repository currently uses two canonical YAML configs:

- `configs/world.yaml` for world-model training (`train_world.py`)
- `configs/plan.yaml` for latent-space MPC-CEM planning (`plan.py`)

The fields below are the source of truth for the current implementation.

## Design principles

- Strict schema: unknown/no-op keys are rejected.
- No silent clamping: out-of-range values raise errors.
- Mode-specific behavior is explicit (`fixed` vs `linear` vs `uncertainty_downshift`).
- Checkpoint metadata is inferred from run directories by default.

## `configs/world.yaml`

### Top-level

| Field | Type | Notes |
|---|---|---|
| `seed` | int | Experiment seed value stored with run metadata. |
| `data` | mapping | Dataset and split configuration. |
| `model` | mapping | Matryoshka latent and model architecture settings. |
| `train` | mapping | Data loader and checkpoint output settings. |
| `optim` | mapping | Optimizer settings. |
| `loss` | mapping | Loss weights. |
| `schedule` | mapping | Early stopping / epoch limits. |
| `wandb` | mapping | Optional W&B logging. |

### `data`

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `zarr_path` | path | existing zarr root | Required. Real demonstration dataset. |
| `split_ratio` | float | `(0, 1)` | Episode split for train/valid loaders. |
| `action_dim` | int | `> 0` | Must match dataset action width and model action dim. |
| `action_mode` | str | `relative` | `train_world.py` currently requires relative actions. |
| `synthetic.enable` | bool | `true/false` | Enables mixed real+synthetic training set. |
| `synthetic.zarr_path` | path | existing zarr root | Required if `synthetic.enable=true`. |
| `synthetic.frac` | float | `[0, 1]` | Fraction of synthetic samples in train mix. |
| `synthetic.total_train` | int\|null | `>0` or `null` | Total train samples after mixing. `null` -> use real train length. |
| `synthetic.seed` | int | any int | Sampling seed for subsampling/mixing. |
| `synthetic.val_source` | str | `real`\|`synthetic`\|`mixed` | Validation source selection. |

### `model`

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `D` | int | `> 0` | Full latent dimensionality. |
| `K` | list[int] | strictly increasing, final value == `D` | Matryoshka prefix levels. |
| `decoder_mode` | str | `per_level`\|`shared` | Decoder parameter sharing strategy. |
| `dynamics_mode` | str | `per_level`\|`shared` | Dynamics parameter sharing strategy. |

### `train`, `optim`, `loss`, `schedule`, `wandb`

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `train.batch_size` | int | `> 0` | Train/valid loader batch size. |
| `train.num_workers` | int | `>= 0` | Data loader workers. |
| `train.no_cuda` | bool | `true/false` | Forces CPU when true. |
| `train.checkpoint_dir` | path | writable dir | Root for timestamped runs. |
| `train.run_name` | str | non-empty recommended | Prefix for run directory name. |
| `optim.lr` | float | `> 0` | Adam learning rate. |
| `loss.recon_weight` | float | `>= 0` | Reconstruction term weight. |
| `loss.teacher_weight` | float | `>= 0` | Teacher-forcing term weight. |
| `loss.rollout_weight` | float | `>= 0` | Autoregressive rollout term weight. |
| (loss sum) | float | `> 0` | At least one loss term must be active. |
| `schedule.max_epochs` | int | `> 0` | Hard cap on epochs. |
| `schedule.patience` | int | `> 0` | Early-stop patience. |
| `schedule.min_delta` | float | `>= 0` | Improvement threshold. |
| `wandb.enable` | bool | `true/false` | Requires `wandb` installed when true. |
| `wandb.project` | str | any | W&B project name. |
| `wandb.run_name` | str | any | W&B run name. |

## `configs/plan.yaml`

### Top-level

| Field | Type | Notes |
|---|---|---|
| `backend` | str | Planning backend: `wm` (learned world model), `gt_env` (ground-truth env propagation), or `particle_sim` (Warp particle propagation). |
| `env_id` | str | Gym env id (default PushT wrapper id). |
| `env` | mapping | Env constructor kwargs. |
| `world_model` | mapping | Checkpoint/model-loading setup. |
| `mpc` | mapping | Closed-loop control schedule. |
| `cem` | mapping | CEM optimizer settings. |
| `objective` | mapping | Planner cost weights/metric (latent for `wm`, image/state for `gt_env` and `particle_sim`). |
| `fidelity` | mapping | Multi-resolution schedules across MPC/CEM/rollout. |
| `gt_env` | mapping | Ground-truth env backend rollout/fidelity settings. |
| `particle_env` | mapping | Warp particle backend rollout/fidelity settings. |
| `init_goal` | mapping | Initial/goal state sampling configuration. |
| `render` | bool | Enable interactive render. |
| `save` | bool | Save rollout MP4 files under `rollouts/`. |

### `env`

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `with_velocity` | bool | `true/false` | Should match world-model training distribution. |
| `with_target` | bool | `true/false` | PushT env target-state inclusion flag. |
| `add_noise` | int | `0`\|`1`\|`2` | 0 none, 1 action noise, 2 state noise. |
| `noise_std` | float\|list | scalar or per-dim | Noise std interpreted by wrapper mode/dimension. |

### `world_model`

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `device` | str | `auto`\|`cpu`\|`cuda` | Compute device for model + planner. |
| `run_dir` | path\|null | existing run dir or `null` | Used in single-model mode. |
| `checkpoint_root` | path | existing directory | Used when `run_dir: null` in single-model mode. |
| `config_path` | path\|null | existing yaml or `null` | Optional explicit world config override. |
| `ensemble.enabled` | bool | `true/false` | Enables world-model ensemble wrapper. |
| `ensemble.run_dirs` | list[path] | len>=2 when enabled | Member checkpoint directories. |

Loading behavior:

- Single-model mode: `run_dir` if set, else latest run under `checkpoint_root`.
- If `config_path` is null, config is read from `<run_dir>/world.yaml`.
- Ensemble mode: config is inferred from first run dir (or `config_path` if provided), and each member run with `world.yaml` is checked for compatibility.

### `mpc`

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `steps` | int | `> 0` | Max environment steps. |
| `horizon` | int | `> 0` | Planned action horizon per replan. |
| `replan_every` | int | `1..horizon` | Number of actions executed before replanning. |

### `cem`

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `pop_size` | int | `> 0` | Candidate trajectories per CEM iteration. |
| `elite_frac` | float | `(0, 1)` | Elite fraction for distribution update. |
| `n_iter` | int | `> 0` | CEM iterations per replan. |
| `init_std` | float | `> 0` recommended | Initial Gaussian std for action sampling. |
| `warm_start` | bool | `true/false` | Reuse previous replan CEM distribution by shifting it forward by executed MPC actions. |
| `action_low` | float\|null | optional | Lower clip bound on sampled actions. |
| `action_high` | float\|null | optional | Upper clip bound on sampled actions. |

### `objective`

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `latent_metric` | str | `l1`\|`l2` | Distance metric for `wm` latent cost and `gt_env` image cost. |
| `terminal_weight` | float | any real | Terminal distance weight (`wm` latent or `gt_env` image mode). |
| `running_weight` | float | any real | Per-step distance weight (`wm` latent or `gt_env` image mode). |
| `action_l2_weight` | float | any real | Action magnitude regularization weight. |
| `eef_weight` | float | any real | `gt_env` backend: end-effector distance weight. |
| `block_pos_weight` | float | any real | `gt_env` backend: block position distance weight. |
| `block_angle_weight` | float | any real | `gt_env` backend: block angle distance weight. |
| `state_l2_weight` | float | any real | `gt_env` backend: full-state L2 term weight. |

### `fidelity`

`fidelity` uses level indices over `model.K`:

- index `0` = coarsest
- index `len(K)-1` = finest

Level fields accept either:

- integer index
- token `coarsest`
- token `finest`
- token `base` (CEM and rollout fields)

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `enabled` | bool | `true/false` | If false, always use finest level. |
| `mpc.mode` | str | `fixed`\|`linear` | Replan-level schedule mode. |
| `mpc.level` | int\|token | used in `fixed` mode | Requested fixed MPC level. |
| `mpc.start_level` | int\|token | used in `linear` mode | Start level over MPC progress. |
| `mpc.end_level` | int\|token | used in `linear` mode | End level over MPC progress. |
| `cem.mode` | str | `fixed`\|`linear` | Iteration-level schedule mode. |
| `cem.level` | int\|token | used in `fixed` mode | Requested fixed CEM level. |
| `cem.start_level` | int\|token | used in `linear` mode | Start level over CEM iterations. |
| `cem.end_level` | int\|token | used in `linear` mode | End level over CEM iterations. |
| `rollout.mode` | str | `fixed`\|`linear`\|`uncertainty_downshift` | Within-horizon schedule mode. |
| `rollout.level` | int\|token | used in `fixed` mode | Rollout level for all horizon steps. |
| `rollout.start_level` | int\|token | used in `linear` mode | Step-0 rollout level. |
| `rollout.end_level` | int\|token | used in `linear` mode | Last-step rollout level. |
| `rollout.uncertainty.criterion` | str | `mean`\|`percentile` | Aggregation used for downshift trigger. |
| `rollout.uncertainty.threshold` | float | any real | Downshift if score > threshold. |
| `rollout.uncertainty.percentile` | float | `[0, 1]` | Quantile used when criterion is `percentile`. |
| `rollout.uncertainty.min_level` | int\|token | level index | Lowest level allowed by downshift policy. |
| `rollout.uncertainty.max_downshifts_per_step` | int | `> 0` | Max downshift moves at a single timestep. |

`base` token semantics:

- In `fidelity.cem.*` fields: `base` resolves to the current MPC-stage level.
- In `fidelity.rollout.*` fields: `base` resolves to the current CEM-iteration level.

Uncertainty score definition for candidate population:

- Compute predicted latent variance at current level.
- Keep only the tail dimensions that would be dropped by one-level downshift.
- Per candidate: mean of those tail variances.
- Aggregate across candidates by:
  - `mean`: population mean
  - `percentile`: configured quantile

### Invalid combinations (rejected)

- `ensemble.enabled=true` with fewer than 2 `ensemble.run_dirs`.
- `ensemble.enabled=true` with `world_model.run_dir` set.
- `ensemble.enabled=false` with non-empty `ensemble.run_dirs`.
- `backend=gt_env` with `fidelity.rollout.mode=uncertainty_downshift`.
- `backend=particle_sim` with `fidelity.rollout.mode=uncertainty_downshift`.
- `rollout.mode=uncertainty_downshift` with non-ensemble world model.
- Unknown fidelity modes or out-of-range level indices.

### `gt_env`

Only used when `backend: gt_env`.

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `rollout_samples` | int | `>0` | Number of stochastic env rollouts per candidate action sequence. |
| `objective_space` | str | `image`\|`state` | `image`: optimize against visual observations. `state`: optimize against low-dim PushT state terms (`eef/block/angle`). |
| `progress` | bool | `true/false` | Show per-CEM-iteration rollout progress bar for `gt_env` planning. |
| `progress_leave` | bool | `true/false` | Keep completed progress bars in terminal output. |
| `fidelity_env.num_levels` | int | `>0` | Number of discrete planning fidelity levels owned by the `gt_env` backend. |
| `fidelity_env.mode` | str | `blur_avgpool`\|`blur_quantize` | Env-side visual fidelity transform. |
| `fidelity_env.blur_sigma_max` | float | `>=0` | Max blur at coarsest level. |
| `fidelity_env.pool_scale_max` | int | `>=1` | Max pooling/downscale factor for `blur_avgpool`. |
| `fidelity_env.quantize_levels_min` | int | `>=2` | Minimum quantization bins at coarsest level. |
| `fidelity_env.quantize_levels_max` | int | `>=2` | Maximum quantization bins at finest level. |
| `fidelity_env.action_noise_std_max` | float | `>=0` | Coarsest-level action noise std (linearly decays to 0 at finest). |
| `fidelity_env.downsample_output` | bool | `true/false` | If true, returns genuinely lower-resolution images at coarse levels. |
| `fidelity_env.min_downsample_size` | int | `>=4` | Smallest output side length used at coarsest level when downsampling is enabled. |

### `particle_env`

Only used when `backend: particle_sim`.

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `rollout_samples` | int | `>0` | Number of stochastic particle rollouts per candidate action sequence. |
| `objective_space` | str | `image`\|`state` | `state`: optimize pose-based PushT metrics. `image`: optimize rendered visual distance. |
| `progress` | bool | `true/false` | Show per-CEM-iteration rollout progress bar for particle planning. |
| `progress_leave` | bool | `true/false` | Keep completed progress bars in terminal output. |
| `fidelity_env.spacings` | list[float] | positive values | Coarsest->finest spacing per fidelity level (larger spacing => fewer particles). The particle backend derives its level count from this list. |
| `fidelity_env.device` | str | `auto`\|`cpu`\|`cuda`\|`cuda:N` | Warp device selection. |
| `fidelity_env.{xmin,xmax,ymin,ymax}` | float | any real | World bounds for particle simulator. |
| `fidelity_env.min_particles` | int | `>=1` | Minimum sampled particles; very coarse levels can collapse to `N=1`. |
| `fidelity_env.coarsest_single_particle` | bool | `true/false` | If true, the coarsest level (`spacings[0]`) is forced to one particle (`N=1`). Enabled by default. |
| `fidelity_env.particle_radius` | float\|null | `>0` or `null` | `null` enables auto radius scaling from particle count (`N`); float forces fixed radius. |
| `fidelity_env.radius_scale` | float | `>0` | Multiplier for auto-scaled radius when `particle_radius=null`. |
| `fidelity_env.radius_clip_spacing` | bool | `true/false` | Clamp auto radius relative to spacing (`N>1`) for stability. |
| `fidelity_env.stem_w` | float | `>0` | T stem width in world units. |
| `fidelity_env.stem_h` | float | `>0` | T stem height in world units. |
| `fidelity_env.bar_w` | float | `>0` | T bar width in world units. |
| `fidelity_env.bar_h` | float | `>0` | T bar height in world units. |
| `fidelity_env.pusher_radius` | float | `>0` | Kinematic pusher radius in world units. |
| `fidelity_env.sim_hz` | int | `>0` | GT-matched pusher microstep rate; must be divisible by `control_hz`. |
| `fidelity_env.control_hz` | int | `>0` | Planner/control-step rate for one backend action. |
| `fidelity_env.pusher_k_p` | float | `>=0` | GT-style proportional gain for the kinematic pusher PD controller. |
| `fidelity_env.pusher_k_v` | float | `>=0` | GT-style damping gain for the kinematic pusher PD controller. |
| `fidelity_env.substeps` | int | `>=1` | Solver substeps per control frame while sampling the PD pusher path continuously. |
| `fidelity_env.iters` | int | `>=1` | Solver iterations per substep. |
| `fidelity_env.mu` | float | `>=0` | Position-level friction factor for pusher contact. |
| `fidelity_env.contact_alpha` | float | `(0,1]` | Contact projection gain; lower values reduce bounce from light touches. |
| `fidelity_env.ground_friction_accel` | float | `>=0` | Per-particle ground-friction deceleration (world units/s²) applied each substep. |
| `fidelity_env.rest_speed_eps` | float | `>=0` | Static-friction threshold; particle speeds below this are snapped to zero. |
| `fidelity_env.lin_damp` | float | any real | Linear damping during prediction. |
| `fidelity_env.vel_damp` | float | any real | Velocity damping during finalize. |
| `fidelity_env.alpha_rigid` | float | `>=0` | Shape-matching projection blend (`1.0` = rigid). |

### `init_goal`

Controls where planning initial/goal states come from.

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `source` | str | `random`\|`dataset` | Randomly sampled states or dataset trajectory endpoints. |
| `dataset.zarr_path` | path\|null | existing zarr or `null` | If `null`, planner falls back to world config `data.zarr_path` when available. |
| `dataset.split` | str | `train`\|`valid`\|`val` | Split used for dataset-based state sampling. |
| `dataset.split_ratio` | float\|null | `(0,1)` or `null` | Episode split ratio; fallback to world split ratio (or 0.8). |
| `dataset.trajectory_len` | int | `>0` | Number of transitions between sampled start and goal states. |
| `dataset.seed` | int\|str | int or `"random"` | Sampling seed for episode and start index selection. |
| `dataset.reconstruct_goal_state` | int | `0`\|`1`\|`2`\|`3` | `0`: do not reconstruct goal. `1`: reconstruct from actions and error if it differs from stored state. `2`: reconstruct from actions and use reconstructed goal. `3`: force action replay as canonical GT trajectory (stored in metadata), and use replayed goal state. |
