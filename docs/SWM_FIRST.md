# SWM-First HUDM Reference

This repository is organized around SWM HDF5 data, a learned hierarchical image-latent dynamics model, and HUDM-owned CEM planning.

## Supported V1 Path

V1 supports one SWM environment/task per checkpoint. The environment must expose continuous `gymnasium.spaces.Box` actions and must be restorable for dataset start-goal evaluation.

Planner inputs are RGB pixels only. Optional dataset columns such as `state`, `proprio`, `qpos`, and `qvel` are kept for restore, diagnostics, and eval metadata; they are not planner objective inputs.

The planner objective is terminal latent L2:

```text
z0 = encoder(current_pixels)
z_goal = encoder(goal_pixels)
z_T = rollout_latent_dynamics(z0, candidate_actions, fidelity_level)
cost = || z_T[:K_level] - z_goal[:K_level] ||_2
```

All fidelity levels come from the learned HUDM model. Native multi-fidelity environments and particle simulation are not part of v1.

## Commands

Collect SWM HDF5:

```bash
python collect_swm.py configs/collect_swm.yaml
```

Train a world model from SWM HDF5:

```bash
python train_world_swm.py configs/world_swm.yaml
```

Evaluate HUDM planning from dataset start-goal pairs through SWM `World.evaluate_from_dataset(...)`:

```bash
python plan_swm.py configs/plan_swm.yaml
```

## Config Fields

Collection centers on:

- `env_id`
- `image_shape`
- `output_path`
- `episodes`
- `seed`
- `policy.import_path`, with SWM random policy fallback
- `env_kwargs`

Training centers on:

- `env_id`
- `data.path`
- `data.pixels_key`
- `data.action_key`
- `image_shape`
- `model.K`
- `model.D`
- `train.horizon`
- checkpoint directory and run name

Planning centers on:

- checkpoint run directory and optional epoch
- `data.path`
- `eval.episodes`
- `eval.goal_offset`
- `env.max_episode_steps`
- CEM horizon, population, elite fraction, iterations, action std, and fidelity schedule

The training config defines `K` and `D`. The checkpoint stores those values, and the planner reads fidelity levels from checkpoint metadata rather than accepting planner-side overrides.

## Dataset Format

`datasets/swm_hdf5.py` reads SWM HDF5 files with:

- `pixels`
- `action`
- `ep_len`
- `ep_offset`

Restore and metadata columns are read when present. Training samples contiguous windows from episode offsets and lengths. Planning samples dataset start-goal pairs from valid split episodes by default.

Image shape is configurable but square. Dataset/model metadata records the image shape used for training and evaluation compatibility.

## Restore Whitelist

V1 restore specs are implemented in `hudm/swm_restore.py`.

Supported envs:

- `swm/PushT-v1`
- registered DMControl wrappers using `qpos` and `qvel`
- `swm/TwoRoom-v1`
- `swm/Piecewise-v0`
- `swm/OGBCube-v0`
- `swm/OGBScene-v0`
- `swm/OGBPointMaze-v0`
- `swm/OGBMaze-v0` if present in the installed SWM registry

Restore may use `reset(options={...})` fields such as `state`, `goal_state`, or `target_state`, or SWM dataset-eval callable behavior such as `set_state(qpos, qvel)` when wrapped by the environment.

OGBench tasks use `hudm/swm_wrappers.py` to record `restore_state = concat(qpos, qvel)` during collection and pass it back through SWM eval callables during eval.

Custom restorable SWM environments can provide `restore.import_path`. The imported callable must return a restore spec with a `spec_id`, `required_columns`, and SWM eval `eval_callables` (`callables` is accepted as an alias); collection, training, and planning validate that spec against the recorded dataset columns.

Excluded for v1:

- non-restorable continuous environments
- Fetch without a full restore adapter
- SimplePointMaze without a restore adapter
- RocketLanding
- discrete envs
- ALE
- Craftax
- Classic Control discrete variants

## Checkpoint Metadata

`hudm/world_io.py` writes `world_metadata.json` next to checkpoint weights. The planner validates this metadata before constructing the policy.

Checkpoint metadata includes:

- `env_id`
- `image_shape`
- `action_dim`
- action low/high bounds
- `K`
- `D`
- model architecture settings
- dataset key mapping
- restore spec id used for compatibility

## SWM Policy Interface

`hudm/swm_policy.py` exposes `HUDMLatentCEMPolicy` with:

- `set_env(env)`
- `get_action(infos)`

The policy consumes SWM `infos` containing current pixels and goal pixels. `plan_swm.py` uses SWM `World.evaluate_from_dataset(...)`, which loads dataset start-goal chunks, restores each env, and places the dataset goal pixels in `infos["goal"]` before policy steps.

Official eval reporting comes from SWM results such as `success_rate` and `episode_successes`, plus HUDM planning diagnostics such as replans, fidelity levels, estimated bits, and plan time. The planner's learned latent L2 objective remains internal to CEM and is not reported as an eval metric.

## Tests

Focused unit tests:

```bash
conda run -n hudm python -m unittest tests.test_swm_framework
```

Compile/import smoke test:

```bash
conda run -n hudm python -m py_compile \
  collect_swm.py train_world_swm.py plan_swm.py \
  datasets/swm_hdf5.py hudm/swm_envs.py hudm/swm_policy.py \
  hudm/swm_restore.py hudm/swm_wrappers.py planning/swm_latent_cem.py
```

Integration smoke targets:

- collect a tiny SWM PushT HDF5 dataset
- train a one-epoch HUDM world model
- run `plan_swm.py` on dataset start-goal eval
- run restore smoke tests for PushT, TwoRoom/Piecewise, DMControl, and OGBench restore recorder
- verify unsupported discrete/non-restorable envs fail validation clearly
