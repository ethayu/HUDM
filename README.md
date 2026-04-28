# HUDM

HUDM is a learned-world-model planning framework for SWM environments.

The active path in this repository is SWM-first:

- collect SWM HDF5 data with `collect_swm.py`
- train an image-latent hierarchical dynamics model with `train_world_swm.py`
- run HUDM CEM planning as an SWM-compatible policy with `plan_swm.py`

Legacy environment-specific planning, native multi-fidelity environment backends, particle simulation, discrete action spaces, and state-space planner objectives are out of scope for v1.

## V1 Scope

HUDM v1 supports SWM environments that have:

- continuous `gymnasium.spaces.Box` actions
- RGB pixel observations
- a usable restore path for dataset start-goal evaluation

All fidelity levels come from the trained hierarchical dynamics model. The planner objective is terminal L2 distance in learned latent space, using the prefix associated with the selected fidelity level.

## Quick Start

Collect a dataset:

```bash
python collect_swm.py configs/collect_swm.yaml
```

Train a HUDM world model:

```bash
python train_world_swm.py configs/world_swm.yaml
```

Run planning/evaluation from dataset start-goal pairs:

```bash
python plan_swm.py configs/plan_swm.yaml
```

The example configs default to `swm/PushT-v1`. Change `env_id`, `data.path`, `image_shape`, restore settings, and CEM settings in the YAML files for other supported SWM tasks.

## Repository Layout

```text
HUDM/
├─ collect_swm.py            # SWM HDF5 collection entrypoint
├─ train_world_swm.py        # SWM HDF5 world-model training entrypoint
├─ plan_swm.py               # HUDM latent CEM planner as an SWM policy
├─ configs/
│  ├─ collect_swm.yaml
│  ├─ world_swm.yaml
│  └─ plan_swm.yaml
├─ datasets/
│  └─ swm_hdf5.py            # SWM HDF5 episode/window reader
├─ hudm/
│  ├─ swm_envs.py            # SWM World construction and action-space checks
│  ├─ swm_policy.py          # SWM-compatible HUDM planner policy
│  ├─ swm_restore.py         # restore whitelist and validation
│  ├─ swm_wrappers.py        # OGBench restore recorder wrapper
│  └─ world_io.py            # checkpoint and metadata IO
├─ planning/
│  ├─ cem_core.py            # reusable CEM optimizer
│  └─ swm_latent_cem.py      # latent terminal-L2 planner
└─ models/world/             # configurable image-latent world model
```

## Reference

See [docs/SWM_FIRST.md](docs/SWM_FIRST.md) for supported environments, dataset/checkpoint metadata, command config fields, restore behavior, and test commands.

## License

HUDM is released under the MIT License.
