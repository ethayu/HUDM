# HUDM — Hierarchical, Uncertainty‑aware Dynamics Models

Research code for dimension‑level dropout in model‑based RL, centered on **PushT** planar manipulation.
---

## 📝 Paper draft

A draft of our accompanying paper describing the methods and experiments in this repository is available here:

👉 **[Draft PDF](./draft.pdf)** — _Hierarchical, Uncertainty-aware Dynamics Models_
---

## 📂 Repository layout

```
HUDM/
├─ checkpoints_world/     # saved world model runs (each with world.yaml & .pt weights)
├─ configs/               # experiment YAMLs (world.yaml)
├─ docs/                  # configuration and usage references
├─ datasets/              # Zarr dataset utilities
│   ├─ zarr_episodes.py  # Zarr-backed full-episode dataset for world training
│   └─ mixed_zarr.py     # Optional mixing of real + synthetic Zarr datasets
├─ models/                # world model components
│   └─ world/             # CNN encoder, upconv decoders, tiny transformer dynamics
├─ scripts/
│   ├─ generate_synth.py  # Generate synthetic rollouts in Zarr format
│   ├─ visualize_rollouts.py  # Visualize rollouts from Zarr datasets
│   └─ visualize_world_decoder.py  # Visualize decoder reconstructions per level
└─ train_world.py         # Hierarchical world model training
```

---

## Training

Train the hierarchical world model:

```bash
python train_world.py configs/world.yaml
```

Log files & checkpoints are written to `checkpoints_world/<run-name>_TIMESTAMP/`.

---

## Synthetic Data Generation

Generate synthetic rollouts in Zarr format:

```yaml
data:
  synthetic:
    enable: true
    zarr_path: "synthetic/pusht_synth.zarr"  # generated via scripts/generate_synth.py
    frac: 0.5
    val_source: mixed
```

```bash
python scripts/generate_synth.py synthetic/pusht_synth.zarr \
  --train_eps 200 --val_eps 50 --len_min 50 --len_max 160 --with_velocity \
  --policy ou --ou-theta 0.15 --ou-sigma 0.2 --img-size 96
```

---

## Visualization

Visualize rollouts from Zarr datasets:

```bash
python scripts/visualize_rollouts.py \
  --config configs/world.yaml \
  --source synthetic \
  --count 5 \
  --fps 15
```

Visualize per-level decoder reconstructions:

```bash
python scripts/visualize_world_decoder.py configs/world.yaml --count 5 --out rollouts/decoder_grid.png
```

---

## Planning (MPC-CEM)

Run closed-loop planning:

```bash
python plan.py configs/plan.yaml
```

Key planner controls in `configs/plan.yaml`:

- `backend`: select planning backend (`wm` or `gt_env`).
- `world_model.*`: world-model config/checkpoint selection.
- `mpc.*`: horizon, total steps, and replan cadence.
- `cem.*`: CEM population and sampling controls.
- `objective.*`: cost weights/metric (latent-space for `wm`, image- or state-space for `gt_env`).
- `gt_env.*`: env-propagation controls (`objective_space`, rollout samples, rollout progress bar, env-side fidelity/noise).
- `init_goal.*`: random or dataset-trajectory init/goal sampling.
- `fidelity.mpc`, `fidelity.cem`, `fidelity.rollout`: unified schedule blocks with `mode` (`fixed`/`linear`, plus `uncertainty_downshift` for rollout).
- `fidelity`: stage-independent scheduling (MPC sets replan-stage base; CEM schedules within replan; rollout schedules within trajectory).
- `fidelity.rollout.uncertainty.criterion`: choose `mean` or `percentile` when using `uncertainty_downshift`.

---

## Configuration

See full YAML field reference and validation rules:

👉 **[YAML Config Reference](./docs/CONFIG_REFERENCE.md)**

Script command reference:

👉 **[Script CLI Reference](./docs/SCRIPT_CLI_REFERENCE.md)**

For world-model training, configure `configs/world.yaml`:

```yaml
data:
  zarr_path: "pusht/pusht_cchi_v7_replay.zarr"  # Real dataset
  synthetic:
    enable: true
    zarr_path: "synthetic/pusht_synth.zarr"    # Synthetic dataset
    frac: 0.5                                  # 50% synthetic in training mix
    val_source: mixed                          # Validation source: real | synthetic | mixed
```

---

## License & Acknowledgements

HUDM is released under the MIT License. PushT code and dataset are distributed under the original DINO‑WM terms.

We thank the authors of **PETS**, **MOPO**, **MBDP**, and **DINO‑WM** for open‑sourcing their work.
