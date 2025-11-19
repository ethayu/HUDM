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
├─ datasets/              # Zarr dataset utilities
│   ├─ zarr_rollouts.py  # Zarr-backed image+action windows for world training
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

data:
  synthetic:
    enable: true
    zarr_path: "synthetic/pusht_synth.zarr"  # generated via scripts/generate_synth.py
    frac: 0.5
    val_source: mixed

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

## Configuration

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
