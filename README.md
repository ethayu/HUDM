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
├─ checkpoints/           # saved runs (each with config.yaml & .pt weights)
├─ configs/               # experiment YAMLs (train.yaml, sim.yaml)
├─ datasets/              # PushT dataset + slicing utilities
│   ├─ pusht_dset.py      # PushT loading + normalization + entrypoints
│   └─ traj_dset.py       # slicing + PadRolloutDataset
├─ models/                # dynamics networks + ensemble wrapper
│   ├─ masked_dynamics.py
│   └─ ensemble.py
├─ planning/              # CEM planner (variance‑aware)
│   └─ cem.py
├─ scripts/
│   └─ test_ensemble.py   # quick sanity check
├─ simulate.py            # rollout & planning driver
└─ train.py               # curriculum‑masked training loop
```

---

## 1  Training

```bash
# train a 5‑member transformer ensemble (config in configs/train.yaml)
python train.py configs/train.yaml
```

Log files & checkpoints are written to `checkpoints/<run‑name>_TIMESTAMP/`.

---

## 2  Simulation / Planning

```bash
# visualise teacher‑forced & free rollouts, or run CEM planning
python simulate.py configs/sim.yaml
```

Key toggles in `configs/sim.yaml`:

* `use_planner`:  false → pure rollout, true → CEM planning.
* `planner_kwargs.agg_mode` & `n_impute`:  cost aggregation over imputations.
* `var_threshold`: per‑dim dropout threshold on ensemble variance.

---

## 3  Video demos

| Model / Mode                 | Clip 1                                  | Clip 2                                  |
| ---------------------------- | --------------------------------------- | --------------------------------------- |
| Transformer — teacher‑forced | [YouTube](https://youtu.be/W39ael3hxlA) | [YouTube](https://youtu.be/mlkAgkUWCq4) |
| Transformer — free rollout   | [YouTube](https://youtu.be/txZVXWEfFX4) | [YouTube](https://youtu.be/UjgS8dc8hBY) |

---

## 4  Configuration fields

| YAML section | Purpose                                                                            |
| ------------ | ---------------------------------------------------------------------------------- |
| `model.*`    | network sizes, ensemble count                                                      |
| `data.*`     | dataset path, history length `num_hist`; normalization for PushT is fixed inside `datasets/pusht_dset.py` |
| `train.*`    | batch size, learning rate, mask curriculum (`max_mask_prob`, `mask_warmup_epochs`) |
| `sim.*`      | env kwargs, planner settings, rendering interval                                   |

Angle representation
- Controlled by `data.use_sincos` (true = use sin/cos embedding). Training and simulation share this.
- Conversion helpers live in `datasets/state_repr.py` with `angle_to_sincos` and `sincos_to_angle`.
- Note: `model.state_dim` must match the chosen representation (7 for angle, 8 for sin/cos when velocities are included).

---

## 5  License & acknowledgements

HUDM is released under the MIT License.  PushT code and dataset are distributed under the original DINO‑WM terms.

We thank the authors of **PETS**, **MOPO**, **MBDP**, and **DINO‑WM** for open‑sourcing their work.
To mix in environment-generated synthetic data for better coverage, set in `configs/train.yaml`:

data:
  synthetic:
    enable: true
    path: "synthetic/pusht_dataset"   # generated via scripts/generate_synth.py
    frac: 0.3                         # 30% of training drawn from synthetic
    total_train: null                 # cap total mixed train samples (optional)
    val_source: real                  # real | synthetic | mixed

Generate the synthetic dataset (OU action noise by default):

python scripts/generate_synth.py --out synthetic/pusht_dataset \
  --train_eps 200 --val_eps 50 --len_min 50 --len_max 160 --with_velocity \
  --policy ou --ou-theta 0.15 --ou-sigma 0.2
