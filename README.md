# HUDM — Hierarchical, Uncertainty‑aware Dynamics Models

Research code for dimension‑level dropout in model‑based RL, centred on **PushT** planar manipulation.

---

## 📂 Repository layout

```
HUDM/
├─ checkpoints/           # saved runs (each with config.yaml & .pt weights)
├─ configs/               # experiment YAMLs (train.yaml, sim.yaml)
├─ data/
│   └─ dataset.py         # thin wrapper around pusht_dset
├─ datasets/              # PushT dataset + slicing utilities
│   ├─ pusht_dset.py
│   └─ traj_dset.py
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

## 1  Training

```bash
# train a 5‑member transformer ensemble (config in configs/train.yaml)
python train.py configs/train.yaml
```

Log files & checkpoints are written to `checkpoints/<run‑name>_TIMESTAMP/`.

---

## 2  Simulation / Planning

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
| `data.*`     | dataset path, history length `num_hist`, normalisation stats                       |
| `train.*`    | batch size, learning rate, mask curriculum (`max_mask_prob`, `mask_warmup_epochs`) |
| `sim.*`      | env kwargs, planner settings, rendering interval                                   |

---

## 5  License & acknowledgements

HUDM is released under the MIT License.  PushT code and dataset are distributed under the original DINO‑WM terms.

We thank the authors of **PETS**, **MOPO**, **MBDP**, and **DINO‑WM** for open‑sourcing their work.
