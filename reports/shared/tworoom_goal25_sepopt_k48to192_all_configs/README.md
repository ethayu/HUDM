# TwoRoom goal25 -- sepopt K48-192 all-configs sweep

Reproduces the original `tworoom_release20260728_goal25_100ep_all_configs.png`
figure (adaptive fidelity scheduling vs. fixed-K, TwoRoom, goal_offset=25, 100
episodes, horizon=2), but with the "winning adaptive schedules" series computed
from a new checkpoint that extends the K range down to 48:

- **New checkpoint**: `mwm_paper10_tworoom_k48_72_96_120_144_168_192_sepopt_20260917`
  (K=[48,72,96,120,144,168,192], epoch 9 / fully trained), vs. the original
  paper10 dense checkpoint (K=[96,120,144,168,192]).
- The two individually-trained fixed-K baseline series (**Fixed K=192**,
  **Fixed K<192**) are unchanged -- those runs use individually-trained
  single-K checkpoints that the new checkpoint doesn't touch, so their
  results were reused from the original sweep rather than recomputed. See
  `data/fixed_k_baselines_summary.json`.
- A fourth series, **Fixed K=96 (dense checkpoint, no scheduling)**, reuses
  `release20260728_tworoom_goal25_densek96_fixed_cem_grid`: the *original*
  paper10 dense checkpoint (K=[96,120,144,168,192] -- the same checkpoint the
  original adaptive frontier used) held fixed at K=96 for the entire plan
  (no fidelity transitions), across the same CEM grid. See
  `data/dense_k96_fixed_cem_grid_summary.json` and
  `config_dense_k96_fixed_cem_grid.yaml`.

## Contents

- `plot.png` -- the figure.
- `generate_plot.py` -- self-contained script that reproduces `plot.png` from
  the files in `data/`. Run with `python generate_plot.py`.
- `config_sepopt_k48to192.yaml` -- the exact benchmark config used to run the
  adaptive-schedule sweep (`mwm.benchmark.matrix config_sepopt_k48to192.yaml`
  in the [HUDM-mwm-ethan](.) repo). Contains all 24 schedule definitions;
  runs `02`-`26` use the new sepopt checkpoint, runs `27`-`31` are the
  (reused, not rerun) fixed-K baselines.
- `config_dense_k96_fixed_cem_grid.yaml` -- the config for the fourth
  (dense-checkpoint, fixed K=96) series.
- `data/sepopt_k48to192_adaptive_summary.json` -- full results (418 runs: 19
  adaptive schedules x 22 CEM pop_size/n_iter combos) for the new checkpoint.
- `data/fixed_k_baselines_summary.json` -- the 125 fixed-K baseline runs
  (K=192, K=168, K=144, K=120, K=96; 25 CEM combos each) reused from the
  original sweep.
- `data/dense_k96_fixed_cem_grid_summary.json` -- the 25 fixed-K=96
  dense-checkpoint runs.

## Reproducing

```bash
python -m mwm.benchmark.matrix config_sepopt_k48to192.yaml --roles \
  release20260728_dense_schedule_02_rollout_fine_to_coarse \
  release20260728_dense_schedule_05_cem_fixed_rollout_base_to_coarse \
  release20260728_dense_schedule_06_cem_coarse_to_fine_rollout_base \
  release20260728_dense_schedule_07_cem_coarse_to_fine_rollout_fine_to_base \
  release20260728_dense_schedule_08_cem_coarse_to_fine_rollout_base_to_coarse \
  release20260728_dense_schedule_11_mpc_fixed_cem_base_rollout_base_to_coarse \
  release20260728_dense_schedule_12_mpc_fixed_cem_coarse_to_base_rollout_base \
  release20260728_dense_schedule_13_mpc_fixed_cem_coarse_to_base_rollout_fine_to_base \
  release20260728_dense_schedule_14_mpc_fixed_cem_coarse_to_base_rollout_base_to_coarse \
  release20260728_dense_schedule_17_mpc_fixed_cem_base_to_fine_rollout_base_to_coarse \
  release20260728_dense_schedule_18_mpc_coarse_to_fine_cem_base_rollout_base \
  release20260728_dense_schedule_19_mpc_coarse_to_fine_cem_base_rollout_fine_to_base \
  release20260728_dense_schedule_20_mpc_coarse_to_fine_cem_base_rollout_base_to_coarse \
  release20260728_dense_schedule_21_mpc_coarse_to_fine_cem_coarse_to_base_rollout_base \
  release20260728_dense_schedule_22_mpc_coarse_to_fine_cem_coarse_to_base_rollout_fine_to_base \
  release20260728_dense_schedule_23_mpc_coarse_to_fine_cem_coarse_to_base_rollout_base_to_coarse \
  release20260728_dense_schedule_24_mpc_coarse_to_fine_cem_base_to_fine_rollout_base \
  release20260728_dense_schedule_25_mpc_coarse_to_fine_cem_base_to_fine_rollout_fine_to_base \
  release20260728_dense_schedule_26_mpc_coarse_to_fine_cem_base_to_fine_rollout_base_to_coarse

python generate_plot.py
```

## Headline result

Best adaptive-schedule point: **98.0%** success
(`release20260728_dense_schedule_25_mpc_coarse_to_fine_cem_base_to_fine_rollout_fine_to_base`),
matching/slightly exceeding the original paper10-dense checkpoint's frontier
ceiling (92-98%), while the new checkpoint's frontier also reaches much
cheaper operating points (as low as ~2M bits/episode at 33% success) than the
K=96-floor checkpoint could access at all.

**But the "Fixed K=96, dense checkpoint, no scheduling" series is fully
competitive with it**: 97% at just 7.7M bits/episode, and 98% at 124.4M
bits/episode -- matching the best adaptive schedule's peak while sitting at
or above the blue frontier through most of the cheap-to-mid cost range (see
plot). At goal25/horizon=2 on TwoRoom, simply fixing K=96 on the *original*
dense checkpoint (no fidelity scheduling, no K=48-192 retraining) already
gets most of the way to what the new checkpoint's best adaptive schedule
achieves -- worth flagging before claiming the wider K range or the
scheduling itself is what's driving the win here.
