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
- A fourth series, **Fixed K=192 (sepopt checkpoint, no scheduling)**, is a
  NEW eval (not reused): the SAME sepopt checkpoint as the adaptive frontier,
  held fixed at K=192 (its own finest/native level) for the entire plan (no
  fidelity transitions at all), across the same CEM grid. This is a true
  within-checkpoint ablation -- "does scheduling help *this* checkpoint, or
  would just running it at max fidelity the whole time be just as good?" --
  unlike an earlier version of this series which compared against the
  *original* dense checkpoint fixed at K=96 instead (removed, since that
  wasn't a fair same-model comparison). See
  `data/sepopt_k192_fixed_cem_grid_summary.json` and
  `config_sepopt_k192_fixed_cem_grid.yaml`.

## Contents

- `plot.png` -- the figure.
- `generate_plot.py` -- self-contained script that reproduces `plot.png` from
  the files in `data/`. Run with `python generate_plot.py`.
- `config_sepopt_k48to192.yaml` -- the exact benchmark config used to run the
  adaptive-schedule sweep (`mwm.benchmark.matrix config_sepopt_k48to192.yaml`
  in the [HUDM-mwm-ethan](.) repo). Contains all 24 schedule definitions;
  runs `02`-`26` use the new sepopt checkpoint, runs `27`-`31` are the
  (reused, not rerun) fixed-K baselines.
- `config_sepopt_k192_fixed_cem_grid.yaml` -- the config for the fourth
  (sepopt-checkpoint, fixed K=192) series.
- `data/sepopt_k48to192_adaptive_summary.json` -- full results (418 runs: 19
  adaptive schedules x 22 CEM pop_size/n_iter combos) for the new checkpoint.
- `data/fixed_k_baselines_summary.json` -- the 125 fixed-K baseline runs
  (K=192, K=168, K=144, K=120, K=96; 25 CEM combos each) reused from the
  original sweep.
- `data/sepopt_k192_fixed_cem_grid_summary.json` -- the 22 fixed-K=192
  sepopt-checkpoint runs.

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

**The within-checkpoint ablation is decisive, though**: "Fixed K=192, sepopt
checkpoint, no scheduling" only reaches 70-81% success across its whole CEM
grid, well below the same checkpoint's own adaptive-scheduling frontier
(89-98%). Adaptive scheduling beats fixed-K=192-no-scheduling in **22/22
(100%)** of matched cells. So while a *different*, individually-trained
K=192 checkpoint can still match the adaptive frontier at high cost (the
black "Fixed K=192 (baseline)" series does, see plot), simply running the
*same* sepopt checkpoint without scheduling leaves a lot of performance on
the table -- for this checkpoint specifically, the scheduling is doing real
work, not just riding on the wider K range or the retraining alone.
