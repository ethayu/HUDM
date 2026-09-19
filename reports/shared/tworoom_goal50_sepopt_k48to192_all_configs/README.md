# TwoRoom goal50 -- sepopt K48-192 all-configs sweep (PARTIAL)

Same structure as `tworoom_goal25_sepopt_k48to192_all_configs`, for TwoRoom
goal_offset=50 (horizon=4, budget=100) instead of goal25.

## Read this before using it for the paper

**1. The adaptive-schedule sweep is PARTIAL.** Only 131/418 cells were
complete when this was generated (19 adaptive schedules x 22 CEM combos =
418 target). The Pareto frontier, and any "best result" number quoted here,
will very likely shift -- probably upward -- as the remaining cells finish.
Regenerate (`python generate_plot.py`) once `data/sepopt_k48to192_adaptive_summary.json`'s
`runs_completed` reaches 418, or ask for a refresh.

**2. Episode count: trimmed to 100, not rerun.** The adaptive sweep was
actually run at `episodes=200` (the only TwoRoom-goal50 manifest with an
exact matching pair count is the 200-pair one). To match the three reused
reference series (Fixed K=192, Fixed K<192, Fixed K=96 dense; all
`episodes=100`), `data/sepopt_k48to192_adaptive_summary.json` was **trimmed**,
not recomputed: each row's `success_rate` is over just the first 100 of its
200 evaluated episodes (a deterministic subset, not a resample), and
`bits_used_total` is halved. Per-row provenance is in each row's
`_trim_note` field. All four series are `episodes=100` in this file.

## What's in it

- **New checkpoint**: `mwm_paper10_tworoom_k48_72_96_120_144_168_192_sepopt_20260917`
  (K=[48,72,96,120,144,168,192], epoch 9), same checkpoint as the goal25 folder.
- **Fixed K=192 / Fixed K<192**: individually-trained single-K checkpoints,
  untouched by the checkpoint swap, reused from the original goal50 sweep
  (episodes=100).
- **Fixed K=96 (dense checkpoint, no scheduling)**: the *original* paper10
  dense checkpoint (K=[96,120,144,168,192]) held fixed at K=96 for the whole
  plan, reused from `release20260728_tworoom_goal50_densek96_fixed_cem_grid_horizon4`
  (episodes=100).

## Contents

- `plot.png` -- the figure (title states the partial-completion fraction).
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `config_sepopt_k48to192.yaml` -- the benchmark config for the adaptive sweep.
- `config_dense_k96_fixed_cem_grid.yaml` -- the config for the Fixed K=96 (dense) series.
- `data/sepopt_k48to192_adaptive_summary.json` -- 131 completed runs so far
  (of 418 target), trimmed to episodes=100 (see caveat 2 above); has
  `runs_completed` / `runs_target` fields.
- `data/fixed_k_baselines_summary.json` -- 125 fixed-K baseline runs (K=192,
  168, 144, 120, 96; 25 CEM combos each), episodes=100.
- `data/dense_k96_fixed_cem_grid_summary.json` -- 25 fixed-K=96
  dense-checkpoint runs, episodes=100.

## Current (partial) headline numbers

All numbers below use the trimmed episodes=100 adaptive data (caveat 2).

- Best adaptive-schedule point so far: **77.0%** at 49.9M bits/episode --
  likely to improve as the remaining ~69% of cells finish.
- Fixed K=96 (dense, no scheduling) reaches **87.0%** at 276.5M bits/episode,
  and 86% as cheap as 138M bits/episode.
- **Win-rate check** (does K96-dense beat the adaptive frontier at
  matched-or-cheaper cost?): **25/25 (100%)** of the K96-dense cells beat the
  current partial adaptive frontier. Even stronger than goal25's 19/25.
  As with goal25, this isn't final -- the adaptive sweep is only 31% done,
  so this gap may close as more schedules complete -- but at 131/418 cells
  it isn't close, either.
