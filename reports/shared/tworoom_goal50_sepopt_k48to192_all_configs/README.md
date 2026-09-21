# TwoRoom goal50 -- sepopt K48-192 all-configs sweep (PARTIAL)

Same structure as `tworoom_goal25_sepopt_k48to192_all_configs`, for TwoRoom
goal_offset=50 (horizon=4, budget=100) instead of goal25. All four series
use episodes=100 (natively -- an earlier version of this data was trimmed
down from a 200-episode run; the adaptive sweep has since been relaunched at
a native episodes=100 using a purpose-built manifest
`configs/manifest/release20260728_tworoom_goal50_exact_seed42_n100.yaml`).

## Read this before using it for the paper

**The adaptive-schedule sweep is PARTIAL.** Only 93/418 cells were complete
when this was generated (19 adaptive schedules x 22 CEM combos = 418
target, ~22%). The Pareto frontier, and any "best result" number quoted
here, will very likely shift -- probably upward -- as the remaining cells
finish. Regenerate (`python generate_plot.py`) once
`data/sepopt_k48to192_adaptive_summary.json`'s `runs_completed` reaches 418,
or ask for a refresh.

## What's in it

- **New checkpoint**: `mwm_paper10_tworoom_k48_72_96_120_144_168_192_sepopt_20260917`
  (K=[48,72,96,120,144,168,192], epoch 9), same checkpoint as the goal25 folder.
- **Fixed K=192 / Fixed K<192**: individually-trained single-K checkpoints,
  untouched by the checkpoint swap, reused from the original goal50 sweep.
- **Fixed K=192 (sepopt checkpoint, no scheduling)**: the SAME sepopt
  checkpoint as the adaptive frontier, held fixed at K=192 (its own
  finest/native level) for the whole plan -- a within-checkpoint ablation
  ("does scheduling help *this* checkpoint?"), not reused, freshly run. An
  earlier version of this series compared against the *original* dense
  checkpoint fixed at K=96 instead -- removed, since that wasn't a fair
  same-model comparison.

## Contents

- `plot.png` -- the figure (title states the partial-completion fraction).
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `config_sepopt_k48to192.yaml` -- the benchmark config for the adaptive sweep.
- `config_sepopt_k192_fixed_cem_grid.yaml` -- the config for the Fixed K=192
  (sepopt) series.
- `data/sepopt_k48to192_adaptive_summary.json` -- 93 completed runs so far
  (of 418 target); has `runs_completed` / `runs_target` fields.
- `data/fixed_k_baselines_summary.json` -- 125 fixed-K baseline runs (K=192,
  168, 144, 120, 96; 25 CEM combos each), episodes=100.
- `data/sepopt_k192_fixed_cem_grid_summary.json` -- 22 fixed-K=192
  sepopt-checkpoint runs (complete).

## Current (partial) headline numbers

- Best adaptive-schedule point so far: **77.0%** at 49.9M bits/episode --
  likely to improve as the remaining ~78% of cells finish.
- Fixed K=192 (sepopt, no scheduling) ranges **48-73%** across its whole CEM
  grid (complete, 22/22) -- below the adaptive frontier everywhere so far.
- **Win-rate check** (does fixed-K=192-no-scheduling beat the adaptive
  frontier on the *same checkpoint* at matched-or-cheaper cost?): **0/22
  (0%)** -- adaptive scheduling is winning every matched cell so far. Same
  qualitative result as goal25 (22/22 adaptive wins there), though this one
  isn't final since the adaptive sweep is only 22% done.
