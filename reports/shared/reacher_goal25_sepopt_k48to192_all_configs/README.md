# Reacher goal25 -- sepopt K48-192 all-configs sweep

Same structure as `tworoom_goal25_sepopt_k48to192_all_configs`, for Reacher
(swm/ReacherDMControl-v0) instead of TwoRoom. goal_offset=25, 100 episodes,
horizon=2 -- all four series use episodes=100.

## Status: COMPLETE

Both the adaptive-schedule sweep (418/418) and the Fixed K=192 (sepopt
checkpoint, no scheduling) grid (22/22) are done. This is the final data.

**Note on how the adaptive data was assembled**: the sweep ran as two
concurrent `--roles`-filtered halves on separate GPUs; each half's finish
wrote its own aggregate `summary.json` to the same output_dir, so the
second half to finish overwrote the first half's aggregate (only the
individual per-cell files were safe). `data/sepopt_k48to192_adaptive_summary.json`
here was rebuilt from all 418 individual per-cell summaries, not from that
clobbered top-level file.

## What's in it

- **New checkpoint**: `mwm_paper10_reacher_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
  (K=[48,72,96,120,144,168,192], epoch 9), the Reacher counterpart to the
  TwoRoom sepopt checkpoint.
- **Fixed K=192 / Fixed K<192**: individually-trained single-K checkpoints,
  untouched by the checkpoint swap, reused from the original
  `release20260728_dense_reacher_all_fidelity_schedules` sweep.
- **Fixed K=192 (sepopt checkpoint, no scheduling)**: the SAME sepopt
  checkpoint as the adaptive frontier, held fixed at K=192 (its own
  finest/native level) for the whole plan -- a within-checkpoint ablation,
  freshly run (not reused). An earlier version of this plot had a "Fixed
  K=96, original dense checkpoint" series here instead -- removed, since
  that wasn't a fair same-model comparison.

## Contents

- `plot.png` -- the final figure.
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `config_sepopt_k48to192.yaml` -- the benchmark config for the adaptive sweep.
- `config_sepopt_k192_fixed_cem_grid.yaml` -- the config for the Fixed K=192
  (sepopt) series.
- `data/sepopt_k48to192_adaptive_summary.json` -- all 418 runs, complete and
  rebuilt (see Status above).
- `data/fixed_k_baselines_summary.json` -- 125 fixed-K baseline runs (K=192,
  168, 144, 120, 96; 25 CEM combos each).
- `data/sepopt_k192_fixed_cem_grid_summary.json` -- the 22 fixed-K=192
  sepopt-checkpoint runs.

## Final headline numbers

- **Final adaptive Pareto frontier**: 85.0% @ 2.12M bits/episode, 98.0% @
  2.69M bits/episode, **100.0% @ 3.43M bits/episode** -- and it stays at
  100% from there (can't go higher).
- **Median success rate across all 418 adaptive cells: 99%.** The
  overwhelming majority of schedule/CEM-budget combinations land
  near-ceiling on this checkpoint.
- **Fixed K=192 (sepopt, no scheduling) ranges 93-97%** across its whole
  CEM grid -- clearly below the adaptive frontier's 100% ceiling.
  **Win-rate check** (does fixed-K=192-no-scheduling beat the adaptive
  frontier on the *same checkpoint* at matched-or-cheaper cost?): **0/22
  (0%)** -- adaptive scheduling wins every matched cell. Same qualitative
  result as both TwoRoom folders (22/22 and 0/22 there too).
- **Fixed K<192 (individually-trained) is bimodal**: success rates range
  from 9% to 99% with a median of just 30%, clustering into a "good"
  group near the top and a "bad" group in the 10-40% band (visible as the
  split orange cloud in the plot). Some of the individually-trained
  single-K Reacher checkpoints (K=96/120/144/168) are simply much weaker
  models than others, independent of any scheduling question -- worth
  checking which specific K values land in which cluster before citing
  "Fixed K<192" as a single number.

## Bottom line

Reacher tells a cleaner, more extreme version of the same story as
TwoRoom: adaptive scheduling on the sepopt K48-192 checkpoint reaches a
100% ceiling that fixed-K=192-no-scheduling (93-97%) cannot match, even
though both use the identical checkpoint. Unlike TwoRoom -- where the
*original* dense checkpoint's fixed-K=96 baseline was competitive with (and
often beat) the new checkpoint's adaptive frontier -- on Reacher the
comparison that matters is the within-checkpoint one, and there, scheduling
is unambiguously winning in every cell tested (0/22 and 0/22 across both
TwoRoom folders plus this one, 3-for-3).
