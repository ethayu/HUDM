# Reacher goal25 -- sepopt K48-192 all-configs sweep (PARTIAL)

Same structure as `tworoom_goal25_sepopt_k48to192_all_configs`, for Reacher
(swm/ReacherDMControl-v0) instead of TwoRoom. goal_offset=25, 100 episodes,
horizon=2 -- all four series use episodes=100 (no trimming needed here).

## Read this before using it for the paper

**The adaptive-schedule sweep is PARTIAL.** Only 104/418 cells were complete
when this was generated (19 adaptive schedules x 22 CEM combos = 418
target). Regenerate (`python generate_plot.py`) once
`data/sepopt_k48to192_adaptive_summary.json`'s `runs_completed` reaches 418,
or ask for a refresh.

## What's in it

- **New checkpoint**: `mwm_paper10_reacher_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
  (K=[48,72,96,120,144,168,192], epoch 9), the Reacher counterpart to the
  TwoRoom sepopt checkpoint.
- **Fixed K=192 / Fixed K<192**: individually-trained single-K checkpoints,
  untouched by the checkpoint swap, reused from the original
  `release20260728_dense_reacher_all_fidelity_schedules` sweep.
- **Fixed K=96 (dense checkpoint, no scheduling)**: the *original* paper10
  dense checkpoint (K=[96,120,144,168,192]) held fixed at K=96 for the whole
  plan, reused from `release20260728_reacher_goal25_densek96_fixed_cem_grid`.

## Contents

- `plot.png` -- the figure (title states the partial-completion fraction).
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `config_sepopt_k48to192.yaml` -- the benchmark config for the adaptive sweep.
- `config_dense_k96_fixed_cem_grid.yaml` -- the config for the Fixed K=96 (dense) series.
- `data/sepopt_k48to192_adaptive_summary.json` -- 104 completed runs so far
  (of 418 target); has `runs_completed` / `runs_target` fields.
- `data/fixed_k_baselines_summary.json` -- 125 fixed-K baseline runs (K=192,
  168, 144, 120, 96; 25 CEM combos each).
- `data/dense_k96_fixed_cem_grid_summary.json` -- 25 fixed-K=96
  dense-checkpoint runs.

## Current (partial) headline numbers -- a very different story from TwoRoom

- **The adaptive frontier already hits 100% success at just 3.5M
  bits/episode** -- from only 104/418 cells so far. Since nothing can beat
  100%, this single point *is* the current Pareto frontier; more cells will
  only add points at higher cost or extend it cheaper, not raise the
  ceiling.
- **Win-rate check** (does K96-dense beat the adaptive frontier at
  matched-or-cheaper cost?): K96-dense wins essentially **0/25** meaningful
  cells (1/25 by the strict "matched-or-cheaper" rule, and that's a
  tie/near-tie against the 100% ceiling, not a real win) -- the opposite
  result from TwoRoom, where K96-dense beat the frontier in 19-25/25 cells.
  On Reacher, the sepopt checkpoint's adaptive scheduling is decisively
  ahead, not the other way around.
- **Fixed K<192 (individually-trained) is bimodal**: success rates range
  from 9% to 99% with a median of just 30%, clustering into a "good"
  group near the top and a "bad" group in the 10-40% band (visible as the
  split orange cloud in the plot). This means some of the individually-
  trained single-K Reacher checkpoints (K=96/120/144/168) are simply much
  weaker models than others, independent of any scheduling question --
  worth checking which specific K values land in which cluster before
  citing "Fixed K<192" as a single number.

**Bottom line so far**: unlike TwoRoom (where a trivial fixed-K=96 policy on
the dense checkpoint matched or beat the best adaptive schedule almost
everywhere), Reacher's sepopt K48-192 checkpoint with adaptive scheduling is
clearly winning, hitting ceiling performance at very low cost. Worth
confirming once the sweep finishes, but the gap is large enough at 25% completion
that it's unlikely to reverse.
