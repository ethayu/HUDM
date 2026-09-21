# Reacher goal50 (plan20/execute4) -- sepopt K48-192 adaptive vs. fixed K=192

Same sepopt K48-192 checkpoint as `reacher_goal25_sepopt_k48to192_all_configs`,
different benchmark config: `goal_offset=50`, 100 episodes, horizon=4
(20-step plan blocks, execute 4 steps / replan).

Both series are **COMPLETE**.

## What's in it

- **Checkpoint**: `checkpoints_mwm/mwm_paper10_reacher_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
  (K=[48,72,96,120,144,168,192], epoch 9) -- same checkpoint used by both
  series below, so this is a clean adaptive-vs-no-scheduling comparison
  rather than a checkpoint-quality comparison.
- **Adaptive schedules** (475/475 cells): 19 schedules x 25 CEM combos
  (pop_size x n_iter sweep), from
  `configs/research/release20260728_dense_reacher_goal50_plan50_execute20_all_fidelity_schedules.yaml`
  (`config_sepopt_k48to192.yaml` in this folder). Note the fixed-K baseline
  runs are commented out of *this* config -- they don't ship in the same
  file as the goal25 example.
- **Fixed K=192, no scheduling** (25/25 cells): the same checkpoint held at
  full fidelity (finest level) for the whole plan, swept over the same 25
  CEM combos, from a separate config,
  `configs/research/release20260728_dense_reacher_goal50_plan50_execute20_fixed_k192_sepopt.yaml`.
  This isolates the effect of *scheduling* from the effect of *checkpoint
  choice* -- unlike the goal25 example's "Fixed K=192" series, which used a
  different, individually-trained K192 checkpoint.

## Contents

- `plot.png` -- adaptive scatter + Pareto frontier (blue) with the fixed-K192
  baseline overlaid (black squares).
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `config_sepopt_k48to192.yaml` -- the adaptive-sweep config actually used.
- `data/sepopt_k48to192_adaptive_summary.json` -- 475/475 completed runs;
  built by walking every `eval.json` under
  `reports/research/release20260728_dense_reacher_goal50_plan100_execute20_all_fidelity_schedules/`
  through `mwm.benchmark.summary.eval_summary_row`.
- `data/fixed_k192_sepopt_summary.json` -- 25/25 completed runs from
  `reports/research/release20260728_dense_reacher_goal50_plan100_execute20_fixed_k192_sepopt/`,
  built the same way.

## Headline numbers

- The adaptive Pareto frontier reaches its max observed success, **92%**, at
  just **~50M bits/episode** (cell
  `24_mpc_coarse_to_fine_cem_base_to_fine_rollout_base__pop100__elite0p1__iter5`)
  -- no completed adaptive cell anywhere in the sweep (up to ~690M
  bits/episode) beats 92%.
- Fixed K=192 (no scheduling) success ranges **63%-83%** across the 25 CEM
  combos, at costs from **~12M to ~737M bits/episode**.
- **Win-rate check** (does fixed K=192 beat the adaptive frontier at
  matched-or-cheaper cost?): fixed K=192 wins **0/25** cells. Every single
  fixed-K192 point is dominated by some adaptive-schedule cell that gets
  equal-or-better success at equal-or-lower cost. Same qualitative result as
  the goal25 sweep, where the sepopt checkpoint's adaptive scheduling was
  also decisively ahead.

**Bottom line**: on this checkpoint, adaptive fidelity scheduling strictly
dominates running the same model at fixed full fidelity -- it's never worse
and typically much cheaper for the same success rate (the frontier's whole
useful range sits in the 0-50M bits/episode region; the fixed-K192 baseline
spends up to ~15x that for worse-or-equal success).
