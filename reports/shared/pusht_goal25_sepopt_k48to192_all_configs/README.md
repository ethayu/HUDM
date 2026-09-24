# PushT goal25 -- sepopt K48-192 all-configs sweep

Same structure as the TwoRoom/Reacher/OGB-Cube `*_sepopt_k48to192_all_configs`
shared folders, for PushT goal_offset=25 (horizon=5, budget=50), on the
`mwm_paper10_pusht_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
checkpoint (K=[48,72,96,120,144,168,192], epoch 9).

## Status: COMPLETE (682/682 main sweep + 100/100 sepopt-fixed-K ablation)

The full-range adaptive-schedule sweep (26 schedules x 22 CEM combos = 572
cells) and the individually-trained fixed-K baselines (27_all_fixed_finest
= K=192, 28-31 = K=168/144/120/96; 5 levels x 22 combos = 110 cells) have
all finished. A separate same-checkpoint fixed-K ablation (sepopt checkpoint
held fixed at K=120/144/168/192, full CEM grid, 100 cells) was added after,
to isolate scheduling from checkpoint identity -- see headline numbers below.

## What's in it

- **Adaptive-schedule sweep (COMPLETE, 572/572)**: horizon=5, budget=50, the
  sepopt K48-192 checkpoint, on the pinned
  `configs/manifest/release20260728_pusht_goal25_exact_seed42_n100.yaml`
  manifest (100 episodes). horizon=5 was chosen after a K120-192-restricted
  probe (see below) showed a dramatic improvement over the same K-range probe
  at horizon=2 (84% vs 79% best success) -- that finding motivated relaunching
  the full K48-192 range at horizon=5 instead of the goal25 default of
  horizon=2.
- **K=120-192 restricted adaptive schedules (probe, 12/12, complete)**: same
  checkpoint/manifest/horizon, scheduler restricted to levels 3-6 (K=120
  through K=192) instead of the full 0-6 (K=48-192) range. It's what
  motivated relaunching the full range at horizon=5 (see below), but is
  **not plotted** here by request -- the frontier reflects only the
  full-range K=48-192 adaptive schedules. Removing it doesn't change any
  headline number: the frontier's best point already came from a
  full-range schedule, not the probe. See
  `data/sepopt_k120_192_probe_summary.json` if needed.
- **Fixed-K baselines (COMPLETE, 110/110)**: individually-trained checkpoints
  `checkpoints_mwm/mwm_paper10_pusht_k{K}_release20260728` (K=96/120/144/168/192),
  run fresh in this same sweep at horizon=5 on the pinned n100 manifest --
  not reused/manifest-drifted. K=192 (schedule 27, the checkpoint's own
  finest level) plotted separately (black squares) from K=168/144/120/96
  (schedules 28-31, orange dots), since K=192 turns out to behave very
  differently from the other four (see headline numbers).
  See `data/fixed_k192_baseline_summary.json` and
  `data/fixed_k_lt192_baseline_summary.json`.
- `data/fixed_k_baselines_summary.json` (125 cells, old pre-swap sweep) is
  superseded by the fresh baselines above and NOT plotted -- kept only for
  reference (old checkpoint naming, horizon=2, unsuffixed non-pinned
  manifest, a three-way mismatch against the current sepopt frontier).
- **Sepopt-checkpoint fixed-K ablation (COMPLETE, 100/100)**: the SAME sepopt
  checkpoint as the adaptive frontier, held fixed at one level for the whole
  plan (no fidelity transitions) -- the same-checkpoint "does scheduling help
  *this* checkpoint" ablation TwoRoom/Reacher already have, extended here to
  all four upper levels (K=120/144/168/192) since that's where the K120-192
  probe showed the adaptive frontier's action concentrated. Full CEM grid at
  each level (pop_size:[20,50,100,150,200] x n_iter:[5,10,15,20,30], 25
  combos/level). Plotted as aqua triangles. See
  `data/sepopt_k{120,144,168,192}_fixed_cem_grid_summary.json`.

## Contents

- `plot.png` -- the figure, final.
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `data/sepopt_k48to192_adaptive_summary.json` -- all 572 adaptive-schedule
  cells.
- `data/sepopt_k120_192_probe_summary.json` -- the 12 K=120-192 restricted
  adaptive-schedule probe runs.
- `data/fixed_k192_baseline_summary.json` -- the 22 individually-trained
  K=192 cells.
- `data/fixed_k_lt192_baseline_summary.json` -- the 88 individually-trained
  K=168/144/120/96 cells (22 combos x 4 levels).
- `data/sepopt_k120_fixed_cem_grid_summary.json`, `sepopt_k144_...`,
  `sepopt_k168_...`, `sepopt_k192_...` -- the 100 sepopt-checkpoint
  fixed-K ablation cells (25 combos x 4 levels).

## Headline numbers (final, 682/682)

- **Best point on the adaptive Pareto frontier: 84.0%** at 50.1M bits/episode
  (`MPC=fixed | CEM=coarse->base | Rollout=fine->base`, a full-range
  K=48-192 schedule -- the probe's best point doesn't beat this, so
  excluding the probe from the plot doesn't change the frontier at all).
- **Fixed K=192 (individually-trained) ranges 64-94%** across its 22 CEM
  combos -- comparable to or *above* the adaptive frontier's ceiling, unlike
  every other environment in this project.
- **Win-rate check** (does fixed K=192 beat the adaptive frontier at
  matched-or-cheaper cost?): **17/22 (77%) -- fixed K=192 WINS**, mean margin
  **+2.2pt**. This is the opposite result from TwoRoom, Reacher, and OGB
  Cube, where adaptive scheduling won essentially every matched cell against
  fixed K=192 (sepopt) or was tightly contested. For PushT specifically, the
  individually-trained K=192 checkpoint is strong enough on its own that
  scheduling doesn't clearly buy anything over just running it at max
  fidelity throughout.
- **Fixed K<192 (individually-trained K=168/144/120/96) is uniformly weak**:
  1-16% across all 88 cells, median 5% -- consistent with the same-checkpoint-family
  low-K collapse seen everywhere else in this project. K=192 is a clear
  outlier among the five individually-trained PushT checkpoints, not
  representative of "individually-trained" as a whole.
- **Sepopt-checkpoint fixed-K ablation (same checkpoint as the adaptive
  frontier, no scheduling) tops out at 85.0%** (K=168, pop=100, n_iter=30,
  121.0M bits/ep) -- close to but never beating the adaptive frontier's 84.0%
  ceiling at matched-or-cheaper cost. **Win-rate check: 3/100 (3%) -- adaptive
  scheduling wins 97/100 matched cells**, mean margin **-13.0pt in the
  sepopt-fixed-K series' favor being rare** (i.e. scheduling usually wins by
  ~13pt at the same cost). This matches the TwoRoom/Reacher/OGB-Cube pattern:
  *within this checkpoint*, scheduling clearly helps. K=120 is the weakest
  level here (66% best), consistent with sitting right at the edge of the
  K120-192 probe's frontier-extension effect.

## Bottom line

PushT breaks the pattern established by TwoRoom, Reacher, and OGB Cube --
but only when you compare against a *different*, individually-trained K=192
checkpoint. That old checkpoint (not the sepopt one) matches or beats the
sepopt checkpoint's own adaptive-scheduling frontier most of the time (17/22
matched cells, +2.2pt mean margin). Once you isolate checkpoint identity from
scheduling by comparing the sepopt checkpoint against *itself* (adaptive vs.
fixed-K, no scheduling, same checkpoint, same manifest), the usual pattern
reappears: adaptive scheduling wins 97/100 matched cells against the sepopt
checkpoint held fixed at K=120/144/168/192, by a comfortable ~13pt margin.
So two separate effects were entangled: (1) scheduling helps *any given*
checkpoint, PushT's sepopt one included -- same as every other environment;
and (2) the individually-trained PushT K=192 checkpoint specifically is
unusually strong (94% best vs. ≤16% for its own siblings at
K=168/144/120/96), strong enough to beat the sepopt checkpoint's entire
adaptive frontier on its own. Whatever makes that one checkpoint special is
a separate question from whether scheduling works, and doesn't appear to
generalize even within PushT's own individually-trained checkpoint family.
