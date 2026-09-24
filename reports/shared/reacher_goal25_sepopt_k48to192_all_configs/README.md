# Reacher goal25 (plan10/execute2) -- sepopt K48-192 checkpoint: adaptive scheduling vs. fixed-K, shared vs. individually-trained

`goal_offset=25`, 100 episodes, horizon=2 (10-step plan blocks, execute 2
steps / replan). Same checkpoint family and report structure as the goal50
counterpart, `../reacher_goal50_plan100_execute20_sepopt_k48to192_adaptive/`.

`plot.png` plots **four** series, all **COMPLETE**:

| # | Series | Cells | Config | Checkpoint |
|---|---|---|---|---|
| 1 | Runs 02-26, K48-192: adaptive schedules (blue) | 418 | `release20260728_dense_reacher_all_fidelity_schedules.yaml` | shared sepopt, scheduler ranges K48-192 |
| 2 | Runs 27-31: individually-trained fixed-K (orange) | 110 | `..._goal25_individual_fixed_k_baselines.yaml` | 5 separate checkpoints, one per K |
| 3 | Runs 02-26, K96-192 floor: adaptive schedules (purple) | 418 | `..._goal25_k96to192slice_sepopt_all_fidelity_schedules.yaml` | shared sepopt, scheduler's "coarsest" floored at K=96 |
| 4 | Dense checkpoint, individual fixed levels (teal) | 154 | `..._goal25_dense_checkpoint_individual_levels.yaml` | shared sepopt, sliced to one fixed level at a time (no scheduling) |

All four use `checkpoints_mwm/mwm_paper10_reacher_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
(K=[48,72,96,120,144,168,192], epoch 9), **except** series #2, which uses 5
separately-trained single-fidelity checkpoints
(`checkpoints_mwm/mwm_paper10_reacher_k{192,168,144,120,96}_release20260728`).
Series #2 was freshly re-run on the same 22-combo sweep grid as series
#1/#3 (superseding an older 125-cell/25-combo run of the same checkpoints --
see "Older, separate-config reference data" below for a discrepancy between
the two worth flagging).

**Note on how series #1 was assembled** (kept from the earlier version of
this report): the sweep ran as two concurrent `--roles`-filtered halves on
separate GPUs; each half's finish wrote its own aggregate `summary.json` to
the same output_dir, so the second half to finish overwrote the first
half's aggregate (only the individual per-cell files were safe).
`data/sepopt_k48to192_adaptive_summary.json` was rebuilt from all 418
individual per-cell summaries, not from that clobbered top-level file.

## Contents

- `plot.png` / `generate_plot.py` -- the four-series plot.
- `data/sepopt_k48to192_adaptive_summary.json` (418/418, series #1) --
  pre-existing, reused as-is.
- `data/k96to192_slice_summary.json` (528/528, series #3 + a redundant
  re-run of series #2's cells, filtered out in the plot),
  `data/dense_checkpoint_individual_levels_summary.json` (154/154, series
  #4), and `data/individual_fixed_k_baselines_summary.json` (110/110, series
  #2), all built by `build_k96to192_and_dense_levels_summary.py`.

Rebuild with:

```bash
python build_k96to192_and_dense_levels_summary.py   # refresh series #2/#3/#4
python generate_plot.py                             # redraw plot.png
```

## Headline numbers

**Adaptive scheduling: K48-192 floor vs. K96-192 floor (series #1 vs #3) --
here the floor costs real success, not just budget.** Unlike the goal50
report (where both floors reached the same 92% ceiling and the K96-192 floor
only cost budget), on goal25 the K48-192 frontier reaches **100%** success
(first hit at 3.4M bits/episode) while the **K96-192 floor caps out at 91%**
and never reaches ceiling, no matter the CEM budget. Win-rate check: **0/418**
K96-192-floor cells beat the K48-192 frontier. On this task/horizon,
scheduling access to K<96 isn't just cheaper -- for the last few points of
success rate, it's necessary.

**Fixed-K, no scheduling: individually-trained (#2) vs. dense-sliced (#4) --
here the dense-sliced checkpoint is equal-or-better at every level, not just
at low K:**

| K   | Individually-trained (#2) | Dense-sliced (#4) |
|-----|---------------------------|--------------------|
| 192 | 63%-83%, mean 75.0%       | 67%-87%, mean 78.5% |
| 168 | 62%-83%, mean 72.5%       | 67%-87%, mean 78.0% |
| 144 | **18%-36%, mean 28.9%**   | **58%-86%, mean 78.0%** |
| 120 | **16%-26%, mean 20.8%**   | **65%-86%, mean 76.0%** |
| 96  | **18%-26%, mean 22.2%**   | **48%-74%, mean 64.6%** |
| 72  | n/a (no individually-trained K72 checkpoint) | 51%-65%, mean 59.3% |
| 48  | n/a (no individually-trained K48 checkpoint) | 37%-53%, mean 45.7% |

K144 and below collapse to near-floor success for the individually-trained
checkpoints (21-29% mean), same cliff shape as goal50 -- but on goal25 even
K192/K168 individually-trained (75.0%/72.5% mean) sit slightly *below* the
dense-sliced checkpoint at those same levels (78.5%/78.0%), rather than
edging it out. Dense-sliced degrades smoothly from 78.5% at K192 down to
45.7% at K48 with no cliff, and beats individually-trained at every level
tested.

**Win-rate vs. the K48-192 adaptive frontier (series #1):** K96-192-floor
0/418 (0%) = individually-trained fixed-K 0/110 (0%) < dense-sliced fixed-K
1/154 (0.6%, marginal). Adaptive scheduling on the shared checkpoint is
undefeated in every practical sense here -- goal25's frontier reaches a hard
100% ceiling that nothing else in this report touches.

## Bottom line

Reacher goal25 tells a sharper version of the goal50 story: (1) restricting
the scheduler to K96-192 doesn't just cost budget here, it costs the last
9 points of success rate (100% -> 91%) -- the frontier needs access to
K<96 to reach ceiling, not merely to reach it cheaply; (2) the dense-sliced
checkpoint beats individually-trained fixed-K at *every* level tested here
(not just low K as in goal50), and the individually-trained checkpoints
still collapse below K168 the same way they do on goal50. Both point the
same direction: joint multi-K training produces a more robust model than
training a dedicated single-K checkpoint, and scheduling on top of that
shared checkpoint beats running it at any single fixed level.

## Older, separate-config reference data (not plotted) -- and an open discrepancy

Two files sit in `data/` from earlier, independently-collected runs, not
drawn in the current `plot.png`:

- `data/fixed_k_baselines_summary.json` (125/125): an earlier run of the
  same 5 individually-trained checkpoints, on a wider 25-combo grid (no
  exclusions) instead of series #2's 22-combo grid. Reports K192 91%-95%
  (mean 93.4%) and K168 97%-99% (mean 98.0%) -- well above series #2's
  re-run of the same checkpoints (K192 75.0%, K168 72.5%). **Resolved**:
  this file's `manifest_sha256` does *not* match the `_n100` manifest used
  by every currently-plotted series -- it predates the goal25 config's
  switch to that manifest (the config's own comment notes the unsuffixed
  manifest "now has 250 pairs in this repo state" vs. the 100 used
  everywhere else). Different eval episodes entirely, not a fair
  comparison -- this is why it was replaced.
- `data/sepopt_k192_fixed_cem_grid_summary.json` (22/22): the shared sepopt
  checkpoint held fixed at K=192 (no scheduling), from a separate config
  `config_sepopt_k192_fixed_cem_grid.yaml`. Reports 93%-97% (mean 95.8%),
  well above series #4's K=192 row (67%-87%, mean 78.5%). **Still
  unresolved**: unlike the file above, this one's `manifest_sha256` *does*
  match the current `_n100` manifest exactly (verified by hash), so the
  manifest isn't the explanation here. It points at the checkpoint via a
  stale nested path (`checkpoints_dense_k48to192_sepopt_20260917_reacher/
  checkpoints_mwm/...`) that no longer exists on disk, so it's not possible
  to directly re-verify whether it was really reading the same checkpoint
  weights as the current path used everywhere else in this report. Worth
  investigating (a different checkpoint version at the old path, an
  eval-protocol change, or a bug) before citing either number as
  authoritative for "fixed K=192, sepopt checkpoint" on goal25.
