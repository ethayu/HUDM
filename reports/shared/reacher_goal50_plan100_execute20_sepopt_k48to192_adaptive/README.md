# Reacher goal50 (plan20/execute4) -- sepopt K48-192 checkpoint: adaptive scheduling vs. fixed-K, shared vs. individually-trained

Same sepopt K48-192 checkpoint as `reacher_goal25_sepopt_k48to192_all_configs`,
different benchmark config: `goal_offset=50`, 100 episodes, horizon=4
(20-step plan blocks, execute 4 steps / replan).

`plot.png` now plots **four** series, all **COMPLETE**, covering every
reacher-goal50 sweep run on this checkpoint so far:

| # | Series | Cells | Config | Checkpoint |
|---|---|---|---|---|
| 1 | Runs 02-26, K48-192: adaptive schedules (blue) | 475 | `release20260728_dense_reacher_goal50_plan50_execute20_all_fidelity_schedules.yaml` | shared sepopt, scheduler ranges K48-192 |
| 2 | Runs 27-31: individually-trained fixed-K (orange) | 125 | same config as #1 | 5 separate checkpoints, one per K |
| 3 | Runs 02-26, K96-192 floor: adaptive schedules (purple) | 475 | `..._k96to192slice_sepopt_all_fidelity_schedules.yaml` | shared sepopt, scheduler's "coarsest" floored at K=96 |
| 4 | Dense checkpoint, individual fixed levels (teal) | 175 | `..._dense_checkpoint_individual_levels.yaml` | shared sepopt, sliced to one fixed level at a time (no scheduling) |

All four use the same checkpoint,
`checkpoints_mwm/mwm_paper10_reacher_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
(K=[48,72,96,120,144,168,192], epoch 9), **except** series #2, which uses 5
separately-trained single-fidelity checkpoints
(`checkpoints_mwm/mwm_paper10_reacher_k{192,168,144,120,96}_release20260728`,
each verified via `config.json` to declare `K=[<single value>]` only -- not
slices of the shared checkpoint). That's the key axis series #2 and #4 test:
same fixed K, shared/scheduled-capable checkpoint (#4) vs. a checkpoint
dedicated to that one K (#2).

All four were run end-to-end at
`/vast/projects/dineshj/lab/aurora/reports/research/` (see each config's
`output_dir` for the exact subdirectory), 40/40 or 14/14 shards each.

## Contents

- `plot.png` / `generate_plot.py` -- the four-series plot.
- `data/new_sweep_partial_adaptive_summary.json` (475/475, series #1) and
  `data/new_sweep_partial_singlek_summary.json` (125/125, series #2), built
  by `build_new_sweep_partial_summary.py` (despite the "partial" filename,
  both complete).
- `data/k96to192_slice_summary.json` (600/600, series #3 + a redundant
  re-run of series #2's cells, filtered out in the plot) and
  `data/dense_checkpoint_individual_levels_summary.json` (175/175, series
  #4), built by `build_k96to192_and_dense_levels_summary.py`.

Rebuild with:

```bash
python build_new_sweep_partial_summary.py           # refresh series #1/#2
python build_k96to192_and_dense_levels_summary.py   # refresh series #3/#4
python generate_plot.py                             # redraw plot.png
```

## Headline numbers

**Adaptive scheduling: K48-192 floor vs. K96-192 floor (series #1 vs #3).**
Both reach the same max success, **92%**, but the K96-192 floor pays **4.3x**
the cost to get there: 50M bits/episode (K48-192) vs. 215M bits/episode
(K96-192). Forcing the scheduler to never drop below K96 doesn't cost
success -- it costs budget, because the scheduler's cheapest good schedules
spend real time below K96 when the K48-192 floor allows it. Win-rate check:
only 33/475 K96-192-floor cells (7%) beat the K48-192 frontier at
matched-or-cheaper cost, and those are marginal (noise-level, not a
systematic win).

**Fixed-K, no scheduling: individually-trained (#2) vs. dense-sliced (#4)
-- the shared checkpoint's low-K slices are dramatically more robust than a
checkpoint trained from scratch at that K:**

| K   | Individually-trained (#2) | Dense-sliced (#4) |
|-----|---------------------------|--------------------|
| 192 | 57%-86%, mean 75.4%       | 63%-83%, mean 75.2% |
| 168 | 53%-82%, mean 71.0%       | 59%-81%, mean 71.6% |
| 144 | **22%-41%, mean 29.5%**   | **61%-86%, mean 73.8%** |
| 120 | **7%-20%, mean 13.5%**    | **60%-83%, mean 75.1%** |
| 96  | **8%-21%, mean 15.1%**    | **61%-75%, mean 67.0%** |

At K192/K168 the two are about equal (the individually-trained model has no
disadvantage there). But at K144 and below, the individually-trained models
collapse to near-floor success while the *same shared checkpoint*, sliced to
that exact level, keeps performing about as well as it does at K192. This
strongly suggests joint multi-K (matryoshka) training produces far more
robust low-K representations than training a dedicated low-K model from
scratch -- the shared encoder/latent space learned at K=192 doesn't degrade
when truncated to K=96, but a model trained to only ever see K=96 apparently
learns something much weaker.

**Win-rate vs. the K48-192 adaptive frontier (series #1):**
K96-192-floor 33/475 (7%, marginal) > dense-sliced fixed-K 3/175 (1.7%) >
individually-trained fixed-K 1/125 (0.8%). Adaptive scheduling on the shared
checkpoint remains strictly the best option; the gap between "beats it
occasionally" (K96-192 floor) and "essentially never beats it" (either
flavor of fixed-K) tracks how much of the scheduler's freedom each series
gives up.

**Bottom line**: (1) scheduling freedom below K96 is what buys the K48-192
frontier's ~4x cost advantage over a K96-192 floor, at matched success; (2)
fixed-K success or failure depends much more on *whether the checkpoint was
jointly trained across K* than on which single K a checkpoint targets --
individually-trained low-K checkpoints are much weaker than the same low K
sliced from a jointly-trained checkpoint.

## Older, separate-config reference data (not plotted)

Two more files sit in `data/` from earlier, independently-collected runs,
not drawn in the current `plot.png` (superseded by series #1/#2 above, which
come from one unified config and are the current source of truth):

- `data/sepopt_k48to192_adaptive_summary.json` (475/475): an earlier
  complete run of the same 19 adaptive schedules, same checkpoint --
  reproduces series #1 almost exactly.
- `data/fixed_k192_sepopt_summary.json` (25/25): the shared sepopt
  checkpoint held at fixed K=192 (no scheduling), from a separate config.
  Consistent with series #4's K=192 row (63%-83% here vs. 63%-83% there).
