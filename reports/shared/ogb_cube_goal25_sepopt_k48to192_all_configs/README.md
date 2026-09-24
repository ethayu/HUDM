# OGB-Cube goal25 -- sepopt K48-192 all-configs sweep

Same structure as `tworoom_goal25_sepopt_k48to192_all_configs`, for OGB-Cube
(swm/OGBCube-v0) instead of TwoRoom. goal_offset=25, 100 episodes,
horizon=2. All series use episodes=100.

## Read this before using it for the paper

**The adaptive-schedule sweep is COMPLETE** (418/418 cells: 19 adaptive
schedules x 22 CEM combos). It ran as two concurrent `--roles`-filtered
halves on separate GPUs -- `data/sepopt_k48to192_adaptive_summary.json` is
built directly from individual per-cell `summary.json` sidecars rather than
any top-level aggregate (the aggregate gets overwritten by whichever half
finishes last; see the reacher folder's README for the full story on that
bug), then re-run once more after the sweep's last 9 cells landed.

**Two fixed-K/no-scheduling series are now COMPLETE, replacing the older
partial/mismatched-grid pieces:**
- **Individually-trained fixed-K, 110/110** (`data/individual_fixed_k_baselines_summary.json`)
  -- 5 separately-trained single-fidelity checkpoints (K=192/168/144/120/96),
  re-run on the same 22-combo grid as the adaptive sweep. Replaces the older
  `data/fixed_k_baselines_summary.json` (125 cells, a different 25-combo
  grid; kept for reference, no longer plotted in `plot.png`/`plot_flops.png`).
- **Dense checkpoint, individual fixed levels, 154/154**
  (`data/dense_checkpoint_individual_levels_summary.json`) -- the SAME
  shared sepopt checkpoint, held fixed (no scheduling) at each of its 7
  declared levels (K=48,72,96,120,144,168,192) in turn. Did not exist at
  all in the earlier version of this report; supersedes the old
  single-level `data/sepopt_k96_fixed_cem_grid_summary.json` (22 cells,
  K=96 only) and `data/sepopt_k192_fixed_cem_grid_summary.json` (never
  completed, still absent) -- both kept for reference, no longer plotted.
  Both new series already carry real audited `dynamics_flops_total` from
  the start (run after the FLOP-audit bug fix below), so no post-hoc
  reconstruction was needed for them.

Built by `build_dense_levels_and_refreshed_fixedk_summary.py`.

**A K=48-96 restricted adaptive-schedule probe is also present in `data/`,
but is NOT drawn in either plot** (kept out to keep both figures to the
three main series -- adaptive, individually-trained fixed-K, dense-sliced
fixed-level -- each shown as a dimmed all-cells scatter plus its own solid
Pareto frontier line). The probe is complete (12/12: 2 schedules x 6 CEM
combos, same sepopt checkpoint but scheduler restricted to levels 0-2
instead of 0-6) and directly answers "would restricting adaptive scheduling
to K=48-96 expand the Pareto frontier further, since those levels are
cheaper than reaching K=192?" -- using only OGB-Cube's own data (not
extrapolated from TwoRoom, whose individually-trained per-K checkpoints
have a different, idiosyncratic pattern). **Answer: yes** -- see the
frontier-extension table under headline numbers below (computed from
`data/sepopt_k48_96_probe_summary.json` directly, independent of what's
plotted).

## What's in it

- **New checkpoint**: `mwm_paper10_ogb_cube_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
  (K=[48,72,96,120,144,168,192], epoch 9), the OGB-Cube counterpart to the
  TwoRoom/Reacher sepopt checkpoints.
- **Individually-trained fixed-K, 110/110**: 5 separately-trained single-K
  checkpoints (K=192/168/144/120/96), re-run on the adaptive sweep's pinned
  `ogb_cube_goal25_exact_seed42_n100.json` manifest and 22-combo grid --
  same manifest SHA as the adaptive sweep and the dense-sliced series
  below, byte-identical episodes across all three (unlike the older
  `fixed_k_baselines_summary.json`, which used the since-drifted unsuffixed
  manifest -- see that file's note if you still need it).
- **Dense checkpoint, individual fixed levels, 154/154**: the SAME sepopt
  checkpoint as the adaptive frontier, held fixed (no scheduling) at each of
  its 7 declared levels in turn -- both a within-checkpoint ablation ("does
  scheduling help, at every level, not just K=96?") and, at the K=96 slice
  specifically, via comparison against
  `release20260728_ogb_cube_goal25_densek96_fixed_cem_grid` (the *original*
  paper10 checkpoint fixed at the same K=96), an isolated read on "does the
  sepopt retraining itself help, independent of scheduling or K range?" See
  headline numbers below for both comparisons.

## Contents

- `plot.png` -- the figure, bits-per-episode cost axis.
- `plot_flops.png` / `generate_plot_flops.py` -- same series, real audited
  dynamics-GFLOPs cost axis instead of bits (see "Real FLOPs vs. bits"
  below). The K=48-96 probe series is excluded from this plot.
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `build_dense_levels_and_refreshed_fixedk_summary.py` -- rebuilds the two
  now-complete fixed-K/dense-sliced series from their source `eval.json`
  directories.
- `config_sepopt_k48to192.yaml` -- the benchmark config for the adaptive sweep.
- `data/sepopt_k48to192_adaptive_summary.json` -- all 418 completed runs;
  has `runs_completed` / `runs_target` fields (both 418).
- `data/individual_fixed_k_baselines_summary.json` -- 110/110 fixed-K
  baseline runs (K=192, 168, 144, 120, 96; 22 CEM combos each, matching the
  adaptive grid). Plotted.
- `data/dense_checkpoint_individual_levels_summary.json` -- 154/154
  dense-sliced fixed-level runs (7 levels x 22 combos). Plotted.
- `data/fixed_k_baselines_summary.json` -- superseded, kept for reference:
  125 fixed-K baseline runs on the older, since-drifted unsuffixed manifest
  (25 CEM combos each). Not plotted.
- `data/sepopt_k96_fixed_cem_grid_summary.json` -- superseded, kept for
  reference: the 22 fixed-K=96 sepopt-checkpoint runs this report used
  before the full 7-level dense-sliced series existed. Not plotted.
- `config_sepopt_k48_96_probe.yaml` -- the config for the K=48-96
  restricted adaptive-schedule probe.
- `data/sepopt_k48_96_probe_summary.json` -- the 12 K=48-96 probe runs
  (complete).

## Real FLOPs vs. bits

`flop_accounting: dynamics_audit` was set correctly on every run here, but a
bug meant `dynamics_flops_total` was silently recorded as 0 for runs before
2026-09-2x (`torch.inference_mode()` tensors bypass the `FlopCounterMode`
dispatch hook the audit relies on -- fixed in `mwm/diagnostics/flops.py`).
For the plotted series (adaptive, individually-trained fixed-K, dense-sliced
fixed-level), `dynamics_flops_total` is **real audited data**: the adaptive
sweep and both new fixed-K/dense-sliced series (110/154 cells) all ran after
the fix. The two superseded reference files (`fixed_k_baselines_summary.json`,
`sepopt_k96_fixed_cem_grid_summary.json`, 125 + 22 cells, pre-fix) instead
carry FLOPs reconstructed post-hoc from each `eval.json`'s saved
per-CEM-iteration trace (shape-only calibration against the real
checkpoints, no re-execution of the benchmark/env/data pipeline needed --
see the Reacher shared folder's README for the full method writeup), with
**zero** shape-match mismatches. The K=48-96 probe (12 cells) was
intentionally left unreconstructed/unplotted in `plot_flops.png` (excluded
on request).

**Does the bits-vs-FLOPs choice change the story?** No, checked against the
old (pre-supersession) Fixed K=96 sepopt series specifically, since unlike
Reacher/TwoRoom this was already the tightest of the three envs and worth
checking carefully rather than assuming: **zero win/loss flips** between
bits and FLOPs axes across all 22 of those cells. Adaptive still wins 21/22
matched cells either way (1 cell -- pop20/iter5, the cheapest -- still goes
to Fixed K=96 under both metrics). The margin distribution barely moves:

| | bits-based | FLOPs-based |
|---|---|---|
| Fixed K=96 wins (of 22) | 1 | 1 |
| Mean margin (adaptive advantage) | -3.64pt | -4.00pt |
| Cells within noise (\|margin\|<=2pt) | 11/22 | 10/22 |

So the original, already-honest "adaptive wins narrowly, with about half
the cells statistically indistinguishable from noise" read on OGB-Cube
holds under real FLOPs too -- this is not a case where switching cost
metrics reveals a bigger gap (as it did for Reacher/TwoRoom's more
lopsided K=192 comparisons). OGB-Cube's Fixed K=96 baseline never strays
far from K=192 in cost terms (it's already at the cheap end of the K
range), so the bits-vs-FLOPs scaling difference that mattered for the
K=192 comparisons has much less room to bite here. The full 110/154-cell
new series (both bits- and FLOPs-native) tell the same qualitative story --
see win-rate numbers below.

## Headline numbers (final, all cells, 418/418 adaptive + 110/110 fixed-K + 154/154 dense-sliced)

- **OGB-Cube is a notably harder task than TwoRoom/Reacher for this
  checkpoint family** -- nothing in this plot approaches the 90-100%
  ceilings seen there. All series cluster in the 40-80% range.
- Best adaptive-schedule point: **79.0%** at 20.16M bits/episode.
- **Win-rate check** (does a fixed-K/no-scheduling cell beat the adaptive
  frontier at matched-or-cheaper cost? all cells, no `n_iter` filtering --
  matching what's actually plotted in `plot.png`/`plot_flops.png`; not
  merged with the K=48-96 probe, which is excluded from both plots):
  individually-trained fixed-K **0/110 (0%)**; dense-sliced fixed-level
  **6/154 (4%)**. Adaptive scheduling wins the large majority of cells
  either way -- same qualitative result as TwoRoom and Reacher, though the
  margin here (max ~79% either way) is much tighter in absolute terms than
  the 100%-ceiling story on Reacher.

**A note on n_iter=5 (the shallowest CEM budget in every grid):** this
report went through several rounds of trying to exclude `n_iter=5` --
first from dense-sliced only (its win-rate against the adaptive frontier
dropped from 6/154 to 1/119, since n_iter=5 was responsible for almost all
of its wins), then from all three series for consistency. That last step
backfired: `n_iter=5` was *also* propping up the adaptive frontier's own
cheap end (2.12M-3.84M bits/ep, 61-65% success were all `n_iter=5` cells),
so removing it weakened the frontier and let dense-sliced win even more
comparisons against it (up to 23/154) -- the opposite of the intended
effect. **Final state: no `n_iter` filtering anywhere** -- all cells from
every series are plotted, which is both the simplest option and avoids the
asymmetric-filtering questions above. If you want to revisit this, the
three variants tried (dense-only filtered, all-filtered, unfiltered) and
their win-rate numbers (1/119, 23/154, 6/154) are preserved in this file's
git history.

- **Individually-trained fixed-K, non-monotonic in K** -- success rises
  from K=96 to a peak at K=144, then falls back at K=168/192:

  | K   | success rate range | mean |
  |-----|--------------------|------|
  | 96  | 41%-49%            | 44.6% |
  | 120 | 50%-63%            | 56.7% |
  | 144 | 56%-75%            | **68.9%** |
  | 168 | 56%-68%            | 62.7% |
  | 192 | 56%-66%            | 61.5% |

- **Dense-sliced fixed-level, roughly flat with a low-K edge** -- peaks at
  K=96, mildly declines toward K=192, mirroring the same non-monotonic
  shape seen in the separate goal25 all-levels/horizon5/n=200 fixed-K sweep
  (`reports/research/ogb_cube_goal25_k48to192_sepopt_20260917_all_levels_horizon5_n200`):

  | K   | success rate range | mean |
  |-----|--------------------|------|
  | 48  | 55%-77%            | 71.9% |
  | 72  | 58%-78%            | 72.3% |
  | 96  | 62%-79%            | **73.8%** |
  | 120 | 54%-73%            | 66.5% |
  | 144 | 55%-76%            | 67.5% |
  | 168 | 56%-75%            | 67.0% |
  | 192 | 56%-72%            | 65.0% |

- **Old vs. new checkpoint at matched K=96, no scheduling** (a separate,
  cleaner ablation isolating retraining from scheduling, using the 22-cell
  K=96 slice of the dense-sliced series above): the sepopt checkpoint
  wins 16/22 matched CEM cells, the original paper10 checkpoint wins 4, and
  2 tie. Mean success 73.8% (new) vs 72.3% (old) -- a small but consistent
  improvement from the retraining itself, on top of whatever scheduling
  adds separately.
- Individually-trained fixed-K and dense-sliced fixed-level overlap
  substantially in the 45-74% range, with the dense-sliced checkpoint
  clearly ahead at low K (K=48/72/96 have no individually-trained
  counterpart cheap enough to compare directly) and roughly comparable at
  K=144-192 -- less of a one-sided win for the dense-sliced checkpoint than
  on Reacher, where it dominates at every level.
- **K=48-96 restricted adaptive schedules genuinely extend the cheap end
  of the frontier.** Merging the 12-cell probe into the full-range
  adaptive frontier (the full 418/418 sweep added two intermediate frontier
  points, 73%@5.30M and 78%@10.60M, without moving the endpoints):

  | Success tier | Full K48-192 frontier | + K48-96 probe |
  |---|---|---|
  | 63% | 2.69M bits/ep | **1.54M bits/ep** (43% cheaper) |
  | 65% | 3.84M bits/ep | 3.84M bits/ep (unchanged) |
  | 74% | 6.72M bits/ep | 6.72M bits/ep (unchanged) |
  | *(new)* 76% | -- | **7.68M bits/ep** (fills a gap) |
  | 79% | 20.16M bits/ep | **11.52M bits/ep** (43% cheaper) |

  The probe's 79%@11.52M point also dominates the old frontier's
  78%@13.44M point outright (higher success, lower cost). This is the
  opposite of what the (retracted) TwoRoom-based extrapolation suggested --
  on OGB-Cube specifically, capping fidelity at K=96 doesn't crater
  quality, so restricting the *adaptive* schedule to K=48-96 buys real
  savings at the cheap end without giving up much success. Only a 12-cell
  probe so far (not a full 418-cell sweep); worth a fuller sweep if this
  matters for the paper.
