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

**The Fixed K=192 (sepopt checkpoint, no scheduling) series still doesn't
exist and isn't currently queued.** `generate_plot.py` skips it gracefully
if `data/sepopt_k192_fixed_cem_grid_summary.json` is absent -- run it and
regenerate for the full 5-series plot if this matters.

**The Fixed K=96 (sepopt checkpoint, no scheduling) series IS present** --
complete (22/22), added on request to directly compare against the
existing K=96 baseline from the *original* paper10 checkpoint.

**A K=48-96 restricted adaptive-schedule probe is also present** --
complete (12/12: 2 schedules x 6 CEM combos, same sepopt checkpoint but
scheduler restricted to levels 0-2 instead of 0-6). This directly answers
"would restricting adaptive scheduling to K=48-96 expand the Pareto
frontier further, since those levels are cheaper than reaching K=192?" --
using only OGB-Cube's own data (not extrapolated from TwoRoom, whose
individually-trained per-K checkpoints have a different, idiosyncratic
pattern). **Answer: yes.** See headline numbers below.

## What's in it

- **New checkpoint**: `mwm_paper10_ogb_cube_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
  (K=[48,72,96,120,144,168,192], epoch 9), the OGB-Cube counterpart to the
  TwoRoom/Reacher sepopt checkpoints.
- **Fixed K=192 / Fixed K<192**: individually-trained single-K checkpoints,
  untouched by the checkpoint swap, reused from the original
  `release20260728_dense_ogb_cube_all_fidelity_schedules` sweep.
  **Caveat**: this reused sweep recorded manifest path
  `ogb_cube_goal25_exact_seed42.json` (the unsuffixed manifest, which had
  100 pairs at the time it ran), while the adaptive sweep uses the pinned
  `ogb_cube_goal25_exact_seed42_n100.json`. Same seed=42/goal_offset=25/
  episodes=100 protocol, but not byte-identical episodes -- the unsuffixed
  file has since drifted to 250 pairs in this repo, which is exactly why
  the adaptive sweep was pointed at the pinned `_n100` file instead.
- **Fixed K=96 (sepopt checkpoint, no scheduling)**: the SAME sepopt
  checkpoint as the adaptive frontier, held fixed at K=96 (level 2, NOT its
  finest level) for the whole plan -- both a within-checkpoint ablation
  ("does scheduling help?") and, via comparison against
  `release20260728_ogb_cube_goal25_densek96_fixed_cem_grid` (the *original*
  paper10 checkpoint fixed at the same K=96), an isolated read on "does the
  sepopt retraining itself help, independent of scheduling or K range?"
  See headline numbers below for both comparisons.
- **Fixed K=192 (sepopt checkpoint, no scheduling)**: pending (see above).

## Contents

- `plot.png` -- the figure.
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `config_sepopt_k48to192.yaml` -- the benchmark config for the adaptive sweep.
- `config_sepopt_k96_fixed_cem_grid.yaml` -- the config for the Fixed K=96
  (sepopt) series.
- `config_sepopt_k192_fixed_cem_grid.yaml` -- the config for the Fixed K=192
  (sepopt) series (not yet run).
- `data/sepopt_k48to192_adaptive_summary.json` -- all 418 completed runs;
  has `runs_completed` / `runs_target` fields (both 418).
- `data/fixed_k_baselines_summary.json` -- 125 fixed-K baseline runs (K=192,
  168, 144, 120, 96; 25 CEM combos each).
- `data/sepopt_k96_fixed_cem_grid_summary.json` -- the 22 fixed-K=96
  sepopt-checkpoint runs (complete).
- `config_sepopt_k48_96_probe.yaml` -- the config for the K=48-96
  restricted adaptive-schedule probe.
- `data/sepopt_k48_96_probe_summary.json` -- the 12 K=48-96 probe runs
  (complete).
- `plot_flops.png` / `generate_plot_flops.py` -- same data, real audited
  dynamics-FLOPs cost axis instead of bits (see "Real FLOPs vs. bits"
  below). The K=48-96 probe series is excluded from this plot.

## Real FLOPs vs. bits (post-hoc reconstruction)

`flop_accounting: dynamics_audit` was set correctly on every run here, but a
bug meant `dynamics_flops_total` was silently recorded as 0 everywhere in
this repo (`torch.inference_mode()` tensors bypass the `FlopCounterMode`
dispatch hook the audit relies on -- fixed in `mwm/diagnostics/flops.py`).
Real FLOPs were reconstructed post-hoc from each `eval.json`'s saved
per-CEM-iteration trace (shape-only calibration against the real
checkpoints, no re-execution of the benchmark/env/data pipeline needed --
see the Reacher shared folder's README for the full method writeup). All
565 cells here (418 adaptive, complete + 125 fixed-K baselines + 22 Fixed
K=96 sepopt) reconstructed with **zero** shape-match mismatches. The
K=48-96 probe (12 cells) was intentionally left unreconstructed/unplotted
(excluded on request).

**Does the story change?** No, and unlike Reacher/TwoRoom this was already
the tightest of the three envs, so it's worth checking carefully rather
than assuming: **zero win/loss flips** between bits and FLOPs axes across
all 22 Fixed K=96 (sepopt) cells. Adaptive still wins 21/22 matched cells
either way (1 cell -- pop20/iter5, the cheapest -- still goes to Fixed
K=96 under both metrics). The margin distribution barely moves:

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
K=192 comparisons has much less room to bite here.

## Headline numbers (final, 418/418)

- **OGB-Cube is a notably harder task than TwoRoom/Reacher for this
  checkpoint family** -- nothing in this plot approaches the 90-100%
  ceilings seen there. All series cluster in the 40-80% range.
- Best adaptive-schedule point across the full sweep: **79.0%** at
  20.16M bits/episode -- unchanged from the partial-sweep read; the last
  ~9 cells to finish didn't move the frontier at all.
- **Win-rate check** (does Fixed K=96-sepopt-no-scheduling beat the
  adaptive frontier on the *same checkpoint* at matched-or-cheaper cost?):
  **1/22 (5%)** -- adaptive scheduling wins 21/22 cells. Same qualitative
  result as TwoRoom and Reacher: scheduling is winning, though the margin
  here (mean -3.64pt, 11/22 cells within plausible n=100 noise) is much
  tighter than the 100%-ceiling story on Reacher.
- **Old vs. new checkpoint at matched K=96, no scheduling** (a separate,
  cleaner ablation isolating retraining from scheduling): the sepopt
  checkpoint wins **16/22 matched CEM cells**, the original paper10
  checkpoint wins 4, and 2 tie. Mean success 73.7% (new) vs 72.3% (old) --
  a small but consistent improvement from the retraining itself, on top of
  whatever scheduling adds separately.
- Fixed K=192 (baseline, individually-trained) and Fixed K<192 both range
  roughly 42-72%, broadly overlapping with the adaptive frontier and the
  new Fixed K=96 (sepopt) series rather than being clearly dominated --
  unlike Reacher's bimodal split, no obvious "good" vs "bad" cluster is
  visible in these individually-trained baselines.
- **K=48-96 restricted adaptive schedules genuinely extend the cheap end
  of the frontier.** Merging the 12-cell probe into the full-range
  adaptive frontier:

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
