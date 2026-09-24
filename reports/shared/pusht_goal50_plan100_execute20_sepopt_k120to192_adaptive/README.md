# PushT goal50 (plan125/execute25) -- sepopt K48-192 checkpoint, K120-192 range

`goal_offset=50`, 100 episodes, horizon=5 (25-step plan blocks, execute 25
steps / replan). Same checkpoint family and report structure as the reacher/
OGB-Cube goal50 reports, but **PushT has no full K48-192 adaptive series** --
that sweep (`release20260728_dense_pusht_goal50_plan50_execute20_all_fidelity_schedules.yaml`)
was cancelled before it ever started running, so this report only covers the
K120-192 range.

`plot.png` (bits-per-episode axis) and `plot_flops.png` (real audited
dynamics-GFLOPs axis, log scale) both plot the same **three** series, all
now **COMPLETE**, from two configs:

| # | Series | Cells plotted | Config |
|---|---|---|---|
| 1 | Runs 02-26, K120-192 floor: adaptive schedules (purple) | 475/475 | `..._goal50_plan50_execute20_k120to192slice_sepopt_all_fidelity_schedules.yaml` |
| 2 | Runs 27-30: individually-trained fixed-K, **K96 dropped** (orange) | 100/100 | same config as #1 |
| 3 | Dense checkpoint, individual levels, K>=120 only (teal) | 100/100 | `..._goal50_plan50_execute20_dense_checkpoint_k120to192_levels.yaml` |

Series #1/#2 share one config: the shared sepopt K48-192 checkpoint
(`checkpoints_mwm/mwm_paper10_pusht_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`)
with every "coarsest" schedule endpoint floored at K=120 (level index 3)
instead of K=48, **except** series #2, which uses 4 separately-trained
single-fidelity checkpoints
(`checkpoints_mwm/mwm_paper10_pusht_k{192,168,144,120}_release20260728`).
Series #3 is the same shared checkpoint sliced to just its K=120/144/168/192
levels (K48/72/96 excluded from this sweep on request).

**Every series on both plots is now restricted to K>=120**: series #2's
K=96 run (`31_single_k96_all_finest`, 25 cells) is filtered out in
`generate_plot.py`/`generate_plot_flops.py` (there are no individually-trained
K=48/K=72 checkpoints to begin with, so K=96 was the only fixed-K<120 point
left to drop). The filtered cells are still present, unfiltered, in
`data/singlek_summary.json` -- only the plot excludes them.

## Contents

- `plot.png` / `generate_plot.py` -- bits-per-episode cost axis.
- `plot_flops.png` / `generate_plot_flops.py` -- real audited dynamics-GFLOPs
  cost axis (log scale). No post-hoc FLOP reconstruction needed -- this
  sweep ran after the FLOP-audit bug fix, so `dynamics_flops_total` is real
  from the start.
- `data/adaptive_summary.json` (475/475, series #1),
  `data/singlek_summary.json` (125/125, series #2), split from the K120-192
  slice sweep output.
- `data/dense_checkpoint_individual_levels_summary.json` (100/100, series
  #3).

Rebuild with:

```bash
python build_summary.py           # refresh all three data/*.json
python generate_plot.py           # redraw plot.png
python generate_plot_flops.py     # redraw plot_flops.png
```

## Headline numbers

- **PushT is much harder than reacher or OGB-Cube on this checkpoint.** The
  K120-192 adaptive frontier tops out at just **48%** success, at
  ~502M bits/episode -- far below reacher's 92-100% ceiling and OGB-Cube's
  67%. Whether the true K48-192 frontier (not run here) would do
  meaningfully better is an open question this report can't answer.
- **Win-rate vs. the adaptive frontier** (K96-dropped individually-trained
  series): **2/100 (2%)** on both the bits axis and the real-FLOPs axis
  (`plot_flops.png`) -- adaptive scheduling still wins in practice but by a
  much narrower margin than on reacher. Dense-sliced fixed-level: 3/100 (3%)
  on both axes. Unlike OGB-Cube, switching cost axis doesn't meaningfully
  change the picture here, even though the fixed-K series visually looks
  closer to the frontier on the log-FLOPs plot.
- **Individually-trained fixed-K collapses immediately below K192** -- the
  sharpest cliff seen across any of these reports (K=96 row kept here for
  context even though it's dropped from the plots):

  | K   | success rate range | mean |
  |-----|--------------------|------|
  | 192 | 20%-37%            | 30.0% |
  | 168 | 1%-5%              | **3.2%** |
  | 144 | 0%-3%              | **1.4%** |
  | 120 | 0%-3%              | **0.7%** |
  | 96  | 0%-2%              | 0.6% (not plotted) |

  On PushT, an individually-trained checkpoint below K192 is essentially
  non-functional (<=3.2% mean success) -- there's no gradual degradation,
  just a cliff at the very top of the range.
- **Dense-sliced fixed-level (K>=120 only) shows a much gentler decline**:

  | K   | success rate range | mean |
  |-----|--------------------|------|
  | 120 | 5%-22%             | 15.8% |
  | 144 | 14%-36%            | 27.2% |
  | 168 | 18%-42%            | 30.2% |
  | 192 | 21%-44%            | 31.2% |

  Even at K120, the shared checkpoint still reaches 15.8% mean success --
  roughly **22x** the individually-trained K120 baseline (0.7%). This is
  the most extreme version of the "joint multi-K training beats a dedicated
  single-K model" finding across all the reports built this session.

## Bottom line

PushT is the hardest task tested on this checkpoint family (48% ceiling vs.
67-100% elsewhere), and it shows the sharpest version of two now-consistent
findings: (1) adaptive scheduling beats fixed-K in practice, though the
margin is narrower here than on reacher; (2) individually-trained fixed-K
checkpoints are dramatically more fragile than the same K sliced from a
jointly-trained checkpoint -- on PushT specifically, an individually-trained
checkpoint is only usable at its native K=192 and collapses to near-zero
success anywhere below it, while the shared checkpoint degrades gracefully
across the same range. Caveat: this report only covers K120-192 -- the full
K48-192 picture (would the adaptive frontier clear 48% with access to lower
K, the way K<96 helped reacher's ceiling) was never run.
