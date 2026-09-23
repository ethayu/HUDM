# OGB-Cube goal50 (plan20/execute4) -- sepopt K48-192 checkpoint: adaptive scheduling vs. fixed-K, shared vs. individually-trained

`goal_offset=50`, 100 episodes, horizon=4 (20-step plan blocks, execute 4
steps / replan). Same report structure as the reacher goal50 counterpart,
`../reacher_goal50_plan100_execute20_sepopt_k48to192_adaptive/`, for
OGB-Cube (`swm/OGBCube-v0`) instead.

`plot.png` plots **three** series, all now COMPLETE (no K96-192-floor
series yet -- that sweep hasn't been run for OGB-Cube):

| # | Series | Cells | Status | Config | Checkpoint |
|---|---|---|---|---|---|
| 1 | Runs 02-26: adaptive schedules (blue) | 475 | COMPLETE | `release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules.yaml` | shared sepopt, scheduler ranges K48-192 |
| 2 | Runs 27-31: individually-trained fixed-K (orange) | 125 | COMPLETE | same config as #1 | 5 separate checkpoints, one per K |
| 3 | Dense checkpoint, individual fixed levels (teal) | 175/175 | COMPLETE | `..._dense_checkpoint_individual_levels.yaml` | shared sepopt, sliced to one fixed level at a time (no scheduling) |

All three use `checkpoints_dense_k48to192_sepopt_20260917/checkpoints_mwm/
mwm_paper10_ogb_cube_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
(K=[48,72,96,120,144,168,192]), **except** series #2, which uses 5
separately-trained single-fidelity checkpoints
(`checkpoints_mwm/mwm_paper10_ogb_cube_k{192,168,144,120,96}_release20260728`).
Unlike reacher, OGB-Cube's goal50 all_fidelity_schedules config already
ships runs 27-31 uncommented in the same file as runs 02-26 -- no separate
standalone config was needed for series #2.

## Contents

- `plot.png` / `generate_plot.py` -- the three-series plot, bits-per-episode
  cost axis. Each series (adaptive, individually-trained fixed-K,
  dense-sliced fixed-level) now gets its own Pareto frontier line + solid
  frontier markers, with non-frontier cells dimmed -- same visual language
  across all three (matching `../ogb_cube_goal25_sepopt_k48to192_all_configs/`).
- `plot_flops.png` / `generate_plot_flops.py` -- same three series, real
  audited dynamics-GFLOPs cost axis (log scale) instead of bits. No
  post-hoc FLOP reconstruction needed here -- all three sweeps ran after
  the FLOP-audit bug fix, so `dynamics_flops_total` is real from the start.
- `data/adaptive_summary.json` (475/475, series #1) and
  `data/singlek_summary.json` (125/125, series #2), split from the same
  600-cell all_fidelity_schedules sweep output.
- `data/dense_checkpoint_individual_levels_summary.json` (175/175, series
  #3, COMPLETE).

Rebuild with:

```bash
python build_summary.py           # refresh all three data/*.json
python generate_plot.py           # redraw plot.png
python generate_plot_flops.py     # redraw plot_flops.png
```

## Headline numbers

- **Adaptive Pareto frontier** (series #1) reaches its max observed
  success, **67%**, at **~323M bits/episode** -- notably higher cost than
  reacher's frontier elbow, and OGB-Cube's ceiling (67%) is far below
  reacher's (92-100%). This is a harder task for this checkpoint family.
  The frontier is fairly flat above ~13M bits/episode (56M -> 322M bits
  only buys 59% -> 67%), so most of the useful range is still cheap.
- **Win-rate vs. the adaptive frontier** (cell success beats the
  frontier's achieved success at matched-or-cheaper cost, bits axis):
  individually-trained fixed-K **0/125 (0%)**; dense-sliced fixed-level
  **3/175 (2%, marginal)**. On the real-FLOPs axis (`plot_flops.png`) the
  dense-sliced win-rate is a bit higher, **9/175 (5%)** -- the dense-sliced
  frontier's cheap end (low-K slices) is more competitive in raw compute
  terms than in bits terms, though still a small minority of cells.
  Adaptive scheduling remains undefeated in practice on either axis, same
  as both reacher reports, but the margin over fixed-K is much narrower on
  this task (fixed-K success rates aren't collapsing the way they do on
  reacher -- see below).
- **Individually-trained fixed-K (series #2), no sharp cliff** (unlike
  reacher, where K144-and-below collapses hard):

  | K   | success rate range | mean |
  |-----|--------------------|------|
  | 192 | 37%-54%            | 48.3% |
  | 168 | 41%-51%            | 45.5% |
  | 144 | 38%-53%            | 46.2% |
  | 120 | 32%-44%            | 37.2% |
  | 96  | 28%-38%            | 31.8% |

  A gradual decline from K192 to K96 (48.3% -> 31.8%), not the sharp
  cliff reacher shows between K168 and K144.
- **Dense-sliced fixed-level (series #3, COMPLETE, 175/175)**:

  | K   | success rate range | mean | n |
  |-----|--------------------|------|---|
  | 48  | 39%-59%            | 49.5% | 25/25 |
  | 72  | 40%-64%            | 53.3% | 25/25 |
  | 96  | 37%-60%            | 51.4% | 25/25 |
  | 120 | 38%-58%            | 49.3% | 25/25 |
  | 144 | 41%-58%            | 49.9% | 25/25 |
  | 168 | 40%-56%            | 48.8% | 25/25 |
  | 192 | 41%-54%            | 48.0% | 25/25 |

  Roughly flat across all levels (~48-53% mean), with low-K (K48/K72/K96)
  if anything scoring *slightly higher* than high-K -- consistent with the
  reacher finding that the shared checkpoint's low-K slices don't collapse
  the way individually-trained low-K models do (K96 dense-sliced 51.4% vs.
  individually-trained 31.8%), but the gap is smaller here than on
  reacher, and there is no clear monotonic trend with K in either
  direction once every level has a full n=25 sample.

## Bottom line

OGB-Cube is a harder task for this checkpoint than reacher (67% ceiling vs.
92-100%), and adaptive scheduling still wins in practice (0/125 and 3/175
fixed-K cells beat the frontier), but the underlying fixed-K story is more
muted here: individually-trained checkpoints degrade gradually with K
rather than collapsing sharply, and the shared checkpoint's low-K slices
don't show as dramatic an advantage over individually-trained low-K models
as they do on reacher.
