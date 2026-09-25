# PushT goal25 -- sepopt K48-192 checkpoint, K120-192 range

`goal_offset=25`, budget=50, 100 episodes, horizon=5 (5 model steps x
action_block 5 = 25-action plan, all 25 executed before replanning), pinned
`configs/manifest/release20260728_pusht_goal25_exact_seed42_n100.yaml`
manifest. goal25 counterpart of
`../pusht_goal50_plan100_execute20_sepopt_k120to192_adaptive/`, same three
series and report structure. The full K48-192 goal25 figure is
`../pusht_goal25_sepopt_k48to192_all_configs/`.

`plot.png` (bits-per-episode axis) and `plot_flops.png` (real audited
dynamics-GFLOPs axis, log scale) both plot the same **three** series, all
**COMPLETE**, from two configs:

| # | Series | Cells plotted | Config |
|---|---|---|---|
| 1 | Runs 02-26, K120-192 floor: adaptive schedules (purple) | 475/475 | `release20260728_dense_pusht_goal25_k120to192slice_sepopt_all_fidelity_schedules.yaml` |
| 2 | Runs 27-30: individually-trained fixed-K, **sepopt retrains** (orange) | 100/100 | `release20260728_dense_pusht_goal25_singlek_sepopt_retrain20260922_k120to192.yaml` |
| 3 | Dense checkpoint, individual levels, K>=120 only (teal) | 100/100 | `release20260728_dense_pusht_goal25_dense_checkpoint_k120to192_levels.yaml` |

Series #1 uses the shared sepopt K48-192 checkpoint
(`checkpoints_mwm/mwm_paper10_pusht_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`)
with every "coarsest" schedule endpoint floored at K=120 (level index 3)
instead of K=48. Series #2 uses the separate-optimizer single-K retrains from
GitHub release `hudm-mwm-single-k48to192-sepopt-retrain-20260922` (canonical
epoch-9 checkpoints, extracted to
`checkpoints_single_k48to192_sepopt_retrain_20260922_pusht/k{K}/checkpoints_mwm/mwm_paper10_pusht_k{K}_sepopt_retrain_20260922`),
run with the same manifest/eval/planner/sweep settings as #1. It replaces the
old release20260728 single-K checkpoints
(`checkpoints_mwm/mwm_paper10_pusht_k{192,168,144,120,96}_release20260728`),
whose cells are kept, unplotted, in `data/singlek_summary.json`.
Series #3 is the shared checkpoint held fixed at each of its K=120/144/168/192
levels.

The bits-axis inset zooms on 0-60M bits / 30-95% success instead of the
goal50 report's 0-2.5x-frontier upper-right inset: here the single-d frontier
runs out to ~370M bits, so the goal50 inset would have hidden it.

## Contents

- `plot.png` / `generate_plot.py` -- bits-per-episode cost axis.
- `plot_flops.png` / `generate_plot_flops.py` -- audited dynamics-GFLOPs
  cost axis (log scale).
- `data/adaptive_summary.json` (475, series #1),
  `data/singlek_retrain_summary.json` (100, series #2),
  `data/dense_checkpoint_individual_levels_summary.json` (100, series #3).
- `data/singlek_summary.json` (125 incl. K96) -- old release20260728 single-K
  cells, not plotted, kept for comparison.

Rebuild with (from the repo root, `PYTHONPATH=.`):

```bash
python reports/shared/pusht_goal25_sepopt_k120to192_adaptive/build_summary.py
python reports/shared/pusht_goal25_sepopt_k120to192_adaptive/generate_plot.py
python reports/shared/pusht_goal25_sepopt_k120to192_adaptive/generate_plot_flops.py
```

## Headline numbers

- **Best adaptive cell: 87%**, reached first on the frontier by
  `25_mpc_coarse_to_fine_cem_base_to_fine_rollout_fine_to_base` (pop=100,
  n_iter=15) at 87.8M bits / 749 GFLOPs per episode; schedule 24 also hits
  87% but only at 167M+ bits/episode.
- **Individually-trained K=192 beats the adaptive frontier on goal25.**
  Cells above the adaptive frontier at matched-or-cheaper cost:

  | Series | bits axis | FLOPs axis |
  |---|---|---|
  | Single-d, sepopt retrains K120-192 (plotted) | 19/100 | 18/100 |
  | ... of which K=192 only | 18/25 (72%) | 17/25 (68%) |
  | Single-d, old release20260728 K120-192 (not plotted) | 19/100 | 17/100 |
  | Dense checkpoint fixed levels | 5/100 | 6/100 |

  Same pattern as the K48-192 goal25 report (17/22 K=192 cells beat that
  frontier), and the opposite of goal50. Swapping in the retrains doesn't
  change this. Within the shared checkpoint, scheduling still beats fixed
  levels (5-6/100 fixed-level wins).
- **Individually-trained fixed-K collapses below K192** (sepopt retrains; old
  release20260728 checkpoints in brackets; "paired" = mean per-cell change,
  same 25 CEM cells):

  | K   | success range | mean | old mean | paired change |
  |-----|---------------|------|----------|---------------|
  | 192 | 60%-93%       | 81.2% | 82.1% | -0.9pt |
  | 168 | 2%-11%        | 7.3%  | 10.6% | -3.3pt |
  | 144 | 1%-12%        | 5.3%  | 6.5%  | -1.2pt |
  | 120 | 0%-6%         | 2.7%  | 3.7%  | -1.0pt |

  The retrains are marginally weaker than the old checkpoints at every K
  (best cell 93% vs 97%), but the picture is unchanged: K=192 is strong and
  everything below it is near-unusable.
- **Dense checkpoint fixed levels degrade gently:**

  | K   | success range | mean |
  |-----|---------------|------|
  | 120 | 33%-63%       | 48.7% |
  | 144 | 47%-84%       | 69.8% |
  | 168 | 51%-84%       | 70.2% |
  | 192 | 51%-85%       | 70.2% |

  At K=120 the shared checkpoint averages 48.7%, ~18x the
  individually-trained K=120 retrain (2.7%).
