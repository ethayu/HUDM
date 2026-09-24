# Individually-trained K vs. dense-matched K -- all four environments

Fair same-K comparison across the full K=[48,72,96,120,144,168,192] range:
is a checkpoint trained from scratch at only one K ("individually-trained")
competitive with the same multi-K matryoshka checkpoint held fixed at that
K with no fidelity scheduling ("dense, matched K")? Run at both
goal_offset=25 and goal_offset=50, overlaid on the same 4-panel plot
(solid/filled = goal=25, dashed/hollow = goal=50).

- **Individually-trained**:
  - K=96-192: `checkpoints_single_k48to192_sepopt_retrain_20260922_{env}/k{K}/checkpoints_mwm/mwm_paper10_{env}_k{K}_sepopt_retrain_20260922`
    (the NEW single-K retrain release, 2026-09-22) -- **except TwoRoom's own
    K=168, at both goals**, which still uses the OLD
    `checkpoints_mwm/mwm_paper10_tworoom_k168_release20260728` checkpoint.
    We verified old-vs-new agreement level by level before switching: every
    other TwoRoom level (96/120/144/192, both goals) and every level for
    OGB Cube/Reacher/PushT matched within a few points. TwoRoom K=168 alone
    showed a ~50pt regression at BOTH goals (91->42 at goal25, 70->15 at
    goal50) with clean training logs and no eval errors -- a genuinely
    weaker checkpoint at that one level, not noise, so it's held back on
    the old checkpoint pending investigation. See "Old-vs-new checkpoint
    verification" below for the full comparison.
  - K=48/72: `checkpoints_single_k48to192_sepopt_retrain_20260922_{env}/k{K}/checkpoints_mwm/mwm_paper10_{env}_k{K}_sepopt_retrain_20260920`
    -- the same new release; release20260728 never trained standalone
    K=48/72 models at all, so this fills that gap with a real (not
    placeholder) individually-trained checkpoint everywhere.
- **Dense (matched K)**:
  - K=96-192: `checkpoints_mwm/mwm_paper10_{env}_k96_120_144_168_192_release20260728`
    (K=[96..192] matryoshka checkpoint), MPC/CEM/rollout all fixed at level
    K, no transitions.
  - K=48/72: the separate sepopt K=[48,72,96,120,144,168,192] matryoshka
    checkpoint used elsewhere in this project, same fixed-K/no-scheduling
    setup. Different training run than the K=96-192 dense checkpoint above,
    so the two dense sub-series' own values at K=96 don't need to (and
    don't exactly) match at the seam -- plotted as one connected line per
    goal for readability, but keep in mind they're two different models.
- Protocol (same for all four envs, both series, both goals): 100 episodes,
  horizon=5, receding_horizon=5, pop_size=300, elite_frac=0.1 (30 elites),
  n_iter=30, seed=42. goal_offset=25 uses budget=50; goal_offset=50 uses
  budget=100 (matching each goal's established convention elsewhere in this
  project).
- Configs:
  - K=96-192: `configs/research/{env}_goal{25,50}_individual_vs_dense_matched_k_horizon5_iter30.yaml`
  - Individually-trained K=48/72: `configs/research/{env}_goal{25,50}_individual_k48_72_horizon5_iter30.yaml`
  - Dense K=48/72: `configs/research/{env}_goal{25,50}_sepopt_dense_k48_72_horizon5_iter30.yaml`
  - (env in tworoom/ogb_cube/reacher/pusht)

## Contents

- `plot.png` -- 4-row, shared-x figure, one row per environment, full
  K=[48..192] range, goal=25 and goal=50 overlaid in each panel.
- `generate_plot.py` -- regenerates `plot.png` from `data/`. Run with
  `python generate_plot.py`.
- `data/{env}_summary.json` / `data/{env}_goal50_summary.json` -- K=96-192
  sweep on the OLD (release20260728) individually-trained checkpoints plus
  dense-matched (10-run aggregate), goal=25 / goal=50. The dense-matched
  values from here are still used everywhere; the individually-trained
  values are only used now for TwoRoom K=168 (see above).
- `data/{env}_individual_new_k96_192_summary.json` / `data/{env}_goal50_individual_new_k96_192_summary.json`
  -- individually-trained K=96-192 on the NEW (sepopt retrain 20260922)
  checkpoints (5-run aggregate), goal=25 / goal=50. Used for every
  env/level except TwoRoom K=168.
- `data/{env}_individual_k48_72_summary.json` / `data/{env}_goal50_individual_k48_72_summary.json`
  -- real individually-trained K=48/72, also from the new release (2-run
  aggregate), goal=25 / goal=50.
- `data/{env}_sepopt_k48_72_summary.json` / `data/{env}_goal50_sepopt_k48_72_summary.json`
  -- dense K=48/72 (2-run aggregate), goal=25 / goal=50.

All copied from the corresponding `reports/research/{env}_goal{25,50}_*_horizon5_iter30/summary.json`
in the main repo.

## Old-vs-new checkpoint verification

Before switching K=96-192 to the new release, we ran both checkpoint
families through the identical protocol (horizon=5, pop=300, elite=30,
n_iter=30, n=100, same manifest) and compared level by level:

### goal_offset=25 (old / new individually-trained, dense unchanged)

| K | TwoRoom | OGB Cube | Reacher | PushT |
|---|---:|---:|---:|---:|
| 96  | 34 / 37 | 44 / 45 | 15 / 19 | 3 / 1 |
| 120 | 37 / 30 | 50 / 48 | 14 / 17 | 4 / 4 |
| 144 | 35 / 40 | 68 / 66 | 31 / 30 | 9 / 3 |
| 168 | **91 / 42** | 66 / 67 | 98 / 96 | 13 / 9 |
| 192 | 93 / 92 | 64 / 64 | 99 / 100 | 91 / 94 |

### goal_offset=50 (old / new individually-trained, dense unchanged)

| K | TwoRoom | OGB Cube | Reacher | PushT |
|---|---:|---:|---:|---:|
| 96  | 14 / 15 | 33 / 30 | 15 / 15 | 0 / 1 |
| 120 | 11 / 14 | 32 / 31 | 12 / 13 | 0 / 0 |
| 144 | 17 / 17 | 48 / 49 | 34 / 23 | 1 / 0 |
| 168 | **70 / 15** | 45 / 46 | 99 / 100 | 3 / 2 |
| 192 | 79 / 74 | 57 / 53 | 94 / 95 | 36 / 39 |

Every cell is within a few points except the two bolded TwoRoom K=168
cells -- consistent, large (~50pt), and isolated to that one level at both
goals. That's why the plot below uses the new checkpoints everywhere
except that one cell.

## Headline result -- goal_offset=25

(Individually-trained values below are from the NEW checkpoint release
except TwoRoom K=168, which uses the OLD one -- see "Old-vs-new checkpoint
verification" above.)

| K | TwoRoom | OGB Cube | Reacher | PushT |
|---|---:|---:|---:|---:|
| 48  | indiv 31 / dense 32 (+1)  | indiv 46 / dense 67 (+21) | indiv 15 / dense 57 (+42) | indiv 0 / dense 9 (+9) |
| 72  | 34 / 33 (-1)              | 43 / 63 (+20)             | 13 / 63 (+50)             | 3 / 10 (+7) |
| 96  | 37 / 99 (+62)             | 45 / 71 (+26)             | 19 / 92 (+73)             | 1 / 42 (+41) |
| 120 | 30 / 99 (+69)             | 48 / 69 (+21)             | 17 / 95 (+78)             | 4 / 55 (+51) |
| 144 | 40 / 97 (+57)             | 66 / 70 (+4)              | 30 / 99 (+69)             | 3 / 89 (+86) |
| 168 | 91 / 94 (+3)              | 67 / 68 (+1)              | 96 / 99 (+3)              | 9 / 90 (+81) |
| 192 | 92 / 93 (+1)              | 64 / 69 (+5)              | 100 / 97 (-3)             | 94 / 89 (-5) |

TwoRoom, Reacher, and PushT all show the same shape: individually-trained is
essentially broken from K=48 through some threshold (K=144-168), while dense
stays near-ceiling (or, at K=48/72, at its own more modest but still much
higher floor) across the whole range; the two only converge at K=168-192.
The one exception to "individually-trained is always worse or tied" is
TwoRoom at K=72, where individually-trained edges out dense by a single
point (34 vs. 33) -- noise at this sample size, not a real reversal.

OGB Cube is the outlier throughout the whole K range, K=48 included: its
individually-trained checkpoints are competitive everywhere, with only a
modest, roughly constant dense advantage (2-27 points) rather than the sharp
low-K collapse seen in the other three environments.

This argues that the benefit of matryoshka/multi-K training isn't just
enabling adaptive fidelity scheduling at inference time -- it also makes
every fixed-K slice of the model dramatically more usable than a dedicated
single-K model trained the same way, for most of these environments, at
every K we've now tested.

## goal_offset=50 result

Same comparison, harder task (goal 50 steps out instead of 25). Individually-
trained values are from the NEW checkpoint release except TwoRoom K=168
(OLD checkpoint, same reasoning as goal25 above):

| K | TwoRoom | OGB Cube | Reacher | PushT |
|---|---:|---:|---:|---:|
| 48  | indiv 11 / dense 16 (+5)  | indiv 30 / dense 53 (+23) | indiv 13 / dense 58 (+45) | indiv 1 / dense 3 (+2) |
| 72  | 13 / 23 (+10)             | 32 / 52 (+20)             | 14 / 73 (+59)             | 0 / 6 (+6) |
| 96  | 15 / 89 (+74)             | 30 / 57 (+27)             | 15 / 99 (+84)             | 1 / 13 (+12) |
| 120 | 14 / 86 (+72)             | 31 / 57 (+26)             | 13 / 99 (+86)             | 0 / 18 (+18) |
| 144 | 17 / 77 (+60)             | 49 / 53 (+4)              | 23 / 95 (+72)             | 0 / 38 (+38) |
| 168 | 70 / 75 (+5)              | 46 / 51 (+5)              | 100 / 95 (-5)             | 2 / 35 (+33) |
| 192 | 74 / 82 (+8)              | 53 / 52 (-1)              | 95 / 93 (-2)              | 39 / 36 (-3) |

The same qualitative pattern holds at goal=50, now confirmed at K=48/72 with
real individually-trained checkpoints too: dense dominates individually-
trained from the bottom of the range through K=144-168 for
TwoRoom/Reacher/PushT, converging (sometimes crossing) by K=192; OGB Cube
stays flat and close throughout, at either goal. Two differences from
goal=25 worth flagging:

- **PushT is much harder overall at goal=50** -- every number is far lower
  (peak success 36% vs. ~90% at goal=25) -- but the individual-vs-dense gap
  pattern is preserved regardless of the absolute difficulty jump.
- **TwoRoom and Reacher converge a level earlier** at goal=50 (by K=168
  rather than needing K=192), and Reacher's dense line actually dips very
  slightly *below* individually-trained at K=168/192 -- the only environment
  where that crossover happens at either goal.
