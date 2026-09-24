# PushT goal50 -- sepopt K48-192 all-configs sweep

Same structure as `pusht_goal25_sepopt_k48to192_all_configs`, for PushT
goal_offset=50 (horizon=5, budget=100) instead of goal25, on the same
`mwm_paper10_pusht_k48_72_96_120_144_168_192_sepopt_actckpt_20260917`
checkpoint (K=[48,72,96,120,144,168,192], epoch 9).

## Read this before using it for the paper

**THIS IS STILL RUNNING, and earlier in its sweep than the goal25 folder.**
As of generation time, the adaptive sweep had **249/680 cells done (37%)** --
schedules 01-11 of 26 adaptive schedules have landed cells; schedules 12-26,
plus the 5 individually-trained fixed-K baselines (27-31), have not started
yet. Numbers below will shift substantially, likely upward, once the rest
lands -- schedule 23
(`mpc_coarse_to_fine_cem_coarse_to_base_rollout_base_to_coarse`), the
strongest performer across every other environment probed in this project,
hasn't run here yet either. Regenerate `plot.png` (`python generate_plot.py`)
once the sweep finishes.

## What's in it

- **Adaptive-schedule sweep (PARTIAL, 249/680)**: horizon=5, budget=100, the
  sepopt K48-192 checkpoint, on the pinned
  `configs/manifest/release20260728_pusht_goal50_exact_seed42_n100.yaml`
  manifest (100 episodes). horizon=5 was chosen after a K120-192-restricted
  probe (see below) showed a clear improvement over the same K-range probe
  at horizon=4 (41% vs 34% best success) -- that finding motivated
  relaunching the full K48-192 range at horizon=5 instead of the goal50
  default of horizon=4.
- **K=120-192 restricted adaptive schedules (probe, 12/12, complete)**: same
  checkpoint/manifest/horizon, scheduler restricted to levels 3-6 (K=120
  through K=192) instead of the full 0-6 (K=48-192) range. Merged into the
  plotted Pareto frontier (blue "+" markers) rather than shown separately,
  same treatment as the K96-192 probes in the TwoRoom/Reacher shared folders.
  See `data/sepopt_k120_192_probe_summary.json`.
- **No fixed-K baselines plotted yet.** Schedules 27-31 in the same sweep
  (individually-trained fixed-K checkpoints -- K=192/168/144/120/96 -- run
  fresh at horizon=5 on the current pinned n100 manifest, so they will NOT
  have the manifest-drift problem other shared folders' reused baselines have)
  haven't started. `data/fixed_k_baselines_summary.json` (125 cells) is NOT
  plotted and should NOT be used for comparison in the meantime: it's data
  reused from the OLD pre-checkpoint-swap sweep, which used (a) the old
  individually-trained-per-K checkpoints (same checkpoints 27-31 will use, but
  a stale run of them), (b) horizon=4 (not 5), and (c) the unsuffixed,
  non-pinned manifest -- a three-way mismatch against the current sepopt
  adaptive frontier, not just a manifest drift. Kept only for reference; will
  be superseded once schedules 27-31 land in the live sweep.

## Contents

- `plot.png` -- the figure (partial data, see caveat above).
- `generate_plot.py` -- self-contained, regenerates `plot.png` from `data/`.
- `data/sepopt_k48to192_adaptive_summary.json` -- adaptive sweep cells
  completed so far (249/680); has `runs_completed`/`runs_target` fields.
- `data/sepopt_k120_192_probe_summary.json` -- the 12 K=120-192 restricted
  adaptive-schedule probe runs (complete).
- `data/fixed_k_baselines_summary.json` -- 125 fixed-K baseline runs reused
  from the old pre-swap sweep. Not plotted (see manifest/horizon/checkpoint
  mismatch caveat above); kept for reference only.

## Headline numbers (partial, 249/680 -- will shift substantially)

- Best point on the current Pareto frontier: **49.0%** at 300.0M
  bits/episode.
- Cheapest frontier point: 1% at 5.4M bits/episode -- essentially a floor
  result, since most of the strong schedules and all fixed-K baselines
  haven't run yet.
- Too early to draw conclusions about K-range restriction or scheduling
  effectiveness here -- less than 40% of the sweep is in. Revisit once
  materially more of the sweep lands.
