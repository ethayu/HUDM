# Planning frontiers across compute budgets (paper Fig. `fig:planning_frontiers`)

2x4 panels (rows: goal offset 25/50; columns: TwoRoom, PushT, OGBench-Cube,
Reacher). Empirical Pareto frontiers of success rate vs. dynamics GFLOPs per
episode for Baseline (independently trained single-K checkpoints, K96-192),
MWM (fixed level; sepopt K48-192 checkpoint sliced to one level), and MWM
(scheduled; same checkpoint, all schedules). All inputs are existing
summaries in sibling `reports/shared/*` folders -- see `SOURCES` in
`generate_paper_plot.py`. Per panel, all series share the pinned n100
manifest and horizon, and only CEM cells common to all three are used.

- `reconstruct_flops.py` + `flop_calib.py` -- fill `dynamics_flops_total` for
  the three scheduled sweeps whose audit recorded 0 (TwoRoom goal25/goal50,
  Reacher goal25) via a shape-only FlopCounterMode calibration on the real
  checkpoints (needs the checkpoints locally). Validated against the natively
  audited Reacher goal50 scheduled sweep: max |rel err| 0.5% (475 runs).
  Writes `data/*_flops_reconstructed.json`.
- `generate_paper_plot.py` -- writes `planning_frontiers.pdf/.png` and
  `frontier_stats.json` (numbers quoted in the paper text).

Caveats: Baseline has no K48/K72 CEM grids; PushT fixed-level covers K>=120
only and PushT goal50 schedules are floored at K120; TwoRoom goal50
scheduled sweep is partial (93 cells, 5 schedules); horizons are H=2
(goal25) / H=4 (goal50), H=5 for PushT.

## Update: paper now uses the per-folder plot_flops.png images

The paper figure is the eight per-folder `plot_flops.png` images (PushT goal25's
was added via `../pusht_goal25_sepopt_k48to192_all_configs/generate_plot_flops.py`).
`stats_from_plot_flops.py` computes the quoted numbers from exactly those
plotted series (writes `plot_flops_stats.json`). `generate_paper_plot.py` /
`reconstruct_flops.py` are the earlier combined-figure variant, kept for reference.
