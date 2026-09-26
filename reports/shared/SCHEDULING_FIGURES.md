# Scheduling figure reproduction packages

Each folder contains independent generation code, copies of all input evaluation
summaries, a provenance/checksum manifest, a README, and regenerated outputs.

| Figure | Package | Generated files |
|---|---|---|
| Single-scale frontiers (Figure 7) | [single_scale_frontiers](single_scale_frontiers/README.md) | `outputs/single_scale_frontiers.pdf`, `.png`, `analysis.json` |
| Factorial scheduling gains | [factorial_scheduling_gains](factorial_scheduling_gains/README.md) | `outputs/factorial_scheduling_gains.pdf`, `.png`, `analysis.json` |
| Per-schedule frontiers | [schedule_frontiers](schedule_frontiers/README.md) | `outputs/schedule_frontiers.pdf`, `.png`, `analysis.json` |

From the repository root, run `.venv/bin/python reports/shared/<package>/generate.py`.
Set `MPLCONFIGDIR=/tmp/mwm-mpl` if the default Matplotlib cache is not writable.
Each package can also be copied elsewhere and run independently with Python
3.10+ and its listed dependencies. No original sibling report directories or
model checkpoints are needed.

These are snapshots of the current analysis, including the Figure 7 percent
labels and interpolated per-schedule display curves. Raw inputs are copied
byte-for-byte. All three regenerated numerical analyses were verified against
the original processed results, ignoring only relocated source paths.

The existing `planning_frontiers_all_envs` workflow and paper assets are retained.
