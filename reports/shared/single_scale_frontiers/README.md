# single_scale_frontiers

Self-contained snapshot of the data and code used for this figure. All eight
task/goal-offset settings are included. No checkpoints, network access, or
files from sibling report folders are required to regenerate it.

From the repository root:

```bash
MPLCONFIGDIR=/tmp/mwm-mpl .venv/bin/python reports/shared/single_scale_frontiers/generate.py
```

Or use Python 3.10+ with the versions in `requirements.txt`, and run
`python generate.py [OUTPUT_DIRECTORY]` from this folder.

- `data/<task>_goal<offset>/scheduled.json`: original scheduled evaluation summary.
- `data/<task>_goal<offset>/data/dense_checkpoint_individual_levels_summary.json`:
  original fixed-MWM evaluation summary.
- `data/sources.json`: local input mapping and minimum scheduling dimensions.
- `data/manifest.json`: original repository-relative paths and SHA-256 checksums.
- `data/reference_analysis.json`: original processed analysis for comparison.
- `analysis.py`, `frontier_utils.py`: matched-grid analysis and numerical helpers.
- `generate.py`: validates input checksums, recomputes results, and renders only this figure.
- `outputs/`: generated PDF, PNG, and `analysis.json`.

The code is a standalone snapshot of `planning_frontiers_all_envs`, adapted
only to use bundled paths and independent entry points. Raw input JSON files
are copied byte-for-byte; historical paths recorded inside them are provenance,
not runtime dependencies. Original reports and paper figures are unchanged.

Numerical comparisons use empirical best-success-within-budget frontiers.
The original matched-grid and metadata checks are retained. TwoRoom uses
reconstructed FLOP summaries; PushT uses K120–192 and Reacher K96–192.
OGBench excludes five-iteration CEM cells, matching the existing analysis.

Validation: regenerated successfully from the bundled inputs. All numerical
results match `data/reference_analysis.json` exactly; only source paths differ.
