# MWM Review Guide

This repo is intentionally narrow while the MWM framework is under active
research development. Review against the current contract, not against legacy
compatibility behavior.

## Current Contract

- Runtime construction goes through `mwm.adapters.builder.build_mwm_from_stable_config`.
- `mwm.adapters.lewm.LeWMStableWMAdapter` is the only implemented base adapter.
- Le-WM MWM shares the latent producer (`encoder`, `projector`) and duplicates
  the non-encoder transition tail (`action_encoder`, `predictor`, `pred_proj`)
  per configured `K`.
- The training recipe is inherited from the base and applied at every level;
  MWM only aggregates the per-level losses matryoshka-style.
- `K=[D]` is an identity-parity check, not a special implementation path.
- Inference uses the standard MWM policy/planner path with the base-aligned
  action preprocessing, action block, frame skip, image preprocessing, and
  rollout semantics.
- Checkpoints are canonical directories containing only `config.json`,
  `weights.pt`, and `world_metadata.json`.
- Datasets are Lance-only.

## Removed Legacy Surface

The following should stay absent:

- Le-WM family-specific builder facades.
- Runtime upstream-object checkpoint targets.
- Runtime source-object delegation.
- Stable-WM reference-policy evaluator fallbacks.
- Placeholder adapters for bases that have not been implemented.
- HDF5 data paths.
- Old identity-mode or experiment-versioned scripts and docs.

## Validation Commands

Use the conda `mwm` environment:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m py_compile $(rg --files -g '*.py')
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_data.py --paper-parity
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/benchmark/paper_parity_pusht.yaml --static-only --roles upstream_lewm_converted retrained_lewm_identity
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/benchmark/scheduled_pusht.yaml --static-only --roles upstream_lewm_converted mwm_scheduled
```

### Local Desktop Workflow

Use local scripts when reviewing on a desktop or laptop without Slurm:

```bash
scripts/local/local_verify.sh
scripts/local/local_benchmark_smoke.sh
```

These scripts do not require `SLURM_JOB_ID` and default to
`${MWM_PYTHON:-python}`. They are smoke workflows only; do not treat CPU smoke
numbers as paper-scale benchmark evidence.

Before launching GPU work on PARCC/Betty, submit through Slurm. The identity
parity split jobs are:

```bash
sbatch scripts/slurm/slurm_mwm_train_pusht_identity.sbatch
sbatch scripts/slurm/slurm_mwm_train_tworoom_identity.sbatch
sbatch --dependency=afterok:<pusht_job>:<tworoom_job> scripts/slurm/slurm_mwm_identity_parity.sbatch
```

The scheduled-MWM comparison split jobs are:

```bash
sbatch scripts/slurm/slurm_mwm_train_pusht_scheduled.sbatch
sbatch scripts/slurm/slurm_mwm_train_tworoom_scheduled.sbatch
sbatch --dependency=afterok:<pusht_job>:<tworoom_job> scripts/slurm/slurm_mwm_scheduled_comparison.sbatch
```
