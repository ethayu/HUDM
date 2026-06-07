# Dense MWM Performance Debug Report

Date: 2026-06-07
Branch: `codex/dense-mwm-performance-debug`
Base SHA: `98fb81042ad7fcfce4be963f3d5a5c97e1174d04`
Latest analysis SHA before this report edit: `4a7489a2e73fe25a22d5f155f1d5e06cee7dbde3`
Runtime: `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`

## Executive Summary

Dense MWM's benchmark underperformance is primarily a planner scheduler problem, not evidence that the canonical dense checkpoint is unusable. Fixed-finest dense planning is strong: PushT `96.0` and TwoRoom `84.0`, close to upstream `98.0` and `86.0`. The current dense benchmark scheduler starts CEM at the coarsest prefixes K=`6,12,48` for half of each 30-iteration solve; those low-K scorers are bad enough that they steer CEM into poor action distributions before the high-K scorers appear. Starting the dense scheduler at K=96 and ramping to K=192 recovers performance: PushT `94.0`, TwoRoom `100.0`.

The follow-up K sweep also shows that increasing K is not guaranteed to improve planning performance under the current terminal latent cost. PushT usually benefits from larger K, but TwoRoom gets worse as K increases from 96 to 192. The diagnostic that rolls out the K=192 transition head while scoring terminal cost on only the first 96 dimensions recovers TwoRoom strongly, so the non-monotonicity is mostly in the extra cost dimensions, not in the K=192 transition dynamics or the fixed CEM budget.

Longer equal-weight dense training did not fix this. The 80-epoch/all-level-plateau probe completed and exported checkpoints, but those checkpoints evaluate worse than the canonical 10-epoch dense checkpoint, especially on TwoRoom. The next training experiment should change matryoshka loss weighting and checkpoint selection for high-K levels rather than simply training the same equal-weight objective longer.

## Primary Results

| env | upstream | current dense `linear_cem` | scheduled bad | dense fixed finest | dense fixed K=96 | K=96-to-finest scheduler |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PushT | 98.0 | 60.0 | 8.0 | 96.0 | 74.0 | 94.0 |
| TwoRoom | 86.0 | 64.0 | 30.0 | 84.0 | 100.0 | 100.0 |

Fixed-level ladder:

| env | fixed coarsest K=6 | fixed K=96 | fixed finest K=192 |
| --- | ---: | ---: | ---: |
| PushT | 4.0 | 74.0 | 96.0 |
| TwoRoom | 34.0 | 100.0 | 84.0 |

Canonical dense K sweep, 3 seeds per condition:

| env | condition | K96 | K144 | K192 | K192 transition + 96D cost |
| --- | --- | ---: | ---: | ---: | ---: |
| PushT | offset 25, horizon 5 | 63.3 | 84.7 | 89.3 | 87.3 |
| PushT | all six offset/horizon cells | 22.6 | 32.7 | 32.1 | 32.8 |
| TwoRoom | offset 25, horizon 5 | 100.0 | 93.3 | 86.0 | 99.3 |
| TwoRoom | all six offset/horizon cells | 74.4 | 60.8 | 46.1 | 79.0 |

Longer equal-weight dense training, same K sweep:

| env | condition | K96 | K144 | K192 | K192 transition + 96D cost |
| --- | --- | ---: | ---: | ---: | ---: |
| PushT | offset 25, horizon 5 | 62.7 | 66.0 | 65.3 | 70.7 |
| PushT | all six offset/horizon cells | 23.9 | 26.6 | 24.1 | 30.0 |
| TwoRoom | offset 25, horizon 5 | 29.3 | 24.0 | 23.3 | 20.7 |
| TwoRoom | all six offset/horizon cells | 13.9 | 12.8 | 11.9 | 14.4 |

Identity MWM/Le-WM baseline on the same six-cell matrix:

| env | offset 25, horizon 5 | all six offset/horizon cells |
| --- | ---: | ---: |
| PushT | 90.0 | 36.4 |
| TwoRoom | 92.0 | 49.9 |

## Direct Answers

1. Dense fixed-finest performance:
   - PushT: `96.0`
   - TwoRoom: `84.0`

2. Scheduler behavior causing the benchmark drop:
   - The current dense `linear_cem` scheduler for a 30-iteration CEM solve uses levels `[0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5]`, i.e. K=`6,12,48,96,144,192`.
   - Per solve, CEM spends 15/30 iterations at K<=48 before reaching K>=96.
   - Fixed coarsest scores are very weak: PushT `4.0`, TwoRoom `34.0`.
   - Therefore early CEM selection is optimizing candidates under low-K scorers that do not preserve enough task-relevant terminal information. The later high-K scorer inherits a distorted candidate distribution.
   - A K=96-to-finest schedule fixes this without retraining: PushT `94.0`, TwoRoom `100.0`.

3. If fixed-finest is bad, is dense undertrained, stale, or configured wrong?
   - Fixed-finest is not bad for the canonical checkpoint.
   - Audit found expected dense levels `[6,12,48,96,144,192]`, `D=192`, `adapter_family=lewm`, `architecture_version=lewm_base_adapter_v1`, and `training_backend=stable_worldmodel_lewm`.
   - Canonical dense training logs completed normally: PushT 10 epochs / 139k steps, TwoRoom 10 epochs / 51k steps.
   - Checkpoint metadata records a dirty export at commit `136d0023451356a9ca07d0f7c3a08807ac407df2` with diff hash `9929b0a877dddd85ec302819fbdd5cb6fee526ecb3d30cc1c8c4b39e1a813229`, but fixed-finest recovery makes this unlikely to be the primary benchmark-drop cause.

4. Does dense need longer train time?
   - Not as "same equal-weight objective, longer wall-clock."
   - The equal-weight 80-epoch/all-level-plateau probe completed, exported checkpoints, and evaluated worse than the canonical dense checkpoint.
   - PushT easy K192 fell to `65.3` versus canonical fixed-finest `96.0`; TwoRoom easy K192 fell to `23.3` versus canonical fixed-finest `84.0`.
   - Final equal-long validation losses show PushT high-K continued improving (`l5=0.00347`), but TwoRoom high-K ended poor (`l4=0.34476`, `l5=0.77931`). This points toward checkpoint selection, overtraining/instability, data/order sensitivity, or high-K weighting, not simply insufficient training time.

5. Exact next experiment:
   - For the benchmark drop: run a multi-seed dense benchmark replacing coarsest-to-finest with K96-to-finest.
   - For K monotonicity: train the prepared high-K-weighted TwoRoom dense config, then rerun the same K monotonicity matrix on that checkpoint.
   - Exact high-K training command is listed below.

## Why K Is Not Monotonic Here

The user's monotonicity intuition is right at the architecture class level: the K+1 representation has more dimensions, and a larger head class can represent the smaller behavior. But the evaluated planner is not testing an oracle over the function class. It is testing the learned transition head plus a squared-error terminal cost in the chosen latent prefix.

The decisive ablation is `dense_fixed_k192_cost_k96`: roll out using the K=192 transition head, but compute the planner terminal cost over only the first 96 latent dimensions. It uses the same CEM budget as full K=192.

Mean diagnostic contrasts across the six offset/horizon cells:

| env | K192 - K96 | K192 cost96 - K192 | K192 cost96 - K96 |
| --- | ---: | ---: | ---: |
| PushT | +9.6 | +0.7 | +10.2 |
| TwoRoom | -28.3 | +32.9 | +4.6 |

For TwoRoom, the K=192 transition head is not the main failure. With a 96D cost it recovers from `46.1` to `79.0` averaged across conditions, beating K96's `74.4`. The full 192D cost is what misranks CEM candidates. The effect is strongest when K96 already contains enough task state and the extra 96 suffix dimensions add terminal-cost terms that are not planner-aligned.

The effect is teased out by harder offsets/horizons. TwoRoom K192 under K96 by `-14.0` at offset 25/horizon 5, but by `-40.7` at offset 100/horizon 9. On PushT, harder conditions lower absolute success for all K and reduce the K192 advantage, but do not show the same harmful suffix-cost signature.

Caveat: the corrected K sweep materialized per-run `eval` and `planner` overrides. `benchmark_mwm.py` does not apply per-run `env` overrides the same way, so the offset-100 probes used the intended larger goal offsets/budgets while retaining the base eval config's environment step cap.

## Experiments Run

Checkpoint/training audit:

- Script: `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python scripts/research_dense_debug_audit.py`
- Outputs: `checkpoint_audit.md`, `scheduler_diagnostics.md`, `training_diagnostics.md`, `audit_summary.json`
- Canonical dense train logs audited: `logs/mwm_train_pusht_dense_6196243.out`, `logs/mwm_train_tworoom_dense_6196244.out`

Fixed-level ladder:

- Command: `sbatch scripts/research_dense_fixed_ladder.sbatch`
- Job: `6200972`
- Output: `reports/research/dense_debug/fixed_ladder_results.md`
- Result: fixed finest recovers PushT `96.0`, TwoRoom `84.0`; fixed coarsest is bad.

Scheduler ablation:

- Command: `sbatch scripts/research_dense_scheduler_ablation.sbatch`
- Job: `6201098`
- Output: `reports/research/dense_debug/rollouts/dense_scheduler_ablation_*`
- Result: K96-to-finest recovers PushT `94.0`, TwoRoom `100.0`.

Canonical dense K monotonicity matrix:

- Superseded/cancelled jobs: `6203086`, `6203087` after identifying wrapper override issues.
- Corrected commands:
  - `DENSE_K_ENV=pusht sbatch scripts/research_dense_k_monotonicity.sbatch`
  - `DENSE_K_ENV=tworoom sbatch scripts/research_dense_k_monotonicity.sbatch`
- Corrected jobs: PushT `6206283`, TwoRoom `6206284`, both `COMPLETED`.
- Output: `k_monotonicity_results.md`, `k_monotonicity_summary.json`

Equal-weight longer training:

- Configs:
  - `configs/research/train_mwm_dense_pusht_equal_long.yaml`
  - `configs/research/train_mwm_dense_tworoom_equal_long.yaml`
- Commands:
  - `DENSE_LONG_ENV=pusht sbatch scripts/research_train_dense_equal_long.sbatch`
  - `DENSE_LONG_ENV=tworoom sbatch scripts/research_train_dense_equal_long.sbatch`
- Superseded/cancelled jobs: PushT `6234160`, TwoRoom `6234159`.
- Final jobs: PushT `6234245`, TwoRoom `6234246`, both `COMPLETED`.
- Checkpoints written under `reports/research/dense_debug/checkpoints_mwm/`; canonical checkpoints were not overwritten.
- Initial dependent eval job `6234464` failed immediately because the wrapper checked for `metadata.json` instead of canonical `world_metadata.json`; fixed and verified.

Equal-weight longer evaluation:

- Sequential eval job: `6381687`, started then cancelled after `00:32:02` because it was inefficient.
- Replacement array:
  - `DENSE_LONG_EVAL_OUTPUT_ROOT=reports/research/dense_debug/rollouts/equal_long/k_monotonicity sbatch scripts/research_dense_equal_long_eval_array.sbatch`
  - Job `6382857`, 36/36 tasks `COMPLETED`.
  - Summary job `6382869`, `COMPLETED`.
- Output: `reports/research/dense_debug/equal_long_eval/k_monotonicity_results.md`

Identity MWM matrix:

- Checkpoints:
  - `checkpoints_mwm/retrained_lewm_identity_pusht_upstream`
  - `checkpoints_mwm/retrained_lewm_identity_tworoom_upstream`
- Array command:
  - `IDENTITY_MWM_OUTPUT_ROOT=reports/research/dense_debug/rollouts/identity_mwm/matrix sbatch scripts/research_identity_mwm_eval_array.sbatch`
- Array job: `6383534`, 36/36 tasks `COMPLETED`.
- Summary command:
  - `IDENTITY_MWM_OUTPUT_ROOT=reports/research/dense_debug/rollouts/identity_mwm/matrix IDENTITY_MWM_SUMMARY_DIR=reports/research/dense_debug/identity_mwm_eval sbatch --dependency=afterok:6383534 scripts/research_identity_mwm_summary.sbatch`
- Summary job: `6383537`, `COMPLETED`.
- Output: `reports/research/dense_debug/identity_mwm_eval/identity_mwm_results.md`

## Slurm Compliance

PARCC docs were inspected before GPU submissions and recorded in:

- `reports/research/dense_debug/equal_long_training_slurm.md`
- `reports/research/dense_debug/k_monotonicity_slurm.md`
- `reports/research/dense_debug/identity_mwm_slurm.md`

Docs inspected:

- `https://parcc.upenn.edu/training/getting-started/looking-around/`
- `https://parcc.upenn.edu/training/getting-started/zero-to-mnist/`
- `https://parcc.upenn.edu/systems/betty/`
- `https://parcc.upenn.edu/about/rates/`

All GPU jobs were run through Slurm on `b200-mig90` with `--gres=gpu:90gb:1` or equivalent job scripts. No long GPU jobs were run on login nodes.

## Matryoshka Weights

All dense results above use equal matryoshka level weighting unless explicitly marked as a prepared future config. The longer training probe also kept equal weights; it changed only training horizon/checkpoint output path and added an all-level plateau gate. Therefore the current data does not show that equal weights converge to monotonic K behavior.

The prepared next training configs change weights but have not been run:

- `configs/research/train_mwm_dense_pusht_highk_weighted.yaml`
- `configs/research/train_mwm_dense_tworoom_highk_weighted.yaml`

They keep levels `[6,12,48,96,144,192]` and use `loss.level_weights: [0.25, 0.25, 0.5, 1.0, 2.0, 4.0]`.

## Recommended Next Experiments

Immediate scheduler validation:

```yaml
planner:
  scheduler:
    policy: linear_cem
    start_level: 3
    end_level: finest
    rollout_level:
      mode: fixed
      level: base
```

Run this multi-seed on the standard benchmark manifests for PushT and TwoRoom and compare against upstream, fixed finest, and current coarsest-to-finest dense.

High-K training probe:

```bash
sbatch --job-name=mwm_dense_tworoom_highk \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --partition=dgx-b200 \
  --gres=gpu:B200:1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=4-00:00:00 \
  --wrap='cd "$SLURM_SUBMIT_DIR" && export MWM_PYTHON=/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python && export PYTHONUNBUFFERED=1 && export TOKENIZERS_PARALLELISM=false && export MPLBACKEND=Agg && export ARTIFACT_ROOT=${MWM_ARTIFACT_ROOT} && [[ -e data ]] || ln -s "$ARTIFACT_ROOT/data" data && $MWM_PYTHON verify_mwm_data.py --paper-parity && $MWM_PYTHON train_mwm.py configs/research/train_mwm_dense_tworoom_highk_weighted.yaml'
```

Then rerun the K matrix on the new checkpoint:

```bash
DENSE_K_ENV=tworoom \
DENSE_K_EXTRA_ARGS='--tworoom-checkpoint <new_checkpoint_dir> --output-root reports/research/dense_debug/rollouts/k_monotonicity_tworoom_highk' \
sbatch --export=ALL,DENSE_K_ENV,DENSE_K_EXTRA_ARGS scripts/research_dense_k_monotonicity.sbatch
```

## Blockers and Limitations

- `git fetch origin multienv-support` failed earlier with SSH publickey permission denied; this worktree used the existing local `origin/multienv-support` at `98fb81042ad7fcfce4be963f3d5a5c97e1174d04`.
- Fresh worktree did not include all large canonical artifacts, so Slurm jobs used symlinks to `${MWM_ARTIFACT_ROOT}` for `data` and `checkpoints_mwm` while writing research outputs under `reports/research/dense_debug/`.
- Large rollout/checkpoint artifacts are present locally and intentionally not part of the compact report commit unless explicitly requested.
- Dense checkpoint metadata records a dirty export; fixed-finest recovery makes that a provenance caveat, not the main explanation.
