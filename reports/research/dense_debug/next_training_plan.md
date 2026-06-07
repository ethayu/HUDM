# Dense High-K Training Next Step

Date: 2026-06-07

## Investigation Result

The next dense training experiment should not be another equal-weight long run. The safer intended recipe is:

- keep the dense levels `[6, 12, 48, 96, 144, 192]`;
- train with high-K-weighted matryoshka loss `[0.25, 0.25, 0.5, 1.0, 2.0, 4.0]`;
- monitor validation loss with `validate/pred_loss_epoch`;
- save top validation checkpoints and export the best Lightning checkpoint to canonical MWM format;
- stop only after all per-level validation losses plateau after warmup;
- keep `schedule.lr_max_epochs` explicit so LR horizon is not silently coupled to the stopping cap.

The prior equal-long probe kept only `last.ckpt`, so it could not recover the best TwoRoom validation point after late collapse. The trainer now supports explicit best-checkpoint export for research runs through:

```yaml
train:
  checkpoint_monitor: validate/pred_loss_epoch
  checkpoint_mode: min
  save_top_k: 2
  export_checkpoint: best
```

## Configs

- `configs/research/train_mwm_dense_tworoom_highk_weighted_converge.yaml`
- `configs/research/train_mwm_dense_pusht_highk_weighted_converge.yaml`

TwoRoom should run first because it showed the clearest equal-long failure and the strongest K192 cost-space pathology.

## Cluster Check

Before preparing the GPU command, I inspected PARCC Slurm guidance on 2026-06-07:

- PARCC getting-started docs say GPU and other nontrivial work should use Slurm (`srun`, `sbatch`, or `salloc`).
- PARCC examples use `dgx-b200` for GPU jobs, and PARCC rates list `betty.b200.mig90` as a 90 GB B200 slice.
- Local `sinfo` confirms `b200-mig90` with `gpu:90gb` and a 7 day partition limit.
- Local `parcc_free.py` showed `b200-mig90` with 15/16 GPUs free at inspection time.

## Exact Next Command

Not launched yet.

```bash
DENSE_HIGHK_ENV=tworoom \
sbatch --export=ALL,DENSE_HIGHK_ENV scripts/research_train_dense_highk_converge.sbatch
```

Expected output checkpoint root:

```text
reports/research/dense_debug/checkpoints_mwm/mwm_dense_tworoom_highk_weighted_converge_<timestamp>
```

After the checkpoint is exported, evaluate the newest timestamped checkpoint with the dense K monotonicity matrix, prioritizing TwoRoom fixed K96/K144/K192 and K192 transition with 96D cost.

## Blockers

- No GPU job has been submitted from this plan yet.
- The efficient K-monotonicity array evaluator from the dense debug worktree has not been merged into `multienv-support`; it should be restored or recreated before the post-training evaluation.
