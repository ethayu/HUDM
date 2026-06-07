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

## Submitted Command

Submitted as Slurm job `6384263`.

```bash
DENSE_HIGHK_ENV=tworoom \
sbatch --parsable --export=ALL,DENSE_HIGHK_ENV scripts/research_train_dense_highk_converge.sbatch
```

Submit output:

```text
sbatch: lua: cli_filter: defaulting --qos=mig for partition 'b200-mig90' (acct=dineshj-lab)
6384263
```

Current queue status at submission check:

```text
6384263|mwm_dense_highk_conv|PENDING|0:00|4-00:00:00|(ReqNodeNotAvail, Reserved for maintenance)
```

## Pending Debug

The pending reason is not a bad checkpoint/config/script request. `scontrol show job -dd 6384263` shows:

```text
JobState=PENDING Reason=ReqNodeNotAvail,_Reserved_for_maintenance
StartTime=2026-06-09T20:00:00
Partition=b200-mig90
QOS=mig
ReqNodeList=(null)
SchedNodeList=dgx029
ReqTRES=cpu=16,mem=128G,node=1,billing=571,gres/gpu=1,gres/gpu:90gb=1
```

`b200-mig90` has one node:

```text
PartitionName=b200-mig90
Nodes=dgx029
AllowQos=normal,mig,wharton,mig-max,maxwall
TRES=cpu=224,mem=1857524M,node=1,billing=8098,gres/gpu=16,gres/gpu:90gb=16
```

The blocking reservation is:

```text
ReservationName=scheduled-maintenance-2026-06-09
StartTime=2026-06-09T08:00:00
EndTime=2026-06-09T20:00:00
Nodes=...dgx[001-029]...
Flags=MAINT,SPEC_NODES,ALL_NODES
State=INACTIVE
```

Interpretation: the job requests a 4-day walltime and cannot fit safely before the June 9 maintenance reservation on the only `b200-mig90` node. Slurm therefore schedules it for `2026-06-09T20:00:00`, immediately after maintenance. The job's QOS/account/partition/GRES look valid.

I also tested `dgx-b200` as a possible alternative. A 4-day full-B200 test-only submission also starts at `2026-06-09T20:00:00`, so switching to full B200 does not improve the safe 4-day start time. Shorter walltimes can produce earlier hypothetical starts, but they risk killing the convergence run before canonical export.

Expected output checkpoint root:

```text
reports/research/dense_debug/checkpoints_mwm/mwm_dense_tworoom_highk_weighted_converge_<timestamp>
```

After the checkpoint is exported, evaluate the newest timestamped checkpoint with the dense K monotonicity matrix, prioritizing TwoRoom fixed K96/K144/K192 and K192 transition with 96D cost.

## Blockers

- Slurm job `6384263` is pending because the requested node class is currently reserved for maintenance.
- The efficient K-monotonicity array evaluator from the dense debug worktree has not been merged into `multienv-support`; it should be restored or recreated before the post-training evaluation.
