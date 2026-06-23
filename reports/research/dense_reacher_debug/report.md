# Dense Reacher MWM Debug Report

## Objective

Debug why dense MWM underperforms on `swm/ReacherDMControl-v0` even though training/validation loss looked strong and the previous `[D]` identity parity artifact had reached 86%.

The primary fork was:

- If dense fixed-finest planning is strong, debug the scheduler.
- If dense fixed-finest planning is weak, debug convergence, checkpoint selection, restore/action preprocessing, or config mismatch.

## Workspace

- Worktree: `/vast/projects/dineshj/lab/ethanyu/code/HUDM/.worktrees/codex-dense-mwm-reacher-performance-debug`
- Branch: `codex/dense-mwm-reacher-performance-debug`
- Base: `origin/multienv-support` at `2f54b271d7eeb3c77cb0a03b7299d9f9adb8948b`
- Python: `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`

## Slurm Notes

Before submitting the GPU job I inspected the live Slurm state:

- `sinfo -p b200-mig90,dgx-b200`
- `scontrol show partition b200-mig90`
- `scontrol show partition dgx-b200`
- `squeue -u "$USER" -o '%.18i %.10P %.30j %.8T %.10M %.9l %.6D %R'`

I also checked the PARCC docs:

- PARCC Slurm overview: https://parcc.upenn.edu/training/slurm/
- PARCC container/GPU examples: https://parcc.upenn.edu/training/software/containers/

The submitted command was:

```bash
sbatch scripts/research_dense_reacher_debug.sbatch
```

The reusable committed script path is now `scripts/research/research_dense_reacher_debug.sbatch`; I moved it after the run to satisfy the repo's script grouping hygiene test.

Job result:

- Job ID: `6791697`
- Partition: `b200-mig90`
- Resources: `cpu=16,mem=128G,gres/gpu:90gb=1`
- State: `COMPLETED`
- Elapsed: `00:20:11`
- Exit code: `0:0`

The script ran:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/research/dense_reacher_planner_ablation.yaml --static-only
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python benchmark_mwm.py configs/research/dense_reacher_planner_ablation.yaml
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/research/dense_reacher_planner_ablation.yaml
```

Raw run artifacts are on disk under:

```text
reports/research/dense_reacher_debug/rollouts/planner_ablation/
```

## Configs Added

- `configs/research/dense_reacher_planner_ablation.yaml`
- `configs/research/dense_reacher_high_fidelity_schedule.yaml`
- `scripts/research/research_dense_reacher_debug.sbatch`

The ablation config compares upstream, identity, final dense linear, final dense fixed levels, a high-only dense schedule, and older dense checkpoints on the same Reacher manifest.

## Baseline Artifact Checks

All relevant checkpoint metadata used `env_id: swm/ReacherDMControl-v0`, restore spec `reacher_qpos_match_qpos_qvel`, and `action_spec.dim: 10`.

Dense final metadata:

- checkpoint: `checkpoints_mwm/mwm_dense_reacher`
- epoch: `9`
- levels: `[6, 12, 48, 96, 144, 192]`
- action spec: base dim `2`, block `5`, dim `10`

Earlier dense checkpoints:

- `checkpoints_mwm/mwm_dense_reacher_interim`: epoch `1`
- `checkpoints_mwm/mwm_dense_reacher_step31000_snapshot`: epoch `2`

This makes restore/action preprocessing or an obvious config mismatch unlikely as the primary explanation.

## Results

| Run | Success | Scheduler counts | Notes |
|---|---:|---|---|
| upstream_fixed_finest | 86.0 | `{0: 2310}` | Upstream control |
| identity_fixed_finest | 60.0 | `{0: 2340}` | Same-run identity control; caveat below |
| dense_linear_final | 32.0 | `{0: 273, 1: 546, 2: 546, 3: 546, 4: 546, 5: 273}` | Canonical dense benchmark scheduler |
| dense_fixed_finest_final | 62.0 | `{5: 2310}` | Primary fixed-finest answer |
| dense_fixed_l4_final | 58.0 | `{4: 2280}` | Level 4 alone is usable |
| dense_fixed_l3_final | 58.0 | `{3: 2400}` | Level 3 alone is usable |
| dense_linear_l4_to_l5_final | 62.0 | `{4: 1170, 5: 1170}` | High-only schedule recovers fixed-finest |
| dense_fixed_finest_step31000 | 18.0 | `{5: 2790}` | Earlier snapshot is worse |
| dense_fixed_finest_interim_epoch1 | 44.0 | `{5: 2610}` | Interim is worse than final |

Episode-level comparison:

- Canonical dense linear succeeds on `16/50`.
- Dense fixed finest succeeds on `31/50`.
- Dense `l4 -> l5` succeeds on `31/50`.
- Fixed finest gains 21 episodes relative to canonical linear and loses 6, for a net gain of 15 episodes.
- `l4 -> l5` gains 24 episodes relative to canonical linear and loses 9, also net +15.

## Answers

### 1. What is dense fixed-finest performance on Reacher?

Dense fixed-finest final checkpoint performance is **62.0%** on the 50-episode Reacher manifest.

This is a 30 point improvement over the canonical dense linear scheduler result of 32.0%.

### 2. If fixed-finest is good, what scheduler behavior causes the benchmark drop?

Fixed-finest is good enough to identify scheduler behavior as the immediate benchmark-drop cause.

The canonical dense schedule is `linear_cem` from `coarsest` to `finest`. With `n_iter: 30`, it spends only three CEM cost calls per solve at level 0, six each at levels 1-4, and three at level 5. Aggregated over the benchmark this is:

```text
{0: 273, 1: 546, 2: 546, 3: 546, 4: 546, 5: 273}
```

Those low-fidelity early CEM iterations shape the action distribution before the high-fidelity levels see candidates. Reacher appears sensitive to that early coarse shaping.

Evidence:

- `fixed finest`: 62.0
- `fixed level 4`: 58.0
- `fixed level 3`: 58.0
- `linear level 4 -> level 5`: 62.0
- `linear level 0 -> level 5`: 32.0

So the problem is not merely "not using finest." It is specifically the canonical schedule's inclusion of the very coarse levels, especially levels 0-2, during early CEM refinement.

### 3. If fixed-finest is bad, is dense undertrained, stale, checkpoint-selected poorly, or configured wrong?

Fixed-finest is not bad relative to the same-run identity control: dense fixed-finest is 62.0 and identity fixed-finest is 60.0.

The final checkpoint is also not obviously stale or poorly selected among inspected dense artifacts:

- epoch 1 fixed finest: 44.0
- step-31k/epoch 2 fixed finest: 18.0
- final epoch 9 fixed finest: 62.0

This progression argues against exporting the wrong old checkpoint as the primary issue.

The config also looks aligned: all inspected Reacher checkpoints use qpos-match restore, action dim 10, and the expected Reacher env id.

### 4. Does dense Reacher need longer train time, best-checkpoint export, different loss weighting, or a Reacher-specific planner schedule?

For the observed benchmark drop from 32.0 to the dense model's available fixed-finest performance, Reacher needs a **Reacher-specific planner schedule** first.

Recommended priority:

1. Use a high-fidelity dense Reacher schedule, starting at level 4 and ending at level 5.
2. Repeat the identity control because this run produced 60.0 while an earlier artifact reported 86.0 for the same named identity checkpoint.
3. Only after the schedule/control repeatability issue is settled, consider longer training or loss weighting to close the remaining gap to upstream 86.0.

Best-checkpoint export is not the first lever based on the inspected snapshots, because the final checkpoint is better than the earlier dense checkpoints under fixed-finest planning.

### 5. What concrete config should run next?

Run:

```text
configs/research/dense_reacher_high_fidelity_schedule.yaml
```

This config focuses on the important rows only:

- upstream fixed finest
- retrained identity fixed finest
- dense canonical `level 0 -> level 5`
- dense fixed finest
- dense high-only `level 4 -> level 5`

### 6. What exact next experiment should be run?

Submit:

```bash
sbatch --export=ALL,MWM_REACHER_CONFIG=configs/research/dense_reacher_high_fidelity_schedule.yaml scripts/research/research_dense_reacher_debug.sbatch
```

If that reproduces the same pattern, promote the Reacher dense benchmark scheduler from `coarsest -> finest` to `4 -> finest`, then run a multi-seed validation before making it canonical.

## Caveats And Blockers

- Identity control repeatability needs follow-up. Earlier recorded identity parity was 86.0, but this matrix reran `checkpoints_mwm/retrained_lewm_identity_reacher_upstream` at 60.0 on the same manifest hash. The dense fixed-finest result is still interpretable because it tied this same-run identity control, but the absolute identity number should be repeated.
- The benchmark uses 50 episodes. A 15 episode swing is large enough to trust directionally, but multi-seed or larger-episode confirmation is still needed before canonical benchmark changes.
- EGL emitted `/dev/dri` permission warnings in stderr. The job completed successfully and wrote valid benchmark artifacts, so these warnings were not fatal.

## Verification

Passed before GPU work:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m py_compile $(rg --files -g '*.py')
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_local_workflow.py tests/test_mwm_repo_hygiene.py tests/test_mwm_core.py tests/test_mwm_artifacts.py
```

Result:

```text
101 passed, 14 subtests passed
```

Passed after adding research configs/scripts:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/research/dense_reacher_planner_ablation.yaml --static-only
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/research/dense_reacher_planner_ablation.yaml
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/research/dense_reacher_high_fidelity_schedule.yaml --static-only
bash -n scripts/research/research_dense_reacher_debug.sbatch
```
